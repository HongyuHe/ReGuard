import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import load_trace
#import a2c as network
import ppo2 as network
import fixed_env as env
from util import *
from pgd import PGDAttack
from smt import synthesize_bandwidth_trace
from fixed_env import BUFFER_THRESH, MILLISECONDS_IN_SECOND

# NN_MODEL = sys.argv[1]
# NN_MODEL = "pretrain/nn_model_ep_155400.pth"
NN_MODEL = "ppo/nn_model_ep_426300.pth"


def generate_adv_chunk(video_chunk_size, initial_buffer_ms, target_buffer_norm,
                        target_throughput_norm, target_delay_norm,
                        original_buffer_norm, original_throughput_norm, original_delay_norm,
                        duration):
    # load the trace
    bandwidth_trace, time_trace = synthesize_bandwidth_trace(
        video_chunk_size=int(video_chunk_size),
        initial_buffer_ms=float(initial_buffer_ms),
        target_buffer_norm=float(target_buffer_norm),
        target_throughput_norm=float(target_throughput_norm),
        target_delay_norm=float(target_delay_norm),
        num_bw_samples=20,
    )
    
    return bandwidth_trace, time_trace

def get_necessary_info(step_idx, all_cooked_time, all_cooked_bw):
    
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)
    
    actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
        learning_rate=ACTOR_LR_RATE)
    
    actor.load_model(NN_MODEL)
    
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    perturbed_bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1
    
    index = 0

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    while True:  # serve video forever
        prev_buffer_size = net_env.buffer_size
        prev_time_stamp = time_stamp
        
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)
            
        if end_of_video:
            return None
    
        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms
        after_time_stamp = time_stamp
        
        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)
            
        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # throughput (MB/s = Mbps/8)
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        
        if index == step_idx:
            # PGD Attack to find perturbed buffer_size and delay
            pgd_attacker = PGDAttack(actor, num_iterations=5)
            perturbed_state = pgd_attacker.pgd_attack(
                state,
                target_bitrate=bit_rate,
                BUFFER_NORM_FACTOR=BUFFER_THRESH / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR,
                M_IN_K=M_IN_K,
                video_chunk_size=video_chunk_size,
                oracle_env_snapshot=net_env,
                oracle_k_steps=K_steps,
            )
            perturbed_action_prob = actor.predict(np.reshape(perturbed_state, (1, S_INFO, S_LEN)))

            prev_perturbed_bit_rate = perturbed_bit_rate
            prev_video_chunk_counter = net_env.video_chunk_counter - 1
            perturbed_bit_rate = np.argmax(np.log(perturbed_action_prob))
        
            
            returned_video_chunk_size = video_chunk_size
            returned_buffer_size = prev_buffer_size
            
            returned_buffer_norm = perturbed_state[1, -1]
            returned_throughput_norm = perturbed_state[2, -1]
            returned_delay_norm = perturbed_state[3, -1]

            
            replaced_time_stamp_start = prev_time_stamp
            replaced_time_stamp_end = after_time_stamp
            
            return {
                "video_chunk_size": returned_video_chunk_size,
                "buffer_size": returned_buffer_size,
                "buffer_norm": returned_buffer_norm,
                "throughput_norm": returned_throughput_norm,
                "delay_norm": returned_delay_norm,
                "original_buffer_norm": state[1, -1],
                "original_throughput_norm": state[2, -1],
                "original_delay_norm": state[3, -1],
                "time_stamp_start": replaced_time_stamp_start,
                "time_stamp_end": replaced_time_stamp_end
            }
            
        index += 1

all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace("./test1/")
benign_all_cooked_time, benign_all_cooked_bw, benign_all_file_names = all_cooked_time[:1], all_cooked_bw[:1], all_file_names[:1]


# I need to know which chunk to replace

modified_all_cooked_time = benign_all_cooked_time.copy()
modified_all_cooked_bw = benign_all_cooked_bw.copy()

step_idx = 0

while True:
    print(f"Generating adversarial chunk for step {step_idx}...")

    necessary_info = get_necessary_info(step_idx=step_idx, all_cooked_time=modified_all_cooked_time, all_cooked_bw=modified_all_cooked_bw)

    if necessary_info == None:
        break
    adv_cooked_bw_chunk, adv_cooked_time_chunk = generate_adv_chunk(
        video_chunk_size=necessary_info["video_chunk_size"],
        initial_buffer_ms=necessary_info["buffer_size"],
        target_buffer_norm=necessary_info["buffer_norm"],
        target_throughput_norm=necessary_info["throughput_norm"],
        target_delay_norm=necessary_info["delay_norm"],
        original_buffer_norm=necessary_info["original_buffer_norm"],
        original_throughput_norm=necessary_info["original_throughput_norm"],
        original_delay_norm=necessary_info["original_delay_norm"],
        duration = (necessary_info["time_stamp_end"] - necessary_info["time_stamp_start"]) / 1000
    )

    if adv_cooked_bw_chunk != None:
        # find the time
        removed_idx = [i for i, t in enumerate(modified_all_cooked_time[0]) if t >= necessary_info["time_stamp_start"] / 1000 and t < necessary_info["time_stamp_end"] / 1000]

        modified_interarrival = np.diff([0] + list(modified_all_cooked_time[0]))
        if len(removed_idx) > 0:
            modified_bw_chunk1 = modified_all_cooked_bw[0][:removed_idx[0]]
            
            modified_interarrival_chunk1 = modified_interarrival[:removed_idx[0]]

            modified_bw_chunk2 = modified_all_cooked_bw[0][removed_idx[-1]+1:]
            modified_interarrival_chunk2 = modified_interarrival[removed_idx[-1]+1:]
        else:
            split_idx = next(i for i, t in enumerate(modified_all_cooked_time[0]) if t >= necessary_info["time_stamp_end"] / 1000)
            modified_bw_chunk1 = modified_all_cooked_bw[0][:split_idx]
            modified_interarrival_chunk1 = modified_interarrival[:split_idx]
            
            modified_bw_chunk2 = modified_all_cooked_bw[0][split_idx:]
            modified_interarrival_chunk2 = modified_interarrival[split_idx:]

        added_bw_chunk = adv_cooked_bw_chunk
        added_time_chunk = adv_cooked_time_chunk
        
        if (step_idx == 0):
            added_bw_chunk = np.concatenate(([0], adv_cooked_bw_chunk))
            added_time_chunk = np.concatenate(([0], adv_cooked_time_chunk))

        modified_all_cooked_bw[0] = np.concatenate([modified_bw_chunk1, added_bw_chunk, modified_bw_chunk2]).tolist()
        modified_interarrival = np.concatenate([modified_interarrival_chunk1, added_time_chunk, modified_interarrival_chunk2])
        modified_all_cooked_time[0] = np.cumsum(modified_interarrival).tolist()
        
    else:
        print("Failed to generate adversarial bandwidth trace.")
        
        
    step_idx += 1 
        
        
    with open("adv_trace/norway_bus_14", "w") as f:
        for t, bw in zip(modified_all_cooked_time[0], modified_all_cooked_bw[0]):
            f.write(f"{t:.4f}\t{bw:.4f}\n")
