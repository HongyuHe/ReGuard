import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import load_trace
#import a2c as network
# import ppo2 as network
import a3c_genet as network
import fixed_env as env
from util_genet import *
from pgd import PGDAttack

# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size delay reward
# NN_MODEL = sys.argv[1]
# NN_MODEL = "pretrain/nn_model_ep_155400.pth"
# NN_MODEL = "ppo/nn_model_ep_426300.pth"
NN_MODEL = "genet/nn_model_ep_9900_torch.pth"

    
def main():

    np.random.seed(RANDOM_SEED)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace("./test1/")
    # all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace("./adv_trace_1/")
    
    all_cooked_time, all_cooked_bw, all_file_names = all_cooked_time[:1], all_cooked_bw[:1], all_file_names[:1]
    
    # np.random.shuffle(all_cooked_bw[0])
    # # randomly change a bit on all_cooked_bw[0]
    # all_cooked_bw[0] = all_cooked_bw[0] * np.random.uniform(0.5, 1.5, size=len(all_cooked_bw[0]))

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    BITRATE_DIM = len(VIDEO_BIT_RATE)
    actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
        learning_rate=ACTOR_LR_RATE, bitrate_dim=BITRATE_DIM)

    # restore neural net parameters
    if NN_MODEL is not None:  # NN_MODEL is the path to file
        # actor.load_model(NN_MODEL)
        actor.load_model_from_tf_converted(NN_MODEL)
        print("Testing model restored.")

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    perturbed_bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    index = 0
    performance_gap1 = []
    performance_gap2 = []
    k_step_reward = []
    perturbed_k_step_reward = []
    k_step_env_snapshots = []  # Store environment snapshots
    k_step_bitrates = []  # Store bitrates before each step
    
    while True:  # serve video forever
        env_snapshot = copy_env(net_env, RANDOM_SEED)
        k_step_env_snapshots.append(env_snapshot)
        k_step_bitrates.append(last_bit_rate)
                     
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                            VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        r_batch.append(reward)
        k_step_reward.append(reward)
        
        adv_env_copy = copy_env(net_env, RANDOM_SEED)
        adv_out = adv_env_copy.get_video_chunk(perturbed_bit_rate)
        _, _, _, adv_rebuf, _, _, _, _ = adv_out
        adv_reward = calculate_reward(perturbed_bit_rate, last_bit_rate, adv_rebuf)
        perturbed_k_step_reward.append(adv_reward)
        
        if len(k_step_reward) == K_steps:
            avg_k_reward = np.sum(k_step_reward) / K_steps
            avg_perturbed_k_reward = np.sum(perturbed_k_step_reward) / K_steps
            
            # Get the environment from K steps ago (first snapshot in the list)
            env_k_steps_ago = k_step_env_snapshots[0]
            bitrate_before_window = k_step_bitrates[0]
            
            optimal_reward, optimal_sequence, worst_reward, worst_sequence = find_optimal_k_step_reward(
                env_k_steps_ago, bitrate_before_window, K_steps
            )
            avg_optimal_reward = optimal_reward / K_steps
            avg_worst_reward = worst_reward / K_steps
            
            performance_gap1.append((avg_optimal_reward - avg_k_reward) / (avg_optimal_reward - avg_worst_reward + 1e-6))
            performance_gap2.append((avg_optimal_reward - avg_k_reward) / abs(avg_optimal_reward))
            # print(f"K-step avg reward: {avg_k_reward}, optimal avg reward: {avg_optimal_reward}, Worst avg reward: {avg_worst_reward}, Perturbed avg reward: {avg_perturbed_k_reward}")
            print(f"K-step avg reward: {avg_k_reward}, optimal avg reward: {avg_optimal_reward}, Worst avg reward: {avg_worst_reward}, xx: {np.mean(performance_gap2)}")

            
            perturbed_k_step_reward.pop(0)
            k_step_reward.pop(0)
            k_step_env_snapshots.pop(0)
            k_step_bitrates.pop(0)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                        str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(video_chunk_size) + '\t' +
                        str(delay) + '\t' +
                        str(entropy_) + '\t' + 
                        str(reward) + '\n')
        log_file.flush()

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
        state[4, :BITRATE_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))        
        noise = np.random.gumbel(size=len(action_prob))
        # bit_rate = np.argmax(np.log(action_prob) + noise)
        action_cumsum = np.cumsum( action_prob )
        selection = (action_cumsum > np.random.randint(
            1 ,RAND_RANGE ) / float( RAND_RANGE )).argmax()
        bit_rate = calculate_from_selection( selection ,last_bit_rate )
        
        s_batch.append(state)
        entropy_ = -np.dot(action_prob, np.log(action_prob))
        entropy_record.append(entropy_)
        
        index += 1

        if end_of_video:
            print("performance gap: ", np.mean(performance_gap1), np.mean(performance_gap2))
            print("reward avg: " + str(np.mean(r_batch)))
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            # print(np.mean(entropy_record))
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
