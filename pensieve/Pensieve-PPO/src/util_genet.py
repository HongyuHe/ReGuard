import fixed_env as env
from itertools import product
import numpy as np

K_steps = 5


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and delay), chunk_til_video_end
S_LEN = 6  # take how many frames in the past
A_DIM = 3
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './mytest_results/log_sim_ppo'
TEST_TRACES = './test/'

def calculate_from_selection(selected ,last_bit_rate):
    # selected_action is 0-5
    # naive step implementation
    if selected == 1:
        bit_rate = last_bit_rate
    elif selected == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    if bit_rate < 0:
        bit_rate = 0
    if bit_rate > 5:
        bit_rate = 5

    # print(bit_rate)
    return bit_rate

def calculate_reward(bit_rate, last_bit_rate, rebuf):
    """Calculate reward for a given action"""
    reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                        VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
    return reward

def copy_env(env_original, RANDOM_SEED=42):
    """
    Create a deep copy of the environment to simulate different action sequences.
    """
    # Create a new environment with the same traces
    env_copy = env.Environment(
        all_cooked_time=env_original.all_cooked_time,
        all_cooked_bw=env_original.all_cooked_bw,
        random_seed=RANDOM_SEED
    )
    
    # Copy all the state variables
    env_copy.video_chunk_counter = env_original.video_chunk_counter
    env_copy.buffer_size = env_original.buffer_size
    env_copy.trace_idx = env_original.trace_idx
    env_copy.cooked_time = env_original.cooked_time
    env_copy.cooked_bw = env_original.cooked_bw
    env_copy.mahimahi_start_ptr = env_original.mahimahi_start_ptr
    env_copy.mahimahi_ptr = env_original.mahimahi_ptr
    env_copy.last_mahimahi_time = env_original.last_mahimahi_time
    env_copy.video_size = env_original.video_size  # This is read-only, so can share
    
    return env_copy

def find_optimal_k_step_reward(env_snapshot, initial_bit_rate, k_steps):
    """
    Find the optimal reward by trying all bitrate combinations over k steps.
    Starting from the saved environment state, try all possible combinations.
    
    Args:
        env_snapshot: Environment state from k steps ago
        initial_bit_rate: The bitrate from before the first step
        k_steps: Number of steps to look ahead
    
    Returns:
        best_reward: The maximum cumulative reward over k steps
        best_sequence: The bitrate sequence that achieves the best reward
    """
    # Generate all possible bitrate combinations for k steps
    all_combinations = list(product(range(A_DIM), repeat=k_steps))
    
    best_reward = float('-inf')
    best_sequence = None
    
    worst_reward = float('inf')
    worst_sequence = None
    
    # Try each combination
    for idx, combination in enumerate(all_combinations):
        
        # Create a copy of the environment to simulate
        temp_env = copy_env(env_snapshot)
        
        total_reward = 0
        last_br = initial_bit_rate
        
        valid_sequence = True
        for step_idx, action in enumerate(combination):
            # Simulate the action
            bit_rate = calculate_from_selection(action, last_br)
            try:
                delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                    temp_env.get_video_chunk(bit_rate)
            except Exception as e:
                print(f"Error simulating step {step_idx}: {e}")
                valid_sequence = False
                break
            
            # Calculate reward for this step
            reward = calculate_reward(bit_rate, last_br, rebuf)
            total_reward += reward
            
            last_br = bit_rate
            
            # If video ends before k steps, this sequence is invalid for comparison
            if end_of_video and step_idx < k_steps - 1:
                valid_sequence = False
                break
        
        # Update best if this is better
        if valid_sequence and total_reward > best_reward:
            best_reward = total_reward
            best_sequence = combination
            
        if valid_sequence and total_reward < worst_reward:
            worst_reward = total_reward
            worst_sequence = combination
    
    # print(f"Optimal search complete. Best reward: {best_reward:.4f}")
    return best_reward, best_sequence, worst_reward, worst_sequence
