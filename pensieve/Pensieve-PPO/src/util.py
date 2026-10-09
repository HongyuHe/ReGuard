import fixed_env as env
from itertools import product
import numpy as np

K_steps = 5


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and delay), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
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
    # NOTE: We avoid calling `env.Environment(...)` because it reloads `video_size_*`
    # from disk each time. For oracle search this would dominate runtime.
    env_copy = object.__new__(env.Environment)

    # Shared references (read-only during simulation).
    env_copy.all_cooked_time = env_original.all_cooked_time
    env_copy.all_cooked_bw = env_original.all_cooked_bw
    env_copy.video_size = env_original.video_size

    # Runtime state.
    env_copy.video_chunk_counter = env_original.video_chunk_counter
    env_copy.buffer_size = env_original.buffer_size
    env_copy.trace_idx = env_original.trace_idx
    env_copy.cooked_time = env_original.cooked_time
    env_copy.cooked_bw = env_original.cooked_bw
    env_copy.mahimahi_start_ptr = env_original.mahimahi_start_ptr
    env_copy.mahimahi_ptr = env_original.mahimahi_ptr
    env_copy.last_mahimahi_time = env_original.last_mahimahi_time

    return env_copy


def find_optimal_k_step_reward_oracle(env_snapshot, initial_bit_rate, k_steps):
    """
    Exact K-step oracle via prefix-reuse search (much faster than enumerating full sequences).

    Returns:
        best_reward, best_sequence, worst_reward, worst_sequence
    """
    k = int(k_steps)
    if k <= 0:
        empty = tuple()
        return 0.0, empty, 0.0, empty

    # Safe per-step upper bound (ignores penalties).
    max_step_reward = float(max(VIDEO_BIT_RATE)) / float(M_IN_K)

    best_reward = float("-inf")
    best_sequence = None
    worst_reward = float("inf")
    worst_sequence = None

    def dfs(depth, env_state, last_br, total_reward, seq):
        nonlocal best_reward, best_sequence, worst_reward, worst_sequence

        if depth >= k:
            if total_reward > best_reward:
                best_reward = float(total_reward)
                best_sequence = tuple(seq)
            if total_reward < worst_reward:
                worst_reward = float(total_reward)
                worst_sequence = tuple(seq)
            return

        # Branch-and-bound for the maximum (exact): if even the best-case remaining reward
        # can't beat the current best, stop exploring this prefix.
        remaining = k - depth
        if best_sequence is not None and total_reward + remaining * max_step_reward <= best_reward:
            return

        for bit_rate in range(A_DIM):
            env_next = copy_env(env_state)
            try:
                _, _, _, rebuf, _, _, end_of_video, _ = env_next.get_video_chunk(bit_rate)
            except Exception:
                continue

            # Match the "valid sequence" rule in the original oracle: if the video ends
            # before reaching K steps, skip this branch.
            if end_of_video and depth < k - 1:
                continue

            step_reward = calculate_reward(bit_rate, last_br, rebuf)
            seq.append(bit_rate)
            dfs(depth + 1, env_next, bit_rate, total_reward + step_reward, seq)
            seq.pop()

    dfs(0, env_snapshot, int(initial_bit_rate), 0.0, [])
    if best_sequence is None or worst_sequence is None:
        return None, None, None, None
    return best_reward, best_sequence, worst_reward, worst_sequence


def find_optimal_k_step_reward_oracle_mpc(env_snapshot, initial_bit_rate, k_steps, horizon=1):
    """
    Approximate K-step oracle using the simulator, but much cheaper than the exact search.

    - `horizon=1` reduces to greedy 1-step lookahead.
    - `horizon>1` uses a receding-horizon (MPC-style) planner: at each step, pick the first
      action of the best/worst `horizon`-step sequence, then re-plan from the next state.

    Returns:
        best_reward, best_sequence, worst_reward, worst_sequence
        (or (None, None, None, None) if no valid K-step sequence exists)
    """
    k = int(k_steps)
    if k <= 0:
        empty = tuple()
        return 0.0, empty, 0.0, empty

    h = int(horizon)
    if h <= 0:
        h = 1

    # If the remaining video is shorter than K, there is no valid sequence under the
    # "do not end the video before finishing K steps" comparison rule.
    try:
        total_chunks = int(getattr(env, "TOTAL_VIDEO_CHUNCK", 48))
        start_counter = int(getattr(env_snapshot, "video_chunk_counter", 0))
        if start_counter + k > total_chunks:
            return None, None, None, None
    except Exception:
        pass

    def search_return(env_state, last_br, depth_taken, horizon_remaining, overall_remaining, mode):
        if horizon_remaining <= 0:
            return 0.0

        best_val = None
        for bit_rate in range(A_DIM):
            env_next = copy_env(env_state)
            try:
                _, _, _, rebuf, _, _, end_of_video, _ = env_next.get_video_chunk(bit_rate)
            except Exception:
                continue

            next_depth = int(depth_taken) + 1
            if end_of_video and next_depth < int(overall_remaining):
                continue

            step_reward = calculate_reward(bit_rate, int(last_br), rebuf)
            future = search_return(
                env_next,
                bit_rate,
                next_depth,
                int(horizon_remaining) - 1,
                int(overall_remaining),
                mode,
            )
            if future is None:
                continue

            total = float(step_reward) + float(future)
            if best_val is None:
                best_val = total
                continue
            if mode == "best":
                if total > best_val:
                    best_val = total
            else:
                if total < best_val:
                    best_val = total
        return best_val

    def rollout(mode):
        temp_env = copy_env(env_snapshot)
        total_reward = 0.0
        last_br = int(initial_bit_rate)
        seq = []

        for step_idx in range(k):
            remaining = int(k - step_idx)
            lookahead = int(min(h, remaining))

            chosen = None
            chosen_score = None
            chosen_env = None
            chosen_step_reward = None

            for bit_rate in range(A_DIM):
                env_next = copy_env(temp_env)
                try:
                    _, _, _, rebuf, _, _, end_of_video, _ = env_next.get_video_chunk(bit_rate)
                except Exception:
                    continue

                # Same validity rule as the exact oracle: end-of-video before the final step
                # makes this sequence invalid for K-step comparison.
                if end_of_video and remaining > 1:
                    continue

                step_reward = calculate_reward(bit_rate, last_br, rebuf)
                if lookahead > 1:
                    future = search_return(
                        env_next,
                        bit_rate,
                        1,
                        lookahead - 1,
                        remaining,
                        mode,
                    )
                    if future is None:
                        continue
                    score = float(step_reward) + float(future)
                else:
                    score = float(step_reward)

                if chosen is None:
                    chosen = bit_rate
                    chosen_score = score
                    chosen_env = env_next
                    chosen_step_reward = float(step_reward)
                    continue

                if mode == "best":
                    if score > chosen_score:
                        chosen = bit_rate
                        chosen_score = score
                        chosen_env = env_next
                        chosen_step_reward = float(step_reward)
                else:
                    if score < chosen_score:
                        chosen = bit_rate
                        chosen_score = score
                        chosen_env = env_next
                        chosen_step_reward = float(step_reward)

            if chosen is None:
                return None, None

            total_reward += float(chosen_step_reward)
            seq.append(int(chosen))
            temp_env = chosen_env
            last_br = int(chosen)

        return float(total_reward), tuple(seq)

    best_reward, best_sequence = rollout("best")
    if best_sequence is None:
        return None, None, None, None
    worst_reward, worst_sequence = rollout("worst")
    if worst_sequence is None:
        return None, None, None, None
    return float(best_reward), best_sequence, float(worst_reward), worst_sequence

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
        for step_idx, bit_rate in enumerate(combination):
            # Simulate the action
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
