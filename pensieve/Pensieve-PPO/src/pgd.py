import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import torch
import copy
from functools import reduce
from typing import Optional, Sequence
from decimal import Decimal
import itertools
import z3
from util import *
from fixed_env import *

class PGDAttack:
    
    def __init__(self, actor_network, 
                 alpha=0.001, num_iterations=10):
        self.actor = actor_network
        self.alpha = alpha
        self.num_iterations = num_iterations
        self._last_video_chunk_size = None
        self._objective_id = 3
        self._oracle_best_reward = None
        self._oracle_worst_reward = None
        self._oracle_k_steps = 0
        self._oracle_mpc_horizon = 1
        self._history_attack_enabled = False
        self._surrogate_k_steps = 0
        self._surrogate_chunk_sizes_mb = None
        self._surrogate_oracle_tau = 0.1
        self._surrogate_oracle_max_sequences = 50000
        self._surrogate_oracle_seq_cache = {}
        self._baseline_state_tensor = None
        self._perturb_reg_weight = 0.0
        self._action_gap_target_action = None
        self._action_gap_target_gap = None
        self._action_gap_oracle_best_mean = None
        self._decision_flip_clean_action = None
        self._rules_file = None
        self._rules_solver = None
        self._rules_vars = None
        self._rules_cache = {}
        self._rules_projection_calls_total = 0
        self._rules_fully_relaxed_calls_total = 0
        self._rules_projection_calls_last_attack = 0
        self._rules_fully_relaxed_calls_last_attack = 0
        self._rules_projection_success_total = 0
        self._rules_projection_fail_total = 0
        self._rules_projection_fallback_total = 0
        self._rules_projection_last_attack_stats = {
            "pgd/rules_projection_attempts": 0,
            "pgd/rules_projection_success": 0,
            "pgd/rules_projection_fail": 0,
            "pgd/rules_projection_success_rate": 0.0,
            "pgd/rules_projection_window_attempts": 0,
            "pgd/rules_projection_window_success": 0,
            "pgd/rules_projection_window_fail": 0,
            "pgd/rules_projection_latest_attempts": 0,
            "pgd/rules_projection_latest_success": 0,
            "pgd/rules_projection_latest_fail": 0,
            "pgd/rules_projection_fallbacks": 0,
        }
        self._history_sparse_last_attack = {
            "pgd/history_sparse_applied": 0,
            "pgd/history_sparse_k": 0,
            "pgd/history_sparse_k_unique": 0,
            "pgd/history_sparse_candidates_total": 0,
            "pgd/history_sparse_selected_delay": 0,
            "pgd/history_sparse_selected_throughput": 0,
            "pgd/history_sparse_full_loss": float("nan"),
            "pgd/history_sparse_loss": float("nan"),
            "pgd/history_sparse_band": float("nan"),
        }
        self._early_stop_last_attack = {
            "pgd/early_stop_enabled": 0,
            "pgd/early_stop_triggered": 0,
            "pgd/early_stop_iter": 0,
            "pgd/early_stop_iters_run": 0,
            "pgd/early_stop_patience": 0,
            "pgd/early_stop_margin": float("nan"),
            "pgd/early_stop_gap": float("nan"),
            "pgd/early_stop_target_action": -1,
        }
        self._last_loss_value = float("nan")

    def get_rules_projection_totals(self) -> tuple[int, int]:
        return int(self._rules_projection_calls_total), int(self._rules_fully_relaxed_calls_total)

    def get_rules_projection_last_attack(self) -> tuple[int, int]:
        return int(self._rules_projection_calls_last_attack), int(self._rules_fully_relaxed_calls_last_attack)

    def get_rules_projection_success_totals(self) -> tuple[int, int, int]:
        return (
            int(self._rules_projection_success_total),
            int(self._rules_projection_fail_total),
            int(self._rules_projection_fallback_total),
        )

    def get_rules_projection_last_attack_stats(self) -> dict[str, float | int]:
        return dict(self._rules_projection_last_attack_stats)

    def get_history_sparse_last_attack(self) -> dict[str, float | int]:
        return dict(self._history_sparse_last_attack)

    def get_early_stop_last_attack(self) -> dict[str, float | int]:
        return dict(self._early_stop_last_attack)

    def _prepare_k_step_chunk_sizes_mb(
        self, oracle_env_snapshot, *, k_steps: int, num_actions: int
    ) -> Optional[torch.Tensor]:
        """
        Pre-compute per-step chunk sizes (MB) for a K-step differentiable surrogate rollout.

        Returns:
            Tensor shape (H, A), where H <= k_steps due to end-of-video, or None on failure.
        """
        k = int(k_steps)
        a_dim = int(num_actions)
        if k <= 0 or a_dim <= 0 or oracle_env_snapshot is None:
            return None

        try:
            start_idx = int(getattr(oracle_env_snapshot, "video_chunk_counter"))
            video_size = getattr(oracle_env_snapshot, "video_size")
        except Exception:
            return None

        try:
            max_chunks = int(len(video_size[0]))
        except Exception:
            try:
                max_chunks = int(len(next(iter(video_size.values()))))
            except Exception:
                return None

        horizon = int(max(0, min(k, max_chunks - start_idx)))
        if horizon <= 0:
            return None

        chunk_sizes_mb = np.zeros((horizon, a_dim), dtype=np.float32)
        for t in range(horizon):
            idx = start_idx + t
            for a in range(a_dim):
                try:
                    chunk_sizes_mb[t, a] = float(video_size[a][idx]) / float(M_IN_K) / float(M_IN_K)
                except Exception:
                    chunk_sizes_mb[t, a] = 0.0
        chunk_sizes_mb[~np.isfinite(chunk_sizes_mb)] = 0.0
        return torch.tensor(chunk_sizes_mb, dtype=torch.float32)

    def _expected_k_step_return_surrogate(
        self,
        state_tensor: torch.Tensor,
        *,
        chunk_sizes_mb: torch.Tensor,
        k_steps: int,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Differentiable K-step expected QoE return under the policy using a lightweight
        environment surrogate.

        Key idea:
        - For each step, estimate an RTT-free link rate from the *full* (throughput, delay)
          history window, then predict per-action delay and rebuffer.
        - Roll the state window forward using expected next-state features.

        Returns:
            Tensor shape (batch,) of expected cumulative reward over H steps.
        """
        if chunk_sizes_mb is None:
            raise ValueError("* chunk_sizes_mb must be provided")

        k = int(k_steps)
        if k <= 0:
            return state_tensor.new_zeros((state_tensor.shape[0],))

        # Clamp horizon to what we actually have precomputed.
        horizon = int(min(k, int(chunk_sizes_mb.shape[0])))
        if horizon <= 0:
            return state_tensor.new_zeros((state_tensor.shape[0],))

        batch = int(state_tensor.shape[0])
        s_len = int(state_tensor.shape[2])
        device = state_tensor.device
        dtype = state_tensor.dtype

        # Constants (all in seconds/MB/Mbps-like units, matching the existing surrogate).
        rtt_s = float(LINK_RTT) / float(MILLISECONDS_IN_SECOND)
        chunk_len_s = float(VIDEO_CHUNCK_LEN) / float(MILLISECONDS_IN_SECOND)
        buffer_thresh_s = float(BUFFER_THRESH) / float(MILLISECONDS_IN_SECOND)

        num_actions = int(chunk_sizes_mb.shape[1])
        br_kbps = torch.tensor(
            VIDEO_BIT_RATE[:num_actions], dtype=dtype, device=device
        )  # (A,)
        max_br = float(np.max(VIDEO_BIT_RATE))

        # Rolling state for the surrogate unroll.
        s = state_tensor

        # Persistent scalars used by reward dynamics.
        buffer_s = torch.clamp(s[:, 1, -1], min=0.0) * float(BUFFER_NORM_FACTOR)  # (B,)
        last_br_kbps = torch.clamp(s[:, 0, -1], min=0.0) * float(max_br)  # (B,)

        total_reward = state_tensor.new_zeros((batch,))

        # Start from the current "chunks remaining" feature and decrement each step.
        remain_norm = torch.clamp(s[:, 5, -1], min=0.0)
        step_remain_delta = float(1.0 / float(CHUNK_TIL_VIDEO_END_CAP))

        for t in range(horizon):
            # Policy at this step.
            action_prob = self.actor.predict_tensor(s)  # (B, A)

            # Estimate RTT-free link rate (MB/s) from the full history window.
            delay_s_hist = torch.clamp(s[:, 3, :], min=0.0) * float(BUFFER_NORM_FACTOR)  # (B, S_LEN)
            chunk_mb_hist = torch.clamp(s[:, 2, :], min=0.0) * delay_s_hist  # (B, S_LEN)
            rtt_free_delay_s_hist = torch.clamp(delay_s_hist - float(rtt_s), min=float(eps))
            link_mb_s_hist = chunk_mb_hist / rtt_free_delay_s_hist  # (B, S_LEN)
            link_mb_s = torch.clamp(link_mb_s_hist.mean(dim=-1, keepdim=True), min=float(eps))  # (B, 1)

            # Per-action chunk size (MB) for this step.
            chunk_mb = (
                chunk_sizes_mb[t].to(device=device, dtype=dtype).view(1, -1).expand(batch, -1)
            )  # (B, A)

            # Predict per-action delay (s): (chunk / link_rate) + RTT.
            pred_delay_s = chunk_mb / link_mb_s + float(rtt_s)  # (B, A)
            pred_rebuf_s = torch.relu(pred_delay_s - buffer_s.view(-1, 1))  # (B, A)

            quality_mbps = (br_kbps / float(M_IN_K)).view(1, -1)  # (1, A)
            smoothness_mbps = (
                torch.abs(br_kbps.view(1, -1) - last_br_kbps.view(-1, 1)) / float(M_IN_K)
            )  # (B, A)

            expected_quality = (action_prob * quality_mbps).sum(dim=-1)  # (B,)
            expected_rebuf = (action_prob * pred_rebuf_s).sum(dim=-1)  # (B,)
            expected_smooth = (action_prob * smoothness_mbps).sum(dim=-1)  # (B,)

            step_reward = (
                expected_quality
                - float(REBUF_PENALTY) * expected_rebuf
                - float(SMOOTH_PENALTY) * expected_smooth
            )
            total_reward = total_reward + step_reward

            # Expected buffer transition (delay affects buffer; sleep clamps to BUFFER_THRESH).
            buffer_after_delay = torch.relu(buffer_s.view(-1, 1) - pred_delay_s)  # (B, A)
            buffer_after_add = buffer_after_delay + float(chunk_len_s)  # (B, A)
            buffer_next = torch.minimum(buffer_after_add, state_tensor.new_tensor(float(buffer_thresh_s)))  # (B, A)
            buffer_s = (action_prob * buffer_next).sum(dim=-1)  # (B,)

            # Update last bitrate (soft expected bitrate).
            last_br_kbps = (action_prob * br_kbps.view(1, -1)).sum(dim=-1)  # (B,)

            # Expected next-state throughput/delay features for this downloaded chunk.
            expected_delay_s = (action_prob * pred_delay_s).sum(dim=-1)  # (B,)
            next_delay_norm = expected_delay_s / float(BUFFER_NORM_FACTOR)  # delay_s / 10

            throughput_norm_per_action = chunk_mb / torch.clamp(pred_delay_s, min=float(eps))  # (B, A)
            next_throughput_norm = (action_prob * throughput_norm_per_action).sum(dim=-1)  # (B,)

            # Next "chunks remaining" feature (soft, but monotonically decreasing).
            remain_norm = torch.clamp(remain_norm - float(step_remain_delta), min=0.0)

            # Roll the state window forward by one and write expected next features.
            row0 = torch.cat(
                [s[:, 0, 1:], torch.clamp(last_br_kbps / float(max_br), min=0.0, max=1.0).unsqueeze(-1)], dim=-1
            )
            row1 = torch.cat(
                [s[:, 1, 1:], torch.clamp(buffer_s / float(BUFFER_NORM_FACTOR), min=0.0).unsqueeze(-1)], dim=-1
            )
            row2 = torch.cat(
                [s[:, 2, 1:], torch.clamp(next_throughput_norm, min=0.0).unsqueeze(-1)], dim=-1
            )
            row3 = torch.cat([s[:, 3, 1:], torch.clamp(next_delay_norm, min=0.0).unsqueeze(-1)], dim=-1)

            # Row 4 stores next-chunk sizes for each action (not a history feature).
            if t + 1 < horizon:
                next_chunk_mb = chunk_sizes_mb[t + 1].to(device=device, dtype=dtype)
            else:
                next_chunk_mb = chunk_sizes_mb[t].to(device=device, dtype=dtype)
            next_chunk_mb = next_chunk_mb.view(1, -1).expand(batch, -1)  # (B, A)
            row4 = s.new_zeros((batch, s_len))
            fill = int(min(num_actions, s_len))
            if fill > 0:
                row4[:, :fill] = next_chunk_mb[:, :fill]

            row5 = torch.cat([s[:, 5, 1:], remain_norm.unsqueeze(-1)], dim=-1)

            s = torch.stack([row0, row1, row2, row3, row4, row5], dim=1)

        return total_reward

    def _oracle_best_k_step_return_surrogate(
        self,
        state_tensor: torch.Tensor,
        *,
        chunk_sizes_mb: torch.Tensor,
        k_steps: int,
        tau: Optional[float] = None,
        max_sequences: Optional[int] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Differentiable approximation of the K-step *oracle best* return under the same surrogate
        dynamics used by `_expected_k_step_return_surrogate`.

        Implementation:
        - If `A ** H` is small, enumerate all action sequences of length H and return:
              tau * logsumexp(R_seq / tau)
          which is a smooth upper bound that approaches max(R_seq) as tau → 0.
        - Otherwise, fall back to a soft-greedy surrogate oracle rollout.

        Returns:
            Tensor shape (batch,) of the soft oracle-best cumulative return.
        """
        if chunk_sizes_mb is None:
            raise ValueError("* chunk_sizes_mb must be provided")

        k = int(k_steps)
        if k <= 0:
            return state_tensor.new_zeros((state_tensor.shape[0],))

        horizon = int(min(k, int(chunk_sizes_mb.shape[0])))
        if horizon <= 0:
            return state_tensor.new_zeros((state_tensor.shape[0],))

        a_dim = int(chunk_sizes_mb.shape[1])
        total_sequences = int(a_dim) ** int(horizon)
        tau_val = float(tau) if tau is not None else float(getattr(self, "_surrogate_oracle_tau", 0.1))
        tau_val = float(max(tau_val, 1e-6))
        max_seq = (
            int(max_sequences)
            if max_sequences is not None
            else int(getattr(self, "_surrogate_oracle_max_sequences", 50000))
        )

        if total_sequences > int(max_seq):
            return self._oracle_best_k_step_return_surrogate_soft_greedy(
                state_tensor,
                chunk_sizes_mb=chunk_sizes_mb,
                k_steps=int(horizon),
                tau=float(tau_val),
                eps=float(eps),
            )

        cache_key = (int(horizon), int(a_dim))
        seq_cache = getattr(self, "_surrogate_oracle_seq_cache", None)
        if not isinstance(seq_cache, dict):
            seq_cache = {}
            self._surrogate_oracle_seq_cache = seq_cache

        action_seqs = seq_cache.get(cache_key)
        if action_seqs is None:
            action_seqs = torch.tensor(
                list(itertools.product(range(int(a_dim)), repeat=int(horizon))),
                dtype=torch.long,
            )
            seq_cache[cache_key] = action_seqs

        returns = self._k_step_return_surrogate_for_action_sequences(
            state_tensor,
            chunk_sizes_mb=chunk_sizes_mb,
            action_seqs=action_seqs,
            eps=float(eps),
        )

        return (returns / float(tau_val)).logsumexp(dim=1) * float(tau_val)

    def _k_step_return_surrogate_for_action_sequences(
        self,
        state_tensor: torch.Tensor,
        *,
        chunk_sizes_mb: torch.Tensor,
        action_seqs: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute surrogate K-step returns for a batch of fixed action sequences.

        Args:
            state_tensor: shape (B, S_INFO, S_LEN)
            chunk_sizes_mb: shape (H, A)
            action_seqs: shape (N, H), entries in [0, A-1]

        Returns:
            Tensor shape (B, N) of cumulative returns for each sequence.
        """
        if chunk_sizes_mb is None:
            raise ValueError("* chunk_sizes_mb must be provided")

        batch = int(state_tensor.shape[0])
        s_info = int(state_tensor.shape[1])
        s_len = int(state_tensor.shape[2])
        device = state_tensor.device
        dtype = state_tensor.dtype

        seqs = action_seqs.to(device=device)
        num_seqs = int(seqs.shape[0])
        horizon = int(min(int(seqs.shape[1]), int(chunk_sizes_mb.shape[0])))
        if horizon <= 0 or num_seqs <= 0:
            return state_tensor.new_zeros((batch, max(num_seqs, 1)))

        chunk_sizes_mb_dev = chunk_sizes_mb.to(device=device, dtype=dtype)
        num_actions = int(chunk_sizes_mb_dev.shape[1])

        rtt_s = float(LINK_RTT) / float(MILLISECONDS_IN_SECOND)
        chunk_len_s = float(VIDEO_CHUNCK_LEN) / float(MILLISECONDS_IN_SECOND)
        buffer_thresh_s = float(BUFFER_THRESH) / float(MILLISECONDS_IN_SECOND)
        buffer_thresh_tensor = state_tensor.new_tensor(float(buffer_thresh_s))

        br_kbps = torch.tensor(VIDEO_BIT_RATE[:num_actions], dtype=dtype, device=device)
        max_br = float(np.max(VIDEO_BIT_RATE))

        s = (
            state_tensor.unsqueeze(1)
            .expand(batch, num_seqs, s_info, s_len)
            .reshape(batch * num_seqs, s_info, s_len)
        )

        buffer_s = torch.clamp(s[:, 1, -1], min=0.0) * float(BUFFER_NORM_FACTOR)
        last_br_kbps = torch.clamp(s[:, 0, -1], min=0.0) * float(max_br)
        remain_norm = torch.clamp(s[:, 5, -1], min=0.0)
        step_remain_delta = float(1.0 / float(CHUNK_TIL_VIDEO_END_CAP))

        if batch == 1:
            actions_flat = seqs
        else:
            actions_flat = seqs.unsqueeze(0).expand(batch, num_seqs, horizon).reshape(batch * num_seqs, horizon)

        total_reward = state_tensor.new_zeros((batch * num_seqs,))

        for t in range(horizon):
            a_t = actions_flat[:, t].to(dtype=torch.long)

            delay_s_hist = torch.clamp(s[:, 3, :], min=0.0) * float(BUFFER_NORM_FACTOR)
            chunk_mb_hist = torch.clamp(s[:, 2, :], min=0.0) * delay_s_hist
            rtt_free_delay_s_hist = torch.clamp(delay_s_hist - float(rtt_s), min=float(eps))
            link_mb_s_hist = chunk_mb_hist / rtt_free_delay_s_hist
            link_mb_s = torch.clamp(link_mb_s_hist.mean(dim=-1), min=float(eps))

            chunk_mb_all = chunk_sizes_mb_dev[t]
            chunk_mb = chunk_mb_all[a_t]

            pred_delay_s = chunk_mb / link_mb_s + float(rtt_s)
            pred_rebuf_s = torch.relu(pred_delay_s - buffer_s)

            quality_mbps = br_kbps[a_t] / float(M_IN_K)
            smoothness_mbps = torch.abs(br_kbps[a_t] - last_br_kbps) / float(M_IN_K)
            step_reward = (
                quality_mbps
                - float(REBUF_PENALTY) * pred_rebuf_s
                - float(SMOOTH_PENALTY) * smoothness_mbps
            )
            total_reward = total_reward + step_reward

            buffer_after_delay = torch.relu(buffer_s - pred_delay_s)
            buffer_after_add = buffer_after_delay + float(chunk_len_s)
            buffer_s = torch.minimum(buffer_after_add, buffer_thresh_tensor)

            last_br_kbps = br_kbps[a_t]

            next_delay_norm = pred_delay_s / float(BUFFER_NORM_FACTOR)
            next_throughput_norm = chunk_mb / torch.clamp(pred_delay_s, min=float(eps))

            remain_norm = torch.clamp(remain_norm - float(step_remain_delta), min=0.0)

            row0 = torch.cat(
                [
                    s[:, 0, 1:],
                    torch.clamp(last_br_kbps / float(max_br), min=0.0, max=1.0).unsqueeze(-1),
                ],
                dim=-1,
            )
            row1 = torch.cat(
                [
                    s[:, 1, 1:],
                    torch.clamp(buffer_s / float(BUFFER_NORM_FACTOR), min=0.0).unsqueeze(-1),
                ],
                dim=-1,
            )
            row2 = torch.cat([s[:, 2, 1:], torch.clamp(next_throughput_norm, min=0.0).unsqueeze(-1)], dim=-1)
            row3 = torch.cat([s[:, 3, 1:], torch.clamp(next_delay_norm, min=0.0).unsqueeze(-1)], dim=-1)

            if t + 1 < horizon:
                next_chunk_mb = chunk_sizes_mb_dev[t + 1]
            else:
                next_chunk_mb = chunk_sizes_mb_dev[t]
            next_chunk_mb = next_chunk_mb.view(1, -1).expand(batch * num_seqs, -1)
            row4 = s.new_zeros((batch * num_seqs, s_len))
            fill = int(min(num_actions, s_len))
            if fill > 0:
                row4[:, :fill] = next_chunk_mb[:, :fill]

            row5 = torch.cat([s[:, 5, 1:], remain_norm.unsqueeze(-1)], dim=-1)

            s = torch.stack([row0, row1, row2, row3, row4, row5], dim=1)

        return total_reward.view(batch, num_seqs)

    def _oracle_best_k_step_return_surrogate_soft_greedy(
        self,
        state_tensor: torch.Tensor,
        *,
        chunk_sizes_mb: torch.Tensor,
        k_steps: int,
        tau: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Fallback differentiable oracle-best approximation when enumerating all action sequences is too large.
        Uses a soft-greedy (soft argmax) selection based on immediate surrogate reward.
        """
        if chunk_sizes_mb is None:
            raise ValueError("* chunk_sizes_mb must be provided")

        k = int(k_steps)
        if k <= 0:
            return state_tensor.new_zeros((state_tensor.shape[0],))

        horizon = int(min(k, int(chunk_sizes_mb.shape[0])))
        if horizon <= 0:
            return state_tensor.new_zeros((state_tensor.shape[0],))

        tau_val = float(max(float(tau), 1e-6))

        batch = int(state_tensor.shape[0])
        s_len = int(state_tensor.shape[2])
        device = state_tensor.device
        dtype = state_tensor.dtype

        rtt_s = float(LINK_RTT) / float(MILLISECONDS_IN_SECOND)
        chunk_len_s = float(VIDEO_CHUNCK_LEN) / float(MILLISECONDS_IN_SECOND)
        buffer_thresh_s = float(BUFFER_THRESH) / float(MILLISECONDS_IN_SECOND)
        buffer_thresh_tensor = state_tensor.new_tensor(float(buffer_thresh_s))

        chunk_sizes_mb_dev = chunk_sizes_mb.to(device=device, dtype=dtype)
        num_actions = int(chunk_sizes_mb_dev.shape[1])
        br_kbps = torch.tensor(VIDEO_BIT_RATE[:num_actions], dtype=dtype, device=device)
        max_br = float(np.max(VIDEO_BIT_RATE))

        s = state_tensor
        buffer_s = torch.clamp(s[:, 1, -1], min=0.0) * float(BUFFER_NORM_FACTOR)
        last_br_kbps = torch.clamp(s[:, 0, -1], min=0.0) * float(max_br)

        total_reward = state_tensor.new_zeros((batch,))
        remain_norm = torch.clamp(s[:, 5, -1], min=0.0)
        step_remain_delta = float(1.0 / float(CHUNK_TIL_VIDEO_END_CAP))

        quality_mbps = (br_kbps / float(M_IN_K)).view(1, -1)

        for t in range(horizon):
            delay_s_hist = torch.clamp(s[:, 3, :], min=0.0) * float(BUFFER_NORM_FACTOR)
            chunk_mb_hist = torch.clamp(s[:, 2, :], min=0.0) * delay_s_hist
            rtt_free_delay_s_hist = torch.clamp(delay_s_hist - float(rtt_s), min=float(eps))
            link_mb_s_hist = chunk_mb_hist / rtt_free_delay_s_hist
            link_mb_s = torch.clamp(link_mb_s_hist.mean(dim=-1, keepdim=True), min=float(eps))

            chunk_mb = chunk_sizes_mb_dev[t].view(1, -1).expand(batch, -1)
            pred_delay_s = chunk_mb / link_mb_s + float(rtt_s)
            pred_rebuf_s = torch.relu(pred_delay_s - buffer_s.view(-1, 1))

            smoothness_mbps = (
                torch.abs(br_kbps.view(1, -1) - last_br_kbps.view(-1, 1)) / float(M_IN_K)
            )
            reward_per_action = (
                quality_mbps
                - float(REBUF_PENALTY) * pred_rebuf_s
                - float(SMOOTH_PENALTY) * smoothness_mbps
            )

            step_soft = (reward_per_action / float(tau_val)).logsumexp(dim=-1) * float(tau_val)
            total_reward = total_reward + step_soft

            action_w = torch.softmax(reward_per_action / float(tau_val), dim=-1)

            buffer_after_delay = torch.relu(buffer_s.view(-1, 1) - pred_delay_s)
            buffer_after_add = buffer_after_delay + float(chunk_len_s)
            buffer_next = torch.minimum(buffer_after_add, buffer_thresh_tensor)
            buffer_s = (action_w * buffer_next).sum(dim=-1)

            last_br_kbps = (action_w * br_kbps.view(1, -1)).sum(dim=-1)

            expected_delay_s = (action_w * pred_delay_s).sum(dim=-1)
            next_delay_norm = expected_delay_s / float(BUFFER_NORM_FACTOR)

            throughput_norm_per_action = chunk_mb / torch.clamp(pred_delay_s, min=float(eps))
            next_throughput_norm = (action_w * throughput_norm_per_action).sum(dim=-1)

            remain_norm = torch.clamp(remain_norm - float(step_remain_delta), min=0.0)

            row0 = torch.cat(
                [s[:, 0, 1:], torch.clamp(last_br_kbps / float(max_br), min=0.0, max=1.0).unsqueeze(-1)],
                dim=-1,
            )
            row1 = torch.cat(
                [s[:, 1, 1:], torch.clamp(buffer_s / float(BUFFER_NORM_FACTOR), min=0.0).unsqueeze(-1)],
                dim=-1,
            )
            row2 = torch.cat([s[:, 2, 1:], torch.clamp(next_throughput_norm, min=0.0).unsqueeze(-1)], dim=-1)
            row3 = torch.cat([s[:, 3, 1:], torch.clamp(next_delay_norm, min=0.0).unsqueeze(-1)], dim=-1)

            if t + 1 < horizon:
                next_chunk_mb = chunk_sizes_mb_dev[t + 1]
            else:
                next_chunk_mb = chunk_sizes_mb_dev[t]
            next_chunk_mb = next_chunk_mb.view(1, -1).expand(batch, -1)
            row4 = s.new_zeros((batch, s_len))
            fill = int(min(num_actions, s_len))
            if fill > 0:
                row4[:, :fill] = next_chunk_mb[:, :fill]

            row5 = torch.cat([s[:, 5, 1:], remain_norm.unsqueeze(-1)], dim=-1)

            s = torch.stack([row0, row1, row2, row3, row4, row5], dim=1)

        return total_reward

    def _z3_to_float(self, value) -> float:
        
        if value is None:
            return float("nan")
        if isinstance(value, z3.RatNumRef):
            return float(value.as_fraction())
        try:
            s = value.as_decimal(30)
            if s.endswith("?"):
                s = s[:-1]
            return float(s)
        except Exception:
            return float(str(value))

    def _load_rules(self, rules_file: str, *, latest_only: bool = False) -> None:
        
        cache_key = (str(rules_file), bool(latest_only))
        rules_cache = getattr(self, "_rules_cache", None)
        if isinstance(rules_cache, dict) and cache_key in rules_cache:
            solver, variable_cache = rules_cache[cache_key]
            self._rules_file = str(rules_file)
            self._rules_solver = solver
            self._rules_vars = variable_cache
            return

        variable_cache: dict[str, z3.ArithRef] = {}

        def Symbol(name: str, *args, **kwargs):
            if name not in variable_cache:
                want_int = bool(
                    name.startswith("BitrateKbps")
                    or name.startswith("ChosenChunkBytes")
                    or kwargs.get("int", False)
                    or kwargs.get("integer", False)
                )
                want_real = bool(kwargs.get("real", False))
                if want_real:
                    want_int = False
                variable_cache[name] = z3.Int(name) if want_int else z3.Real(name)
            return variable_cache[name]

        def Integer(v):
            return z3.IntVal(str(int(v)))

        def Float(v, *args, **kwargs):
            return z3.RealVal(str(v))

        def Mul(*args):
            return reduce(lambda a, b: a * b, args, z3.RealVal("1"))

        def Add(*args):
            return reduce(lambda a, b: a + b, args, z3.RealVal("0"))

        def Abs(x):
            return z3.Abs(x)

        def Or(*args):
            return z3.Or(*args)

        def And(*args):
            return z3.And(*args)

        def Max(*args):
            if len(args) == 0:
                raise ValueError("* Max() requires at least 1 argument")
            cur = args[0]
            for nxt in args[1:]:
                cur = z3.If(cur >= nxt, cur, nxt)
            return cur

        def Min(*args):
            if len(args) == 0:
                raise ValueError("* Min() requires at least 1 argument")
            cur = args[0]
            for nxt in args[1:]:
                cur = z3.If(cur <= nxt, cur, nxt)
            return cur

        def Pow(base, exp):
            exp_int = None
            if isinstance(exp, int):
                exp_int = int(exp)
            elif isinstance(exp, z3.IntNumRef):
                try:
                    exp_int = int(exp.as_long())
                except Exception:
                    exp_int = None

            if exp_int == -1:
                return z3.RealVal("1") / base
            if exp_int is not None and exp_int >= 0 and exp_int <= 5:
                out = z3.RealVal("1")
                for _ in range(int(exp_int)):
                    out = out * base
                return out
            return base**exp

        def Equality(a, b):
            return a == b

        def Unequality(a, b):
            return a != b

        def StrictLessThan(a, b):
            return a < b

        def StrictGreaterThan(a, b):
            return a > b

        def LessThan(a, b):
            return a <= b

        def GreaterThan(a, b):
            return a >= b

        eval_env = {
            "__builtins__": {},
            "Symbol": Symbol,
            "Integer": Integer,
            "Float": Float,
            "Mul": Mul,
            "Add": Add,
            "Abs": Abs,
            "Or": Or,
            "And": And,
            "Max": Max,
            "Min": Min,
            "Pow": Pow,
            "Equality": Equality,
            "Unequality": Unequality,
            "StrictLessThan": StrictLessThan,
            "StrictGreaterThan": StrictGreaterThan,
            "LessThan": LessThan,
            "GreaterThan": GreaterThan,
        }

        def _suffix_int(name: str) -> Optional[int]:
            i = int(len(name))
            while i > 0 and name[i - 1].isdigit():
                i -= 1
            if i == int(len(name)):
                return None
            try:
                return int(name[i:])
            except Exception:
                return None

        def _rule_is_latest_only(expr: z3.BoolRef) -> bool:
            stack = [expr]
            while stack:
                node = stack.pop()
                if z3.is_const(node) and node.num_args() == 0 and node.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                    name = str(node.decl().name())
                    suffix = _suffix_int(name)
                    if suffix is not None and int(suffix) != 7:
                        return False
                else:
                    stack.extend(list(node.children()))
            return True

        rules: list[tuple[str, z3.BoolRef]] = []
        with open(rules_file, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                expr = eval(line, eval_env)  # nosec - trusted local rules file
                if isinstance(expr, bool):
                    expr = z3.BoolVal(expr)
                expr = z3.simplify(expr)
                if latest_only and not _rule_is_latest_only(expr):
                    continue
                rules.append((line, expr))

        #* Avoid stalling the whole rollout if the non-linear rule set is hard.
        solver = z3.SolverFor("QF_NRA")
        timeout_ms = 1000 if not bool(latest_only) else 50
        solver.set(timeout=int(timeout_ms))
        solver.set(unsat_core=True)
        seen: dict[str, int] = {}
        for rule_str, rule in rules:
            count = int(seen.get(rule_str, 0))
            seen[rule_str] = count + 1
            name = rule_str if count == 0 else f"{rule_str} (dup={count})"
            solver.assert_and_track(rule, name)

        if isinstance(rules_cache, dict):
            rules_cache[cache_key] = (solver, variable_cache)
        self._rules_file = str(rules_file)
        self._rules_solver = solver
        self._rules_vars = variable_cache

    def _project_with_rules(
        self,
        *,
        rules_file: str,
        bitrate_kbps: Optional[int] = None,
        buffer_seconds: Optional[float] = None,
        buffer_seconds_max: Optional[float] = None,
        chosen_chunk_bytes: Optional[int] = None,
        delay_ms: Optional[float] = None,
        throughput_mbps: Optional[float] = None,
        bitrate_kbps_hist: Optional[Sequence[int]] = None,
        buffer_seconds_hist: Optional[Sequence[float]] = None,
        fix_buffer_hist: Optional[Sequence[bool]] = None,
        fix_delay_hist: Optional[Sequence[bool]] = None,
        fix_throughput_hist: Optional[Sequence[bool]] = None,
        chosen_chunk_bytes_hist: Optional[Sequence[int]] = None,
        delay_ms_hist: Optional[Sequence[float]] = None,
        throughput_mbps_hist: Optional[Sequence[float]] = None,
        max_relax_steps: int = 7,
        fix_buffer: bool = False,
        fix_delay: bool = False,
        fix_throughput: bool = False,
        latest_only: bool = False,
    ) -> Optional[tuple[object, object, object]]:
        """
        * Project onto the mined rule set.
          - Scalar mode (default): project the latest-step variables (e.g., DelayMs / ThroughputMbps).
          - History-window mode: project the full window jointly (e.g., DelayMs0..DelayMs7).
          If the candidate already satisfies the rules, return it unchanged. Otherwise,
          search for a satisfying assignment close to the candidate by expanding a
          tolerance window; if that fails, fall back to any satisfying assignment.
        """
        

        self._rules_projection_calls_total = int(self._rules_projection_calls_total) + 1
        self._rules_projection_calls_last_attack = int(self._rules_projection_calls_last_attack) + 1
        is_window = (
            delay_ms_hist is not None
            or throughput_mbps_hist is not None
            or chosen_chunk_bytes_hist is not None
            or bitrate_kbps_hist is not None
            or buffer_seconds_hist is not None
        )
        if is_window:
            self._load_rules(rules_file, latest_only=False)
        else:
            self._load_rules(rules_file, latest_only=bool(latest_only))
        assert self._rules_solver is not None
        assert self._rules_vars is not None

        solver = self._rules_solver
        v = self._rules_vars

        def _realval(x: float) -> z3.ArithRef:
            #* `RealVal` does not accept scientific notation; force fixed-point.
            return z3.RealVal(format(Decimal(str(float(x))), "f"))

        def _first_var(*names: str) -> Optional[z3.ArithRef]:
            for name in names:
                if name in v:
                    return v[name]
            return None

        solver.push()
        try:
            if is_window:
                if delay_ms_hist is None or throughput_mbps_hist is None or chosen_chunk_bytes_hist is None:
                    raise ValueError("* Window projection requires delay_ms_hist/throughput_mbps_hist/chosen_chunk_bytes_hist")

                history_len = int(len(delay_ms_hist))
                if int(len(throughput_mbps_hist)) != history_len or int(len(chosen_chunk_bytes_hist)) != history_len:
                    raise ValueError("* Window projection inputs must have the same length")

                delay_vars = [v.get(f"DelayMs{i}") for i in range(history_len)]
                thr_vars = [v.get(f"ThroughputMbps{i}") for i in range(history_len)]
                chosen_vars = [v.get(f"ChosenChunkBytes{i}") for i in range(history_len)]

                has_window_vars = all(x is not None for x in delay_vars) and all(x is not None for x in thr_vars)
                if not has_window_vars:
                    #* Fallback: rules file does not define window-suffixed variables; project each step independently.
                    out_delay: list[float] = []
                    out_thr: list[float] = []
                    out_buf: list[float] = []
                    for i in range(history_len):
                        step_bitrate = int(bitrate_kbps_hist[i]) if bitrate_kbps_hist is not None else bitrate_kbps
                        step_buf = float(buffer_seconds_hist[i]) if buffer_seconds_hist is not None else buffer_seconds
                        step_fix_buf = bool(fix_buffer_hist[i]) if fix_buffer_hist is not None else bool(fix_buffer)
                        step_fix_delay = bool(fix_delay_hist[i]) if fix_delay_hist is not None else bool(fix_delay)
                        step_fix_thr = bool(fix_throughput_hist[i]) if fix_throughput_hist is not None else bool(
                            fix_throughput
                        )
                        sol = self._project_with_rules(
                            rules_file=str(rules_file),
                            bitrate_kbps=int(step_bitrate) if step_bitrate is not None else None,
                            buffer_seconds=float(step_buf) if step_buf is not None else None,
                            buffer_seconds_max=buffer_seconds_max,
                            chosen_chunk_bytes=int(chosen_chunk_bytes_hist[i]),
                            delay_ms=float(delay_ms_hist[i]),
                            throughput_mbps=float(throughput_mbps_hist[i]),
                            max_relax_steps=int(max_relax_steps),
                            fix_buffer=bool(step_fix_buf),
                            fix_delay=bool(step_fix_delay),
                            fix_throughput=bool(step_fix_thr),
                            latest_only=bool(latest_only),
                        )
                        if sol is None:
                            return None
                        sol_delay, sol_thr, sol_buf = sol
                        out_delay.append(float(sol_delay))
                        out_thr.append(float(sol_thr))
                        out_buf.append(float(sol_buf))
                    return out_delay, out_thr, out_buf

                for i in range(history_len):
                    if chosen_vars[i] is not None:
                        solver.add(chosen_vars[i] == int(chosen_chunk_bytes_hist[i]))
                    if bitrate_kbps_hist is not None:
                        br_var = v.get(f"BitrateKbps{i}")
                        if br_var is not None:
                            solver.add(br_var == int(bitrate_kbps_hist[i]))
                    if buffer_seconds_hist is not None:
                        buf_var = v.get(f"BufferSeconds{i}")
                        if buf_var is not None:
                            fix_i = bool(fix_buffer_hist[i]) if fix_buffer_hist is not None else False
                            if fix_i:
                                solver.add(buf_var == _realval(float(buffer_seconds_hist[i])))

                cand_delay = [float(x) for x in delay_ms_hist]
                cand_thr = [float(x) for x in throughput_mbps_hist]
                cand_buf = [float(x) for x in buffer_seconds_hist] if buffer_seconds_hist is not None else [0.0] * history_len

                #* First check: if the candidate already satisfies the mined rules, do not change it.
                solver.push()
                for i in range(history_len):
                    solver.add(delay_vars[i] == _realval(cand_delay[i]))
                    solver.add(thr_vars[i] == _realval(cand_thr[i]))
                    if buffer_seconds_hist is not None:
                        buf_var = v.get(f"BufferSeconds{i}")
                        if buf_var is not None:
                            fix_i = bool(fix_buffer_hist[i]) if fix_buffer_hist is not None else False
                            if not fix_i:
                                solver.add(buf_var == _realval(cand_buf[i]))
                if solver.check() == z3.sat:
                    solver.pop()
                    return cand_delay, cand_thr, cand_buf
                solver.pop()

                #* Freeze non-perturbed history indices if requested (L0 / sparse attacks).
                for i in range(history_len):
                    if fix_delay_hist is not None and bool(fix_delay_hist[i]):
                        solver.add(delay_vars[i] == _realval(cand_delay[i]))
                    if fix_throughput_hist is not None and bool(fix_throughput_hist[i]):
                        solver.add(thr_vars[i] == _realval(cand_thr[i]))

                #* If even the rule set is infeasible (given fixed params), abort.
                if solver.check() != z3.sat:
                    return None

                base_model = solver.model()
                base_delay = [
                    self._z3_to_float(base_model.evaluate(delay_vars[i], model_completion=True)) for i in range(history_len)
                ]
                base_thr = [
                    self._z3_to_float(base_model.evaluate(thr_vars[i], model_completion=True)) for i in range(history_len)
                ]
                base_buf = list(cand_buf)
                if buffer_seconds_hist is not None:
                    for i in range(history_len):
                        buf_var = v.get(f"BufferSeconds{i}")
                        if buf_var is None:
                            continue
                        fix_i = bool(fix_buffer_hist[i]) if fix_buffer_hist is not None else False
                        if fix_i:
                            continue
                        base_buf[i] = self._z3_to_float(base_model.evaluate(buf_var, model_completion=True))

                base_tol_delay = [max(1.0, abs(x) * 0.05) for x in cand_delay]
                base_tol_thr = [max(0.01, abs(x) * 0.05) for x in cand_thr]
                base_tol_buf = [max(0.1, abs(x) * 0.05) for x in cand_buf]

                #* Prefer a solution close to the candidate, by expanding a tolerance window.
                for step in range(int(max_relax_steps)):
                    solver.push()
                    for i in range(history_len):
                        tol_delay = float(base_tol_delay[i]) * (2.0**step)
                        tol_thr = float(base_tol_thr[i]) * (2.0**step)
                        if fix_delay_hist is not None and bool(fix_delay_hist[i]):
                            solver.add(delay_vars[i] == _realval(cand_delay[i]))
                        else:
                            solver.add(z3.Abs(delay_vars[i] - _realval(cand_delay[i])) <= _realval(tol_delay))
                        if fix_throughput_hist is not None and bool(fix_throughput_hist[i]):
                            solver.add(thr_vars[i] == _realval(cand_thr[i]))
                        else:
                            solver.add(z3.Abs(thr_vars[i] - _realval(cand_thr[i])) <= _realval(tol_thr))

                        if buffer_seconds_hist is not None:
                            buf_var = v.get(f"BufferSeconds{i}")
                            if buf_var is None:
                                continue
                            fix_i = bool(fix_buffer_hist[i]) if fix_buffer_hist is not None else False
                            if fix_i:
                                continue
                            tol_buf = float(base_tol_buf[i]) * (2.0**step)
                            solver.add(z3.Abs(buf_var - _realval(cand_buf[i])) <= _realval(tol_buf))

                    if solver.check() == z3.sat:
                        model = solver.model()
                        out_delay = [
                            self._z3_to_float(model.evaluate(delay_vars[i], model_completion=True))
                            for i in range(history_len)
                        ]
                        out_thr = [
                            self._z3_to_float(model.evaluate(thr_vars[i], model_completion=True))
                            for i in range(history_len)
                        ]
                        out_buf = list(cand_buf)
                        if buffer_seconds_hist is not None:
                            for i in range(history_len):
                                buf_var = v.get(f"BufferSeconds{i}")
                                if buf_var is None:
                                    continue
                                fix_i = bool(fix_buffer_hist[i]) if fix_buffer_hist is not None else False
                                if fix_i:
                                    continue
                                out_buf[i] = self._z3_to_float(model.evaluate(buf_var, model_completion=True))
                        solver.pop()
                        return out_delay, out_thr, out_buf
                    solver.pop()

                #* Fully relaxed: any satisfying assignment (can be far from the candidate).
                self._rules_fully_relaxed_calls_total = int(self._rules_fully_relaxed_calls_total) + 1
                self._rules_fully_relaxed_calls_last_attack = int(self._rules_fully_relaxed_calls_last_attack) + 1
                print("WARNING: Using fully-relaxed rule projection to find any satisfying assignment")
                return base_delay, base_thr, base_buf

            DelayMs = _first_var("DelayMs", "DelayMs7")
            ThroughputMbps = _first_var("ThroughputMbps", "ThroughputMbps7")
            BufferSeconds = _first_var("BufferSeconds", "BufferSeconds7")
            BitrateKbps = _first_var("BitrateKbps", "BitrateKbps7")
            ChosenChunkBytes = _first_var("ChosenChunkBytes", "ChosenChunkBytes7")

            if DelayMs is None or ThroughputMbps is None:
                return None

            if BitrateKbps is not None and bitrate_kbps is not None:
                solver.add(BitrateKbps == int(bitrate_kbps))
            if ChosenChunkBytes is not None and chosen_chunk_bytes is not None:
                solver.add(ChosenChunkBytes == int(chosen_chunk_bytes))

            cand_delay = float(delay_ms) if delay_ms is not None else 0.0
            cand_thr = float(throughput_mbps) if throughput_mbps is not None else 0.0
            cand_buf = float(buffer_seconds) if buffer_seconds is not None else 0.0

            #* If buffer perturbation is disabled, keep buffer fixed under rules projection.
            if fix_buffer and BufferSeconds is not None:
                solver.add(BufferSeconds == _realval(cand_buf))
            if fix_delay and DelayMs is not None:
                solver.add(DelayMs == _realval(cand_delay))
            if fix_throughput and ThroughputMbps is not None:
                solver.add(ThroughputMbps == _realval(cand_thr))

            #* First check: if the candidate already satisfies the mined rules, do not change it.
            solver.push()
            solver.add(DelayMs == _realval(cand_delay))
            solver.add(ThroughputMbps == _realval(cand_thr))
            if BufferSeconds is not None and not fix_buffer:
                solver.add(BufferSeconds == _realval(cand_buf))
            if solver.check() == z3.sat:
                solver.pop()
                return float(cand_delay), float(cand_thr), float(cand_buf)
            solver.pop()

            #* Projection: search for a satisfying assignment close to the candidate.
            base_tol_delay = max(1.0, abs(cand_delay) * 0.05)
            base_tol_thr = max(0.01, abs(cand_thr) * 0.05)
            base_tol_buf = max(0.1, abs(cand_buf) * 0.05)

            #* If even the rule set is infeasible (given fixed params), abort.
            if solver.check() != z3.sat:
                return None
            base_model = solver.model()
            base_d = self._z3_to_float(base_model.evaluate(DelayMs, model_completion=True))
            base_t = self._z3_to_float(base_model.evaluate(ThroughputMbps, model_completion=True))
            base_b = cand_buf
            if BufferSeconds is not None and not fix_buffer:
                base_b = self._z3_to_float(base_model.evaluate(BufferSeconds, model_completion=True))

            #* Prefer a solution close to the candidate, by expanding a tolerance window.
            for step in range(int(max_relax_steps)):
                tol_delay = base_tol_delay * (2.0**step)
                tol_thr = base_tol_thr * (2.0**step)
                tol_buf = base_tol_buf * (2.0**step)
                solver.push()
                if fix_delay:
                    solver.add(DelayMs == _realval(cand_delay))
                else:
                    solver.add(z3.Abs(DelayMs - _realval(cand_delay)) <= _realval(tol_delay))
                if fix_throughput:
                    solver.add(ThroughputMbps == _realval(cand_thr))
                else:
                    solver.add(z3.Abs(ThroughputMbps - _realval(cand_thr)) <= _realval(tol_thr))
                if BufferSeconds is not None and not fix_buffer:
                    solver.add(z3.Abs(BufferSeconds - _realval(cand_buf)) <= _realval(tol_buf))
                if solver.check() == z3.sat:
                    model = solver.model()
                    d = self._z3_to_float(model.evaluate(DelayMs, model_completion=True))
                    t = self._z3_to_float(model.evaluate(ThroughputMbps, model_completion=True))
                    b = cand_buf
                    if BufferSeconds is not None and not fix_buffer:
                        b = self._z3_to_float(model.evaluate(BufferSeconds, model_completion=True))
                    solver.pop()
                    return float(d), float(t), float(b)
                solver.pop()

            #* Fully relaxed: any satisfying assignment (can be far from the candidate).
            self._rules_fully_relaxed_calls_total = int(self._rules_fully_relaxed_calls_total) + 1
            self._rules_fully_relaxed_calls_last_attack = int(self._rules_fully_relaxed_calls_last_attack) + 1
            print("WARNING: Using fully-relaxed rule projection to find any satisfying assignment")
            return float(base_d), float(base_t), float(base_b)
        finally:
            solver.pop()

    def _update_state_like_simulator(
        self,
        state: np.ndarray,
        *,
        bit_rate: int,
        buffer_s: float,
        video_chunk_size: int,
        delay_ms: float,
        next_video_chunk_sizes: Sequence[int],
        video_chunk_remain: int,
    ) -> np.ndarray:
        #* Mirror `inference._update_state`: roll window and overwrite latest column.
        new_state = np.roll(state, -1, axis=1).astype(np.float64, copy=False)
        new_state[0, -1] = VIDEO_BIT_RATE[int(bit_rate)] / float(np.max(VIDEO_BIT_RATE))
        new_state[1, -1] = float(buffer_s) / float(BUFFER_NORM_FACTOR)
        new_state[2, -1] = float(video_chunk_size) / float(delay_ms) / float(M_IN_K)
        new_state[3, -1] = float(delay_ms) / float(M_IN_K) / float(BUFFER_NORM_FACTOR)
        new_state[4, : int(A_DIM)] = np.array(list(next_video_chunk_sizes), dtype=np.float64) / float(M_IN_K) / float(M_IN_K)
        new_state[5, -1] = np.minimum(float(video_chunk_remain), float(CHUNK_TIL_VIDEO_END_CAP)) / float(CHUNK_TIL_VIDEO_END_CAP)
        return new_state

    def _policy_k_step_reward_mean_argmax(
        self,
        env_snapshot,
        state_snapshot: np.ndarray,
        *,
        last_bitrate: int,
        start_action: int,
        k_steps: int,
    ) -> Optional[float]:
        #* Simulator rollout of the policy return when we force the first action to `start_action`.
        k = int(k_steps)
        if k <= 0:
            return None

        env_sim = copy_env(env_snapshot)
        state_sim = np.array(state_snapshot, copy=True)

        total_reward = 0.0
        last = int(last_bitrate)
        a = int(start_action)

        for i in range(int(k)):
            try:
                delay_ms, _sleep_ms, buffer_s, rebuf_s, chunk_size, next_sizes, end, remain = env_sim.get_video_chunk(int(a))
            except Exception:
                return None

            #* Match the oracle validity rule: sequences that end early are invalid for K-step comparison.
            if bool(end) and i < int(k) - 1:
                return None

            total_reward += float(calculate_reward(int(a), int(last), float(rebuf_s)))
            state_sim = self._update_state_like_simulator(
                state_sim,
                bit_rate=int(a),
                buffer_s=float(buffer_s),
                video_chunk_size=int(chunk_size),
                delay_ms=float(delay_ms),
                next_video_chunk_sizes=[int(x) for x in next_sizes],
                video_chunk_remain=int(remain),
            )
            last = int(a)

            if i < int(k) - 1:
                p = self.actor.predict(np.reshape(state_sim, (1, int(S_INFO), int(S_LEN)))).astype(np.float64).reshape(-1)
                a = int(np.argmax(p))

        return float(total_reward) / float(k)

    def _policy_k_step_rebuf_mean_argmax(
        self,
        env_snapshot,
        state_snapshot: np.ndarray,
        *,
        start_action: int,
        k_steps: int,
    ) -> Optional[float]:
        #* Simulator rollout of the mean rebuffering time when we force the first action to `start_action`
        #* and then follow the policy with argmax.
        k = int(k_steps)
        if k <= 0:
            return None

        env_sim = copy_env(env_snapshot)
        state_sim = np.array(state_snapshot, copy=True)

        total_rebuf_s = 0.0
        a = int(start_action)

        for i in range(int(k)):
            try:
                delay_ms, _sleep_ms, buffer_s, rebuf_s, chunk_size, next_sizes, end, remain = env_sim.get_video_chunk(int(a))
            except Exception:
                return None

            #* Match the oracle validity rule: sequences that end early are invalid for K-step comparison.
            if bool(end) and i < int(k) - 1:
                return None

            total_rebuf_s += float(rebuf_s)
            state_sim = self._update_state_like_simulator(
                state_sim,
                bit_rate=int(a),
                buffer_s=float(buffer_s),
                video_chunk_size=int(chunk_size),
                delay_ms=float(delay_ms),
                next_video_chunk_sizes=[int(x) for x in next_sizes],
                video_chunk_remain=int(remain),
            )

            if i < int(k) - 1:
                p = self.actor.predict(np.reshape(state_sim, (1, int(S_INFO), int(S_LEN)))).astype(np.float64).reshape(-1)
                a = int(np.argmax(p))

        return float(total_rebuf_s) / float(k)

    def _select_action_rebuf_target_action(
        self, env_snapshot, state_snapshot: np.ndarray, *, last_bitrate: int, k_steps: int
    ) -> tuple[Optional[int], float]:
        #* Target action selection for Objective 6: pick the first action with the largest K-step mean rebuffering.
        k = int(k_steps)
        if k <= 0:
            return None, float("nan")

        _ = int(last_bitrate)
        best_rebuf_mean = None
        best_action = None
        for a in range(int(A_DIM)):
            r = self._policy_k_step_rebuf_mean_argmax(
                env_snapshot,
                state_snapshot,
                start_action=int(a),
                k_steps=int(k),
            )
            if r is None or not np.isfinite(float(r)):
                continue
            if best_rebuf_mean is None or float(r) > float(best_rebuf_mean):
                best_rebuf_mean = float(r)
                best_action = int(a)

        if best_action is None or best_rebuf_mean is None:
            return None, float("nan")
        return int(best_action), float(best_rebuf_mean)

    def _select_action_gap_target_action(
        self, env_snapshot, state_snapshot: np.ndarray, *, last_bitrate: int, k_steps: int
    ) -> tuple[Optional[int], float, float]:
        #* Target action selection for Objective 5: pick the first action with the largest K-step oracle gap.
        k = int(k_steps)
        if k <= 0:
            return None, float("nan"), float("nan")

        best_r, _best_seq, _worst_r, _worst_seq = find_optimal_k_step_reward_oracle(env_snapshot, int(last_bitrate), int(k))
        if best_r is None:
            return None, float("nan"), float("nan")

        oracle_best_mean = float(best_r) / float(k)
        best_gap = None
        best_action = None
        for a in range(int(A_DIM)):
            pr = self._policy_k_step_reward_mean_argmax(
                env_snapshot,
                state_snapshot,
                last_bitrate=int(last_bitrate),
                start_action=int(a),
                k_steps=int(k),
            )
            if pr is None or not np.isfinite(float(pr)):
                continue
            gap = float(oracle_best_mean) - float(pr)
            if not np.isfinite(gap):
                continue
            if best_gap is None or gap > float(best_gap):
                best_gap = float(gap)
                best_action = int(a)

        if best_action is None or best_gap is None:
            return None, float(oracle_best_mean), float("nan")
        return int(best_action), float(oracle_best_mean), float(best_gap)
        
    def compute_gradient(self, state):
        """
        Compute gradient of loss w.r.t. state features (buffer_size and delay)
        
        Args:
            state: Current state (S_INFO, S_LEN) - numpy array
            
        Returns:
            Gradients for buffer_size and delay
        """
        if hasattr(self.actor, "actor"):
            self.actor.actor.zero_grad()
        # Convert to PyTorch tensor
        state_tensor = torch.FloatTensor(state.reshape(1, state.shape[0], state.shape[1]))
        state_tensor.requires_grad = True
        baseline_tensor = getattr(self, "_baseline_state_tensor", None)
        
        # Forward pass
        action_prob = self.actor.predict_tensor(state_tensor)

        objective_id = int(getattr(self, "_objective_id", 3))
        history_attack_enabled = bool(getattr(self, "_history_attack_enabled", False))

        #**************** Objective 1: Push action distribution toward uniform ****************#
        # Loss = KL(uniform || π(a|s_adv)). Minimizing pushes π toward uniform (high entropy).
        if objective_id == 1:
            eps = 1e-8
            num_actions = int(action_prob.shape[-1])
            uniform_dist = torch.ones_like(action_prob) / float(num_actions)
            loss = (uniform_dist * torch.log(uniform_dist / (action_prob + eps))).sum(dim=-1).mean()
            if baseline_tensor is not None and baseline_tensor.shape == state_tensor.shape:
                #* Small L2 penalty on perturbation magnitude.
                delta = state_tensor - baseline_tensor
                l2 = (delta * delta).mean()
                loss = loss + float(self._perturb_reg_weight) * l2

        elif objective_id in (5, 6):
            #**************** Objectives 5/6: Target-action (argmax) ****************#
            #* Two-stage objective:
            #  (1) Choose a target first action `a*` in `pgd_attack` (objective-specific).
            #  (2) Minimize -log π(a*|s_adv) so PGD pushes the policy toward that action.
            eps = 1e-8
            a_star = getattr(self, "_action_gap_target_action", None)
            if a_star is None or int(a_star) < 0 or int(a_star) >= int(action_prob.shape[-1]):
                #* Keep `loss` connected to the graph so `loss.backward()` succeeds (yields zero grads).
                loss = (action_prob * 0.0).sum()
            else:
                p_star = torch.clamp(action_prob[:, int(a_star)], min=eps)
                loss = (-torch.log(p_star)).mean()
                if baseline_tensor is not None and baseline_tensor.shape == state_tensor.shape:
                    #* Small L2 penalty on perturbation magnitude.
                    delta = state_tensor - baseline_tensor
                    l2 = (delta * delta).mean()
                    loss = loss + float(self._perturb_reg_weight) * l2

            # eps = 1e-8
            # k_steps = int(getattr(self, "_surrogate_k_steps", 0))
            # chunk_sizes_mb = getattr(self, "_surrogate_chunk_sizes_mb", None)
            # if k_steps > 0 and chunk_sizes_mb is not None:
            #     total_gap = self._expected_k_step_rebuf_gap_surrogate(
            #         state_tensor, chunk_sizes_mb=chunk_sizes_mb, k_steps=int(k_steps), eps=float(eps)
            #     )
            #     loss = (-total_gap).mean()
            # else:
            #     num_actions = int(action_prob.shape[-1])
            #     throughput_mb_s = torch.clamp(state_tensor[:, 2, -1], min=eps).view(-1, 1)
            #     next_chunk_mb = torch.clamp(state_tensor[:, 4, :num_actions], min=0.0)
            #     pred_delay_s = next_chunk_mb / throughput_mb_s
            #     buffer_s = (
            #         torch.clamp(state_tensor[:, 1, -1], min=0.0).view(-1, 1) * float(BUFFER_NORM_FACTOR)
            #     )
            #     pred_rebuf_s = torch.relu(pred_delay_s - buffer_s)
            #     expected_rebuf = (action_prob * pred_rebuf_s).sum(dim=-1)
            #     oracle_rebuf = torch.min(pred_rebuf_s, dim=-1).values
            #     loss = (-(expected_rebuf - oracle_rebuf)).mean()

        elif objective_id == 7:
            #* Objective 7: Decision flip (change clean argmax).
            #* Minimize log π(a_clean|s_adv) so PGD pushes probability mass away from the clean argmax.
            eps = 1e-8
            a_clean = getattr(self, "_decision_flip_clean_action", None)
            if a_clean is None or int(a_clean) < 0 or int(a_clean) >= int(action_prob.shape[-1]):
                #* Keep `loss` connected to the graph so `loss.backward()` succeeds (yields zero grads).
                loss = (action_prob * 0.0).sum()
            else:
                p_clean = action_prob[:, int(a_clean)]
                loss = torch.log(p_clean + eps).mean()
                if baseline_tensor is not None and baseline_tensor.shape == state_tensor.shape:
                    #* Small L2 penalty on perturbation magnitude.
                    delta = state_tensor - baseline_tensor
                    l2 = (delta * delta).mean()
                    loss = loss + float(self._perturb_reg_weight) * l2

        else:
            #* Objectives 2/3/4: QoE surrogate objectives (differentiable surrogate dynamics).
            eps = 1e-8
            num_actions = int(action_prob.shape[-1])

            br_kbps = torch.tensor(
                VIDEO_BIT_RATE[:num_actions], dtype=action_prob.dtype, device=action_prob.device
            )  # (A,)
            quality_mbps = (br_kbps / float(M_IN_K)).view(1, -1)  # (1, A)

            #* Last bitrate is encoded in the state as a normalized scalar feature.
            max_br = float(np.max(VIDEO_BIT_RATE))
            last_br_kbps = torch.clamp(state_tensor[:, 0, -1], min=0.0, max=1.0).view(-1, 1) * float(max_br)
            smoothness_mbps = (torch.abs(br_kbps.view(1, -1) - last_br_kbps) / float(M_IN_K))  # (1, A)

            #* Predict next-chunk delay from the throughput estimate in the current state.
            #  Note: state[2] stores throughput in MB/s (i.e., throughput_mbps / 8).
            throughput_mb_s = torch.clamp(state_tensor[:, 2, -1], min=eps).view(-1, 1)  # (1, 1)
            next_chunk_mb = torch.clamp(state_tensor[:, 4, :num_actions], min=0.0)  # (1, A)
            pred_delay_s = next_chunk_mb / throughput_mb_s  # (1, A)

            #* Rebuffering time for the *next* chunk: max(pred_delay_s - buffer_s, 0).
            buffer_s = torch.clamp(state_tensor[:, 1, -1], min=0.0).view(-1, 1) * float(BUFFER_NORM_FACTOR)  # (1, 1)
            pred_rebuf_s = torch.relu(pred_delay_s - buffer_s)  # (1, A)

            expected_quality = (action_prob * quality_mbps).sum(dim=-1)  # (1,)
            expected_rebuf = (action_prob * pred_rebuf_s).sum(dim=-1)  # (1,)
            expected_smoothness = (action_prob * smoothness_mbps).sum(dim=-1)  # (1,)
            expected_reward = (
                expected_quality
                - float(REBUF_PENALTY) * expected_rebuf
                - float(SMOOTH_PENALTY) * expected_smoothness
            )

            #* Objective 2: Minimize K-step expected QoE surrogate return (policy-only).
            if objective_id == 2:
                k_steps = int(getattr(self, "_surrogate_k_steps", 0))
                chunk_sizes_mb = getattr(self, "_surrogate_chunk_sizes_mb", None)
                if k_steps > 0:
                    if chunk_sizes_mb is None:
                        chunk0 = torch.clamp(state_tensor[:, 4, :num_actions], min=0.0).detach().cpu()
                        chunk_sizes_mb = chunk0.repeat(int(k_steps), 1).to(dtype=torch.float32)

                    horizon = int(min(int(k_steps), int(chunk_sizes_mb.shape[0])))
                    horizon = int(max(horizon, 1))
                    policy_total = self._expected_k_step_return_surrogate(
                        state_tensor,
                        chunk_sizes_mb=chunk_sizes_mb,
                        k_steps=int(k_steps),
                    )
                    loss = (policy_total / float(horizon)).mean()
                else:
                    loss = expected_reward.mean()

            #**************** Objective 3: Surrogate oracle-gap (policy vs surrogate oracle-best) ****************#
            elif objective_id == 3:
                k_steps = int(getattr(self, "_surrogate_k_steps", 0))
                chunk_sizes_mb = getattr(self, "_surrogate_chunk_sizes_mb", None)
                if k_steps > 0:
                    if chunk_sizes_mb is None:
                        chunk0 = torch.clamp(state_tensor[:, 4, :num_actions], min=0.0).detach().cpu()
                        chunk_sizes_mb = chunk0.repeat(int(k_steps), 1).to(dtype=torch.float32)

                    policy_total = self._expected_k_step_return_surrogate(
                        state_tensor,
                        chunk_sizes_mb=chunk_sizes_mb,
                        k_steps=int(k_steps),
                    )
                    oracle_total = self._oracle_best_k_step_return_surrogate(
                        state_tensor,
                        chunk_sizes_mb=chunk_sizes_mb,
                        k_steps=int(k_steps),
                        tau=float(getattr(self, "_surrogate_oracle_tau", 0.1)),
                        max_sequences=int(getattr(self, "_surrogate_oracle_max_sequences", 50000)),
                    )
                    loss = (policy_total - oracle_total).mean()
                else:
                    loss = expected_reward.mean()

                if baseline_tensor is not None and baseline_tensor.shape == state_tensor.shape:
                    #* Small L2 penalty on perturbation magnitude.
                    delta = state_tensor - baseline_tensor
                    l2 = (delta * delta).mean()
                    loss = loss + float(self._perturb_reg_weight) * l2

            #**************** Objective 4: Greedy/MPC-oracle-normalized QoE regret ****************#
            elif objective_id == 4:
                if history_attack_enabled:
                    k_steps = int(getattr(self, "_surrogate_k_steps", 0))
                    chunk_sizes_mb = getattr(self, "_surrogate_chunk_sizes_mb", None)
                    if k_steps > 0:
                        if chunk_sizes_mb is None:
                            chunk0 = torch.clamp(state_tensor[:, 4, :num_actions], min=0.0).detach().cpu()
                            chunk_sizes_mb = chunk0.repeat(int(k_steps), 1).to(dtype=torch.float32)
                        current_total = self._expected_k_step_return_surrogate(
                            state_tensor,
                            chunk_sizes_mb=chunk_sizes_mb,
                            k_steps=int(k_steps),
                        )
                    else:
                        current_total = expected_reward.view(-1)
                else:
                    current_total = expected_reward.view(-1)
                    if self._oracle_k_steps:
                        current_total = current_total * float(self._oracle_k_steps)

                if (
                    self._oracle_k_steps
                    and self._oracle_best_reward is not None
                    and self._oracle_worst_reward is not None
                ):
                    best_total = float(self._oracle_best_reward)
                    worst_total = float(self._oracle_worst_reward)
                    denom = float(max(best_total - worst_total, 1e-6))
                    loss = ((current_total - best_total) / denom).mean()
                else:
                    loss = current_total.mean()
            else:
                raise ValueError(f"* Unknown objective_id: {objective_id} (expected 1..7)")

        # Backward pass
        self._last_loss_value = float(loss.detach().item())
        loss.backward()

        return state_tensor.grad

    def compute_loss(self, state) -> float:
        #* Forward-only objective value for a given state (matches `compute_gradient`).
        state_tensor = torch.FloatTensor(state.reshape(1, state.shape[0], state.shape[1]))
        baseline_tensor = getattr(self, "_baseline_state_tensor", None)

        with torch.no_grad():
            action_prob = self.actor.predict_tensor(state_tensor)

            objective_id = int(getattr(self, "_objective_id", 3))
            history_attack_enabled = bool(getattr(self, "_history_attack_enabled", False))

            if objective_id == 1:
                eps = 1e-8
                num_actions = int(action_prob.shape[-1])
                uniform_dist = torch.ones_like(action_prob) / float(num_actions)
                loss = (uniform_dist * torch.log(uniform_dist / (action_prob + eps))).sum(dim=-1).mean()
                if baseline_tensor is not None and baseline_tensor.shape == state_tensor.shape:
                    delta = state_tensor - baseline_tensor
                    l2 = (delta * delta).mean()
                    loss = loss + float(self._perturb_reg_weight) * l2
            elif objective_id in (5, 6):
                eps = 1e-8
                a_star = getattr(self, "_action_gap_target_action", None)
                if a_star is None or int(a_star) < 0 or int(a_star) >= int(action_prob.shape[-1]):
                    loss = torch.zeros((), dtype=action_prob.dtype, device=action_prob.device)
                else:
                    p_star = torch.clamp(action_prob[:, int(a_star)], min=eps)
                    loss = (-torch.log(p_star)).mean()
                    if baseline_tensor is not None and baseline_tensor.shape == state_tensor.shape:
                        delta = state_tensor - baseline_tensor
                        l2 = (delta * delta).mean()
                        loss = loss + float(self._perturb_reg_weight) * l2

            elif objective_id == 7:
                eps = 1e-8
                a_clean = getattr(self, "_decision_flip_clean_action", None)
                if a_clean is None or int(a_clean) < 0 or int(a_clean) >= int(action_prob.shape[-1]):
                    loss = torch.zeros((), dtype=action_prob.dtype, device=action_prob.device)
                else:
                    p_clean = action_prob[:, int(a_clean)]
                    loss = torch.log(p_clean + eps).mean()
                    if baseline_tensor is not None and baseline_tensor.shape == state_tensor.shape:
                        delta = state_tensor - baseline_tensor
                        l2 = (delta * delta).mean()
                        loss = loss + float(self._perturb_reg_weight) * l2

            else:
                eps = 1e-8
                num_actions = int(action_prob.shape[-1])

                br_kbps = torch.tensor(
                    VIDEO_BIT_RATE[:num_actions], dtype=action_prob.dtype, device=action_prob.device
                )
                quality_mbps = (br_kbps / float(M_IN_K)).view(1, -1)

                max_br = float(np.max(VIDEO_BIT_RATE))
                last_br_kbps = torch.clamp(state_tensor[:, 0, -1], min=0.0, max=1.0).view(-1, 1) * float(max_br)
                smoothness_mbps = (torch.abs(br_kbps.view(1, -1) - last_br_kbps) / float(M_IN_K))

                throughput_mb_s = torch.clamp(state_tensor[:, 2, -1], min=eps).view(-1, 1)
                next_chunk_mb = torch.clamp(state_tensor[:, 4, :num_actions], min=0.0)
                pred_delay_s = next_chunk_mb / throughput_mb_s

                buffer_s = torch.clamp(state_tensor[:, 1, -1], min=0.0).view(-1, 1) * float(BUFFER_NORM_FACTOR)
                pred_rebuf_s = torch.relu(pred_delay_s - buffer_s)

                expected_quality = (action_prob * quality_mbps).sum(dim=-1)
                expected_rebuf = (action_prob * pred_rebuf_s).sum(dim=-1)
                expected_smoothness = (action_prob * smoothness_mbps).sum(dim=-1)
                expected_reward = (
                    expected_quality
                    - float(REBUF_PENALTY) * expected_rebuf
                    - float(SMOOTH_PENALTY) * expected_smoothness
                )

                if objective_id == 2:
                    k_steps = int(getattr(self, "_surrogate_k_steps", 0))
                    chunk_sizes_mb = getattr(self, "_surrogate_chunk_sizes_mb", None)
                    if k_steps > 0:
                        if chunk_sizes_mb is None:
                            chunk0 = torch.clamp(state_tensor[:, 4, :num_actions], min=0.0).detach().cpu()
                            chunk_sizes_mb = chunk0.repeat(int(k_steps), 1).to(dtype=torch.float32)

                        horizon = int(min(int(k_steps), int(chunk_sizes_mb.shape[0])))
                        horizon = int(max(horizon, 1))
                        policy_total = self._expected_k_step_return_surrogate(
                            state_tensor,
                            chunk_sizes_mb=chunk_sizes_mb,
                            k_steps=int(k_steps),
                        )
                        loss = (policy_total / float(horizon)).mean()
                    else:
                        loss = expected_reward.mean()
                elif objective_id == 3:
                    k_steps = int(getattr(self, "_surrogate_k_steps", 0))
                    chunk_sizes_mb = getattr(self, "_surrogate_chunk_sizes_mb", None)
                    if k_steps > 0:
                        if chunk_sizes_mb is None:
                            chunk0 = torch.clamp(state_tensor[:, 4, :num_actions], min=0.0).detach().cpu()
                            chunk_sizes_mb = chunk0.repeat(int(k_steps), 1).to(dtype=torch.float32)

                        policy_total = self._expected_k_step_return_surrogate(
                            state_tensor,
                            chunk_sizes_mb=chunk_sizes_mb,
                            k_steps=int(k_steps),
                        )
                        oracle_total = self._oracle_best_k_step_return_surrogate(
                            state_tensor,
                            chunk_sizes_mb=chunk_sizes_mb,
                            k_steps=int(k_steps),
                            tau=float(getattr(self, "_surrogate_oracle_tau", 0.1)),
                            max_sequences=int(getattr(self, "_surrogate_oracle_max_sequences", 50000)),
                        )
                        loss = (policy_total - oracle_total).mean()
                    else:
                        loss = expected_reward.mean()

                    if baseline_tensor is not None and baseline_tensor.shape == state_tensor.shape:
                        delta = state_tensor - baseline_tensor
                        l2 = (delta * delta).mean()
                        loss = loss + float(self._perturb_reg_weight) * l2

                elif objective_id == 4:
                    if history_attack_enabled:
                        k_steps = int(getattr(self, "_surrogate_k_steps", 0))
                        chunk_sizes_mb = getattr(self, "_surrogate_chunk_sizes_mb", None)
                        if k_steps > 0:
                            if chunk_sizes_mb is None:
                                chunk0 = torch.clamp(state_tensor[:, 4, :num_actions], min=0.0).detach().cpu()
                                chunk_sizes_mb = chunk0.repeat(int(k_steps), 1).to(dtype=torch.float32)
                            current_total = self._expected_k_step_return_surrogate(
                                state_tensor,
                                chunk_sizes_mb=chunk_sizes_mb,
                                k_steps=int(k_steps),
                            )
                        else:
                            current_total = expected_reward.view(-1)
                    else:
                        current_total = expected_reward.view(-1)
                        if self._oracle_k_steps:
                            current_total = current_total * float(self._oracle_k_steps)

                    if (
                        self._oracle_k_steps
                        and self._oracle_best_reward is not None
                        and self._oracle_worst_reward is not None
                    ):
                        best_total = float(self._oracle_best_reward)
                        worst_total = float(self._oracle_worst_reward)
                        denom = float(max(best_total - worst_total, 1e-6))
                        loss = ((current_total - best_total) / denom).mean()
                    else:
                        loss = current_total.mean()
                else:
                    raise ValueError(f"* Unknown objective_id: {objective_id} (expected 1..7)")

        return float(loss.item())
    
    def pgd_attack(
        self,
        original_state,
        target_bitrate,
        BUFFER_NORM_FACTOR,
        M_IN_K=1000.0,
        video_chunk_size=None,
        delay_norm_factor=10.0,
        enforce_coupling=True,
        perturb_buffer=False,
        use_rules=False,
        rules_file=None,
        rules_project_every_step=False,
        rules_perturb_both=True,
        perturb_delay_history=False,
        perturb_throughput_history=False,
        history_sparse=False,
        history_sparse_tol=0.05,
        oracle_env_snapshot=None,
        oracle_k_steps=K_steps,
        oracle_mpc_horizon=1,
        objective_id=5,
        early_stop=True,
        early_stop_margin=0.1,
        early_stop_patience=2,
        early_stop_min_delta=1e-6,
        surrogate_oracle_tau=0.1,
        surrogate_oracle_max_sequences=50000,
    ):
        """
        Perform PGD attack by perturbing a subset of state features.

        Notes:
        - By default, only the delay feature is directly perturbed (and buffer if `perturb_buffer=True`).
          Throughput is only directly perturbed when `perturb_throughput_history=True`; otherwise it may still
          change indirectly when coupling or rules are enforced.
        - When `perturb_delay_history` is enabled, PGD may perturb the full S_LEN delay history vector (row 3).
        - When `perturb_throughput_history` is enabled, PGD may perturb the full S_LEN throughput history
          vector (row 2), including the latest column.
        - If `video_chunk_size` is provided and `enforce_coupling` is True, we additionally
          project (throughput_norm, delay_norm) pairs to satisfy:
              throughput_norm = video_chunk_size_bytes / delay_ms / M_IN_K
          where delay_ms = delay_norm * M_IN_K * delay_norm_factor.
          For history columns (< -1) where per-step chunk sizes are not directly available,
          we infer an equivalent coupling constant from the original (throughput_norm, delay_norm)
          product and keep that constant fixed during PGD.
        - If `perturb_buffer` is False, buffer is kept fixed. With coupling enabled,
          this means the only independent PGD update is on delay (throughput follows
          from delay via the coupling constraint).
        
        Args:
            original_state: Original state array
            target_bitrate: Target bitrate index
            BUFFER_NORM_FACTOR: max allowed buffer feature (in normalized units)
            M_IN_K: M_IN_K constant
            video_chunk_size: downloaded chunk size (bytes) for the state being attacked
            delay_norm_factor: normalization factor used for delay (default matches env.py)
            enforce_coupling: whether to enforce throughput-delay coupling
            perturb_buffer: whether to perturb buffer feature
            use_rules: if True, project (delay, buffer, throughput) onto mined logic rules (implies no coupling)
            rules_file: path to rules file in SymPy-like syntax (default: rules/netnomos_abr_100000_p4_abr.pl)
            rules_project_every_step: if True and `use_rules` is enabled, do a Z3 projection after every PGD step
            rules_perturb_both: if True and `use_rules` is enabled, allow PGD to perturb (delay_norm, throughput_norm) before rule projection
            perturb_delay_history: if True, allow PGD to perturb delay history `state[3, :]`
            perturb_throughput_history: if True, allow PGD to perturb throughput history `state[2, :]`
                (if False, throughput is not directly perturbed)
            history_sparse: if True and any history perturbation is enabled, greedily sparsify the set of
                perturbable history values by ranking `abs(grad * delta)` from a full-history PGD run, then
                rerun PGD allowing only the top-k values until the loss is within `history_sparse_tol`.
            history_sparse_tol: relative tolerance for the greedy sparsification stopping criterion.
	            objective_id:
	                - 1: push π(a|s_adv) toward uniform (high entropy)
	                - 2: minimize K-step QoE surrogate reward (policy-only)
	                - 3: minimize surrogate oracle gap (Eπ[R_sur] - OracleBest_sur) under the attacked input
	                - 4: minimize oracle-normalized QoE regret (MPC oracle bounds)
	                - 5: action-gap target (pick worst first action by K-step oracle gap, then maximize π(a*|s_adv))
	                - 6: rebuffer-worst target (pick the first action with the largest K-step mean rebuffering, then maximize π(a*|s_adv))
	                - 7: decision-flip target (minimize log π(a_clean|s_adv) to change the clean argmax)
	            early_stop:
	                - If True and objective_id == 3, stop once the Objective-3 loss stops improving by `early_stop_min_delta`
	                  for `early_stop_patience` consecutive iterations (and return the best iterate).
	                - If True and objective_id in {5,6}, stop once the target action is reached with a log-prob margin.
	                - If True and objective_id == 7, stop once argmax(π(.|s_adv)) differs from the clean argmax.
            early_stop_margin: logit-gap margin threshold for early stopping (computed from log-probs).
            early_stop_patience: number of consecutive satisfied iterations to trigger early stop.
            early_stop_min_delta: Objective-3 early stop improvement threshold on the loss value.
            surrogate_oracle_tau: temperature for the soft surrogate oracle-best aggregation (smaller is closer to max).
            surrogate_oracle_max_sequences: max number of action sequences to enumerate for surrogate oracle-best.
            
        Returns:
            perturbed_state
        """
        state = np.array(original_state, copy=True)
        self._rules_projection_calls_last_attack = 0
        self._rules_fully_relaxed_calls_last_attack = 0
        self._rules_projection_last_attack_stats = {
            "pgd/rules_projection_attempts": 0,
            "pgd/rules_projection_success": 0,
            "pgd/rules_projection_fail": 0,
            "pgd/rules_projection_success_rate": 0.0,
            "pgd/rules_projection_window_attempts": 0,
            "pgd/rules_projection_window_success": 0,
            "pgd/rules_projection_window_fail": 0,
            "pgd/rules_projection_latest_attempts": 0,
            "pgd/rules_projection_latest_success": 0,
            "pgd/rules_projection_latest_fail": 0,
            "pgd/rules_projection_fallbacks": 0,
        }
        self._history_sparse_last_attack = {
            "pgd/history_sparse_applied": 0,
            "pgd/history_sparse_k": 0,
            "pgd/history_sparse_k_unique": 0,
            "pgd/history_sparse_candidates_total": 0,
            "pgd/history_sparse_selected_delay": 0,
            "pgd/history_sparse_selected_throughput": 0,
            "pgd/history_sparse_full_loss": float("nan"),
            "pgd/history_sparse_loss": float("nan"),
            "pgd/history_sparse_band": float("nan"),
        }
        self._early_stop_last_attack = {
            "pgd/early_stop_enabled": 0,
            "pgd/early_stop_triggered": 0,
            "pgd/early_stop_iter": 0,
            "pgd/early_stop_iters_run": 0,
            "pgd/early_stop_patience": 0,
            "pgd/early_stop_margin": float("nan"),
            "pgd/early_stop_gap": float("nan"),
            "pgd/early_stop_target_action": -1,
        }
        self._last_video_chunk_size = video_chunk_size
        self._objective_id = int(objective_id)
        if self._objective_id not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError(f"* objective_id must be in {{1,2,3,4,5,6,7}} (got {objective_id})")
        self._oracle_best_reward = None
        self._oracle_worst_reward = None
        self._oracle_k_steps = 0
        self._oracle_mpc_horizon = int(oracle_mpc_horizon)
        self._baseline_state_tensor = torch.FloatTensor(state.reshape(1, state.shape[0], state.shape[1]))
        self._action_gap_target_action = None
        self._action_gap_oracle_best_mean = None
        self._action_gap_target_gap = None
        self._decision_flip_clean_action = None
        if self._objective_id == 7:
            #* Objective 7 baseline: the clean argmax action that we aim to flip.
            try:
                p_clean = self.actor.predict(np.reshape(state, (1, int(S_INFO), int(S_LEN)))).astype(np.float64).reshape(-1)
                if p_clean.size > 0:
                    self._decision_flip_clean_action = int(np.argmax(p_clean))
            except Exception:
                self._decision_flip_clean_action = None
        try:
            self._surrogate_oracle_tau = float(surrogate_oracle_tau)
        except Exception:
            self._surrogate_oracle_tau = float(getattr(self, "_surrogate_oracle_tau", 0.1))
        self._surrogate_oracle_tau = float(max(self._surrogate_oracle_tau, 1e-6))
        try:
            self._surrogate_oracle_max_sequences = int(surrogate_oracle_max_sequences)
        except Exception:
            self._surrogate_oracle_max_sequences = int(getattr(self, "_surrogate_oracle_max_sequences", 50000))
        self._surrogate_oracle_max_sequences = int(max(self._surrogate_oracle_max_sequences, 1))

        if oracle_env_snapshot is not None and self._objective_id == 4:
            try:
                best_r, _, worst_r, _ = find_optimal_k_step_reward_oracle_mpc(
                    oracle_env_snapshot,
                    int(target_bitrate),
                    int(oracle_k_steps),
                    horizon=int(oracle_mpc_horizon),
                )
                if (
                    best_r is not None
                    and worst_r is not None
                    and np.isfinite(best_r)
                    and np.isfinite(worst_r)
                    and int(oracle_k_steps) > 0
                ):
                    self._oracle_best_reward = float(best_r)
                    self._oracle_worst_reward = float(worst_r)
                    self._oracle_k_steps = int(oracle_k_steps)
            except Exception:
                self._oracle_best_reward = None
                self._oracle_worst_reward = None
                self._oracle_k_steps = 0

        perturb_delay_history = bool(perturb_delay_history)
        perturb_throughput_history = bool(perturb_throughput_history)
        rules_perturb_both = bool(rules_perturb_both)
        if use_rules:
            #* Rules-based projection is the 3rd mode:
            #  - disable coupling (rules may already contain the coupling constraint)
            enforce_coupling = False
            #  - optional: do delay-only PGD by setting `rules_perturb_both=False`
            #  - buffer perturbation remains controlled by `perturb_buffer`
            #
            # If throughput perturbation is disabled (no history throughput), force delay-only mode
            # under rules: throughput is then determined by the (DelayMs, ChosenChunkBytes) coupling rule.
            if not perturb_throughput_history:
                rules_perturb_both = False
            if not rules_perturb_both:
                perturb_buffer = False
            if video_chunk_size is None:
                raise ValueError("* use_rules=True requires `video_chunk_size` (ChosenChunkBytes) to be provided")
            if rules_file is None:
                raise ValueError("* use_rules=True requires `rules_file` to be provided")
        
        # Extract original values (denormalized)
        history_len = int(state.shape[1])
        original_buffer_norm = state[1, -1]
        original_throughput_norm_hist = np.array(state[2, :], copy=True)
        original_delay_norm_hist = np.array(state[3, :], copy=True)
        
        # Initialize perturbations
        perturbed_buffer_norm = original_buffer_norm
        perturbed_throughput_norm_hist = np.array(original_throughput_norm_hist, copy=True)
        perturbed_delay_norm_hist = np.array(original_delay_norm_hist, copy=True)

        # Create state with current perturbations
        temp_state = np.array(state, copy=True)
        eps_delay_norm = 1e-6
        use_coupling = (
            enforce_coupling
            and video_chunk_size is not None
            and delay_norm_factor > 0
            and M_IN_K > 0
        )
        use_rules = bool(use_rules)
        rules_project_every_step = bool(rules_project_every_step)
        if use_rules:
            use_coupling = False

        rule_bitrate_kbps_hist = None
        rule_chunk_bytes_hist = None
        rule_buffer_norm_hist = None
        if use_rules:
            max_br = float(np.max(VIDEO_BIT_RATE))
            bitrate_values = np.array(VIDEO_BIT_RATE, dtype=np.float64)
            bitrate_norm_hist = np.array(state[0, :], dtype=np.float64, copy=False)
            bitrate_kbps_hist: list[int] = []
            for val in bitrate_norm_hist:
                if not np.isfinite(val):
                    bitrate_kbps_hist.append(int(VIDEO_BIT_RATE[int(target_bitrate)]))
                    continue
                br_kbps = float(val) * max_br
                idx = int(np.argmin(np.abs(bitrate_values - br_kbps)))
                bitrate_kbps_hist.append(int(VIDEO_BIT_RATE[idx]))
            rule_bitrate_kbps_hist = np.array(bitrate_kbps_hist, dtype=np.int64)

            delay_ms_hist = np.array(original_delay_norm_hist, dtype=np.float64) * float(M_IN_K) * float(delay_norm_factor)
            chunk_bytes_hist = np.array(original_throughput_norm_hist, dtype=np.float64) * delay_ms_hist * float(M_IN_K)
            chunk_bytes_hist[~np.isfinite(chunk_bytes_hist)] = 0.0
            chunk_bytes_hist = np.maximum(chunk_bytes_hist, 1.0)
            if video_chunk_size is not None:
                chunk_bytes_hist[-1] = float(video_chunk_size)
            rule_chunk_bytes_hist = chunk_bytes_hist

            rule_buffer_norm_hist = np.array(state[1, :], dtype=np.float64, copy=False)

        if use_coupling:
            #& Initial projection to make the starting state consistent
            '''
            delay_ms = delay_norm * M_IN_K * delay_norm_factor
            throughput_norm = video_chunk_size_bytes / (delay_norm * M_IN_K * delay_norm_factor) / M_IN_K
                        = (video_chunk_size_bytes / (M_IN_K^2 * delay_norm_factor)) / delay_norm
                        = C / delay_norm
            '''
            coupling_c_last = float(video_chunk_size) / (
                float(M_IN_K) * float(M_IN_K) * float(delay_norm_factor)
            )

            #* Per-column coupling constants:
            #  throughput_norm[j] * delay_norm[j] = coupling_c_hist[j]
            # For history columns, infer C from the original state (equivalent to inferring the
            # per-step chunk size); for the latest column, prefer the provided chunk size.
            coupling_c_hist = np.array(original_throughput_norm_hist, dtype=np.float64, copy=True)
            coupling_c_hist *= np.array(original_delay_norm_hist, dtype=np.float64, copy=False)
            coupling_c_hist[~np.isfinite(coupling_c_hist)] = 0.0
            coupling_c_hist[-1] = float(coupling_c_last)
            coupling_mask = np.isfinite(coupling_c_hist) & (coupling_c_hist > 0.0)

            if np.any(coupling_mask):
                perturbed_delay_norm_hist[coupling_mask] = np.maximum(
                    perturbed_delay_norm_hist[coupling_mask], eps_delay_norm
                )
                perturbed_throughput_norm_hist[coupling_mask] = (
                    coupling_c_hist[coupling_mask] / perturbed_delay_norm_hist[coupling_mask]
                )
            temp_state[2, :] = np.array(perturbed_throughput_norm_hist, copy=False)
            temp_state[3, :] = np.array(perturbed_delay_norm_hist, copy=False)

        if self._objective_id == 5:
            # try:
            #     if oracle_env_snapshot is None or oracle_k_steps is None or int(oracle_k_steps) <= 0:
            #         self._action_gap_target_action = None
            #         self._action_gap_oracle_best_mean = None
            #         self._action_gap_target_gap = None
            #     else:
            #         a_star, oracle_best_mean, gap = self._select_action_gap_target_action(
            #             oracle_env_snapshot,
            #             np.array(temp_state, copy=True),
            #             last_bitrate=int(target_bitrate),
            #             k_steps=int(oracle_k_steps),
            #         )
            #         self._action_gap_target_action = a_star
            #         self._action_gap_oracle_best_mean = float(oracle_best_mean)
            #         self._action_gap_target_gap = float(gap)
            # except Exception:
            #     self._action_gap_target_action = None
            #     self._action_gap_oracle_best_mean = None
            #     self._action_gap_target_gap = None

            #* Objective 5 target-action selection: pick the first action with the largest K-step oracle gap.
            try:
                if oracle_env_snapshot is None or oracle_k_steps is None or int(oracle_k_steps) <= 0:
                    self._action_gap_target_action = None
                    self._action_gap_oracle_best_mean = None
                    self._action_gap_target_gap = None
                else:
                    a_star, oracle_best_mean, gap = self._select_action_gap_target_action(
                        oracle_env_snapshot,
                        np.array(temp_state, copy=True),
                        last_bitrate=int(target_bitrate),
                        k_steps=int(oracle_k_steps),
                    )
                    self._action_gap_target_action = a_star
                    self._action_gap_oracle_best_mean = float(oracle_best_mean)
                    self._action_gap_target_gap = float(gap)
            except Exception:
                self._action_gap_target_action = None
                self._action_gap_oracle_best_mean = None
                self._action_gap_target_gap = None
        elif self._objective_id == 6:
            #* Objective 6 target-action selection: pick the first action that yields the worst K-step rebuffering.
            try:
                if oracle_env_snapshot is None or oracle_k_steps is None or int(oracle_k_steps) <= 0:
                    self._action_gap_target_action = None
                    self._action_gap_oracle_best_mean = None
                    self._action_gap_target_gap = None
                else:
                    a_star, rebuf_mean = self._select_action_rebuf_target_action(
                        oracle_env_snapshot,
                        np.array(temp_state, copy=True),
                        last_bitrate=int(target_bitrate),
                        k_steps=int(oracle_k_steps),
                    )
                    self._action_gap_target_action = a_star
                    self._action_gap_oracle_best_mean = None
                    self._action_gap_target_gap = float(rebuf_mean)
            except Exception:
                self._action_gap_target_action = None
                self._action_gap_oracle_best_mean = None
                self._action_gap_target_gap = None

        # PGD iterations
        self._history_attack_enabled = bool(perturb_delay_history or perturb_throughput_history)
        self._surrogate_k_steps = int(oracle_k_steps) if oracle_k_steps is not None else 0
        self._surrogate_chunk_sizes_mb = None
        if self._objective_id in (2, 3, 4):
            self._surrogate_chunk_sizes_mb = self._prepare_k_step_chunk_sizes_mb(
                oracle_env_snapshot,
                k_steps=int(self._surrogate_k_steps),
                num_actions=int(A_DIM),
            )
        delay_update_idx_full = list(range(history_len)) if perturb_delay_history else [history_len - 1]
        # If throughput history perturbation is disabled, treat throughput as not directly perturbable.
        # (Under rules/coupling, throughput can still change as a consequence of delay changes.)
        throughput_update_idx_full = list(range(history_len)) if perturb_throughput_history else []

        init_buffer_norm = float(perturbed_buffer_norm)
        init_throughput_norm_hist = np.array(perturbed_throughput_norm_hist, copy=True)
        init_delay_norm_hist = np.array(perturbed_delay_norm_hist, copy=True)
        init_state = np.array(temp_state, copy=True)

        def _run_pgd(
            delay_update_idx: Sequence[int], throughput_update_idx: Sequence[int]
        ) -> tuple[np.ndarray, dict[str, float | int], dict[str, float | int]]:
            #^ Run PGD with an explicit set of perturbable indices (used by history sparsification).
            pert_buffer = float(init_buffer_norm)
            pert_thr_hist = np.array(init_throughput_norm_hist, copy=True)
            pert_delay_hist = np.array(init_delay_norm_hist, copy=True)
            cur_state = np.array(init_state, copy=True)

            delay_update_set = set(int(j) for j in delay_update_idx)
            throughput_update_set = set(int(j) for j in throughput_update_idx)
            last_rule_buffer = float(pert_buffer)
            last_rule_delay_hist = np.array(pert_delay_hist, copy=True)
            last_rule_thr_hist = np.array(pert_thr_hist, copy=True)

            rules_stats: dict[str, float | int] = {
                "pgd/rules_projection_attempts": 0,
                "pgd/rules_projection_success": 0,
                "pgd/rules_projection_fail": 0,
                "pgd/rules_projection_success_rate": 0.0,
                "pgd/rules_projection_window_attempts": 0,
                "pgd/rules_projection_window_success": 0,
                "pgd/rules_projection_window_fail": 0,
                "pgd/rules_projection_latest_attempts": 0,
                "pgd/rules_projection_latest_success": 0,
                "pgd/rules_projection_latest_fail": 0,
                "pgd/rules_projection_fallbacks": 0,
            }

            def _record_rules_projection_attempt(*, success: bool, is_window: bool, fallback: bool) -> None:
                rules_stats["pgd/rules_projection_attempts"] = int(rules_stats["pgd/rules_projection_attempts"]) + 1
                if bool(success):
                    rules_stats["pgd/rules_projection_success"] = int(rules_stats["pgd/rules_projection_success"]) + 1
                    self._rules_projection_success_total = int(self._rules_projection_success_total) + 1
                    if bool(is_window):
                        rules_stats["pgd/rules_projection_window_success"] = int(
                            rules_stats["pgd/rules_projection_window_success"]
                        ) + 1
                    else:
                        rules_stats["pgd/rules_projection_latest_success"] = int(
                            rules_stats["pgd/rules_projection_latest_success"]
                        ) + 1
                else:
                    rules_stats["pgd/rules_projection_fail"] = int(rules_stats["pgd/rules_projection_fail"]) + 1
                    self._rules_projection_fail_total = int(self._rules_projection_fail_total) + 1
                    if bool(is_window):
                        rules_stats["pgd/rules_projection_window_fail"] = int(rules_stats["pgd/rules_projection_window_fail"]) + 1
                    else:
                        rules_stats["pgd/rules_projection_latest_fail"] = int(rules_stats["pgd/rules_projection_latest_fail"]) + 1
                if bool(is_window):
                    rules_stats["pgd/rules_projection_window_attempts"] = int(
                        rules_stats["pgd/rules_projection_window_attempts"]
                    ) + 1
                else:
                    rules_stats["pgd/rules_projection_latest_attempts"] = int(
                        rules_stats["pgd/rules_projection_latest_attempts"]
                    ) + 1
                if bool(fallback):
                    rules_stats["pgd/rules_projection_fallbacks"] = int(rules_stats["pgd/rules_projection_fallbacks"]) + 1
                    self._rules_projection_fallback_total = int(self._rules_projection_fallback_total) + 1

            #^ Early stop modes:
            #  - Objective 3: stop when the loss stops improving
            #  - Objectives 5/6: stop when argmax == a* with margin
            #  - Objective 7: stop when argmax != clean argmax
            objective_id_local = int(getattr(self, "_objective_id", 3))
            a_star = getattr(self, "_action_gap_target_action", None)
            a_clean = getattr(self, "_decision_flip_clean_action", None)

            stop_margin = float(early_stop_margin)
            if not np.isfinite(stop_margin) or stop_margin < 0.0:
                stop_margin = 0.0
            stop_patience = int(early_stop_patience)
            if stop_patience <= 0:
                stop_patience = 1

            stop_min_delta = float(early_stop_min_delta)
            if not np.isfinite(stop_min_delta) or stop_min_delta < 0.0:
                stop_min_delta = 0.0

            enable_action_early_stop = (
                bool(early_stop)
                and objective_id_local in (5, 6)
                and a_star is not None
                and int(a_star) >= 0
                and int(a_star) < int(A_DIM)
            )
            enable_obj3_early_stop = bool(early_stop) and int(objective_id_local) == 3
            enable_obj7_early_stop = (
                bool(early_stop)
                and int(objective_id_local) == 7
                and a_clean is not None
                and int(a_clean) >= 0
                and int(a_clean) < int(A_DIM)
                and (not bool(use_rules) or bool(rules_project_every_step))
            )
            enable_early_stop = bool(enable_action_early_stop or enable_obj3_early_stop or enable_obj7_early_stop)
            stop_metrics: dict[str, float | int] = {
                "pgd/early_stop_enabled": int(enable_early_stop),
                "pgd/early_stop_triggered": 0,
                "pgd/early_stop_iter": 0,
                "pgd/early_stop_iters_run": 0,
                "pgd/early_stop_patience": (
                    int(stop_patience)
                    if (enable_action_early_stop or enable_obj3_early_stop)
                    else (1 if enable_obj7_early_stop else 0)
                ),
                "pgd/early_stop_margin": float(stop_margin) if enable_action_early_stop else float("nan"),
                "pgd/early_stop_gap": float("nan"),
                "pgd/early_stop_target_action": (
                    int(a_star)
                    if enable_action_early_stop
                    else (int(a_clean) if enable_obj7_early_stop else -1)
                ),
                "pgd/early_stop_min_delta": float(stop_min_delta) if enable_obj3_early_stop else float("nan"),
                "pgd/early_stop_best_loss": float("nan"),
                "pgd/early_stop_best_iter": 0,
            }

            consecutive_ok = 0
            last_gap = float("nan")
            best_loss = float("nan")
            best_iter = 0
            no_improve = 0
            best_pert_buffer = float(pert_buffer)
            best_pert_delay_hist = np.array(pert_delay_hist, copy=True)
            best_pert_thr_hist = np.array(pert_thr_hist, copy=True)
            iters_run = 0

            for iter_idx in range(self.num_iterations):
                grad = self.compute_gradient(cur_state)

                if int(objective_id_local) == 3:
                    cur_loss = float(getattr(self, "_last_loss_value", float("nan")))
                    if np.isfinite(cur_loss):
                        if not np.isfinite(best_loss) or cur_loss < float(best_loss) - float(stop_min_delta):
                            best_loss = float(cur_loss)
                            best_iter = int(iters_run)
                            no_improve = 0
                            best_pert_buffer = float(pert_buffer)
                            best_pert_delay_hist = np.array(pert_delay_hist, copy=True)
                            best_pert_thr_hist = np.array(pert_thr_hist, copy=True)
                        else:
                            no_improve += 1

                    if bool(enable_obj3_early_stop) and int(no_improve) >= int(stop_patience):
                        stop_metrics["pgd/early_stop_triggered"] = 1
                        stop_metrics["pgd/early_stop_iter"] = int(iters_run)
                        stop_metrics["pgd/early_stop_best_loss"] = float(best_loss)
                        stop_metrics["pgd/early_stop_best_iter"] = int(best_iter)
                        break

                grad_buffer = float(grad[0, 1, -1].item())
                grad_throughput_hist = grad[0, 2, :].detach().cpu().numpy()
                grad_delay_hist = grad[0, 3, :].detach().cpu().numpy()
                grad_throughput_last = float(grad_throughput_hist[-1])
                grad_delay_last = float(grad_delay_hist[-1])

                if perturb_buffer:
                    # pert_buffer = float(pert_buffer) - self.alpha * float(np.sign(grad_buffer))
                    #* Full-gradient step.
                    pert_buffer = float(pert_buffer) - self.alpha * float(grad_buffer)
                else:
                    pert_buffer = float(original_buffer_norm)

                if use_coupling:
                    if coupling_c_hist is None or coupling_mask is None:  # pragma: no cover
                        raise RuntimeError("* Internal error: coupling constants not initialized")

                    coupled_update_idx = set(int(j) for j in delay_update_set) | set(int(j) for j in throughput_update_set)
                    for j in coupled_update_idx:
                        if not bool(coupling_mask[int(j)]):
                            continue

                        delay_norm_safe = max(float(pert_delay_hist[int(j)]), eps_delay_norm)
                        throughput_norm_safe = float(coupling_c_hist[int(j)]) / delay_norm_safe
                        d_throughput_d_delay = -throughput_norm_safe / delay_norm_safe
                        grad_delay_total = float(grad_delay_hist[int(j)]) + float(grad_throughput_hist[int(j)]) * float(
                            d_throughput_d_delay
                        )

                        # pert_delay_hist[int(j)] = float(pert_delay_hist[int(j)]) - self.alpha * float(
                        #     np.sign(grad_delay_total)
                        # )
                        #* Full-gradient step (on the coupled objective).
                        pert_delay_hist[int(j)] = float(pert_delay_hist[int(j)]) - self.alpha * float(grad_delay_total)
                        pert_delay_hist[int(j)] = max(float(pert_delay_hist[int(j)]), eps_delay_norm)
                        pert_thr_hist[int(j)] = float(coupling_c_hist[int(j)]) / float(pert_delay_hist[int(j)])
                elif use_rules:
                    #^ Optional history updates (projection applies to latest only unless `--rules-every-step`).
                    for j in delay_update_set:
                        if int(j) == history_len - 1:
                            continue
                        # pert_delay_hist[int(j)] = float(pert_delay_hist[int(j)]) - self.alpha * float(
                        #     np.sign(grad_delay_hist[int(j)])
                        # )
                        #* Full-gradient step.
                        pert_delay_hist[int(j)] = float(pert_delay_hist[int(j)]) - self.alpha * float(grad_delay_hist[int(j)])
                    for j in throughput_update_set:
                        if int(j) == history_len - 1:
                            continue
                        # pert_thr_hist[int(j)] = float(pert_thr_hist[int(j)]) - self.alpha * float(
                        #     np.sign(grad_throughput_hist[int(j)])
                        # )
                        #* Full-gradient step.
                        pert_thr_hist[int(j)] = float(pert_thr_hist[int(j)]) - self.alpha * float(
                            grad_throughput_hist[int(j)]
                        )

                    last_idx = int(history_len - 1)
                    if last_idx in delay_update_set:
                        # pert_delay_hist[last_idx] = float(pert_delay_hist[last_idx]) - self.alpha * float(
                        #     np.sign(grad_delay_last)
                        # )
                        #* Full-gradient step.
                        pert_delay_hist[last_idx] = float(pert_delay_hist[last_idx]) - self.alpha * float(grad_delay_last)
                        pert_delay_hist[last_idx] = max(float(pert_delay_hist[last_idx]), eps_delay_norm)

                    if rules_perturb_both:
                        if last_idx in throughput_update_set:
                            # pert_thr_hist[last_idx] = float(pert_thr_hist[last_idx]) - self.alpha * float(
                            #     np.sign(grad_throughput_last)
                            # )
                            #* Full-gradient step.
                            pert_thr_hist[last_idx] = float(pert_thr_hist[last_idx]) - self.alpha * float(
                                grad_throughput_last
                            )
                    else:
                        if not rules_project_every_step:
                            pert_thr_hist[last_idx] = float(original_throughput_norm_hist[last_idx])
                else:
                    for j in throughput_update_set:
                        # pert_thr_hist[int(j)] = float(pert_thr_hist[int(j)]) - self.alpha * float(
                        #     np.sign(grad_throughput_hist[int(j)])
                        # )
                        #* Full-gradient step.
                        pert_thr_hist[int(j)] = float(pert_thr_hist[int(j)]) - self.alpha * float(
                            grad_throughput_hist[int(j)]
                        )
                    for j in delay_update_set:
                        # pert_delay_hist[int(j)] = float(pert_delay_hist[int(j)]) - self.alpha * float(
                        #     np.sign(grad_delay_hist[int(j)])
                        # )
                        #* Full-gradient step.
                        pert_delay_hist[int(j)] = float(pert_delay_hist[int(j)]) - self.alpha * float(grad_delay_hist[int(j)])

                pert_buffer = np.clip(pert_buffer, 0, BUFFER_NORM_FACTOR)
                pert_delay_hist = np.clip(pert_delay_hist, 0, None)
                if use_coupling:
                    pert_delay_hist[coupling_mask] = np.maximum(pert_delay_hist[coupling_mask], eps_delay_norm)
                    pert_thr_hist[coupling_mask] = coupling_c_hist[coupling_mask] / pert_delay_hist[coupling_mask]
                pert_thr_hist = np.clip(pert_thr_hist, 0, None)

                if use_rules and rules_project_every_step:
                    #^ Per-step projection (projected gradient descent). This is expensive because
                    #  it adds one Z3 call per PGD iteration.
                    if rule_bitrate_kbps_hist is None or rule_chunk_bytes_hist is None or rule_buffer_norm_hist is None:
                        raise RuntimeError("* Internal error: rules metadata not initialized")

                    buffer_seconds_max = float(BUFFER_NORM_FACTOR) * float(delay_norm_factor)
                    if self._history_attack_enabled:
                        cand_delay_ms_hist = [float(x) * float(M_IN_K) * float(delay_norm_factor) for x in pert_delay_hist.tolist()]
                        chosen_chunk_bytes_hist = [
                            int(max(1.0, float(x))) for x in np.array(rule_chunk_bytes_hist, copy=False).tolist()
                        ]
                        if rules_perturb_both:
                            cand_thr_mbps_hist = [float(x) * 8.0 for x in pert_thr_hist.tolist()]
                        else:
                            cand_thr_mbps_hist = []
                            for j in range(history_len):
                                delay_ms = max(float(cand_delay_ms_hist[j]), 1e-6)
                                chosen_bytes = int(chosen_chunk_bytes_hist[j])
                                cand_thr_mbps_hist.append(float(chosen_bytes) * 8.0 / (float(M_IN_K) * delay_ms))

                        cand_buffer_seconds_hist = [
                            float(x) * float(delay_norm_factor) for x in np.array(rule_buffer_norm_hist, copy=False).tolist()
                        ]
                        if perturb_buffer:
                            cand_buffer_seconds_hist[-1] = float(pert_buffer) * float(delay_norm_factor)
                        fix_buffer_hist = [True] * int(history_len)
                        fix_buffer_hist[-1] = not bool(perturb_buffer)

                        sol = self._project_with_rules(
                            rules_file=str(rules_file),
                            bitrate_kbps_hist=[int(x) for x in np.array(rule_bitrate_kbps_hist, copy=False).tolist()],
                            buffer_seconds_hist=cand_buffer_seconds_hist,
                            fix_buffer_hist=fix_buffer_hist,
                            buffer_seconds_max=buffer_seconds_max,
                            chosen_chunk_bytes_hist=chosen_chunk_bytes_hist,
                            delay_ms_hist=cand_delay_ms_hist,
                            throughput_mbps_hist=cand_thr_mbps_hist,
                        )
                        if sol is not None:
                            sol_delay_ms_hist, sol_thr_mbps_hist, sol_buffer_s_hist = sol
                            for j in range(history_len):
                                pert_delay_hist[j] = float(sol_delay_ms_hist[j]) / (float(M_IN_K) * float(delay_norm_factor))
                                pert_thr_hist[j] = float(sol_thr_mbps_hist[j]) / 8.0
                            if perturb_buffer:
                                pert_buffer = float(sol_buffer_s_hist[-1]) / float(delay_norm_factor)
                            else:
                                pert_buffer = float(original_buffer_norm)
                            last_rule_buffer = float(pert_buffer)
                            last_rule_delay_hist = np.array(pert_delay_hist, copy=True)
                            last_rule_thr_hist = np.array(pert_thr_hist, copy=True)
                            _record_rules_projection_attempt(success=True, is_window=True, fallback=False)
                        else:
                            pert_buffer = float(last_rule_buffer)
                            pert_delay_hist = np.array(last_rule_delay_hist, copy=True)
                            pert_thr_hist = np.array(last_rule_thr_hist, copy=True)
                            _record_rules_projection_attempt(success=False, is_window=True, fallback=True)
                    else:
                        j = int(history_len - 1)
                        bitrate_kbps = int(rule_bitrate_kbps_hist[j])
                        if perturb_buffer:
                            buffer_seconds = float(pert_buffer) * float(delay_norm_factor)
                            fix_buffer = False
                        else:
                            buffer_seconds = float(rule_buffer_norm_hist[j]) * float(delay_norm_factor)
                            fix_buffer = True

                        cand_delay_ms = float(pert_delay_hist[j]) * float(M_IN_K) * float(delay_norm_factor)
                        chosen_chunk_bytes = int(max(1.0, float(rule_chunk_bytes_hist[j])))
                        if rules_perturb_both:
                            cand_thr_mbps = float(pert_thr_hist[j]) * 8.0
                        else:
                            cand_thr_mbps = float(chosen_chunk_bytes) * 8.0 / (float(M_IN_K) * max(cand_delay_ms, 1e-6))

                        sol = self._project_with_rules(
                            rules_file=str(rules_file),
                            bitrate_kbps=bitrate_kbps,
                            buffer_seconds=buffer_seconds,
                            buffer_seconds_max=buffer_seconds_max,
                            chosen_chunk_bytes=chosen_chunk_bytes,
                            delay_ms=cand_delay_ms,
                            throughput_mbps=cand_thr_mbps,
                            fix_buffer=fix_buffer,
                            latest_only=True,
                        )
                        if sol is not None:
                            sol_delay_ms, sol_thr_mbps, sol_buffer_s = sol
                            pert_delay_hist[j] = float(sol_delay_ms) / (float(M_IN_K) * float(delay_norm_factor))
                            pert_thr_hist[j] = float(sol_thr_mbps) / 8.0
                            if perturb_buffer:
                                pert_buffer = float(sol_buffer_s) / float(delay_norm_factor)
                            else:
                                pert_buffer = float(original_buffer_norm)
                            last_rule_buffer = float(pert_buffer)
                            last_rule_delay_hist = np.array(pert_delay_hist, copy=True)
                            last_rule_thr_hist = np.array(pert_thr_hist, copy=True)
                            _record_rules_projection_attempt(success=True, is_window=False, fallback=False)
                        else:
                            pert_buffer = float(last_rule_buffer)
                            pert_delay_hist = np.array(last_rule_delay_hist, copy=True)
                            pert_thr_hist = np.array(last_rule_thr_hist, copy=True)
                            _record_rules_projection_attempt(success=False, is_window=False, fallback=True)

                cur_state[1, -1] = float(pert_buffer)
                cur_state[2, :] = np.array(pert_thr_hist, copy=False)
                cur_state[3, :] = np.array(pert_delay_hist, copy=False)

                iters_run = int(iter_idx) + 1
                if enable_action_early_stop:
                    #^ Robust early stop: require the target action to be argmax with a margin for
                    #  several consecutive iterations (post-projection) to avoid oscillation.
                    p_obs = self.actor.predict(np.reshape(cur_state, (1, int(S_INFO), int(S_LEN)))).astype(np.float64).reshape(-1)
                    if p_obs.size >= 2:
                        a_hat = int(np.argmax(p_obs))
                        if int(a_hat) == int(a_star):
                            eps = 1e-12
                            p_star = float(p_obs[int(a_hat)])
                            p_other = float(np.max(np.delete(p_obs, int(a_hat))))
                            gap = float(np.log(p_star + eps) - np.log(p_other + eps))
                            last_gap = float(gap)
                            if np.isfinite(gap) and gap >= float(stop_margin):
                                consecutive_ok += 1
                            else:
                                consecutive_ok = 0
                        else:
                            consecutive_ok = 0

                    if int(consecutive_ok) >= int(stop_patience):
                        stop_metrics["pgd/early_stop_triggered"] = 1
                        stop_metrics["pgd/early_stop_iter"] = int(iters_run)
                        stop_metrics["pgd/early_stop_gap"] = float(last_gap)
                        break

                if enable_obj7_early_stop:
                    p_obs = self.actor.predict(np.reshape(cur_state, (1, int(S_INFO), int(S_LEN)))).astype(np.float64).reshape(-1)
                    if p_obs.size >= 2 and a_clean is not None and 0 <= int(a_clean) < int(p_obs.size):
                        a_hat = int(np.argmax(p_obs))
                        if int(a_hat) != int(a_clean):
                            eps = 1e-12
                            p_clean = float(p_obs[int(a_clean)])
                            p_other = float(np.max(np.delete(p_obs, int(a_clean))))
                            gap = float(np.log(p_other + eps) - np.log(p_clean + eps))
                            last_gap = float(gap)
                            stop_metrics["pgd/early_stop_triggered"] = 1
                            stop_metrics["pgd/early_stop_iter"] = int(iters_run)
                            stop_metrics["pgd/early_stop_gap"] = float(last_gap)
                            break

            if int(objective_id_local) == 3 and bool(early_stop) and np.isfinite(best_loss):
                #* Restore the best iterate (in feature space) before any final projection step.
                pert_buffer = float(best_pert_buffer)
                pert_delay_hist = np.array(best_pert_delay_hist, copy=True)
                pert_thr_hist = np.array(best_pert_thr_hist, copy=True)
                cur_state[1, -1] = float(pert_buffer)
                cur_state[2, :] = np.array(pert_thr_hist, copy=False)
                cur_state[3, :] = np.array(pert_delay_hist, copy=False)
                stop_metrics["pgd/early_stop_best_loss"] = float(best_loss)
                stop_metrics["pgd/early_stop_best_iter"] = int(best_iter)

            if use_rules:
                #^ Final projection (one Z3 call per attack) to keep the output state rule-consistent.
                if rule_bitrate_kbps_hist is None or rule_chunk_bytes_hist is None or rule_buffer_norm_hist is None:
                    raise RuntimeError("* Internal error: rules metadata not initialized")

                buffer_seconds_max = float(BUFFER_NORM_FACTOR) * float(delay_norm_factor)
                if self._history_attack_enabled:
                    cand_delay_ms_hist = [float(x) * float(M_IN_K) * float(delay_norm_factor) for x in pert_delay_hist.tolist()]
                    chosen_chunk_bytes_hist = [
                        int(max(1.0, float(x))) for x in np.array(rule_chunk_bytes_hist, copy=False).tolist()
                    ]
                    if rules_perturb_both:
                        cand_thr_mbps_hist = [float(x) * 8.0 for x in pert_thr_hist.tolist()]
                    else:
                        cand_thr_mbps_hist = []
                        for j in range(history_len):
                            delay_ms = max(float(cand_delay_ms_hist[j]), 1e-6)
                            chosen_bytes = int(chosen_chunk_bytes_hist[j])
                            cand_thr_mbps_hist.append(float(chosen_bytes) * 8.0 / (float(M_IN_K) * delay_ms))

                    cand_buffer_seconds_hist = [
                        float(x) * float(delay_norm_factor) for x in np.array(rule_buffer_norm_hist, copy=False).tolist()
                    ]
                    if perturb_buffer:
                        cand_buffer_seconds_hist[-1] = float(pert_buffer) * float(delay_norm_factor)
                    fix_buffer_hist = [True] * int(history_len)
                    fix_buffer_hist[-1] = not bool(perturb_buffer)

                    sol = self._project_with_rules(
                        rules_file=str(rules_file),
                        bitrate_kbps_hist=[int(x) for x in np.array(rule_bitrate_kbps_hist, copy=False).tolist()],
                        buffer_seconds_hist=cand_buffer_seconds_hist,
                        fix_buffer_hist=fix_buffer_hist,
                        buffer_seconds_max=buffer_seconds_max,
                        chosen_chunk_bytes_hist=chosen_chunk_bytes_hist,
                        delay_ms_hist=cand_delay_ms_hist,
                        throughput_mbps_hist=cand_thr_mbps_hist,
                    )
                    if sol is not None:
                        sol_delay_ms_hist, sol_thr_mbps_hist, sol_buffer_s_hist = sol
                        for j in range(history_len):
                            pert_delay_hist[j] = float(sol_delay_ms_hist[j]) / (float(M_IN_K) * float(delay_norm_factor))
                            pert_thr_hist[j] = float(sol_thr_mbps_hist[j]) / 8.0
                        if perturb_buffer:
                            pert_buffer = float(sol_buffer_s_hist[-1]) / float(delay_norm_factor)
                        else:
                            pert_buffer = float(original_buffer_norm)
                        last_rule_buffer = float(pert_buffer)
                        last_rule_delay_hist = np.array(pert_delay_hist, copy=True)
                        last_rule_thr_hist = np.array(pert_thr_hist, copy=True)
                        _record_rules_projection_attempt(success=True, is_window=True, fallback=False)
                    else:
                        pert_buffer = float(last_rule_buffer)
                        pert_delay_hist = np.array(last_rule_delay_hist, copy=True)
                        pert_thr_hist = np.array(last_rule_thr_hist, copy=True)
                        _record_rules_projection_attempt(success=False, is_window=True, fallback=True)
                else:
                    j = int(history_len - 1)
                    bitrate_kbps = int(rule_bitrate_kbps_hist[j])

                    if perturb_buffer:
                        buffer_seconds = float(pert_buffer) * float(delay_norm_factor)
                        fix_buffer = False
                    else:
                        buffer_seconds = float(rule_buffer_norm_hist[j]) * float(delay_norm_factor)
                        fix_buffer = True

                    cand_delay_ms = float(pert_delay_hist[j]) * float(M_IN_K) * float(delay_norm_factor)
                    chosen_chunk_bytes = int(max(1.0, float(rule_chunk_bytes_hist[j])))
                    if rules_perturb_both:
                        cand_thr_mbps = float(pert_thr_hist[j]) * 8.0
                    else:
                        cand_thr_mbps = float(chosen_chunk_bytes) * 8.0 / (float(M_IN_K) * max(cand_delay_ms, 1e-6))

                    sol = self._project_with_rules(
                        rules_file=str(rules_file),
                        bitrate_kbps=bitrate_kbps,
                        buffer_seconds=buffer_seconds,
                        buffer_seconds_max=buffer_seconds_max,
                        chosen_chunk_bytes=chosen_chunk_bytes,
                        delay_ms=cand_delay_ms,
                        throughput_mbps=cand_thr_mbps,
                        fix_buffer=fix_buffer,
                        latest_only=True,
                    )
                    if sol is not None:
                        sol_delay_ms, sol_thr_mbps, sol_buffer_s = sol
                        pert_delay_hist[j] = float(sol_delay_ms) / (float(M_IN_K) * float(delay_norm_factor))
                        pert_thr_hist[j] = float(sol_thr_mbps) / 8.0
                        if perturb_buffer:
                            pert_buffer = float(sol_buffer_s) / float(delay_norm_factor)
                        else:
                            pert_buffer = float(original_buffer_norm)
                        last_rule_buffer = float(pert_buffer)
                        last_rule_delay_hist = np.array(pert_delay_hist, copy=True)
                        last_rule_thr_hist = np.array(pert_thr_hist, copy=True)
                        _record_rules_projection_attempt(success=True, is_window=False, fallback=False)
                    else:
                        pert_buffer = float(last_rule_buffer)
                        pert_delay_hist = np.array(last_rule_delay_hist, copy=True)
                        pert_thr_hist = np.array(last_rule_thr_hist, copy=True)
                        _record_rules_projection_attempt(success=False, is_window=False, fallback=True)

            pert_buffer = np.clip(pert_buffer, 0, BUFFER_NORM_FACTOR)
            pert_thr_hist = np.clip(pert_thr_hist, 0, None)
            pert_delay_hist = np.clip(pert_delay_hist, 0, None)
            cur_state[1, -1] = float(pert_buffer)
            cur_state[2, :] = np.array(pert_thr_hist, copy=False)
            cur_state[3, :] = np.array(pert_delay_hist, copy=False)

            attempts = int(rules_stats["pgd/rules_projection_attempts"])
            success = int(rules_stats["pgd/rules_projection_success"])
            rules_stats["pgd/rules_projection_success_rate"] = float(success) / float(attempts) if attempts > 0 else 0.0
            stop_metrics["pgd/early_stop_iters_run"] = int(iters_run)
            if (
                int(stop_metrics.get("pgd/early_stop_triggered", 0)) == 0
                and enable_early_stop
                and np.isfinite(last_gap)
                and not np.isfinite(float(stop_metrics.get("pgd/early_stop_gap", float("nan"))))
            ):
                stop_metrics["pgd/early_stop_gap"] = float(last_gap)

            return np.array(cur_state, copy=True), stop_metrics, rules_stats

        full_state, full_stop, full_rules = _run_pgd(delay_update_idx_full, throughput_update_idx_full)
        self._early_stop_last_attack = dict(full_stop)
        self._rules_projection_last_attack_stats = dict(full_rules)
        if not bool(history_sparse) or not bool(self._history_attack_enabled):
            return full_state

        full_loss = float(self.compute_loss(full_state))
        if not np.isfinite(full_loss):
            return full_state

        grad_full = self.compute_gradient(full_state)
        grad_throughput_full = grad_full[0, 2, :].detach().cpu().numpy()
        grad_delay_full = grad_full[0, 3, :].detach().cpu().numpy()
        delta_thr_full = np.array(full_state[2, :], copy=False) - np.array(original_throughput_norm_hist, copy=False)
        delta_delay_full = np.array(full_state[3, :], copy=False) - np.array(original_delay_norm_hist, copy=False)

        #^ Rank scalar coordinates by `abs(grad * delta)` (first-order contribution proxy).
        candidates: list[tuple[int, int, float]] = []
        for j in delay_update_idx_full:
            score = abs(float(grad_delay_full[int(j)]) * float(delta_delay_full[int(j)]))
            if not np.isfinite(score):
                score = 0.0
            candidates.append((3, int(j), float(score)))
        for j in throughput_update_idx_full:
            score = abs(float(grad_throughput_full[int(j)]) * float(delta_thr_full[int(j)]))
            if not np.isfinite(score):
                score = 0.0
            candidates.append((2, int(j), float(score)))

        if len(candidates) == 0:
            return full_state

        candidates.sort(key=lambda x: float(x[2]), reverse=True)

        tol = float(history_sparse_tol)
        if not np.isfinite(tol) or tol < 0.0:
            tol = 0.0
        band = float(tol) * float(max(1.0, abs(full_loss)))

        for k in range(1, len(candidates) + 1):
            if k == len(candidates):
                break

            topk = candidates[:k]
            if use_coupling:
                #^ Coupling uses a shared index manifold (delay drives throughput).
                idx_set = sorted(set(int(idx) for _, idx, _ in topk))
                delay_idx_k = idx_set
                thr_idx_k: list[int] = []
            else:
                delay_idx_k = sorted(set(int(idx) for row, idx, _ in topk if int(row) == 3))
                thr_idx_k = sorted(set(int(idx) for row, idx, _ in topk if int(row) == 2))

            cand_state, cand_stop, cand_rules = _run_pgd(delay_idx_k, thr_idx_k)
            cand_loss = float(self.compute_loss(cand_state))
            if np.isfinite(cand_loss) and cand_loss <= float(full_loss + band):
                self._history_sparse_last_attack = {
                    "pgd/history_sparse_applied": 1,
                    "pgd/history_sparse_k": int(k),
                    "pgd/history_sparse_k_unique": int(len(delay_idx_k) + len(thr_idx_k)),
                    "pgd/history_sparse_candidates_total": int(len(candidates)),
                    "pgd/history_sparse_selected_delay": int(len(delay_idx_k)),
                    "pgd/history_sparse_selected_throughput": int(len(thr_idx_k)),
                    "pgd/history_sparse_full_loss": float(full_loss),
                    "pgd/history_sparse_loss": float(cand_loss),
                    "pgd/history_sparse_band": float(band),
                }
                self._early_stop_last_attack = dict(cand_stop)
                self._rules_projection_last_attack_stats = dict(cand_rules)
                return cand_state

        #* If no sparse subset can match the full-history loss, keep the full-history attack.
        if use_coupling:
            delay_all = sorted(set(int(idx) for _, idx, _ in candidates))
            thr_all: list[int] = []
        else:
            delay_all = sorted(set(int(idx) for row, idx, _ in candidates if int(row) == 3))
            thr_all = sorted(set(int(idx) for row, idx, _ in candidates if int(row) == 2))
        self._history_sparse_last_attack = {
            "pgd/history_sparse_applied": 1,
            "pgd/history_sparse_k": int(len(candidates)),
            "pgd/history_sparse_k_unique": int(len(delay_all) + len(thr_all)),
            "pgd/history_sparse_candidates_total": int(len(candidates)),
            "pgd/history_sparse_selected_delay": int(len(delay_all)),
            "pgd/history_sparse_selected_throughput": int(len(thr_all)),
            "pgd/history_sparse_full_loss": float(full_loss),
            "pgd/history_sparse_loss": float(full_loss),
            "pgd/history_sparse_band": float(band),
        }
        self._early_stop_last_attack = dict(full_stop)
        self._rules_projection_last_attack_stats = dict(full_rules)
        return full_state
