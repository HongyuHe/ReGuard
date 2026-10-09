import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
EPS = 0.2  # PPO2 epsilon
CHECKPOINT_FORMAT_VERSION = 2

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # Actor network
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.fc1_actor = nn.Linear(1, FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, FEATURE_NUM)
        self.conv1_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv2_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv3_actor = nn.Linear(self.a_dim, FEATURE_NUM)
        self.fc3_actor = nn.Linear(1, FEATURE_NUM)
        self.fc4_actor = nn.Linear(FEATURE_NUM * self.s_dim[0], FEATURE_NUM)
        self.pi_head = nn.Linear(FEATURE_NUM, action_dim)

    def forward(self, inputs):
        #* Feature 1: last bitrate
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        #* Feature 2: last buffer size
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        #* Feature 3: throughput history
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]).view(-1, FEATURE_NUM))
        #* Feature 4: delay history
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]).view(-1, FEATURE_NUM))
        #* Feature 5: next chunk sizes
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :self.a_dim]).view(-1, FEATURE_NUM))
        #* Feature 6: number of chunks remaining
        split_5 = F.relu(self.fc3_actor(inputs[:, 5:6, -1]))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], 1)

        pi_net = F.relu(self.fc4_actor(merge_net))
        pi = F.softmax(self.pi_head(pi_net), dim=-1)
        pi = torch.clamp(pi, ACTION_EPS, 1. - ACTION_EPS)
        return pi
    
    def forward_without_clamp(self, inputs):
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]).view(-1, FEATURE_NUM))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]).view(-1, FEATURE_NUM))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :self.a_dim]).view(-1, FEATURE_NUM))
        split_5 = F.relu(self.fc3_actor(inputs[:, 5:6, -1]))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], 1)

        pi_net = F.relu(self.fc4_actor(merge_net))
        pi = F.softmax(self.pi_head(pi_net), dim=-1)
        return pi


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic network
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.fc1_actor = nn.Linear(1, FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, FEATURE_NUM)
        self.conv1_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv2_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv3_actor = nn.Linear(self.a_dim, FEATURE_NUM)
        self.fc3_actor = nn.Linear(1, FEATURE_NUM)
        self.fc4_actor = nn.Linear(FEATURE_NUM * self.s_dim[0], FEATURE_NUM)
        self.val_head = nn.Linear(FEATURE_NUM, 1)

    def forward(self, inputs):
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]).view(-1, FEATURE_NUM))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]).view(-1, FEATURE_NUM))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :self.a_dim]).view(-1, FEATURE_NUM))
        split_5 = F.relu(self.fc3_actor(inputs[:, 5:6, -1]))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], 1)

        value_net = F.relu(self.fc4_actor(merge_net))
        value = self.val_head(value_net)
        return value
    
class Network():
    def __init__(self, state_dim, action_dim, learning_rate):

        self.s_dim = state_dim
        self.action_dim = action_dim
        self._entropy_weight = np.log(action_dim)
        self._freeze_entropy_weight = False
        self.H_target = 0.1
        self.PPO_TRAINING_EPO = 5

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.lr_rate = learning_rate
        self.optimizer = optim.Adam(list(self.actor.parameters()) + \
                                    list(self.critic.parameters()), lr=learning_rate)

    def get_network_params(self):
        return [self.actor.state_dict(), self.critic.state_dict()]
    
    def set_network_params(self, input_network_params):
        actor_net_params, critic_net_params = input_network_params
        self.actor.load_state_dict(actor_net_params)
        self.critic.load_state_dict(critic_net_params)

    def set_entropy_weight(self, entropy_weight):
        self._entropy_weight = max(float(entropy_weight), 1e-2)

    def set_freeze_entropy_weight(self, freeze):
        self._freeze_entropy_weight = bool(freeze)

    def r(self, pi_new, pi_old, acts):
        return torch.sum(pi_new * acts, dim=1, keepdim=True) / \
               torch.sum(pi_old * acts, dim=1, keepdim=True)

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):
        s_batch = torch.from_numpy(s_batch).to(torch.float32)
        a_batch = torch.from_numpy(a_batch).to(torch.float32)
        p_batch = torch.from_numpy(p_batch).to(torch.float32)
        v_batch = torch.from_numpy(v_batch).to(torch.float32)

        total_loss_values = []
        policy_loss_values = []
        value_loss_values = []
        entropy_values = []
        ratio_values = []
        clip_frac_values = []
        approx_kl_values = []
        adv_values = []

        for _ in range(self.PPO_TRAINING_EPO):
            pi = self.actor.forward(s_batch)
            val = self.critic.forward(s_batch)

            # loss
            adv = v_batch - val.detach()
            ratio = self.r(pi, p_batch, a_batch)
            ppo2loss = torch.min(ratio * adv, torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv)
            # Dual-PPO
            dual_loss = torch.where(adv < 0, torch.max(ppo2loss, 3. * adv), ppo2loss)
            loss_entropy = torch.sum(-pi * torch.log(pi), dim=1, keepdim=True)
            policy_loss = -dual_loss.mean()
            value_loss = F.mse_loss(val, v_batch)
            entropy_mean = loss_entropy.mean()
            loss = policy_loss + 10. * value_loss - self._entropy_weight * entropy_mean

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                old_logp = torch.log(torch.sum(p_batch * a_batch, dim=1, keepdim=True))
                new_logp = torch.log(torch.sum(pi * a_batch, dim=1, keepdim=True))
                approx_kl = (old_logp - new_logp).mean()
                clip_frac = (torch.abs(ratio - 1.0) > EPS).to(torch.float32).mean()

                total_loss_values.append(float(loss.item()))
                policy_loss_values.append(float(policy_loss.item()))
                value_loss_values.append(float(value_loss.item()))
                entropy_values.append(float(entropy_mean.item()))
                ratio_values.append(float(ratio.mean().item()))
                clip_frac_values.append(float(clip_frac.item()))
                approx_kl_values.append(float(approx_kl.item()))
                adv_values.append(float(adv.mean().item()))

        entropy_weight_before = float(self._entropy_weight)
        if not bool(self._freeze_entropy_weight):
            _H = (-(torch.log(p_batch) * p_batch).sum(dim=1)).mean().item()
            _g = _H - self.H_target
            self._entropy_weight -= self.lr_rate * _g * 0.1 * self.PPO_TRAINING_EPO
            self._entropy_weight = max(self._entropy_weight, 1e-2)
        entropy_weight_after = float(self._entropy_weight)

        def _mean(values):
            if len(values) == 0:
                return float("nan")
            return float(np.mean(values))

        return {
            "train/loss/total": _mean(total_loss_values),
            "train/loss/policy": _mean(policy_loss_values),
            "train/loss/value": _mean(value_loss_values),
            "train/loss/entropy": _mean(entropy_values),
            "train/ratio_mean": _mean(ratio_values),
            "train/clip_frac": _mean(clip_frac_values),
            "train/approx_kl": _mean(approx_kl_values),
            "train/adv_mean": _mean(adv_values),
            "train/entropy_weight_before": float(entropy_weight_before),
            "train/entropy_weight_after": float(entropy_weight_after),
            "train/entropy_weight_frozen": float(1.0 if bool(self._freeze_entropy_weight) else 0.0),
        }

    def predict(self, input):
        with torch.no_grad():
            input = torch.from_numpy(input).to(torch.float32)
            pi = self.actor.forward(input)[0]
            return pi.numpy()
        
    def predict_tensor(self, input):
        pi = self.actor.forward_without_clamp(input)
        return pi

    def load_model(self, nn_model, *, restore_optimizer=True, restore_entropy_weight=True):
        checkpoint = torch.load(nn_model, map_location="cpu")
        load_info = {
            "legacy_checkpoint": False,
            "restored_optimizer": False,
            "restored_entropy_weight": False,
        }

        if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint and "critic_state_dict" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])

            if bool(restore_optimizer) and checkpoint.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = float(self.lr_rate)
                load_info["restored_optimizer"] = True

            if bool(restore_entropy_weight) and checkpoint.get("entropy_weight") is not None:
                self.set_entropy_weight(float(checkpoint["entropy_weight"]))
                load_info["restored_entropy_weight"] = True

            return load_info

        actor_model_params, critic_model_params = checkpoint
        self.actor.load_state_dict(actor_model_params)
        self.critic.load_state_dict(critic_model_params)
        load_info["legacy_checkpoint"] = True
        return load_info

    def save_model(self, nn_model):
        checkpoint = {
            "checkpoint_format_version": int(CHECKPOINT_FORMAT_VERSION),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "entropy_weight": float(self._entropy_weight),
            "lr_rate": float(self.lr_rate),
        }
        torch.save(checkpoint, nn_model)

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        R_batch = np.zeros_like(r_batch)

        if terminal:
            # in this case, the terminal reward will be assigned as r_batch[-1]
            R_batch[-1] = r_batch[-1]  # terminal state
        else:
            val = self.critic.forward(s_batch)
            R_batch[-1] = val[-1]  # bootstrap from last state

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t] = r_batch[t] + GAMMA * R_batch[t + 1]

        return list(R_batch)
           
