import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
EPS = 0.2  # PPO2 epsilon

class Actor(nn.Module):
    """Actor network compatible with TensorFlow-converted weights"""
    
    def __init__(self, state_dim, action_dim, bitrate_dim):
        super(Actor, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.bitrate_dim = bitrate_dim

        # Split 0: fully connected for inputs[:, 0:1, -1]
        self.fc_split_0 = nn.Linear(1, FEATURE_NUM)
        
        # Split 1: fully connected for inputs[:, 1:2, -1]
        self.fc_split_1 = nn.Linear(1, FEATURE_NUM)
        
        # Split 2: conv1d for inputs[:, 2:3, :]
        self.conv_split_2 = nn.Conv1d(1, FEATURE_NUM, kernel_size=4)
        
        # Split 3: conv1d for inputs[:, 3:4, :]
        self.conv_split_3 = nn.Conv1d(1, FEATURE_NUM, kernel_size=4)
        
        # Split 4: conv1d for inputs[:, 4:5, :bitrate_dim]
        self.conv_split_4 = nn.Conv1d(1, FEATURE_NUM, kernel_size=4)
        
        # Split 5: fully connected for inputs[:, 5:6, -1]
        self.fc_split_5 = nn.Linear(1, FEATURE_NUM)
        
        # Calculate flattened sizes after conv
        conv_output_2_3 = (state_dim[1] - 4 + 1) * FEATURE_NUM
        conv_output_4 = (bitrate_dim - 4 + 1) * FEATURE_NUM
        
        # Merge layer input size
        merge_size = FEATURE_NUM + FEATURE_NUM + conv_output_2_3 + conv_output_2_3 + conv_output_4 + FEATURE_NUM
        
        # Dense layers
        self.fc_merge = nn.Linear(merge_size, FEATURE_NUM)
        self.fc_out = nn.Linear(FEATURE_NUM, action_dim)

    def forward(self, inputs):
        # Split 0: [batch_size, 1, 1] -> [batch_size, 128]
        split_0 = F.relu(self.fc_split_0(inputs[:, 0:1, -1]))
        
        # Split 1: [batch_size, 1, 1] -> [batch_size, 128]
        split_1 = F.relu(self.fc_split_1(inputs[:, 1:2, -1]))
        
        # Split 2: [batch_size, 1, seq_len] -> [batch_size, 128, seq_len-3]
        split_2 = F.relu(self.conv_split_2(inputs[:, 2:3, :]))
        split_2_flat = split_2.view(split_2.size(0), -1)
        
        # Split 3: [batch_size, 1, seq_len] -> [batch_size, 128, seq_len-3]
        split_3 = F.relu(self.conv_split_3(inputs[:, 3:4, :]))
        split_3_flat = split_3.view(split_3.size(0), -1)
        
        # Split 4: [batch_size, 1, bitrate_dim] -> [batch_size, 128, bitrate_dim-3]
        split_4 = F.relu(self.conv_split_4(inputs[:, 4:5, :self.bitrate_dim]))
        split_4_flat = split_4.view(split_4.size(0), -1)
        
        # Split 5: [batch_size, 1, 1] -> [batch_size, 128]
        split_5 = F.relu(self.fc_split_5(inputs[:, 5:6, -1]))
        
        # Merge all splits
        merge_net = torch.cat([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 1)
        
        # Dense layers
        pi_net = F.relu(self.fc_merge(merge_net))
        pi = F.softmax(self.fc_out(pi_net), dim=-1)
        pi = torch.clamp(pi, ACTION_EPS, 1. - ACTION_EPS)
        return pi
    
    def forward_without_clamp(self, inputs):
        split_0 = F.relu(self.fc_split_0(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc_split_1(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv_split_2(inputs[:, 2:3, :]))
        split_2_flat = split_2.view(split_2.size(0), -1)
        split_3 = F.relu(self.conv_split_3(inputs[:, 3:4, :]))
        split_3_flat = split_3.view(split_3.size(0), -1)
        split_4 = F.relu(self.conv_split_4(inputs[:, 4:5, :self.bitrate_dim]))
        split_4_flat = split_4.view(split_4.size(0), -1)
        split_5 = F.relu(self.fc_split_5(inputs[:, 5:6, -1]))
        
        merge_net = torch.cat([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 1)
        
        pi_net = F.relu(self.fc_merge(merge_net))
        pi = F.softmax(self.fc_out(pi_net), dim=-1)
        return pi


class Critic(nn.Module):
    """Critic network compatible with TensorFlow-converted weights"""
    
    def __init__(self, state_dim, action_dim, bitrate_dim):
        super(Critic, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.bitrate_dim = bitrate_dim

        # Split 0: fully connected for inputs[:, 0:1, -1]
        self.fc_split_0 = nn.Linear(1, FEATURE_NUM)
        
        # Split 1: fully connected for inputs[:, 1:2, -1]
        self.fc_split_1 = nn.Linear(1, FEATURE_NUM)
        
        # Split 2: conv1d for inputs[:, 2:3, :]
        self.conv_split_2 = nn.Conv1d(1, FEATURE_NUM, kernel_size=4)
        
        # Split 3: conv1d for inputs[:, 3:4, :]
        self.conv_split_3 = nn.Conv1d(1, FEATURE_NUM, kernel_size=4)
        
        # Split 4: conv1d for inputs[:, 4:5, :bitrate_dim]
        self.conv_split_4 = nn.Conv1d(1, FEATURE_NUM, kernel_size=4)
        
        # Split 5: fully connected for inputs[:, 5:6, -1]
        self.fc_split_5 = nn.Linear(1, FEATURE_NUM)
        
        # Calculate flattened sizes
        conv_output_2_3 = (state_dim[1] - 4 + 1) * FEATURE_NUM
        conv_output_4 = (bitrate_dim - 4 + 1) * FEATURE_NUM
        
        # Merge layer input size
        merge_size = FEATURE_NUM + FEATURE_NUM + conv_output_2_3 + conv_output_2_3 + conv_output_4 + FEATURE_NUM
        
        # Dense layers
        self.fc_merge = nn.Linear(merge_size, FEATURE_NUM)
        self.fc_out = nn.Linear(FEATURE_NUM, 1)

    def forward(self, inputs):
        split_0 = F.relu(self.fc_split_0(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc_split_1(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv_split_2(inputs[:, 2:3, :]))
        split_2_flat = split_2.view(split_2.size(0), -1)
        split_3 = F.relu(self.conv_split_3(inputs[:, 3:4, :]))
        split_3_flat = split_3.view(split_3.size(0), -1)
        split_4 = F.relu(self.conv_split_4(inputs[:, 4:5, :self.bitrate_dim]))
        split_4_flat = split_4.view(split_4.size(0), -1)
        split_5 = F.relu(self.fc_split_5(inputs[:, 5:6, -1]))
        
        merge_net = torch.cat([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 1)
        
        value_net = F.relu(self.fc_merge(merge_net))
        value = self.fc_out(value_net)
        return value
    
class Network():
    def __init__(self, state_dim, action_dim, bitrate_dim, learning_rate):
        self.s_dim = state_dim
        self.action_dim = action_dim
        self.bitrate_dim = bitrate_dim
        self._entropy_weight = np.log(action_dim)
        self.H_target = 0.1
        self.PPO_TRAINING_EPO = 5

        self.actor = Actor(state_dim, action_dim, bitrate_dim)
        self.critic = Critic(state_dim, action_dim, bitrate_dim)
        self.lr_rate = learning_rate
        self.optimizer = optim.Adam(list(self.actor.parameters()) + \
                                    list(self.critic.parameters()), lr=learning_rate)

    def get_network_params(self):
        return [self.actor.state_dict(), self.critic.state_dict()]
    
    def set_network_params(self, input_network_params):
        actor_net_params, critic_net_params = input_network_params
        self.actor.load_state_dict(actor_net_params)
        self.critic.load_state_dict(critic_net_params)

    def r(self, pi_new, pi_old, acts):
        return torch.sum(pi_new * acts, dim=1, keepdim=True) / \
               torch.sum(pi_old * acts, dim=1, keepdim=True)

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):
        s_batch = torch.from_numpy(s_batch).to(torch.float32)
        a_batch = torch.from_numpy(a_batch).to(torch.float32)
        p_batch = torch.from_numpy(p_batch).to(torch.float32)
        v_batch = torch.from_numpy(v_batch).to(torch.float32)

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

            loss = -dual_loss.mean() + 10. * F.mse_loss(val, v_batch) - self._entropy_weight * loss_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update entropy weight
        _H = (-(torch.log(p_batch) * p_batch).sum(dim=1)).mean().item()
        _g = _H - self.H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1 * self.PPO_TRAINING_EPO
        self._entropy_weight = max(self._entropy_weight, 1e-2)

    def predict(self, input):
        with torch.no_grad():
            input = torch.from_numpy(input).to(torch.float32)
            pi = self.actor.forward(input)[0]
            return pi.numpy()
        
    def predict_tensor(self, input):
        pi = self.actor.forward_without_clamp(input)
        return pi

    def load_model(self, nn_model):
        actor_model_params, critic_model_params = torch.load(nn_model)
        self.actor.load_state_dict(actor_model_params)
        self.critic.load_state_dict(critic_model_params)
        
    def load_model_from_tf_converted(self, nn_model, device='cpu'):
        checkpoint = torch.load(nn_model, map_location=device)
        
        if isinstance(checkpoint, dict):
            # Dictionary format (TensorFlow-converted)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f"✓ Model loaded (dict format) from {nn_model}")
        elif isinstance(checkpoint, (tuple, list)) and len(checkpoint) == 2:
            # Tuple format (PyTorch-native)
            self.actor.load_state_dict(checkpoint[0])
            self.critic.load_state_dict(checkpoint[1])
            print(f"✓ Model loaded (tuple format) from {nn_model}")
        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    def save_model(self, nn_model):
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        torch.save(model_params, nn_model)

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
           
