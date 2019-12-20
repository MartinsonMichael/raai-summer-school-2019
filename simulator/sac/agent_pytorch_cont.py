import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# class ValueNet(nn.Module):
#     def __init__(self, state_size, hidden_size, device):
#         super(ValueNet, self).__init__()
#         self._device = device
#
#         self._dense1 = nn.Linear(in_features=state_size, out_features=hidden_size)
#         torch.nn.init.xavier_uniform_(self._dense1.weight)
#         torch.nn.init.constant_(self._dense1.bias, 0)
#
#         self._dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
#         torch.nn.init.xavier_uniform_(self._dense2.weight)
#         torch.nn.init.constant_(self._dense2.bias, 0)
#
#         self._head1 = nn.Linear(in_features=hidden_size, out_features=1)
#         torch.nn.init.xavier_uniform_(self._head1.weight)
#         torch.nn.init.constant_(self._head1.bias, 0)
#
#     def forward(self, x):
#         x = F.relu(self._dense1(x))
#         x = F.relu(self._dense2(x))
#         x = self._head1(x)
#         return x
from torch.distributions import Normal


class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, device):
        super(QNet, self).__init__()
        self._device = device

        self._dense_s = nn.Linear(in_features=state_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense_s.weight)
        torch.nn.init.constant_(self._dense_s.bias, 0)

        self._dense_a = nn.Linear(in_features=action_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense_a.weight)
        torch.nn.init.constant_(self._dense_a.bias, 0)

        self._dense2 = nn.Linear(in_features=hidden_size + hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head1 = nn.Linear(in_features=hidden_size, out_features=action_size)
        torch.nn.init.xavier_uniform_(self._head1.weight)
        torch.nn.init.constant_(self._head1.bias, 0)

    def forward(self, state, action):
        s = F.relu(self._dense_s(state))
        a = F.relu(self._dense_a(action))
        x = torch.cat((s, a), 1)
        x = F.relu(self._dense2(x))
        x = self._head1(x)
        return x


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, device):
        super(Policy, self).__init__()
        self._device = device
        self._action_size = action_size

        self._dense1 = nn.Linear(in_features=state_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense1.weight)
        torch.nn.init.constant_(self._dense1.bias, 0)

        self._dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head_bias = nn.Linear(in_features=hidden_size, out_features=action_size)
        torch.nn.init.xavier_uniform_(self._head_bias.weight)
        torch.nn.init.constant_(self._head_bias.bias, 0)

        self._head_variance = nn.Linear(in_features=hidden_size, out_features=action_size)
        torch.nn.init.xavier_uniform_(self._head_variance.weight)
        torch.nn.init.constant_(self._head_variance.bias, 0)

    def forward(self, state):
        x = F.relu(self._dense1(state))
        x = F.relu(self._dense2(x))
        bias = F.tanh(self._head_bias(x))
        var = F.tanh(self._head_variance(x))

        action = F.tanh(bias)

        normal = Normal(bias, var)
        sampled_action = normal.sample()

        log_prob = normal.log_prob(sampled_action)
        log_prob -= torch.log(1 - sampled_action.pow(2) + 1e-10)
        log_prob = log_prob.sum(1, keepdim=True)

        sampled_action = F.tanh(sampled_action)

        return sampled_action, log_prob, action


class SAC_Agent_Torch_Continues:

    def __init__(self, state_size, action_size, hidden_size, start_lr=3*10**-4, device='cpu'):
        self._action_size = action_size
        self._hidden_size = hidden_size
        self._device = device

        self._temperature = torch.tensor(data=[1.0], requires_grad=True)
        self._target_entropy = torch.tensor([0.98 * -np.log(1 / action_size) for _ in range(action_size)])
        self._temp_optimizer = optim.Adam([self._temperature], lr=start_lr)

        self._Q1 = QNet(state_size, action_size, hidden_size, device).to(device)
        self._Q2 = QNet(state_size, action_size, hidden_size, device).to(device)

        self._target_Q1 = QNet(state_size, action_size, hidden_size, device).to(device)
        self._target_Q2 = QNet(state_size, action_size, hidden_size, device).to(device)
        SAC_Agent_Torch_Continues.copy_model_over(self._Q1, self._target_Q1, 0)
        SAC_Agent_Torch_Continues.copy_model_over(self._Q2, self._target_Q2, 0)

        self._Policy = Policy(state_size, action_size, hidden_size, device).to(device)

        self._q1_optimizer = optim.Adam(self._Q1.parameters(), lr=start_lr)
        self._q2_optimizer = optim.Adam(self._Q2.parameters(), lr=start_lr)
        self._policy_optimizer = optim.Adam(self._Policy.parameters(), lr=start_lr)

    @staticmethod
    def copy_model_over(from_model, to_model, tau=0.95):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone() * (1 - tau) + to_model.data.clone() * tau)

    def hard_target_update(self):
        SAC_Agent_Torch_Continues.copy_model_over(self._Q1, self._target_Q1, 0)
        SAC_Agent_Torch_Continues.copy_model_over(self._Q2, self._target_Q2, 0)

    def batch_action(self, state):
        # state: [batch_size, state_size]
        action, _, _ = self._Policy(
            torch.tensor(state, requires_grad=False, dtype=torch.float32, device=self._device)
        )
        return action.cpu().detach().numpy()

    def get_single_action(self, state):
        # state: [state_size, ]
        # return [action_size, ]
        action = self.get_batch_actions(np.array([state]))
        return action[0]

    def save(self, folder):
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self._Q1.state_dict(), os.path.join(folder, 'q1'))
        torch.save(self._Q2.state_dict(), os.path.join(folder, 'q2'))
        torch.save(self._V.state_dict(), os.path.join(folder, 'v'))
        torch.save(self._Policy.state_dict(), os.path.join(folder, 'policy'))

    def load(self, folder):
        import os
        if not os.path.exists(folder):
            raise ValueError(f'there is no such folder: {folder}')
        self._Q1.load_state_dict(torch.load(os.path.join(folder, 'q1')))
        self._Q2.load_state_dict(torch.load(os.path.join(folder, 'q2')))
        self._V.load_state_dict(torch.load(os.path.join(folder, 'v')))
        self.update_V_target(1.0)
        self._Policy.load_state_dict(torch.load(os.path.join(folder, 'policy')))

    def update_step(self, replay_batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_batch
        state_batch = torch.stack(tuple(map(torch.from_numpy, np.array(state_batch)))).to(self._device).detach()
        action_batch = torch.FloatTensor(action_batch).to(self._device).detach()
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self._device).detach()
        next_state_batch = torch.stack(tuple(map(torch.from_numpy, np.array(next_state_batch)))).to(self._device).detach()
        done_batch = torch.FloatTensor(done_batch).to(self._device).detach()

        new_action, new_action_log_prob, _ = self._Policy(state_batch)
        new_action_log_prob = torch.clamp((new_action_log_prob + 1e-10).log(), -20, 2)

        print(f'new_action shape : {new_action.size()}')

        new_q_value = torch.min(self._Q1(state_batch, new_action), self._Q2(state_batch, new_action))
        print(f'new_q_value shape : {new_q_value.size()}')
        loss_policy = (new_action_log_prob * self._temperature - new_q_value).mean()

        new_next_action, new_next_action_log_prob, _ = self._Policy(next_state_batch)
        new_next_q_value = torch.min(
            self._target_Q1(next_state_batch, new_next_action),
            self._target_Q2(next_state_batch, new_next_action),
        )
        target_q = new_next_q_value - new_next_action_log_prob * self._temperature
        print(f'target_q shape : {target_q.size()}')
        loss_q1 = F.mse_loss(self._Q1(state_batch, action_batch), target_q.detach())
        loss_q2 = F.mse_loss(self._Q2(state_batch, action_batch), target_q.detach())

        # gradient updates
        self._q1_optimizer.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_value_(self._Q1.parameters(), 1.0)
        self._q1_optimizer.step()

        self._q2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_value_(self._Q2.parameters(), 1.0)
        self._q2_optimizer.step()

        self._policy_optimizer.zero_grad()
        loss_policy.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self._Policy.parameters(), 1.0)
        self._policy_optimizer.step()

        # cumpute gradients of temperature directly
        self._temp_optimizer.zero_grad()
        loss_temp = (-new_action_log_prob * self._temperature - self._target_entropy * self._temperature).mean()
        loss_temp.backward()
        torch.nn.utils.clip_grad_value_(self._temperature, 1.0)
        self._temp_optimizer.step()

        # update V Target
        SAC_Agent_Torch_Continues.copy_model_over(self._Q1, self._target_Q1)
        SAC_Agent_Torch_Continues.copy_model_over(self._Q2, self._target_Q2)

        del state_batch
        del next_state_batch
        del reward_batch
        del done_batch
        del action_batch

        return (
            loss_q1.cpu().detach().numpy(),
            loss_q2.cpu().detach().numpy(),
            -1,
            loss_policy.cpu().detach().numpy(),
            self._temperature.cpu().detach().numpy(),
        )
