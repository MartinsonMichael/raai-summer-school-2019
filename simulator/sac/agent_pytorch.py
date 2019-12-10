import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class PictureProcessor(nn.Module):
    def __init__(self):
        super(PictureProcessor, self).__init__()

        self._conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(8, 8),
            stride=(4, 4),
        )
        torch.nn.init.xavier_uniform_(self._conv1.weight)
        torch.nn.init.uniform_(self._conv1.bias)

        self._conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4),
            stride=(2, 2),
        )
        torch.nn.init.xavier_uniform_(self._conv2.weight)
        torch.nn.init.uniform_(self._conv2.bias)

        self._conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
        )
        torch.nn.init.xavier_uniform_(self._conv3.weight)
        torch.nn.init.uniform_(self._conv3.bias)

    def forward(self, x):
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = F.relu(self._conv3(x))
        return x.view(x.size(0), -1)

    def get_out_shape_for_in(self, input_shape):
        return self.forward(torch.Tensor(np.zeros((1, *input_shape)))).shape[1]


class ValueNet(nn.Module):
    def __init__(self, picture_shape, hidden_size, device):
        super(ValueNet, self).__init__()
        self._device = device
        self._pic_prepros = PictureProcessor()
        pic_out_size = self._pic_prepros.get_out_shape_for_in(picture_shape)

        self._dense1 = nn.Linear(in_features=pic_out_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense1.weight)
        torch.nn.init.uniform_(self._dense1.bias)

        self._dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.uniform_(self._dense2.bias)

        self._head1 = nn.Linear(in_features=hidden_size, out_features=1)
        torch.nn.init.xavier_uniform_(self._head1.weight)
        torch.nn.init.uniform_(self._head1.bias)

    def forward(self, picture_state):
        x = self._pic_prepros(picture_state)
        x = F.relu(self._dense1(x))
        x = F.relu(self._dense2(x))
        x = self._head1(x)
        return x


class QNet(nn.Module):
    def __init__(self, picture_shape, action_size, hidden_size, device):
        super(QNet, self).__init__()
        self._device = device
        self._pic_prepros = PictureProcessor()
        pic_out_size = self._pic_prepros.get_out_shape_for_in(picture_shape)

        self._dense_action = nn.Linear(in_features=action_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense_action.weight)
        torch.nn.init.uniform_(self._dense_action.bias)

        self._dense2 = nn.Linear(in_features=pic_out_size + hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.uniform_(self._dense2.bias)

        self._head1 = nn.Linear(in_features=hidden_size, out_features=1)
        torch.nn.init.xavier_uniform_(self._head1.weight)
        torch.nn.init.uniform_(self._head1.bias)

    def forward(self, picture_state, action):
        # print(f'QNew -> forward -> action : {action}')
        # print(f'QNew -> forward -> action type : {type(action)}')

        s = self._pic_prepros(picture_state)
        # print(f'QNew -> forward -> s : {s}')
        a = self._dense_action(action)
        # print(f'Qnet -> forward -> a : {a}')
        x = torch.cat((s, a), 1)
        # print(f'Qnet -> forward -> x : {x}')
        x = F.relu(self._dense2(x))
        # print(f'Qnet -> forward -> x : {x}')
        x = self._head1(x)
        # print(f'Qnet -> forward -> x : {x}')
        return x


class Policy(nn.Module):
    def __init__(self, picture_shape, action_size, hidden_size, device):
        super(Policy, self).__init__()
        self._device = device
        self._pic_prepros = PictureProcessor()
        pic_out_size = self._pic_prepros.get_out_shape_for_in(picture_shape)

        self._dense1 = nn.Linear(in_features=pic_out_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense1.weight)
        torch.nn.init.uniform_(self._dense1.weight)

        self._dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.uniform_(self._dense2.weight)

        self._head = nn.Linear(in_features=hidden_size, out_features=action_size)
        torch.nn.init.xavier_uniform_(self._head.weight)
        torch.nn.init.uniform_(self._head.weight)

    def forward(self, picture_state):
        x = self._pic_prepros(picture_state)
        x = F.relu(self._dense1(x))
        x = F.relu(self._dense2(x))
        probs = F.softmax(self._head(x), dim=1)
        return probs

    def _sample_gumbel_uniform(self, shape, eps=1e-20):
        u = np.random.uniform(low=0, high=1, size=shape).astype(np.float32)
        u = -np.log(-np.log(u + eps) + eps)
        return torch.from_numpy(u).to(self._device).detach()

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self._sample_gumbel_uniform(logits.size()) * temperature
        return F.softmax(y, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y, (y + 1e-20).log()

    def evaluate_gumbel(self, picture_state, temperature):
        # shape [batch, action]
        probs = torch.clamp((self.forward(picture_state) + 1e-20).log(), min=-20.0, max=2.0)

        # shape [batch, action] and [batch, 1]
        sampled_actions, sampled_log_probs = self.gumbel_softmax(probs, temperature)
        # print(f'policy -> evaluate_gumbel -> sampled_log_probs : {sampled_log_probs}')
        sampled_log_probs = torch.clamp(sampled_log_probs, min=-20.0, max=2.0)
        # print(f'policy -> evaluate_gumbel -> sampled_log_probs.clamped : {sampled_log_probs}')

        return sampled_actions, sampled_log_probs.max(dim=1, keepdim=True)[0]


class SAC_Agent_Torch:

    def __init__(self, picture_shape, action_size, hidden_size, start_lr=3*10**-5, device='cpu'):
        self._picture_shape = picture_shape
        self._action_size = action_size
        self._hidden_size = hidden_size
        self._device = device

        self._Q1 = QNet(picture_shape, action_size, hidden_size, device).to(device)
        self._Q2 = QNet(picture_shape, action_size, hidden_size, device).to(device)
        self._V = ValueNet(picture_shape, hidden_size, device).to(device)
        self._V_target = ValueNet(picture_shape, hidden_size, device).to(device)
        self.update_V_target(1.0)
        self._Policy = Policy(picture_shape, action_size, hidden_size, device).to(device)

        self._v_optimizer = optim.Adam(self._V.parameters(), lr=start_lr)
        self._q1_optimizer = optim.Adam(self._Q1.parameters(), lr=start_lr)
        self._q2_optimizer = optim.Adam(self._Q2.parameters(), lr=start_lr)
        self._policy_optimizer = optim.Adam(self._Policy.parameters(), lr=start_lr)

    def update_V_target(self, smooth_factor=0.95):
        for target_param, param in zip(self._V_target.parameters(), self._V.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - smooth_factor) + param.data * smooth_factor
            )

    def get_batch_actions(self, state, need_argmax=False, use_gumbel=True, temperature=0.5):
        # state: [batch_size, state_size]
        if not use_gumbel:
            temperature = 0.000001
        batch_action, _ = self._Policy.evaluate_gumbel(
            torch.tensor(state / 256, requires_grad=False, dtype=torch.float32, device=self._device),
            temperature,
        )
        if need_argmax:
            return np.argmax(batch_action.cpu().detach().numpy(), axis=1)
        return batch_action.cpu().detach().numpy()

    def get_single_action(self, state, need_argmax=False, use_gumbel=True, temperature=0.5):
        # state: [state_size, ]
        # return [action_szie, ]
        action = self.get_batch_actions(np.array([state]), need_argmax, use_gumbel, temperature)
        return action[0]

    def save(self, folder):
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)
        pass

    def load(self, folder):
        import os
        if not os.path.exists(folder):
            raise ValueError(f'there is no such folder: {folder}')
        pass

    def update_step(
            self,
            replay_batch,
            temperature=0.5,
            gamma=0.7,
            v_exp_smooth_factor=0.995,
    ):
        # shape of treplay_batch : tuple of (
        #     [batch_size, tuple(picture, extra_features)], - state
        #     [batch_size, actoin_size],- action
        #     [batch_size, 1],          - revard
        #     [batch_size, tuple(picture, extra_features)], - new state
        #     [batch_size, 1]           - is it done? (1 for done, 0 for not yet)
        # )
        state, action, reward, next_state, done_flag = replay_batch
        state = torch.stack(tuple(map(torch.from_numpy, np.array(state)))).to(self._device).detach()
        action = torch.FloatTensor(np.array(action)).to(self._device).detach()
        reward = torch.FloatTensor(np.array(reward)).to(self._device).detach()
        next_state = torch.stack(tuple(map(torch.from_numpy, np.array(next_state)))).to(self._device).detach()
        done_flag = torch.FloatTensor(done_flag).to(self._device).detach()

        # print(f'state : {state}')
        # print(f'action: {action}')
        # print(f'reward : {reward}')
        # print(f'next_state : {next_state}')
        # print(f'done_flag : {done_flag}')

        v_next = self._V_target(next_state)
        # print(f'v_next shape : {v_next.size()}')
        # print(f'v_next {v_next}')

        target_q = reward + gamma * (1 - done_flag) * v_next
        new_action, log_prob = self._Policy.evaluate_gumbel(state, temperature)
        # print(f'new_actoin : {new_action}')
        # print(f'log_prob: {log_prob}')
        new_q_value = torch.min(
            self._Q1(state, new_action),
            self._Q2(state, new_action)
        )
        # print(f'new q value shape : {new_q_value.size()}')
        # print(f'new_q_value : {new_q_value}')
        target_v = new_q_value - log_prob


        loss_q1 = F.mse_loss(self._Q1(state, action), target_q.detach())
        loss_q2 = F.mse_loss(self._Q2(state, action), target_q.detach())
        loss_value = F.mse_loss(self._V(state), target_v.detach())
        loss_policy = (log_prob - new_q_value).mean()

        # gradient updates
        self._q1_optimizer.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_value_(self._Q1.parameters(), 1.0)
        self._q1_optimizer.step()

        self._q2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_value_(self._Q2.parameters(), 1.0)
        self._q2_optimizer.step()

        self._v_optimizer.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_value_(self._V.parameters(), 3.0)
        self._v_optimizer.step()

        self._policy_optimizer.zero_grad()
        loss_policy.backward()
        torch.nn.utils.clip_grad_value_(self._Policy.parameters(), 1.0)
        self._policy_optimizer.step()

        # update V Target
        self.update_V_target(v_exp_smooth_factor)

        return (
            loss_q1.cpu().detach().numpy(),
            loss_q2.cpu().detach().numpy(),
            loss_value.cpu().detach().numpy(),
            loss_policy.cpu().detach().numpy(),
        )
