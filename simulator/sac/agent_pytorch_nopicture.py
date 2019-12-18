import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical


class ValueNet(nn.Module):
    def __init__(self, state_size, hidden_size, device):
        super(ValueNet, self).__init__()
        self._device = device

        self._dense1 = nn.Linear(in_features=state_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense1.weight)
        torch.nn.init.constant_(self._dense1.bias, 0)

        self._dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head1 = nn.Linear(in_features=hidden_size, out_features=1)
        torch.nn.init.xavier_uniform_(self._head1.weight)
        torch.nn.init.constant_(self._head1.bias, 0)

    def forward(self, x):
        x = F.relu(self._dense1(x))
        x = F.relu(self._dense2(x))
        x = self._head1(x)
        return x


class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, device):
        super(QNet, self).__init__()
        self._device = device

        self._dense1 = nn.Linear(in_features=state_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense1.weight)
        torch.nn.init.constant_(self._dense1.bias, 0)

        self._dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head1 = nn.Linear(in_features=hidden_size, out_features=action_size)
        torch.nn.init.xavier_uniform_(self._head1.weight)
        torch.nn.init.constant_(self._head1.bias, 0)

    def forward(self, state):
        x = F.relu(self._dense1(state))
        x = F.relu(self._dense2(x))
        x = self._head1(x)
        return x


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, device):
        super(Policy, self).__init__()
        self._device = device

        self._dense1 = nn.Linear(in_features=state_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense1.weight)
        torch.nn.init.constant_(self._dense1.bias, 0)

        self._dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head = nn.Linear(in_features=hidden_size, out_features=action_size)
        torch.nn.init.xavier_uniform_(self._head.weight)
        torch.nn.init.constant_(self._head.bias, 0)

    def forward(self, state):
        x = F.relu(self._dense1(state))
        x = F.relu(self._dense2(x))
        probs = F.softmax(self._head(x), dim=1)
        return probs


class SAC_Agent_Torch_NoPic:

    def __init__(self, state_size, action_size, hidden_size, start_lr=3*10**-4, device='cpu'):
        self._action_size = action_size
        self._hidden_size = hidden_size
        self._device = device

        # self._temperature = torch.tensor(data=1.0, requires_grad=True)
        self._temperature = 1.0
        self._target_temperature = torch.tensor([0.98 * -np.log(1 / action_size) for _ in range(action_size)])
        # self._temperature_optimizer = optim.Adam([self._temperature], lr=start_lr)

        self._Q1 = QNet(state_size, action_size, hidden_size, device).to(device)
        self._Q2 = QNet(state_size, action_size, hidden_size, device).to(device)
        self._V = ValueNet(state_size, hidden_size, device).to(device)
        self._V_target = ValueNet(state_size, hidden_size, device).to(device)
        self.update_V_target(1.0)
        self._Policy = Policy(state_size, action_size, hidden_size, device).to(device)

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
        actions_probs = self._Policy(
            torch.tensor(state, requires_grad=False, dtype=torch.float32, device=self._device)
        )
        if use_gumbel:
            ind_max = Categorical(probs=actions_probs).sample()
        else:
            ind_max = np.argmax(actions_probs.cpu().detach().numpy(), axis=1)
        if need_argmax:
            return ind_max
        onehot_actions = np.eye(actions_probs.shape[1])
        onehot_actions = onehot_actions[ind_max]
        return onehot_actions

    def get_single_action(self, state, need_argmax=False, use_gumbel=True, temperature=0.5):
        # state: [state_size, ]
        # return [action_size, ]
        action = self.get_batch_actions(np.array([state]), need_argmax, use_gumbel, temperature)
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

    def update_step(
            self,
            replay_batch,
            temperature=0.5,
            gamma=0.7,
            v_exp_smooth_factor=0.995,
    ):
        state, action, reward, next_state, done_flag = replay_batch
        state = torch.stack(tuple(map(torch.from_numpy, np.array(state)))).to(self._device).detach()
        action = torch.FloatTensor(np.array(action)).to(self._device).detach()
        reward = torch.FloatTensor(np.array(reward)).to(self._device).detach()
        next_state = torch.stack(tuple(map(torch.from_numpy, np.array(next_state)))).to(self._device).detach()
        done_flag = torch.FloatTensor(done_flag).to(self._device).detach()

        v_next = self._V_target(next_state)

        target_q = reward + gamma * (1 - done_flag) * v_next

        loss_q1 = F.mse_loss((self._Q1(state)[action == 1.0]).unsqueeze_(-1), target_q.detach())
        loss_q2 = F.mse_loss((self._Q2(state)[action == 1.0]).unsqueeze_(-1), target_q.detach())

        probs = self._Policy(state)
        print(f'probs sample : {probs[0].data}')
        log_probs = torch.clamp((probs + 1e-10).log(), -20, 2)

        new_q_value = torch.min(
            self._Q1(state),
            self._Q2(state),
        )
        target_v = torch.sum(
            probs * (new_q_value - log_probs * self._temperature),
            dim=1,
            keepdim=True,
        )
        loss_value = F.mse_loss(self._V(state), target_v.detach())

        loss_policy = torch.sum(
            probs.detach() * (log_probs * self._temperature - new_q_value),
            dim=1,
        ).mean()

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
        torch.nn.utils.clip_grad_value_(self._V.parameters(), 1.0)
        self._v_optimizer.step()

        self._policy_optimizer.zero_grad()
        loss_policy.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self._Policy.parameters(), 1.0)
        self._policy_optimizer.step()

        # cumpute gradients of temperature directly
        self._temperature += 0.001 * np.sum(probs.data.numpy() * log_probs.data.numpy())
        self._temperature = np.clip(self._temperature, 0.001, 10)

        # update V Target
        self.update_V_target(v_exp_smooth_factor)

        del state
        del next_state
        del reward
        del done_flag
        del action

        return (
            loss_q1.cpu().detach().numpy(),
            loss_q2.cpu().detach().numpy(),
            loss_value.cpu().detach().numpy(),
            loss_policy.cpu().detach().numpy(),
            self._temperature,
        )
