import torch
from torch import nn
import torch.nn.functional as F


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
