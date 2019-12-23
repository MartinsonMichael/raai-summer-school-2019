import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


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
        torch.nn.init.constant_(self._conv1.bias, 0)

        self._conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4),
            stride=(2, 2),
        )
        torch.nn.init.xavier_uniform_(self._conv2.weight)
        torch.nn.init.constant_(self._conv2.bias, 0)

        self._conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
        )
        torch.nn.init.xavier_uniform_(self._conv3.weight)
        torch.nn.init.constant_(self._conv3.bias, 0)

    def forward(self, state):
        x = F.relu(self._conv1(state))
        x = F.relu(self._conv2(x))
        x = F.relu(self._conv3(x))
        return x.view(x.size(0), -1)

    def get_out_shape_for_in(self, input_shape):
        return self.forward(torch.Tensor(np.zeros((1, *input_shape)))).shape[1]


class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, device):
        super(QNet, self).__init__()
        self._device = device

        self.is_picture_input = None
        if isinstance(state_size, tuple):
            if state_size.__len__() == 3:
                self.is_picture_input = True
            elif state_size.__len__() == 1:
                state_size = state_size[0]
                self.is_picture_input = False
            else:
                raise ValueError(f'state size dont understood, it is tuple and have {len(state_size)} dims')
        elif isinstance(state_size, int):
            self.is_picture_input = False
        else:
            raise ValueError(f'state size dont understood, expected types int or tuple, have {type(state_size)}')

        if isinstance(action_size, tuple):
            if action_size.__len__() > 1:
                raise ValueError('action shape doesnt understood')
            action_size = action_size[0]
        elif not isinstance(action_size, int):
            raise ValueError('action shape doesnt understood')

        if self.is_picture_input:
            self._dense_s = PictureProcessor()
            self._state_layer_out_size = self._dense_s.get_out_shape_for_in(state_size)
        else:
            self._dense_s = nn.Linear(in_features=state_size, out_features=hidden_size)
            torch.nn.init.xavier_uniform_(self._dense_s.weight)
            torch.nn.init.constant_(self._dense_s.bias, 0)
            self._state_layer_out_size = hidden_size

        self._dense_a = nn.Linear(in_features=action_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense_a.weight)
        torch.nn.init.constant_(self._dense_a.bias, 0)

        self._dense2 = nn.Linear(in_features=hidden_size + self._state_layer_out_size, out_features=hidden_size)
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

        self.is_picture_input = None
        if isinstance(state_size, tuple):
            if state_size.__len__() == 3:
                self.is_picture_input = True
            elif state_size.__len__() == 1:
                state_size = state_size[0]
                self.is_picture_input = False
            else:
                raise ValueError(f'state size dont understood, it is tuple and have {len(state_size)} dims')
        elif isinstance(state_size, int):
            self.is_picture_input = False
        else:
            raise ValueError(f'state size dont understood, expected types int or tuple, have {type(state_size)}')

        if isinstance(action_size, tuple):
            if action_size.__len__() > 1:
                raise ValueError('action shape doesnt understood')
            action_size = action_size[0]
        elif not isinstance(action_size, int):
            raise ValueError('action shape doesnt understood')

        if self.is_picture_input:
            self._dense1 = PictureProcessor()
            self._state_layer_out_size = self._dense1.get_out_shape_for_in(state_size)
        else:
            self._dense1 = nn.Linear(in_features=state_size, out_features=hidden_size)
            torch.nn.init.xavier_uniform_(self._dense1.weight)
            torch.nn.init.constant_(self._dense1.bias, 0)
            self._state_layer_out_size = hidden_size

        self._dense2 = nn.Linear(in_features=self._state_layer_out_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self._dense2.weight)
        torch.nn.init.constant_(self._dense2.bias, 0)

        self._head = nn.Linear(in_features=hidden_size, out_features=2 * action_size)
        torch.nn.init.xavier_uniform_(self._head.weight)
        torch.nn.init.constant_(self._head.bias, 0)

    def forward(self, state):
        x = F.relu(self._dense1(state))
        x = F.relu(self._dense2(x))
        x = self._head(x)
        return x
