import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')


class PolicyVPG(nn.Module):
    def __init__(self, device, state_shape, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        super(PolicyVPG, self).__init__()

        self.device = device
        self.activation = activation
        C, H, W = state_shape

        self.conv1 = nn.Conv2d(in_channels=C, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=64 * H * W, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.out_layer = nn.Linear(hidden_dims[-1], out_dim)
        self.to(self.device)

    def _format(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))

        for fc_hidden in self.hidden_layers:
            x = self.activation(fc_hidden(x))

        x = self.out_layer(x)
        return x

    def full_pass(self, state):
        logits = self.forward(state)  # preferences over actions

        # sample action from the probability distribution
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_p_action = dist.log_prob(action)

        # the entropy term encourage having evenly distributed actions
        entropy = dist.entropy().unsqueeze(-1)

        return action.item(), log_p_action, entropy

    def select_action(self, state):
        """Helper function for when we just need to sample an action"""
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def select_greedy_action(self, state):
        logits = self.forward(state)
        action = np.argmax(logits.detach().cpu().numpy())
        return action


class ValueVPG(nn.Module):
    def __init__(self, device, state_shape, hidden_dims=(32, 32), activation=F.relu) -> None:
        super(ValueVPG, self).__init__()

        self.device = device
        self.activation = activation
        C, H, W = state_shape

        self.conv1 = nn.Conv2d(in_channels=C, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=64 * H * W, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.out_layer = nn.Linear(hidden_dims[-1], 1)
        self.to(self.device)

    def _format(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))

        for fc_hidden in self.hidden_layers:
            x = self.activation(fc_hidden(x))

        x = self.out_layer(x)
        return x


if __name__ == "__main__":
    pass
