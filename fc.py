import warnings

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

AS_NEW_ROW = 0
AS_NEW_COLUMN = 1

################################################################################################### Policy based methods
class FCDAP(nn.Module):  # Fully connected discrete action policy for VPG

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        super(FCDAP, self).__init__()

        self.device = device
        self.activation = activation

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.out_layer = nn.Linear(hidden_dims[-1], out_dim)

    def _format(self, x):
        """
        Convert state to tensor if not and shape it correctly for the training process
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
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
        log_p_action = dist.log_prob(action).unsqueeze(-1)

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
        action = np.argmax(logits.detach().numpy())
        return action

class FCV(nn.Module):  # Fully connected value (state-value) for VPG

    def __init__(self, device, in_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        super(FCV, self).__init__()

        self.device = device
        self.activation = activation

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.out_layer = nn.Linear(hidden_dims[-1], 1)

    def _format(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation(self.fc1(x))

        for fc_hidden in self.hidden_layers:
            x = self.activation(fc_hidden(x))

        x = self.out_layer(x)
        return x

class FCAC(nn.Module):  # Fully connected actor-critic A2C (Discrete action)

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        super(FCAC, self).__init__()

        self.device = device
        self.activation = activation

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.value_out_layer = nn.Linear(hidden_dims[-1], 1)
        self.policy_out_layer = nn.Linear(hidden_dims[-1], out_dim)

        self.to(device)

    def _format(self, x):
        """
        Convert state to tensor if not and shape it correctly for the training process
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation(self.fc1(x))

        for fc_hidden in self.hidden_layers:
            x = self.activation(fc_hidden(x))

        return self.policy_out_layer(x), self.value_out_layer(x)

    def full_pass(self, state):
        logits, value = self.forward(state)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)

        # the entropy term encourage having evenly distributed actions
        entropy = dist.entropy().unsqueeze(-1)
        action = action.item() if len(action) == 1 else action.data.numpy()
        return action, logpa, entropy, value

    def select_action(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action = action.item() if len(action) == 1 else action.data.numpy()
        return action

    def get_state_value(self, state):
        _, value = self.forward(state)
        return value

#################################################################################################### Value based methods

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network Architecture.
    Same as DQN but using 2 outputs : One for the State-value function (return a single number)
    and one for the Action-advantage function (return the advantage value of each action)
    """

    def __init__(self, device, state_shape, out_dim, hidden_dims=(32, 32), activation=F.relu):
        super(DuelingDQN, self).__init__()

        self.device = device
        self.activation = activation
        C, H, W = state_shape

        self.conv1 = nn.Conv2d(in_channels=C, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=64*H*W, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.state_value_output = nn.Linear(hidden_dims[-1], 1)
        self.advantage_value_output = nn.Linear(hidden_dims[-1], out_dim)
        self.to(self.device)

    def forward(self, state):
        x = self._format(state)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))

        for fc_hidden in self.hidden_layers:
            x = self.activation(fc_hidden(x))

        advantage = self.advantage_value_output(x)
        state_value = self.state_value_output(x)

        # The fact that A and V are separated yield the ability to capture different features

        # A = Q - V  --> Q = V + A
        # But once we have Q, we cannot recover uniquely V and A...
        # To address this we will subtract the mean of A from Q, this will shift A and V by a
        # constant and stabilize the optim process

        # expand the scalar to the same size as advantage, so we can add them up to recreate Q
        # because a = q - v so q = v + a
        state_value = state_value.expand_as(advantage)

        q = state_value + advantage - advantage.mean(1, keepdim=True).expand_as(advantage)

        return q

    def _format(self, x):
        """
        Convert state to tensor if not and shape it correctly for the training process
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x


if __name__ == "__main__":
    pass
