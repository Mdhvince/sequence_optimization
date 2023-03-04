import warnings

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

AS_NEW_ROW = 0
AS_NEW_COLUMN = 1


class FCQV(nn.Module):  # Fully connected Q-function Q(s, a)

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation_fc=F.relu):
        super(FCQV, self).__init__()

        self.device = device
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            # here we just increase the dimension of the first hidden layer
            # in order to catch states and actions, see torch.cat in self.forward
            if i == 0:
                in_dim += out_dim

            hidden_layer = nn.Linear(in_dim, hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        # output the value of a state-action pair
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.to(self.device)

    def _format(self, state, action):
        x, u = state, action

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        x = self.activation_fc(self.input_layer(x))

        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, u), dim=AS_NEW_COLUMN)
            x = self.activation_fc(hidden_layer(x))

        return self.output_layer(x)


class FCDP(nn.Module):  # fully connected deterministic policy (for continuous action)
    def __init__(
            self, device, in_dim, action_bounds,
            hidden_dims=(32, 32), activation_fc=F.relu, out_activation_fc=F.tanh):
        """
        In the pendulum env, we need to make the difference between a value of -2 and -2.5,
        that why we use tanh, it allow to map negative strongly negative, 0 to near 0 and so on,
        so it will map the values between -1 and 1 then we will scale back to original values.
        """
        super(FCDP, self).__init__()

        self.device = device
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc

        # min and max value of an action, if we have 2 possible actions [move, jump]
        # move: values can be in range (-30, 30)
        # jump: values can be in range (0, 5)
        # so lower = [-30, 0] & upper = [30, 5]
        self.lower, self.upper = action_bounds
        nA = len(self.upper)

        self.input_layer = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer = nn.Linear(hidden_dims[-1], nA)
        self.to(self.device)

        self.lower = torch.tensor(self.lower, device=self.device, dtype=torch.float32)
        self.upper = torch.tensor(self.upper, device=self.device, dtype=torch.float32)

        # after passing the tanh, outputs will be in range (-1, 1), so we need to scale it back
        # to original values
        self.nn_min = self.out_activation_fc(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = self.out_activation_fc(torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.upper - self.lower) / \
                                    (self.nn_max - self.nn_min) + self.lower

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))

        x = self.output_layer(x)
        x = self.out_activation_fc(x)
        return self.rescale_fn(x)


class FCDAP(nn.Module):  # Fully connected discrete action policy

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
        return np.argmax(logits.detach().numpy())


class FCV(nn.Module):  # Fully connected value (state-value)

    def __init__(self, device, in_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        """
        """
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
    """
    """

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        """
        """
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
        """Helper function for when we just need to sample an action"""
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action = action.item() if len(action) == 1 else action.data.numpy()
        return action

    def get_state_value(self, state):
        _, value = self.forward(state)
        return value


if __name__ == "__main__":
    pass
