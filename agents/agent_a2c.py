import warnings
from itertools import count

import numpy as np
import torch
import torch.optim as optim

from fc import FCAC

warnings.filterwarnings('ignore')


class Agent:
    def __init__(self, n_workers, nS, nA, seed, device):

        self.nS = nS
        self.nA = nA
        self.seed = seed
        self.n_workers = n_workers
        self.gamma = .99
        self.hidden_dims = (256, 128)
        self.lr = .001

        self.ac_model = FCAC(device, self.nS, self.nA, hidden_dims=self.hidden_dims)
        self.optimizer = optim.RMSprop(self.ac_model.parameters(), lr=self.lr)
        self.max_grad = 1.

        self.policy_loss_weight = 1.
        self.value_loss_weight = .6
        self.entropy_loss_weight = .001
        self.lambdaa = .95

        self.values = None
        self.entropies = None
        self.rewards = None
        self.logpas = None

    def interact_with_environment(self, states, mp_env, t_step):
        # Infer on batch of states
        actions, logpas, entropies, values = self.ac_model.full_pass(states)

        # send the 'step' cmd from main process to child process
        new_states, rewards, dones = mp_env.step(states, actions, t_step)

        self.logpas.append(logpas)
        self.entropies.append(entropies)
        self.rewards.append(rewards)
        self.values.append(values)

        return new_states, dones

    def learn(self):
        logpas = torch.stack(self.logpas).squeeze()
        entropies = torch.stack(self.entropies).squeeze()
        values = torch.stack(self.values).squeeze()
        n_step_returns = []
        gaes = []

        T = len(self.rewards)  # length of rewards (+ the bootstrapping state-value)

        # the sequence starts at base**start and ends with base**stop.
        discounts = np.logspace(start=0, stop=T, num=T, base=self.gamma, endpoint=False)
        rewards = np.array(self.rewards).squeeze()

        # compute the n-step return from each t
        for w in range(self.n_workers):
            for t_step in range(T):
                discounted_reward = discounts[: T - t_step] * rewards[t_step:, w]
                n_step_returns.append(np.sum(discounted_reward))

        n_step_returns = np.array(n_step_returns).reshape(self.n_workers, T)

        # T-1 because the recall the last value in T=len(rewards) is a bootstrapping value
        lambda_discounts = np.logspace(
            start=0, stop=T - 1, num=T - 1, base=self.gamma * self.lambdaa, endpoint=False)

        np_values = values.data.numpy()
        # array of TD errors from 0 to T:   ∑ Rγ * V(St+1) - V(St)
        td_errors = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]

        for w in range(self.n_workers):
            for t_step in range(T - 1):
                discounted_advantage = lambda_discounts[: T - 1 - t_step] * td_errors[t_step:, w]
                gaes.append(np.sum(discounted_advantage))

        gaes = np.array(gaes).reshape(self.n_workers, T - 1)
        discounted_gaes = discounts[:-1] * gaes

        # For some tensors we use reshape instead of view because view only works on
        # contiguous tensors. When transposing the tensor, it becomes non-contiguous in memory.
        # we could have used also: x.contiguous().view(-1)

        # [:-1, ...] remove last row on the first dimension but keep all other dimensions
        values = values[:-1, ...].view(-1).unsqueeze(1)
        logpas = logpas.view(-1).unsqueeze(1)
        entropies = entropies.view(-1).unsqueeze(1)
        n_step_returns = torch.FloatTensor(n_step_returns.T[:-1]).reshape(-1, 1)
        discounted_gaes = torch.FloatTensor(discounted_gaes.T).reshape(-1, 1)

        T -= 1
        T *= self.n_workers
        assert n_step_returns.size() == (T, 1)
        assert values.size() == (T, 1)
        assert logpas.size() == (T, 1)
        assert entropies.size() == (T, 1)

        value_error = n_step_returns.detach() - values

        value_loss = value_error.pow(2).mul(0.5).mean()
        policy_loss = -(discounted_gaes.detach() * logpas).mean()
        entropy_loss = -entropies.mean()

        loss = self.policy_loss_weight * policy_loss + \
               self.value_loss_weight * value_loss + \
               self.entropy_loss_weight * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), self.max_grad)
        self.optimizer.step()

    def evaluate_one_episode(self, env):
        self.ac_model.eval()

        total_rewards = 0.0
        n_highs = 0.0

        s, d = env.reset(), False

        for ts in count():
            with torch.no_grad():
                a = self.ac_model.select_action(s)

            s, r, d = env.step(s, a, ts)

            total_rewards += r
            n_highs += env.n_high_adjacent

            if d: break

        self.ac_model.train()
        return total_rewards, n_highs

    def reset_metrics(self):
        self.logpas = []
        self.rewards = []
        self.entropies = []
        self.values = []
