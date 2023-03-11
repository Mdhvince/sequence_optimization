import random
import warnings
from itertools import count
from collections import deque

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment import EnvSeqV1
from networks import PolicyVPG, ValueVPG

warnings.filterwarnings('ignore')


class Agent:
    def __init__(self, state_shape, nA, device):
        self.device = device
        self.gamma = .99
        lr_p = 0.0005
        lr_v = 0.0007
        hidden_dims_p = (512, 256, 128, 32)
        hidden_dims_v = (512, 256, 128, 64, 32)
        self.entropy_loss_weight = 0.001
        self.C = state_shape[0]

        # Define policy network, value network and max gradient for gradient clipping
        self.policy = PolicyVPG(self.device, state_shape, nA, hidden_dims=hidden_dims_p).to(self.device)
        self.p_optimizer = optim.Adam(self.policy.parameters(), lr=lr_p)
        self.p_max_grad = 1

        self.value_model = ValueVPG(self.device, state_shape, hidden_dims=hidden_dims_v).to(self.device)
        self.v_optimizer = optim.Adam(self.value_model.parameters(), lr=lr_v)
        self.v_max_grad = float("inf")

        self.logpas = []
        self.rewards = []
        self.entropies = []
        self.values = []

    def interact_with_environment(self, state, env, t_step):
        action, logpa, entropy = self.policy.full_pass(state)
        next_state, reward, is_terminal = env.step(action, t_step)

        self.logpas.append(logpa.squeeze())
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self.values.append(self.value_model(state))

        return next_state, is_terminal

    def learn(self):
        """
        Learn once full trajectory is collected
        """
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T - t] * self.rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)

        self.logpas = torch.cat(self.logpas)
        self.entropies = torch.cat(self.entropies)
        self.values = torch.cat(self.values)

        # --------------------------------------------------------------------
        # A(St, At) = Gt - V(St)
        # Loss = -1/N * sum_0_to_N( A(St, At) * log πθ(At|St) + βH )

        advantage = returns - self.values
        policy_loss = -(discounts * advantage.detach() * self.logpas).mean()
        entropy_loss_H = -self.entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss_H

        self.p_optimizer.zero_grad()
        loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.p_max_grad)
        self.p_optimizer.step()

        # --------------------------------------------------------------------
        # A(St, At) = Gt - V(St)
        # Loss = 1/N * sum_0_to_N( A(St, At)² )

        value_loss = advantage.pow(2).mul(0.5).mean()
        self.v_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.v_max_grad)
        self.v_optimizer.step()

    def evaluate_one_episode(self, env, shuffle):
        self.policy.eval()
        total_rewards = 0.0
        stoppage_pp = np.zeros(env.N_POSITIONS)
        seq = []

        s, d = env.reset(shuffle), False

        for ts in count():
            with torch.no_grad():
                a = self.policy.select_greedy_action(s)
            s, r, d = env.step(a, ts)

            total_rewards += r
            stoppage_pp += env.stoppage_per_position
            seq.append(a)
            if d: break

        self.policy.train()
        return total_rewards, stoppage_pp, seq

    def reset_metrics(self):
        self.logpas = []
        self.rewards = []
        self.entropies = []
        self.values = []


if __name__ == "__main__":
    writer = SummaryWriter("runs/vpg")

    seed = 42
    model_path = "agent_vpg.pt"
    n_episodes = 15000

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    wc = pd.read_csv("notebooks/WC.csv", sep=";").iloc[:20, :]
    wc = wc.sample(frac=1, random_state=seed)

    print(len(wc))

    positions = wc.columns
    takt_time = np.ones(len(positions)) * 59
    buffer_percent = np.array([1.45, 1.60, 1.25, 1.50, 1.25, 1.25, 1.25, 1.50, 1.25, 1.25, 1.25, 1.25, 1.25])

    env = EnvSeqV1(wc, takt_time, buffer_percent)
    nA = env.action_space
    state_shape = env.reset().shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(state_shape, nA, device)

    last_100_score = deque(maxlen=100)
    last_100_stoppage_duration = deque(maxlen=100)
    last_100_stoppage_pp = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        shuffle = True
        state, is_terminal = env.reset(shuffle), False

        agent.reset_metrics()

        for t_step in count():
            next_state, is_terminal = agent.interact_with_environment(state, env, t_step)
            state = next_state
            if is_terminal: break

        next_value = 0 if is_terminal else agent.value_model(state).detach().item()
        agent.rewards.append(next_value)
        agent.learn()

        total_rewards, stoppage_pp, sequence = agent.evaluate_one_episode(env, shuffle)
        last_100_stoppage_pp.append(stoppage_pp / 60)

        mean_stoppage_per_position = np.mean(np.array(last_100_stoppage_pp), axis=0)
        writer.add_scalars(
            f"Mean down time per position (minutes)",
            dict(zip(positions, mean_stoppage_per_position)), i_episode
        )

    writer.close()
