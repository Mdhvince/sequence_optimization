import random
import warnings
from itertools import count
from collections import deque

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.networks import DuelingDQN
from src.environment import EnvSeqV1, EnvSeqV2
from src.action_selection import EGreedyExpStrategy, GreedyStrategy
from src.replay_buffer import ReplayBuffer

warnings.filterwarnings('ignore')


class Agent:
    def __init__(self, state_shape, nA, seed, device):
        """
        :param state_shape: (CxHxW) shape of the state
        :param nA:
        :param seed:
        :param device:
        """
        self.C = state_shape[0]
        self.seed = seed
        self.batch_size = 256
        self.nA = nA
        # self.nA_step = nA_step
        lr = .001
        self.gamma = .995
        self.device = device
        self.strategy = EGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.1)
        self.use_ddqn = True
        self.tau = 0.1

        self.memory_capacity = 100000
        self.memory = ReplayBuffer(self.memory_capacity, self.batch_size, self.seed)

        hidden_dims = (512, 128)
        # hidden_dims = (1024, 512, 512, 128)

        self.behavior_policy = DuelingDQN(device, state_shape, nA, hidden_dims=hidden_dims).to(device)
        self.target_policy = DuelingDQN(device, state_shape, nA, hidden_dims=hidden_dims).to(device)

        self.optimizer = optim.Adam(self.behavior_policy.parameters(), lr=lr)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def interact_with_environment(self, state, env, t_step, nA):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.strategy.select_action(self.behavior_policy, state, nA)
        next_state, reward, done = env.step(action, t_step)
        return action, reward, next_state, done

    def sample_and_learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.device)
        states = states.reshape(self.batch_size, self.C, states.shape[1], -1)
        next_states = next_states.reshape(self.batch_size, self.C, next_states.shape[1], -1)

        states = states.squeeze(1)
        next_states = next_states.squeeze(1)

        # Action that have the highest value: Index of action ==> FROM THE BEHAVIOR POLICY
        argmax_q_next = self.behavior_policy(next_states).detach().argmax(dim=1).unsqueeze(-1)
        # _, argmax_q_next = torch.topk(self.behavior_policy(next_states).detach(), k=self.nA_step, dim=1)

        # Action-values of "best" actions  ==> FROM THE TARGET POLICY
        Q_targets_next = self.target_policy(next_states).gather(1, argmax_q_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        actions = actions.to(torch.int64)
        Q_expected = self.behavior_policy(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate_one_episode(self, env, shuffle):
        total_rewards = 0.0
        stoppage_pp = np.zeros(env.N_POSITIONS)
        seq = []

        s, d = env.reset(shuffle), False

        for ts in count():
            with torch.no_grad():
                a = GreedyStrategy.select_action(self.behavior_policy, s)

            s, r, d = env.step(a, ts)

            total_rewards += r
            stoppage_pp += env.stoppage_per_position

            if d: break

        return total_rewards, stoppage_pp, seq

    def sync_weights(self, use_polyak_averaging=True):
        if use_polyak_averaging:
            """
            Instead of freezing the target and doing a big update every n steps, we can slow down
            the target by mixing a big % of weight from the target and a small % from the 
            behavior policy. So the update will be smoother and continuous at each time step.
            For example we add 1% of new information learned by the behavior policy to the target
            policy at every step.
            - self.tau: ratio of the behavior network that will be mixed into the target network.
            tau = 1 means full update (100%)
            """
            if self.tau is None:
                raise Exception("You are using Polyak averaging but TAU is None")

            for t, b in zip(self.target_policy.parameters(), self.behavior_policy.parameters()):
                target_ratio = (1.0 - self.tau) * t.data
                behavior_ratio = self.tau * b.data
                mixed_weights = target_ratio + behavior_ratio
                t.data.copy_(mixed_weights.data)
        else:
            """
            target network was frozen during n steps, now we are update it with the behavior network
            weight.
            """
            for t, b in zip(self.target_policy.parameters(), self.behavior_policy.parameters()):
                t.data.copy_(b.data)



if __name__ == "__main__":
    writer = SummaryWriter("../runs/dqns")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    SEED = 42
    MODEL_PATH = "dqn_agent.pt"
    N_EPISODES = 25600
    WARMUP_BS = 5

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    wc = pd.read_csv("../notebooks/WC.csv", sep=";").sample(frac=1, random_state=SEED)
    wc = wc.iloc[:50, :]

    positions = wc.columns
    takt_time = np.ones(len(positions)) * 59
    buffer_percent = np.array([1.45, 1.60, 1.25, 1.50, 1.25, 1.25, 1.25, 1.50, 1.25, 1.25, 1.25, 1.25, 1.25])

    env = EnvSeqV2(wc, takt_time, buffer_percent)
    nA = env.action_space
    s = env.reset()
    state_shape = s.shape

    agent = Agent(state_shape, nA, SEED, device)
    writer.add_graph(
        agent.behavior_policy,
        torch.tensor(s, device=device, dtype=torch.float32).unsqueeze(0)
    )

    last_100_reward = deque(maxlen=100)
    last_100_stoppage_duration = deque(maxlen=100)
    last_100_stoppage_pp = deque(maxlen=100)

    running_mean_100 = -np.inf

    for e in range(1, N_EPISODES + 1):
        shuffle = False
        state, is_terminal = env.reset(shuffle), False

        for t_step in count():
            action, reward, next_state, done = agent.interact_with_environment(state, env, t_step, nA)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > agent.batch_size * WARMUP_BS:
                agent.sample_and_learn()
                agent.sync_weights(use_polyak_averaging=True)
            if done: break

        reward_episode, stoppage_pp, sequence = agent.evaluate_one_episode(env, shuffle)
        last_100_reward.append(reward_episode)
        last_100_stoppage_pp.append(stoppage_pp / 60)

        if e % 100 == 0:
            mean_stoppage_per_position = dict(zip(positions, np.mean(np.array(last_100_stoppage_pp), axis=0)))
            mean_rewards = np.mean(last_100_reward)
            writer.add_scalars("Mean down time per position in minutes", mean_stoppage_per_position, e)
            writer.add_scalar("Mean reward", mean_rewards, e)

    writer.close()
