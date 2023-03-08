import warnings
from itertools import count

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from fc import DuelingDQN, DQN
from action_selection import EGreedyExpStrategy, GreedyStrategy
from replay_buffer import ReplayBuffer

warnings.filterwarnings('ignore')


class Agent:
    def __init__(self, nS, nA, seed, device):
        self.seed = seed
        self.batch_size = 256
        lr = .01
        self.gamma = .99
        self.device = device
        self.strategy = EGreedyExpStrategy()
        self.use_ddqn = True
        self.use_dueling = True
        self.tau = 0.1

        self.memory_capacity = 50000
        self.memory = ReplayBuffer(self.memory_capacity, self.batch_size, self.seed)

        # TODO: Add more layers
        hidden_dims = (512, 128)
        hidden_dims = (1024, 512, 512, 128)

        if self.use_dueling:
            self.behavior_policy = DuelingDQN(self.device, nS, nA, hidden_dims=hidden_dims).to(self.device)
            self.target_policy = DuelingDQN(self.device, nS, nA, hidden_dims=hidden_dims).to(self.device)
        else:
            self.behavior_policy = DQN(self.device, nS, nA, hidden_dims=hidden_dims).to(self.device)
            self.target_policy = DQN(self.device, nS, nA, hidden_dims=hidden_dims).to(self.device)

        self.optimizer = optim.RMSprop(self.behavior_policy.parameters(), lr=lr)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def interact_with_environment(self, state, env, t_step, nA):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.strategy.select_action(self.behavior_policy, state, nA, env.actions_history)
        next_state, reward, done = env.step(action, t_step)
        return action, reward, next_state, done

    def sample_and_learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.device)
        if self.use_ddqn:
            # Action that have the highest value: Index of action ==> FROM THE BEHAVIOR POLICY
            argmax_q_next = self.behavior_policy(next_states).detach().argmax(dim=1).unsqueeze(-1)

            # Action-values of "best" actions  ==> FROM THE TARGET POLICY
            Q_targets_next = self.target_policy(next_states).gather(1, argmax_q_next)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        else:
            # highest action-values : Q(Sₜ₊₁,a)
            Q_targets_next = self.target_policy(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        actions = actions.to(torch.int64)
        Q_expected = self.behavior_policy(states).gather(1, actions)

        loss = F.huber_loss(Q_expected, Q_targets, delta=np.inf)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate_one_episode(self, env, shuffle):
        total_rewards = 0.0
        stoppage_pp = np.zeros(env.N_POSITIONS)

        s, d = env.reset(shuffle), False

        for ts in range(env.N_SEATS):
            with torch.no_grad():
                a = GreedyStrategy.select_action(self.behavior_policy, s, env.actions_history)
            s, r, d = env.step(a, ts)

            total_rewards += r
            stoppage_pp += env.stoppage_duration_pp

            if d: break

        return total_rewards, stoppage_pp

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
    pass
