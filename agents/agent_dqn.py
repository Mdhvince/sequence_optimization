import warnings
from itertools import count

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from fc import DuelingDQN
from action_selection import EGreedyExpStrategy, GreedyStrategy
from replay_buffer import ReplayBuffer

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
        states = states.reshape(self.batch_size, self.C, self.nA, -1)
        next_states = next_states.reshape(self.batch_size, self.C, self.nA, -1)

        states = states.squeeze(1)
        next_states = next_states.squeeze(1)

        # Action that have the highest value: Index of action ==> FROM THE BEHAVIOR POLICY
        argmax_q_next = self.behavior_policy(next_states).detach().argmax(dim=1).unsqueeze(-1)
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
            seq.append(a)

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
    pass
