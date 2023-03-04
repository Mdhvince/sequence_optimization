import warnings
from itertools import count

import torch
import numpy as np
import torch.optim as optim
from fc import FCDP, FCQV, FCDAP, FCV, FCAC
from action_selection import GreedyStrategyContinuous, NormalNoiseStrategyContinuous
from replay_buffer import ReplayBuffer

warnings.filterwarnings('ignore')


class DDPG:
    def __init__(self, action_bounds, nS, nA, seed, device):

        self.device = device
        buffer_size = 100000
        bs = 256
        hidden_dims = 64, 64
        lr = 0.0003
        self.tau = 0.005
        self.gamma = .99
        self.n_warmup_batches = 5

        self.memory = ReplayBuffer(buffer_size, bs, seed)

        self.critic = FCQV(device, nS, nA, hidden_dims)  # using ReLu by default
        self.critic_target = FCQV(device, nS, nA, hidden_dims)

        self.actor = FCDP(device, nS, action_bounds, hidden_dims)  # ReLu + Tanh
        self.actor_target = FCDP(device, nS, action_bounds, hidden_dims)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.max_grad = float('inf')

        self.training_strategy = NormalNoiseStrategyContinuous(action_bounds, exploration_noise_ratio=0.1)
        self.eval_strategy = GreedyStrategyContinuous(action_bounds)

    def interact_with_environment(self, state, env, t_step):
        min_samples = self.memory.batch_size * self.n_warmup_batches
        use_max_exploration = len(self.memory) < min_samples

        action = self.training_strategy.select_action(self.actor, state, use_max_exploration)

        next_state, reward, is_terminal = env.step(state, action, t_step)
        experience = (state, action, reward, next_state, float(is_terminal))

        return experience

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def sample_and_learn(self):
        states, actions, rewards, next_states, is_terminals = self.memory.sample(self.device)

        # update the critic: Li(θ) = ( r + γQ(s′,μ(s′; ϕ); θ) − Q(s,a;θi) )^2

        a_next = self.actor_target(next_states)
        Q_next = self.critic_target(next_states, a_next)
        Q_target = rewards + self.gamma * Q_next * (1 - is_terminals)
        Q = self.critic(states, actions)

        error = Q - Q_target.detach()
        critic_loss = error.pow(2).mul(0.5).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
        self.critic_optimizer.step()

        # update the actor: Li(ϕ) = -1/N * sum of Q(s, μ(s; ϕi); θi)

        a_pred = self.actor(states)
        Q_pred = self.critic(states, a_pred)

        actor_loss = -Q_pred.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
        self.actor_optimizer.step()

    def evaluate_one_episode(self, env):
        total_rewards = 0

        s, d = env.reset(), False

        for ts in count():
            with torch.no_grad():
                a = self.eval_strategy.select_action(self.actor, s)
            s, r, d = env.step(s, a, ts)

            total_rewards += r
            if d: break

        return total_rewards

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

            # mixe value networks
            for t, b in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_ratio = (1.0 - self.tau) * t.data
                behavior_ratio = self.tau * b.data
                mixed_weights = target_ratio + behavior_ratio
                t.data.copy_(mixed_weights.data)

            # mix policy networks
            for t, b in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_ratio = (1.0 - self.tau) * t.data
                behavior_ratio = self.tau * b.data
                mixed_weights = target_ratio + behavior_ratio
                t.data.copy_(mixed_weights.data)
        else:
            """
            target network was frozen during n steps, now we are update it with the behavior network
            weight.
            """
            for t, b in zip(self.critic_target.parameters(), self.critic.parameters()):
                t.data.copy_(b.data)

            for t, b in zip(self.actor_target.parameters(), self.actor.parameters()):
                t.data.copy_(b.data)


class VPG:
    def __init__(self, nS, nA, device):
        self.device = device
        self.gamma = .99
        lr_p = 0.0005
        lr_v = 0.0007
        hidden_dims_p = (128, 64)
        hidden_dims_v = (256, 128)
        self.entropy_loss_weight = 0.001

        # Define policy network, value network and max gradient for gradient clipping
        self.policy = FCDAP(self.device, nS, nA, hidden_dims=hidden_dims_p).to(self.device)
        self.p_optimizer = optim.Adam(self.policy.parameters(), lr=lr_p)
        self.p_max_grad = 1

        self.value_model = FCV(self.device, nS, hidden_dims=hidden_dims_v).to(self.device)
        self.v_optimizer = optim.RMSprop(self.value_model.parameters(), lr=lr_v)
        self.v_max_grad = float("inf")

        self.logpas = []
        self.rewards = []
        self.entropies = []
        self.values = []

    def interact_with_environment(self, state, env, t_step):
        action, logpa, entropy = self.policy.full_pass(state)
        next_state, reward, is_terminal = env.step(state, action, t_step)

        self.logpas.append(logpa)
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

    def evaluate_one_episode(self, env):
        sequence = []
        self.policy.eval()
        total_rewards = 0
        steps_completed = 0

        s, d = env.reset(), False

        for ts in count():
            with torch.no_grad():
                a = self.policy.select_greedy_action(s)
            s, r, d = env.step(s, a, ts)

            total_rewards += r
            steps_completed += 1
            sequence.append(a)
            if d: break

        self.policy.train()
        return total_rewards, steps_completed, sequence

    def reset_metrics(self):
        self.logpas = []
        self.rewards = []
        self.entropies = []
        self.values = []


if __name__ == "__main__":
    pass
