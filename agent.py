import warnings
from collections import deque

import torch
import torch.optim as optim
from fc import FCDP, FCQV
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

    def interact_with_environment(self, state, env, t_step, max_iter):
        min_samples = self.memory.batch_size * self.n_warmup_batches
        use_max_exploration = len(self.memory) < min_samples

        action = self.training_strategy.select_action(self.actor, state, use_max_exploration)
        next_state, reward, is_terminal, _ = env.step(action, state, t_step, max_iter)
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

    def evaluate_one_episode(self, env, max_iter):
        total_rewards = 0
        reward_per_swap = deque(maxlen=5)

        s, d = env.reset(), False

        for ts in range(max_iter):
            with torch.no_grad():
                a = self.eval_strategy.select_action(self.actor, s)

            s, r, d, _ = env.step(a, s, ts, max_iter)

            total_rewards += r
            reward_per_swap.append(r)
            if d: break

        return total_rewards, reward_per_swap

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


if __name__ == "__main__":
    pass
