import random
import warnings
from collections import deque

import numpy as np
import torch

from agent import DDPG
from environment import EnvSeq

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    seed = 42
    model_path = "actor.pt"
    n_episodes = 10000

    batch_of_seats, n_positions = 5, 9
    nS, nA = batch_of_seats * n_positions, 3

    max_iter = batch_of_seats * 2

    env = EnvSeq(batch_of_seats, n_positions, threshold=19)
    action_bounds = env.action_bounds

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = DDPG(action_bounds, nS, nA, seed, device)

    last_100_score = deque(maxlen=100)
    mean_of_last_100 = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        state, is_terminal = env.reset(), False

        for t_step in range(max_iter):
            state, action, reward, next_state, is_terminal = agent.interact_with_environment(state, env, t_step, max_iter)
            agent.store_experience(state, action, reward, next_state, is_terminal)
            state = next_state

            if len(agent.memory) > agent.memory.batch_size * agent.n_warmup_batches:
                agent.sample_and_learn()
                agent.sync_weights(use_polyak_averaging=True)

            if is_terminal: break

        # Evaluate
        total_rewards, reward_per_swap = agent.evaluate_one_episode(env, max_iter)
        last_100_score.append(total_rewards)

        if len(last_100_score) >= 100:
            mean_100_score = np.mean(last_100_score)

            if i_episode % 100 == 0:
                print(f"Episode {i_episode}\t{reward_per_swap}\tAverage mean 100 eval score: {mean_100_score}")

        #
        #     if (mean_100_score >= goal_mean_100_reward):
        #         torch.save(agent.actor.state_dict(), model_path)
        #         old_score = mean_100_score
        #         break
        #
        # if i_episode % 1000 == 0:
        #     print(f"Episode {i_episode}\t{reward_per_swap}\tAverage mean 100 eval score: {mean_100_score}")
