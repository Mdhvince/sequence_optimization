import random
import warnings
from collections import deque
from itertools import count

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.agent_dqn import Agent
from environment import EnvSeq

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    writer = SummaryWriter("runs/dqns")

    seed = 42
    model_path = "dqn_agent.pt"
    n_episodes = 800000
    warmup_bs = 5

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = EnvSeq()
    nS, nA = env.observation_space, env.action_space
    print(f"nS: {nS}\nnA: {nA}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(nS, nA, seed, device)

    last_100_score = deque(maxlen=100)
    last_100_highs = deque(maxlen=100)

    running_mean_100 = -np.inf

    for i_episode in range(1, n_episodes + 1):

        if i_episode < 6000:
            shuffle = i_episode % 50 == 0
        elif 6000 <= i_episode < 10000:
            shuffle = i_episode % 25 == 0
        elif 10000 <= i_episode < 14000:
            shuffle = i_episode % 10 == 0
        else:
            shuffle = True

        state, is_terminal = env.reset(shuffle), False

        for t_step in count():
            action, reward, next_state, done = agent.interact_with_environment(state, env, t_step, nA)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > agent.batch_size * warmup_bs:
                agent.sample_and_learn()  # one-step optimization on the behavior policy
                agent.sync_weights(use_polyak_averaging=True)
            if done: break

        total_rewards, n_highs = agent.evaluate_one_episode(env, shuffle)
        last_100_highs.append(n_highs)
        last_100_score.append(total_rewards)

        if i_episode % 100 == 0:
            mean_100_score = np.mean(last_100_score)
            writer.add_scalar("mean_100_rewards", mean_100_score, i_episode)
            writer.add_scalar("n_high", np.mean(last_100_highs), i_episode)

            if (mean_100_score > running_mean_100) and (mean_100_score > 0):
                print(f"Mean 100 reward: {mean_100_score}\t Saving weights...")
                torch.save(agent.behavior_policy.state_dict(), model_path)

                running_mean_100 = mean_100_score

    writer.close()
