import random
import warnings
from collections import deque
from itertools import count

import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from agents.agent_dqn import Agent
from environment import EnvSeqV0, EnvSeqV1

warnings.filterwarnings('ignore')

# def chunker(seq, size):
#     return [seq[pos:pos + size] for pos in range(0, len(seq), size)]


if __name__ == "__main__":
    writer = SummaryWriter("runs/dqns")

    seed = 42
    model_path = "dqn_agent.pt"
    n_episodes = 3600
    warmup_bs = 5

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    wc = pd.read_csv("notebooks/WC.csv", sep=";")
    positions = wc.columns
    takt_time = np.ones(len(positions)) * 59
    buffer_percent = np.array([1.45, 1.60, 1.25, 1.50, 1.25, 1.25, 1.25, 1.50, 1.25, 1.25, 1.25, 1.25, 1.25])

    env = EnvSeqV1(wc, takt_time, buffer_percent)
    nS, nA = env.observation_space, env.action_space
    print(f"nS: {nS}\nnA: {nA}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(nS, nA, seed, device)

    last_100_score = deque(maxlen=100)
    last_100_stoppage_duration = deque(maxlen=100)
    last_100_stoppage_pp = deque(maxlen=100)

    running_mean_100 = -np.inf

    for i_episode in range(1, n_episodes + 1):
        shuffle = i_episode == 1  # (i_episode % 10 == 0 or i_episode == 1)
        state, is_terminal = env.reset(shuffle), False

        for t_step in range(env.N_SEATS):
            action, reward, next_state, done = agent.interact_with_environment(state, env, t_step, nA)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > agent.batch_size * warmup_bs:
                agent.sample_and_learn()  # one-step optimization on the behavior policy
                agent.sync_weights(use_polyak_averaging=True)
            if done: break

        total_rewards, stoppage_pp = agent.evaluate_one_episode(env, shuffle)
        last_100_score.append(total_rewards)
        last_100_stoppage_pp.append(stoppage_pp / 60)

        # if i_episode % 5 == 0:
        # writer.add_scalar("Mean last 100 rewards", np.mean(last_100_score), i_episode)
        mean_stoppage_per_position = np.mean(np.array(last_100_stoppage_pp), axis=0)
        writer.add_scalars(
            f"Mean down time per position (minutes)",
            dict(zip(positions, mean_stoppage_per_position)), i_episode
        )

        # if (mean_100_score > running_mean_100) and (mean_100_score > 0):
        #     print(f"Mean 100 reward: {mean_100_score}\t Saving weights...")
        #     torch.save(agent.behavior_policy.state_dict(), model_path)
        #     running_mean_100 = mean_100_score

    writer.close()
