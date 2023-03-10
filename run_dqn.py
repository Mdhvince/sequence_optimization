import random
import warnings
from collections import deque
from itertools import count

import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from agents.agent_dqn import Agent
from environment import EnvSeqV1

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    writer = SummaryWriter("runs/dqns")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    SEED = 42
    MODEL_PATH = "dqn_agent.pt"
    N_EPISODES = 25600
    WARMUP_BS = 5

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    wc = pd.read_csv("notebooks/WC.csv", sep=";").iloc[:20, :]
    wc = wc.sample(frac=1, random_state=SEED)

    positions = wc.columns
    takt_time = np.ones(len(positions)) * 59
    buffer_percent = np.array([1.45, 1.60, 1.25, 1.50, 1.25, 1.25, 1.25, 1.50, 1.25, 1.25, 1.25, 1.25, 1.25])

    env = EnvSeqV1(wc, takt_time, buffer_percent)
    nA = env.action_space
    state_shape = env.reset().shape

    agent = Agent(state_shape, nA, SEED, device)

    last_100_reward = deque(maxlen=100)
    last_100_stoppage_duration = deque(maxlen=100)
    last_100_stoppage_pp = deque(maxlen=100)

    running_mean_100 = -np.inf

    for i_episode in range(1, N_EPISODES + 1):
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

        mean_stoppage_per_position = np.mean(np.array(last_100_stoppage_pp), axis=0)
        writer.add_scalars(
            f"Mean down time per position (minutes)", dict(zip(positions, mean_stoppage_per_position)), i_episode
        )
        writer.add_scalar("Mean reward", np.mean(last_100_reward), i_episode)

    writer.close()
