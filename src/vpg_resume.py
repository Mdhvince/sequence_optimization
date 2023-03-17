import random
import warnings
from itertools import count
from collections import deque

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment import EnvSeqV2
from networks import PolicyVPG, ValueVPG
from src.vpg import Agent

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    writer = SummaryWriter("runs/vpg_resume")

    seed = 42
    model_path = "weights/vpg.pt"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    wc = pd.read_csv("../notebooks/WC.csv", sep=";").iloc[:20, :]
    wc = wc.sample(frac=1, random_state=seed)

    positions = wc.columns
    takt_time = np.ones(len(positions)) * 59
    buffer_percent = np.array([1.45, 1.60, 1.25, 1.50, 1.25, 1.25, 1.25, 1.50, 1.25, 1.25, 1.25, 1.25, 1.25])

    env = EnvSeqV2(wc, takt_time, buffer_percent)
    nA = env.action_space
    state_shape = env.reset().shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    last_episode = checkpoint["episode"]
    print(f"last mean reward: {checkpoint['mean_100_reward']}\n")

    agent = Agent(state_shape, nA, device, resume=True, checkpoint=checkpoint)

    # Resume
    n_episodes = 15000
    last_100_reward = deque(maxlen=100)
    last_100_stoppage_pp = deque(maxlen=100)

    for e in range(last_episode, last_episode + n_episodes):
        shuffle = False
        state, is_terminal = env.reset(shuffle), False

        agent.reset_metrics()

        for t_step in count():
            next_state, is_terminal = agent.interact_with_environment(state, env, t_step)
            state = next_state
            if is_terminal: break

        next_value = 0 if is_terminal else agent.value_model(state).detach().item()
        agent.rewards.append(next_value)
        agent.learn()

        reward_episode, stoppage_pp, sequence = agent.evaluate_one_episode(env, shuffle)
        last_100_reward.append(reward_episode)
        last_100_stoppage_pp.append(stoppage_pp / 60)

        if e % 1 == 0:
            mean_stoppage_per_position = dict(zip(positions, np.mean(np.array(last_100_stoppage_pp), axis=0)))
            mean_rewards = np.mean(last_100_reward)

            print(f"Episode {e} \t {mean_rewards}")
            print(f"Episode {e} \t {mean_stoppage_per_position}")
            print("")
            # writer.add_scalars("Mean down time per position in minutes", mean_stoppage_per_position, e)
            # writer.add_scalar("Mean reward", mean_rewards, e)


    writer.close()
