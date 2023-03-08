import random
import warnings
from collections import deque
from itertools import count

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.agent_vpg import Agent
from environment import EnvSeqV0, EnvSeqV1

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    writer = SummaryWriter("runs/vpg")

    seed = 42
    model_path = "agent_vpg.pt"
    n_episodes = "-àà"

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
    agent = Agent(nS, nA, device)

    last_100_score = deque(maxlen=100)
    last_100_stoppage_duration = deque(maxlen=100)
    last_100_stoppage_pp = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        shuffle = i_episode == 1
        state, is_terminal = env.reset(shuffle), False

        agent.reset_metrics()

        for t_step in count():
            next_state, is_terminal = agent.interact_with_environment(state, env, t_step)  # VPG only
            state = next_state
            if is_terminal: break

        next_value = 0 if is_terminal else agent.value_model(state).detach().item()
        agent.rewards.append(next_value)
        agent.learn()

        # Evaluate
        total_rewards, stoppage_pp = agent.evaluate_one_episode(env, shuffle)
        last_100_stoppage_pp.append(stoppage_pp / 60)

        # Stats
        mean_stoppage_per_position = np.mean(np.array(last_100_stoppage_pp), axis=0)
        writer.add_scalars(
            f"Mean down time per position (minutes)",
            dict(zip(positions, mean_stoppage_per_position)), i_episode
        )

        # Save
        # if (mean_100_score >= goal_mean_100_reward):
        #     torch.save(agent.actor.state_dict(), model_path)
        #     break

    writer.close()
