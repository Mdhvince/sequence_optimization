import random
import warnings
from collections import deque
from itertools import count

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent import VPG
from environment import EnvSeq

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    writer = SummaryWriter("runs/vpg")

    seed = 42
    model_path = "actor.pt"
    n_episodes = 300000

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = EnvSeq()
    nS, nA = env.observation_space, env.action_space

    print(f"nS: {nS}\nnA: {nA}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = VPG(nS, nA, device)

    last_100_score = deque(maxlen=100)
    mean_of_last_100 = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        state, is_terminal = env.reset(), False

        agent.reset_metrics()

        for t_step in count():
            next_state, is_terminal = agent.interact_with_environment(state, env, t_step)  # VPG only
            state = next_state
            if is_terminal: break

        next_value = 0 if is_terminal else agent.value_model(state).detach().item()
        agent.rewards.append(next_value)
        agent.learn()

        # Evaluate
        total_rewards, steps_completed, sequence = agent.evaluate_one_episode(env)
        last_100_score.append(total_rewards)

        # print(f"Steps completed: {steps_completed} \t Rewards: {total_rewards}")
        # print(f"Sequence: {sequence} \t Rewards: {total_rewards}")

        if len(last_100_score) >= 100:
            mean_100_score = np.mean(last_100_score)
            if i_episode % 100 == 0:
                # print(f"Episode {i_episode}\tAverage mean 100 eval score: {mean_100_score}")
                writer.add_scalar("mean_100_rewards", mean_100_score, i_episode)

        #
        #     if (mean_100_score >= goal_mean_100_reward):
        #         torch.save(agent.actor.state_dict(), model_path)
        #         old_score = mean_100_score
        #         break

    writer.close()