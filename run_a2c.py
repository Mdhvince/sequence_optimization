import random
import warnings
from itertools import count
from collections import deque

import numpy as np
import torch

from agents.agent_a2c import Agent
from environment import EnvSeq, MultiprocessEnv

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    seed = 42
    n_workers = 8
    model_path = "a2c_agent.pt"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = EnvSeq()
    nS, nA = env.observation_space, env.action_space

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(n_workers, nS, nA, seed, device)


    mp_env = MultiprocessEnv(n_workers, seed)
    states = mp_env.reset([False])                             # TODO: create a loop for episode and add this reset in the loop

    episode, n_steps_start = 0, 0
    max_n_steps = 200                                                  # TODO: this will need do be set as len(sequence)

    last_100_score = deque(maxlen=100)

    # n-step Advantage Estimate :  Aᴳᴬᴱ(Sₜ, Aₜ) = ∑ λⁿ Rₜ₊ₙ - V(Sₜ)
    agent.reset_metrics()
    for t_step in count():
        # ---- From here, everything is stacked (2d arrays of n rows = n_workers)
        states, dones = agent.interact_with_environment(states, mp_env, t_step)

        if dones.sum() or (t_step+1) - n_steps_start == max_n_steps:
            next_values = agent.ac_model.get_state_value(states).detach().numpy() * (1 - dones)

            agent.rewards.append(next_values)  # ∑ Rₜ₊ₙ + V(Sₜ₊ₙ)
            agent.values.append(torch.Tensor(next_values))

            agent.learn()

            agent.reset_metrics()
            n_steps_start = t_step + 1

        if dones.sum() != 0.:  # at least one worker is done
            total_rewards, n_highs = agent.evaluate_one_episode(env)
            last_100_score.append(total_rewards)
            mean_100_eval_score = np.mean(last_100_score)
            print(f"Episode {episode}\tAverage mean 100 eval score: {mean_100_eval_score}")


            # reset state of done workers, so they can restart collecting while others continue.
            for i in range(agent.n_workers):
                if dones[i]:
                    states[i] = mp_env.reset(worker_id=i)
                    episode += 1

    mp_env.close()
