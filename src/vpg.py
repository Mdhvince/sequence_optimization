import random
import warnings
import configparser
from itertools import count
from collections import deque

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment import EnvSeqV2
from networks import PolicyVPG, ValueVPG

warnings.filterwarnings('ignore')


class Agent:
    def __init__(self, state_shape, nA, device, run_type="train", checkpoint=None):
        self.device = device
        self.gamma = .99
        lr_p = 0.0005
        lr_v = 0.0007
        hidden_dims_p = (512, 256, 128, 32)
        hidden_dims_v = (512, 256, 128, 64, 32)
        self.entropy_loss_weight = 0.001
        self.C = state_shape[0]

        # Define policy network, value network, optimizers and max gradient for gradient clipping
        self.policy = PolicyVPG(device, state_shape, nA, hidden_dims=hidden_dims_p)
        self.value_model = ValueVPG(device, state_shape, hidden_dims=hidden_dims_v)

        self.p_optimizer = optim.Adam(self.policy.parameters(), lr=lr_p)
        self.v_optimizer = optim.Adam(self.value_model.parameters(), lr=lr_v)

        self.p_max_grad = 1
        self.v_max_grad = float("inf")

        self.logpas = []
        self.rewards = []
        self.entropies = []
        self.values = []

        if run_type in ["resume", "evaluate"]:
            assert checkpoint is not None
            print("Resume training")
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.value_model.load_state_dict(checkpoint["value_state_dict"])
            self.p_optimizer.load_state_dict(checkpoint["p_optimizer_state_dict"])
            self.v_optimizer.load_state_dict(checkpoint["v_optimizer_state_dict"])

            for s in self.p_optimizer.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        s[k] = v.to(device)

            for s in self.v_optimizer.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        s[k] = v.to(device)

    def interact_with_environment(self, state, env, t_step):
        action, logpa, entropy = self.policy.full_pass(state)
        next_state, reward, is_terminal, _ = env.step(action, t_step)

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
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1).to(self.device)

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

    def evaluate_one_episode(self, env, shuffle):
        self.policy.eval()
        total_rewards = 0.0
        stoppage_pp = np.zeros(env.N_POSITIONS)
        seq = []

        s, d = env.reset(shuffle), False

        for ts in count():
            with torch.no_grad():
                a = self.policy.select_greedy_action(s)
            s, r, d, info = env.step(a, ts)

            total_rewards += r
            stoppage_pp += env.stoppage_per_position
            seq.append(info)
            if d: break

        self.policy.train()
        return total_rewards, stoppage_pp, seq

    def reset_metrics(self):
        self.logpas = []
        self.rewards = []
        self.entropies = []
        self.values = []


# END Agent


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    last_episode = checkpoint["episode"]
    last_best_reward = checkpoint['mean_100_reward']
    print(f"last mean reward: {last_best_reward}\n")
    return checkpoint, last_episode, last_best_reward

def save_model(model_path, agent, mean_rewards, episode):
    torch.save({
        'episode': episode,
        'mean_100_reward': mean_rewards,
        'policy_state_dict': agent.policy.state_dict(),
        'value_state_dict': agent.value_model.state_dict(),
        'p_optimizer_state_dict': agent.p_optimizer.state_dict(),
        'v_optimizer_state_dict': agent.v_optimizer.state_dict(),
    }, model_path)

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def update_results_episode(reward_list, stoppage_list, reward_episode, stoppage_episode):
    reward_list.append(reward_episode)
    stoppage_list.append(stoppage_episode / 60)

def compute_stats(positions, last_100_stoppage_pp, last_100_reward):
    mean_stoppage_per_position = dict(zip(positions, np.mean(np.array(last_100_stoppage_pp), axis=0)))
    mean_rewards = np.mean(last_100_reward)
    return mean_stoppage_per_position, mean_rewards

def print_stats(episode, mean_rewards, mean_stoppage_per_position):
    print(f"Episode {episode} \t {mean_rewards}")
    print(f"Episode {episode} \t {mean_stoppage_per_position}\n")

def broadcast_stats(writer, episode, mean_rewards, mean_stoppage_per_position):
    writer.add_scalars("Mean down time per position in minutes", mean_stoppage_per_position, e)
    writer.add_scalar("Mean reward", mean_rewards, episode)

def setup(config_filepath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()
    config.read(config_filepath)
    return device, config["DEFAULT"], config["TRAIN"], config["TENSORBOARD"]

def make_env(cfg):
    wc = pd.read_csv(cfg.get("workcontent_filepath"), sep=";").iloc[:20, :]
    wc = wc.sample(frac=1, random_state=SEED)
    positions = wc.columns
    takt_time = np.ones(len(positions)) * 59
    buffer_percent = np.array([1.45, 1.60, 1.25, 1.50, 1.25, 1.25, 1.25, 1.50, 1.25, 1.25, 1.25, 1.25, 1.25])
    env = EnvSeqV2(wc, takt_time, buffer_percent)
    return env

def get_writer(cfg_tsb):
    return SummaryWriter(cfg_tsb.get("folder")) if LAUNCH_TSB else None

def load_checkpoint_info(mode, model_path, device):
    if mode == "train":
        checkpoint = None
        last_episode = 0
        last_best_reward = -np.inf
    else:
        checkpoint, last_episode, last_best_reward = load_model(model_path, device)
    return checkpoint, last_episode, last_best_reward



if __name__ == "__main__":
    device, cfg, cfg_train, cfg_tsb = setup(config_filepath="config.ini")

    SEED = cfg.getint("seed")
    MODE = cfg.get("mode")
    MODEL_PATH = cfg.get("model_path")
    N_EPISODES = cfg.getint("n_episodes")
    PRINT_EVERY = cfg.getint("print_every")
    DEBUG = cfg.getboolean("debug")
    TARGET_REWARD = cfg_train.getint("target_reward")
    LAUNCH_TSB = cfg_tsb.getboolean("launch")

    assert MODE in ["train", "evaluation", "resume"]

    seed_everything(SEED)
    checkpoint, last_episode, last_best_reward = load_checkpoint_info(MODE, MODEL_PATH, device)

    env = make_env(cfg)
    state_shape, nA = env.reset().shape, env.action_space
    agent = Agent(state_shape, nA, device, run_type=MODE, checkpoint=checkpoint)

    last_100_reward = deque(maxlen=100)
    last_100_stoppage_pp = deque(maxlen=100)
    mean_rewards = -np.inf

    # curriculum learning is applied via resume training

    if MODE == "evaluate":
        reward_episode, stoppage_pp, sequence = agent.evaluate_one_episode(env, shuffle=False)
    else:
        writer = get_writer(cfg_tsb)
        for e in range(last_episode, last_episode + N_EPISODES):
            shuffle = False

            state, is_terminal = env.reset(shuffle), False
            agent.reset_metrics()

            for t_step in count():
                state, is_terminal = agent.interact_with_environment(state, env, t_step)  # state = next_state (output)
                if is_terminal: break

            next_value = 0 if is_terminal else agent.value_model(state).detach().item()
            agent.rewards.append(next_value)
            agent.learn()

            reward_episode, stoppage_pp, _ = agent.evaluate_one_episode(env, shuffle)
            update_results_episode(last_100_reward, last_100_stoppage_pp, reward_episode, stoppage_pp)

            if e % PRINT_EVERY == 0:
                res, mean_rewards = compute_stats(env.wc.columns, last_100_stoppage_pp, last_100_reward)
                broadcast_stats(writer, e, mean_rewards, res) if LAUNCH_TSB else print_stats(e, mean_rewards, res)

            if DEBUG is False:
                if (MODE == "resume") and (mean_rewards >= last_best_reward):
                    save_model(MODEL_PATH, agent, mean_rewards, e)
                    last_best_reward = mean_rewards

                elif (MODE == "train") and (mean_rewards >= TARGET_REWARD):
                    save_model(MODEL_PATH, agent, mean_rewards, e)
                    break

    print("Finished.")
    if LAUNCH_TSB: writer.close()
