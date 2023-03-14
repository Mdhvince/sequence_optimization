import random
from collections import deque, namedtuple
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = seed

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, device):
        """Randomly sample a batch of experiences from memory."""

        # give me batch_size "randomly" selected <S, A, Rₜ₊₁, Sₜ₊₁> in a list
        experiences = random.sample(self.memory, k=self.batch_size)

        # stack all the states together and convert them to a tensor
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(device)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)

        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        return states, actions, rewards, next_states, dones  # (batch_size x 5)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


if __name__ == "__main__":
    pass
