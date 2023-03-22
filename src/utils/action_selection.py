import numpy as np
import torch


class EGreedyExpStrategy:
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, model, state, nA):
        action = e_greedy_action_selection(model, state, self.epsilon, nA)
        self._epsilon_update()
        return action


class GreedyStrategy:
    def __init__(self):
        self.exploratory_action_taken = False

    @staticmethod
    def select_action(model, state, illegal_actions=None):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()

        # top2actions = q_values.argsort()[-2:][::-1]
        action = np.argmax(q_values)
        return action


def e_greedy_action_selection(model, state, epsilon, nA, illegal_actions=None):
    if np.random.rand() > epsilon:
        model.eval()
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
        model.train()
        action = np.argmax(q_values)

        # top2actions = q_values.argsort()[-2:][::-1]

    else:
        action = np.random.randint(nA)
        # top2actions = np.random.randint(0, nA, size=2)
    return action


if __name__ == "__main__":
    pass
