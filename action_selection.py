import numpy as np
import torch

INF = 9999999999999999

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

    def select_action(self, model, state, nA, illegal_actions=None):
        action = e_greedy_action_selection(model, state, self.epsilon, nA, illegal_actions)
        self._epsilon_update()
        return action


class GreedyStrategy:
    def __init__(self):
        self.exploratory_action_taken = False

    @staticmethod
    def select_action(model, state, illegal_actions=None):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()

        action = np.argmax(q_values)
        # Down the value of illegal actions
        if illegal_actions is not None:
            while action in illegal_actions:
                q_values[action] = -INF
                action = np.argmax(q_values)

        return action


def e_greedy_action_selection(model, state, epsilon, nA, illegal_actions=None):
    if np.random.rand() > epsilon:
        model.eval()
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
        model.train()
        action = np.argmax(q_values)

        # Down the value of illegal actions
        if illegal_actions is not None:
            while action in illegal_actions:
                q_values[action] = -INF
                action = np.argmax(q_values)
    else:
        action = np.random.randint(nA)
    return action


if __name__ == "__main__":
    pass
