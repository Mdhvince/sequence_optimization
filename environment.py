import warnings
import numpy as np

warnings.filterwarnings('ignore')


class EnvSeq:
    def __init__(self, batch_of_seats, n_positions, threshold):
        """
        - sequence_matrix: The matrix to be optimized => 2d matrix of shape (size of seats batch, n_positions)
        - threshold (that discretize high to low complexity)
        """
        self.seat_sequence = {}
        self.delta = None

        self.size_seat_batch = batch_of_seats
        self.n_positions = n_positions
        self.threshold = threshold
        self.scaled_state_2d = None
        self.n_success_moves = 0.0

    def reset(self):
        # initial_state = np.random.randint(low=0, high=101, size=(self.size_seat_batch, self.n_positions))
        initial_state = np.array([
            [10, 15, 20, 21, 40, 18, 12, 16, 18],
            [10, 15, 20, 21, 40, 18, 12, 16, 18],
            [20, 30, 15, 13, 18, 21, 23, 33, 37],
            [20, 30, 15, 13, 18, 21, 23, 33, 37],
            [20, 30, 15, 13, 18, 21, 23, 33, 37],
            # [10, 15, 20, 21, 40, 18, 12, 16, 18],
            # [10, 15, 20, 21, 40, 18, 12, 16, 18],
            # [10, 15, 20, 21, 40, 18, 12, 16, 18],
            # [20, 30, 15, 13, 18, 21, 23, 33, 37],
            # [20, 30, 15, 13, 18, 21, 23, 33, 37],
        ])
        # np.random.shuffle(initial_state)
        self.scaled_state_2d = self.scale_state(initial_state)
        return self.get_state()

    def step(self, action, state, t_step, max_iter):
        """
        Apply action on environment
        """

        is_terminal = False
        state_2d = state.reshape(self.size_seat_batch, -1)

        idx, direction, by = action
        idx, direction, by = round(idx), round(direction), round(by)

        if idx == 0 and direction == -1:
            is_terminal = True
            reward = -1.0
            next_state = state  # state remain unchanged
            self.scaled_state_2d = state_2d
            return next_state, reward, is_terminal, ""

        state_2d_copy = np.copy(state_2d)
        try:
            row_to_move = state_2d[idx]
            to_idx = idx + direction * by

            state_2d = np.delete(state_2d, idx, axis=0)
            next_state_2d = np.insert(state_2d, to_idx, row_to_move, axis=0)

            if t_step == max_iter - 1:
                reward = self.get_reward(next_state_2d)
            else:
                reward = 0.0

            self.scaled_state_2d = next_state_2d
            next_state = self.get_state()

            # print(f"Moving row from position {idx} to position {to_idx}...")
            # self.sequence_matrix = self.unscale_state(next_state_2d)

        except IndexError:
            # if the agent is trying to move the sequence out of boundaries => terminal state with no reward
            is_terminal = True
            reward = -1.0
            next_state = state  # state remain the same
            self.scaled_state_2d = state_2d_copy

        return next_state, reward, is_terminal, ""

    def get_state(self):
        state = self.scaled_state_2d.flatten()
        return state

    @staticmethod
    def get_reward(state_2d):
        """
        Compute the difference between 2 adjacents rows.
        The higher the sum of diffs per columns, the better the sequence is for that position
        The higher the sum of totak reward, the better the overall sequence is,
        """
        diff_adj = np.diff(state_2d, axis=0)
        diffs = np.abs(diff_adj)
        reward = np.sum(diffs, axis=0)
        return np.sum(reward)

    def scale_state(self, state):
        """
        Scale the state
        """
        max_complexity = np.max(state, axis=None)
        self.delta = max_complexity - self.threshold

        # add delta to all high complexity values. This way over complexify already complex workcontent
        # This will allow the algorithm to have a preference for lower complexity to by adjacent to higher ones

        activation = lambda elt, thresh, delta: elt if elt < thresh else elt + delta
        scaled_state = np.array(
            [activation(i, self.threshold, self.delta) for row in state for i in row]
        ).reshape(self.size_seat_batch, -1)

        return scaled_state

    def unscale_state(self, scaled_state_2d):
        """
        Useful to print the sequence
        """
        activation = lambda elt, thresh, delta: elt if elt < thresh else elt - delta
        unscaled_state = np.array(
            [activation(i, self.threshold, self.delta) for row in scaled_state_2d for i in row]
        ).reshape(self.size_seat_batch, -1)

        return unscaled_state

    @property
    def action_bounds(self):
        low_idx, up_idx = 0, self.size_seat_batch - 1  # : for idx selection
        low_dir, up_dir = -1, 1  # : for direction
        low_by, up_by = 1, self.size_seat_batch - 1  # : for "by"

        action_bounds = np.array([low_idx, low_dir, low_by]), np.array([up_idx, up_dir, up_by])
        return action_bounds


if __name__ == "__main__":
    pass
