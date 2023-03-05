import warnings
import numpy as np

warnings.filterwarnings('ignore')


class EnvSeq:
    def __init__(self):
        """
        Inputs:
        - batch_of_seats:
        - n_positions:

        Seat sequence representation: batch seat=3 and nPositions=4
        -------|P0|---|P1|---|P2|---|P3|-
        ---------------------------------
        seat 0 |23|---|10|---|15|---|11|-
        ---------------------------------
        seat 1 |22|---|40|---|39|---|11|-
        ---------------------------------
        seat 2 |34|---|18|---|16|---|32|-
        ---------------------------------

        Goal: Optimize the order of seats (rows) without changing the order or positions (columns)
        """
        self.n_high_adjacent = 0.0
        self.THRESHOLD = 19  # threshold (that discretize high to low complexity)
        self.batch_of_seats = None  # Number of seats in the batch
        self.n_positions = None  # Number of position per seats

        self.remaining_moves = None  # remaining moves to take by the agent
        self.available_actions = None  # actions available for the agent
        self.actions_history = None  # taken actions

        self.initial_seat_sequence = None
        self.final_seat_sequence = None
        self.scaled_initial_seat_sequence = None


    def reset(self, shuffle=False):
        self.initial_seat_sequence = np.array([
            [20, 30, 15, 13, 18, 21, 23, 33, 37],
            [10, 15, 20, 21, 40, 18, 12, 16, 18],
            [33, 46, 15, 12, 11, 45, 22, 21, 40],
            [11, 17, 30, 31, 30, 14, 14, 17, 13],
            [30, 40, 10, 12, 16, 31, 33, 23, 20],
            [20, 30, 15, 13, 18, 21, 23, 33, 37],
            [10, 15, 20, 21, 40, 18, 12, 16, 18],
            [33, 46, 15, 12, 11, 45, 22, 21, 40],
            [11, 17, 30, 31, 30, 14, 14, 17, 13],
            [30, 40, 10, 12, 16, 31, 33, 23, 20],
        ])
        if shuffle:
            np.random.shuffle(self.initial_seat_sequence)
        self.batch_of_seats, self.n_positions = self.initial_seat_sequence.shape

        self.scaled_initial_seat_sequence = self._scale_state(self.initial_seat_sequence, self.THRESHOLD)
        self.final_seat_sequence = np.zeros((self.batch_of_seats, self.n_positions), dtype=int)
        self.actions_history = np.ones(self.batch_of_seats, dtype=int) * -1
        self.available_actions = np.arange(self.batch_of_seats)
        self.remaining_moves = self.batch_of_seats

        state = self._build_state()
        return state


    def step(self, state, action, t_step):
        """
        Apply action on environment and build a new state (next state):
            1 - update the current row (t_step) of the final seq, with the chosen row (action) of the initial seq
            2 - replace the chosen row (action) of the initial seq by 0
            3 - update the history of taken actions
            4 - update the array of available actions by replacing the taken action by -1
            5 - decrease the number of remaining moves

        Reward is -100 for invalid behaviors, 0 for good behavior and R > 0 for reaching completing the task.
        (I can also try to give a reward only if an expected sequence is given but that would a too "supervised")
        """
        is_terminal = False
        reward = 0.0

        # action = round(action[0])

        if self._is_invalid_behavior(action):
            is_terminal = True
            reward = -100.0
            return state, reward, is_terminal

        self.final_seat_sequence[t_step] = self.scaled_initial_seat_sequence[action]
        self.scaled_initial_seat_sequence[action] = np.zeros(self.n_positions, dtype=int)
        self.actions_history[t_step] = action
        self.available_actions[action] = -1
        self.remaining_moves = self.remaining_moves - 1
        next_state = self._build_state()

        if self._is_task_completed():
            is_terminal = True
            reward = self.get_reward(self.final_seat_sequence)

        return next_state, reward, is_terminal


    def get_reward(self, sequence):
        """
        Compute the difference between 2 adjacent rows.
        The higher the sum of diffs per columns, the better the sequence is for that position
        The higher the sum of total reward, the better the overall sequence is,
        """
        self.n_high_adjacent = self.count_adjacent_high_complexity(sequence)
        penalty = -self.n_high_adjacent

        diff_adj = np.diff(sequence, axis=0)
        diffs = np.abs(diff_adj)
        reward = np.sum(diffs, axis=0)
        return np.sum(reward) + penalty * 100

    def count_adjacent_high_complexity(self, sequence):
        mask = sequence >= self.THRESHOLD
        # shift the mask one row down to find consecutive pairs
        shifted_mask = np.vstack((mask[1:], np.zeros_like(mask[-1])))
        # count the number of pairs where both values are True
        count = np.count_nonzero(mask & shifted_mask)
        return count


    @property
    def action_bounds(self):
        """
        Only one action dimension : choose the row from the initial sequence to be placed in the placeholder sequence.
        The action is the row id ranging from 0 to  batch_of_seats-1
        """
        _ = self.reset()  # to update batch_of_seats

        LOW, UP = 0, self.batch_of_seats - 1
        action_bounds = np.array([LOW]), np.array([UP])
        return action_bounds


    @property
    def observation_space(self):
        state = self.reset()
        return len(state)


    @property
    def action_space(self):
        _ = self.reset()    # to update batch_of_seats
        return self.batch_of_seats
        # return 1  # if continuous algo like td3


    def _build_state(self):
        initial_scaled_seq = self.scaled_initial_seat_sequence.flatten()
        final_seq = self.final_seat_sequence.flatten()
        action_hist = self.actions_history
        available_actions = self.available_actions
        remaining_moves = np.array([self.remaining_moves], dtype=int)

        state = np.concatenate((initial_scaled_seq, final_seq, action_hist, available_actions, remaining_moves))
        return state


    def _is_task_completed(self):
        """
        Task is completed if:
        - initial scaled sequence is full of zeros.
        """
        return np.all(self.scaled_initial_seat_sequence == 0)


    def _is_invalid_behavior(self, action):
        """
        Possible invalid behaviors:
            - action has already been taken
        """
        return int(action) in self.actions_history


    def _scale_state(self, seat_sequence, threshold):
        """
        Scale the state:
        - Add delta to all high complexity values. This way over complexity already complex work content, this will
        allow the algorithm to have a preference for lower complexity after a high one
        """
        max_complexity = np.max(seat_sequence, axis=None)
        delta = max_complexity - threshold

        activation = lambda elt, thresh, delta: elt if elt < thresh else elt + delta
        scaled_seat_sequence = np.array(
            [activation(i, threshold, delta) for row in seat_sequence for i in row]
        ).reshape(self.batch_of_seats, self.n_positions)

        return scaled_seat_sequence


if __name__ == "__main__":
    pass
