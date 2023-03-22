import warnings

import numpy as np

warnings.filterwarnings('ignore')


def reward_of_optimized_sequence(wc, takt_time, buffer_percent):
    seat_sequence = np.copy(wc.values)

    # Get the reward of the optimized sequence
    delay_func = lambda x: 0 if x < 0 else x
    stoppage_func = lambda delay, max_delay: 0 if delay < max_delay else delay - max_delay

    N_POSITIONS = len(wc.columns)
    delay_per_position = np.zeros(N_POSITIONS)
    stoppage_per_position = np.zeros(N_POSITIONS)
    max_delay_pp = takt_time * buffer_percent - takt_time

    for n, row in enumerate(seat_sequence):
        delay_tmp = row - takt_time + delay_per_position
        delay_per_position = np.array([delay_func(i) for i in delay_tmp])
        stoppage = np.array([stoppage_func(i, max_delay_pp[n]) for n, i in enumerate(delay_per_position)])
        stoppage_per_position += np.copy(stoppage)

    sequence_stoppage = np.sum(stoppage_per_position)
    r = 1 - sequence_stoppage
    return r


class EnvSeqV2:
    def __init__(self, workcontent, takt_time, buffer_percent):
        """
        :param workcontent: Pandas dataframe where columns are positions dans values represent the workcontent
        :param takt_time: 1D array representing the time required per position
        :param buffer_percent: 1D array representing the buffer length of each position
        """
        self.wc = workcontent
        self.N_POSITIONS = len(workcontent.columns)
        self.N_SEATS = len(workcontent)
        self.TAKT_TIME_PP = takt_time
        self.BUFFER_LENGTH_PP_PERCENT = buffer_percent
        assert len(self.BUFFER_LENGTH_PP_PERCENT) == self.N_POSITIONS

        self.MAX_DELAY_PP = self.TAKT_TIME_PP * self.BUFFER_LENGTH_PP_PERCENT - self.TAKT_TIME_PP

        self.seat_sequence = None
        self.delay_matrix = None
        self.stoppage_matrix = None
        self.stoppage_per_position = None
        self.delay_per_position = None


    def reset(self, shuffle=False):
        self.seat_sequence = np.copy(self.wc.values)
        if shuffle:
            np.random.shuffle(self.seat_sequence)

        self.delay_matrix = np.zeros((self.N_SEATS, self.N_POSITIONS))
        self.stoppage_matrix = np.zeros((self.N_SEATS, self.N_POSITIONS))
        self.stoppage_per_position = np.zeros(self.N_POSITIONS)
        self.delay_per_position = np.zeros(self.N_POSITIONS)

        state = self.build_state()
        return state

    def step(self, action, t_step):
        # is_terminal = t_step == self.N_SEATS - 1
        is_terminal = t_step == self.N_SEATS * self.N_SEATS

        row_id, direction = self.get_movement_from_action(action)
        self.update_sequence(row_id, direction)
        self.update_delay()
        next_state = self.build_state()
        reward = self.reward()

        return next_state, reward, is_terminal, (row_id, direction)

    def get_movement_from_action(self, action):
        if action < self.N_SEATS:
            direction = -1  # UP
        elif self.N_SEATS <= action < 2 * self.N_SEATS:
            direction = 1  # DOWN
            action = action - self.N_SEATS
        else:
            direction = 0  # HOLD
            action = action - 2 * self.N_SEATS

        return action, direction

    def update_sequence(self, row_id, direction):
        """
        :param row_id: Row id to be moved
        :param direction: -1, 1, 0 for (up, down, hold)
        """
        destination_id = (row_id + direction) % len(self.seat_sequence)
        row = self.seat_sequence[row_id]
        self.seat_sequence = np.delete(self.seat_sequence, row_id, axis=0)
        self.seat_sequence = np.insert(self.seat_sequence, destination_id, row, axis=0)

    def update_delay(self):
        """
        Update the delay and stoppage matrices (for the new state) and the total stoppage (for the reward)
        """
        delay_func = lambda x: 0 if x < 0 else x
        stoppage_func = lambda delay, max_delay: 0 if delay < max_delay else delay - max_delay

        self.delay_matrix = np.zeros((self.N_SEATS, self.N_POSITIONS))
        self.delay_per_position = np.zeros(self.N_POSITIONS)
        self.stoppage_matrix = np.zeros((self.N_SEATS, self.N_POSITIONS))
        self.stoppage_per_position = np.zeros(self.N_POSITIONS)

        for n, row in enumerate(self.seat_sequence):
            delay_tmp = row - self.TAKT_TIME_PP + self.delay_per_position
            self.delay_per_position = np.array([delay_func(i) for i in delay_tmp])

            # for the new state
            self.delay_matrix[n] = np.copy(self.delay_per_position)
            stoppage = np.array([stoppage_func(i, self.MAX_DELAY_PP[n]) for n, i in enumerate(self.delay_per_position)])
            self.stoppage_matrix[n] = np.copy(stoppage)

            # for the reward
            self.stoppage_per_position += np.copy(stoppage)

    def reward(self):
        sequence_stoppage = np.sum(self.stoppage_per_position)
        return -sequence_stoppage + 1

    def build_state(self):
        sequence = np.copy(self.seat_sequence)
        delay = np.copy(self.delay_matrix)

        relative_sequence = sequence / np.max(sequence)
        relative_delay = delay / self.MAX_DELAY_PP.reshape(1, -1)  # how close the delay is to become a stoppage
        # stoppage = np.copy(self.stoppage_matrix)

        state = np.stack((relative_sequence, relative_delay), axis=0)
        return state

    @property
    def observation_space(self):
        state = self.reset()
        return len(state)

    @property
    def action_space(self):
        return self.N_SEATS + 2 * self.N_SEATS


if __name__ == "__main__":
    pass
