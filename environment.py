import warnings

import numpy as np


warnings.filterwarnings('ignore')


class EnvSeqV1:
    def __init__(self, workcontent, takt_time, buffer_percent):
        """
        Input:
            - workcontent: Pandas dataframe where columns are positions dans values represent the workcontent
            - takt_time:
            - buffer_percent:

        Goal: Optimize the order of seats (rows) without changing the order or positions (columns)
        """
        self.wc = workcontent
        self.N_POSITIONS = len(workcontent.columns)
        self.N_SEATS = len(workcontent)
        self.TAKT_TIME_PP = takt_time
        self.BUFFER_LENGTH_PP_PERCENT = buffer_percent
        assert len(self.BUFFER_LENGTH_PP_PERCENT) == self.N_POSITIONS

        self.MAX_DELAY_PP = self.TAKT_TIME_PP * self.BUFFER_LENGTH_PP_PERCENT - self.TAKT_TIME_PP

        self.initial_seat_sequence = None
        self.final_seat_sequence = None
        self.stoppage_per_position = None
        self.delay_per_position = None
        self.actions_history = None
        self.INF = 99999999


    def reset(self, shuffle=False):
        self.actions_history = []
        self.initial_seat_sequence = np.copy(self.wc.values)

        if shuffle:
            np.random.shuffle(self.initial_seat_sequence)

        self.final_seat_sequence = np.zeros((self.N_SEATS, self.N_POSITIONS))
        self.stoppage_per_position = np.zeros(self.N_POSITIONS)
        self.delay_per_position = np.zeros(self.N_POSITIONS)

        state = self.build_state()
        return state


    def step(self, state, action, t_step):

        if self.is_invalid_behavior(action):
            return state, -100, True

        self.actions_history.append(action)             # useful for filtering out already taken action from the q-value
        self.update_sequence(action, t_step)
        self.update_delay(t_step)
        self.update_stoppage()

        next_state = self.build_state()
        reward = self.reward()
        is_terminal = self.is_terminal()

        return next_state, reward, is_terminal


    def update_sequence(self, action, t_step):
        self.final_seat_sequence[t_step] = self.initial_seat_sequence[action]
        self.initial_seat_sequence[action] = np.ones(self.N_POSITIONS) * self.INF

    def update_delay(self, t_step):
        delay_tmp = self.final_seat_sequence[t_step] - self.TAKT_TIME_PP + self.delay_per_position
        delay_func = lambda x: 0 if x < 0 else x
        self.delay_per_position = np.array([delay_func(i) for i in delay_tmp])

    def update_stoppage(self):
        stoppage_duration_func = lambda tps, tps_max: 0 if tps < tps_max else tps - tps_max
        self.stoppage_per_position = np.array(
            [stoppage_duration_func(i, self.MAX_DELAY_PP[n]) for n, i in enumerate(self.delay_per_position)]
        )

    def reward(self):
        total_stoppages = np.sum(self.stoppage_per_position)
        return 1 / (total_stoppages + 1)


    def build_state(self):
        state = np.concatenate((
            self.initial_seat_sequence.flatten(),
            self.delay_per_position / self.MAX_DELAY_PP,
        ))
        return state


    def is_terminal(self):
        return np.all(self.initial_seat_sequence == self.INF)

    def is_invalid_behavior(self, action):
        # this can happen during exploration
        return int(action) in self.actions_history

    @property
    def observation_space(self):
        state = self.reset()
        return len(state)

    @property
    def action_space(self):
        return self.N_SEATS



if __name__ == "__main__":
    pass

