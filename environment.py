import warnings
import numpy as np

import torch.multiprocessing as mp


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
        self.n_high_adjacent = 0.0
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

        self.scaled_initial_seat_sequence = self._scale_sequence(self.initial_seat_sequence, self.THRESHOLD)
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
        self.n_high_adjacent = 0.0
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

        diff_adj = np.diff(sequence, axis=0)
        diffs = np.abs(diff_adj)
        reward = np.sum(diffs, axis=0)
        return np.sum(reward) * 1 / (self.n_high_adjacent + 1)

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
        thresh = np.array([self.THRESHOLD], dtype=int)

        state = np.concatenate((initial_scaled_seq, final_seq, action_hist, available_actions, remaining_moves, thresh))
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


    def _scale_sequence(self, seat_sequence, threshold):
        """
        Scale the state:
        - Add delta to all high complexity values. This way over complexity already complex work content, this will
        allow the algorithm to have a preference for lower complexity after a high one
        """
        max_complexity = np.max(seat_sequence, axis=None)
        delta = max_complexity - threshold

        # augment the distance between low WC and high WC
        activation = lambda elt, thresh, delta: 0 if elt < thresh else thresh
        scaled_seat_sequence = np.array(
            [activation(i, threshold, delta) for row in seat_sequence for i in row]
        ).reshape(self.batch_of_seats, self.n_positions)

        return scaled_seat_sequence


class MultiprocessEnv(object):

    def __init__(self, n_workers, seed):
        self.n_workers = n_workers
        self.seed = seed

        # In A2C there is one learner in the main process and several workers in the env.
        # So we need a way for the agent to send command from the main process (parent) to the
        # workers (children). We can achieve this using Pipe.
        self.pipes = [mp.Pipe() for _ in range(self.n_workers)]

        self.workers = []  # hold the workers, so we can use .join() later in .close()

        for worker_id in range(self.n_workers):
            w = mp.Process(target=self.work, args=(worker_id, self.pipes[worker_id][1]))
            self.workers.append(w)
            w.start()

    def work(self, worker_id, child_process):
        seed = self.seed + worker_id
        env = EnvSeq()

        # Execute the received command
        while True:
            cmd, kwargs = child_process.recv()
            if cmd == "reset":
                child_process.send(env.reset(**kwargs))
            elif cmd == "step":
                child_process.send(env.step(**kwargs))
            else:
                del env
                child_process.close()
                break

    def reset(self, shuffle=False, worker_id=None):
        """
        - If worker_id is not None: Send the reset message from the parent to the child.
        The child will receive the message in .work() then send the result of env.reset() to the
        parent here.
        - Otherwise, send the reset to all children and get + stack their results (states)
        """
        # since all workers will not finish at the same time, we need a way to reset one particular
        # worker when he is done, so he can start re-interacting with the environment
        if worker_id is not None:
            main_process, _ = self.pipes[worker_id]
            msg = (
                "reset", {"shuffle": shuffle}
            )
            self.send_msg(msg, worker_id)
            state = main_process.recv()  # [0]
            return state

        self.broadcast_msg(('reset', {}))
        return np.vstack([main_process.recv() for main_process, _ in self.pipes])

    def send_msg(self, msg, worker_id):
        main_process, _ = self.pipes[worker_id]
        main_process.send(msg)

    def step(self, states, actions, t_step):
        # batch of actions. each worker should have take an action, so len(actions) should be
        # equal to the number of workers.
        assert len(actions) == self.n_workers

        for worker_id in range(self.n_workers):
            # dictionary will be pass as kwargs, so argument will be key=value in the env.step()
            msg = (
                "step", {
                    "state": states[worker_id], "action": actions[worker_id], "t_step": t_step
                }
            )
            self.send_msg(msg, worker_id)

        results = []
        for worker_id in range(self.n_workers):
            main_process, _ = self.pipes[worker_id]
            state, reward, done = main_process.recv()

            results.append(
                (state, np.array(reward, dtype=float), np.array(done, dtype=float))
            )

        # return array of 2d arrays.
        # index 0 contains 2d arrays of states
        # index 1 contains 2d arrays of rewards ...
        return [np.vstack(block) for block in np.array(results).T]

    def close(self):
        self.broadcast_msg(("close", {}))
        [w.join() for w in self.workers]

    def broadcast_msg(self, msg):
        [main_process.send(msg) for main_process, _ in self.pipes]


if __name__ == "__main__":
    pass
