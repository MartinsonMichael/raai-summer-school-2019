import numpy as np
from .utils import SAC__Agent
from ..gym_road_cars import CarRacing


class Holder:
    """
    Class to hold agent, environment and replay buffer.
    Also it is a place to controll hyperparameters of learning process.
    """

    def __init__(self, session, batch_size=32, hidden_size=256, buffer_size=30 * 1000):
        self.session = session
        self.batch_size = batch_size

        # for reward history
        self.update_steps_count = 0
        self.history = []

        # init replay buffer
        self.cur_write_index = 0
        self.buffer_size = buffer_size
        self.buffer = [
            np.zeros((buffer_size, 5), dtype=np.float32),
            np.zeros((buffer_size, 5), dtype=np.float32),
            np.zeros((buffer_size, 1), dtype=np.float32),
            np.zeros((buffer_size, 5), dtype=np.float32),
            np.zeros((buffer_size, 1), dtype=np.float32),
        ]

        # init environment and agent
        self.env = CarRacing(num_bots=0, start_file=None)
        self.agent = SAC__Agent(session, state_size=5, action_size=5, hidden_size=hidden_size)
        self.goal = None

    def reset_env(self):
        self.env.reset()
        goal = np.array(self.env.car_goal_poly)
        self.goal = np.array([np.mean(goal[:, 0]), np.mean(goal[:, 1])])

    def insert_N_sample_to_replay_memory(self, N, temperature=0.5):
        for _ in range(N):

            # resen env, if we hanen't goal
            if self.goal is None:
                self.reset_env()

            state = np.hstack([self.env.state[1], self.goal])
            action = self.agent.get_single_action(
                state,
                need_argmax=False,
                temperature=temperature,
            )
            new_state, reward, done, info = self.env.step(np.argmax(action))

            # state
            self.buffer[0][self.cur_write_index] = state
            # action
            self.buffer[1][self.cur_write_index] = action
            # reward
            self.buffer[2][self.cur_write_index] = np.array([reward])
            # new state
            self.buffer[3][self.cur_write_index] = np.hstack([np.array(new_state[1]), self.goal])
            # done flag
            self.buffer[4][self.cur_write_index] = 1.0 if done else 0.0
            self.cur_write_index += 1
            if self.cur_write_index >= self.buffer_size:
                self.cur_write_index = 0

            # reset env if done
            if done:
                self.reset_env()

    def iterate_over_buffer(self, steps):
        cur_steps = 0
        while True:
            indexes = np.arange(len(self.buffer[0]))
            np.random.shuffle(indexes)

            for ind in range(0, len(indexes), self.batch_size):
                yield (
                    self.buffer[i][indexes[ind: ind + self.batch_size]]
                    for i in range(5)
                )
                cur_steps += 1
                if cur_steps >= steps:
                    raise StopIteration

    def update_agent(
            self,
            update_step_num=500,
            temperature=0.5,
            gamma=0.7,
            v_exp_smooth_factor=0.8,
            need_update_VSmooth=False
    ):
        for batch in self.iterate_over_buffer(update_step_num):
            self.update_steps_count += 1
            self.agent.update_step(
                batch,
                temperature=temperature,
                gamma=gamma,
                v_exp_smooth_factor=v_exp_smooth_factor,
                need_update_VSmooth=need_update_VSmooth,
            )

    def get_test_game_total_revard(
            self,
            max_steps=1000,
            temperature=10,
            add_to_memory=True
    ):
        self.reset_env()
        total_revard = 0
        was_game_finit = False

        for _ in range(max_steps):
            # make action
            state = np.hstack([self.env.state[1], self.goal])
            action = self.agent.get_single_action(
                state,
                need_argmax=False,
                temperature=temperature,
            )
            new_state, reward, done, info = self.env.step(np.argmax(action))

            total_revard += reward

            if done:
                was_game_finit = True
                break

        self.reset_env()
        if not was_game_finit:
            total_revard = -9999

        if add_to_memory:
            self.history.append([self.update_steps_count, total_revard])

        return total_revard

    def get_history(self):
        return np.array(self.history)