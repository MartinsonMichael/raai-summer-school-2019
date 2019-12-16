import chainerrl
from sac import SAC__Agent
from sac import SAC__Agent_noV
from sac import SAC_Agent_Torch
from envs.common_envs_utils.env_wrappers import *
from envs.gym_car_intersect import CarRacingHackatonContinuous2
from envs.gym_car_intersect_fixed.environment import CarRacingHackatonContinuousFixed
import argparse

import matplotlib.pyplot as plt

import tensorflow as tf
from matplotlib import animation
from IPython.display import display, HTML
import numpy as np
import datetime
from chainerrl import replay_buffer

from multiprocessing import Process
import pickle

from envs import SubprocVecEnv_tf2

plt.rcParams['animation.ffmpeg_path'] = u'/home/mmartinson/.conda/envs/mmd_default/bin/ffmpeg'


replay_buffer.ReplayBuffer()


def plot_sequence_images(image_array, need_disaply=True, need_save=True):
    ''' Display images sequence as an animation in jupyter notebook

    Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
    '''
    dpi = 72.0
    xpixels, ypixels = image_array[0].shape[:2]
    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(image_array),
        interval=33,
        repeat_delay=1,
        repeat=True
    )
    if need_save:
        import os
        if not os.path.exists('save_animation_folder'):
            os.makedirs('save_animation_folder')
        anim.save(f'./save_animation_folder/{datetime.datetime.now()}.mp4')
    if need_disaply:
        display(HTML(anim.to_html5_video()))


class Holder:
    '''
    Class to hold agent, environment and replay buffer.
    Also it is a place to controll hyperparameters of learning process.
    '''

    def __init__(self,
                 agent_type,
                 name,
                 env_num=32,
                 batch_size=32,
                 hidden_size=256,
                 buffer_size=10 * 1000,
                 learning_rate=3e-4,
                 device='cpu',
                 args=None,
                 ):
        self.agent_type = agent_type
        self.batch_size = batch_size
        self.env_num = env_num

        # init environment and agent
        _make_env = None
        if args.env_type == 'old':
            def f():
                env = CarRacingHackatonContinuous2(num_bots=0, start_file=None)
                env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=500)
                env = MaxAndSkipEnv(env, skip=4)
                env = ExtendedDiscreteWrapper(env)
                env = WarpFrame(env, channel_order='chw')
                return env
            _make_env = f
        if args.env_type == 'new':
            def f():
                env = CarRacingHackatonContinuousFixed(
                    reward_settings_file_path=args.settings_path,
                    num_bots=0,
                )
                env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=250)
                env = MaxAndSkipEnv(env, skip=4)
                # env = ExtendedDiscreteWrapper(env)
                env = DiscreteOnlyLRWrapper(env)
                env = WarpFrame(env, channel_order='chw')
                return env
            _make_env = f

        self.single_test_env = _make_env()

        if self.agent_type == 'V':
            self.agent = SAC__Agent(
                picture_shape=(84, 84, 3),
                extra_size=12,
                action_size=5,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
            )
        if self.agent_type == 'noV':
            self.agent = SAC__Agent_noV(
                picture_shape=(84, 84, 3),
                extra_size=12,
                action_size=5,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
            )
        if self.agent_type == 'torch':
            self.agent = SAC_Agent_Torch(
                picture_shape=(3, 84, 84),
                action_size=self.single_test_env.action_space.n,
                hidden_size=hidden_size,
                start_lr=learning_rate,
                device=device,
            )

        # for reward history
        self.update_steps_count = 0
        self.history = []
        self.name = name
        self._losses = {
            'q1': [],
            'q2': [],
            'v': [],
            'policy': [],
        }

        log_dir = 'logs/' + name
        self.log_summary_writer = tf.summary.create_file_writer(log_dir)

        # init replay buffer
        self.cur_write_index = 0
        self.buffer = replay_buffer.ReplayBuffer(capacity=buffer_size, num_steps=1)

        self.env = SubprocVecEnv_tf2([_make_env for _ in range(self.env_num)])
        self.env_test = SubprocVecEnv_tf2([_make_env for _ in range(10)])

        self.env_state = self.env.reset()
        self._dones = [False for _ in range(self.env_num)]

    def log(self, test_game_mean_rewards, goal_achieve):
        with self.log_summary_writer.as_default():
            tf.summary.scalar(name='goal_achieve', data=goal_achieve, step=self.update_steps_count)
            tf.summary.scalar(name='mean_reward', data=test_game_mean_rewards, step=self.update_steps_count)
            # tf.summary.histogram(name='rewards', data=test_game_rewards, step=self.update_steps_count)
            # tf.summary.scalar(name='update_steps', data=self.update_steps_count, step=self.update_steps_count)
            for loss_name, loss_values in self._losses.items():
                tf.summary.scalar(
                    name=f'loss {loss_name}',
                    data=np.array(loss_values).mean(),
                    step=self.update_steps_count
                )
            self._losses = {
                'q1': [],
                'q2': [],
                'v': [],
                'policy': [],
            }

    def insert_N_sample_to_replay_memory(self, N, temperature=0.5):
        for i in range(N // self.env_num):

            action = self.agent.get_batch_actions(
                self.env_state,
                need_argmax=False,
                temperature=temperature,
            )
            new_state, reward, done, info = self.env.step(np.argmax(action, axis=1))
            # action = np.zeros((128, 5))
            # new_state = np.ones((128, 84, 84, 3), dtype=np.float32)
            # reward = np.ones((128, 1), dtype=np.float32)
            # done = np.zeros((128, 1), dtype=np.float32)
            # info = [{} for _ in range(128)]

            for s, a, r, d, ns, was_prev_done in zip(self.env_state, action, reward, done, new_state, self._dones):
                if was_prev_done:
                    continue

                self.buffer.append(
                    state=np.clip(s * 255.0, 0.0, 255.0).astype(np.uint8),
                    action=a,
                    reward=r,
                    is_state_terminal=d,
                    next_state=np.clip(ns * 255.0, 0.0, 255.0).astype(np.uint8),
                )
            self._dones = done.copy()
            self.env_state = new_state

            ind_to_reset = []
            for index, one_info in enumerate(info):
                if 'needs_reset' in one_info.keys():
                    ind_to_reset.append(index)
            if len(ind_to_reset) != 0:
                obs = self.env.force_reset(ind_to_reset)
                self.env_state[np.array(ind_to_reset)] = obs

            if i % 100 == 99:
                print(f'replay buffer size : {self.buffer.__len__()}')

    def iterate_over_buffer(self, steps):
        for _ in range(steps):
            batch = self.buffer.sample(self.batch_size)
            batch_new = [
                [np.array(item[0]['state']).astype(np.float32) / 255.0 for item in batch],
                [np.array(item[0]['action']) for item in batch],
                [np.array([item[0]['reward']]).astype(np.float32) for item in batch],
                [np.array(item[0]['next_state']).astype(np.float32) / 255.0 for item in batch],
                [np.array([1.0 if item[0]['is_state_terminal'] else 0.0]) for item in batch],
            ]
            yield batch_new
            del batch_new

    def update_agent(
            self,
            update_step_num=500,
            temperature=0.5,
            gamma=0.7,
            v_exp_smooth_factor=0.995,
    ):
        for index, batch in enumerate(self.iterate_over_buffer(update_step_num)):
            self.update_steps_count += 1
            loss_q1, loss_q2, loss_v, loss_policy = -1, -1, -1, -1
            if self.agent_type in {'V', 'torch'}:
                loss_q1, loss_q2, loss_v, loss_policy = self.agent.update_step(
                    batch,
                    temperature=temperature,
                    gamma=gamma,
                    v_exp_smooth_factor=v_exp_smooth_factor,
                )
            if self.agent_type == 'noV':
                loss_q1, loss_q2, loss_policy = self.agent.update_step(
                    batch,
                    temperature=temperature,
                    gamma=gamma,
                    v_exp_smooth_factor=v_exp_smooth_factor,
                )

            print(f'loss Q1, Q2, V, Policy: {loss_q1} {loss_q2} {loss_v} {loss_policy}')

            self._losses['q1'].append(loss_q1)
            self._losses['q2'].append(loss_q2)
            self._losses['v'].append(loss_v)
            self._losses['policy'].append(loss_policy)

    def iterate_over_test_game(self, max_steps=1000, temperature=0.00001):
        state = self.env_test.reset()
        for _ in range(max_steps):
            action = self.agent.get_batch_actions(
                state,
                need_argmax=False,
                temperature=temperature,
            )
            state, reward, done, info = self.env_test.step(np.argmax(action, axis=1))

            yield state, action, reward, done, info

        return None, None, None, np.ones(10), {}

    def get_test_game_mean_reward(self):
        sm = np.zeros(10)
        goal_done = np.zeros(10)
        mask = np.ones(10)
        steps_count = np.ones(10)
        for state, action, reward, done, info in self.iterate_over_test_game(
                max_steps=1000,
                temperature=1.0,
        ):
            # print('*')
            # print(reward)
            # print(done)
            # print(mask)
            assert reward.shape == (10,)
            sm += reward * mask
            steps_count += mask
            for i in range(10):
                if mask[i] == 1:
                    if 'finish' in info[i].keys() and info[i]['finish']:
                        goal_done[i] = 1
                    if 'is_finish' in info[i].keys() and info[i]['is_finish']:
                        goal_done[i] = 1
                    if 'need_restart' in info[i].keys():
                        mask[i] = 0
            mask = mask * (1 - done)

            if mask.sum() == 0:
                break

        self.log(sm.mean(), goal_done.mean())
        return sm.mean(), goal_done.mean()

    def visualize(self):
        state = self.single_test_env.reset()
        ims = [self.single_test_env.state]
        for _ in range(1000):
            action = self.agent.get_single_action(state, need_argmax=True, temperature=1.0)
            state, reward, done, info = self.single_test_env.step(action)
            if done:
                break
            ims.append(self.single_test_env.state)
        return np.array(ims)

    def save(self, folder, need_dump_replay_buffer):
        import os
        folder = os.path.join(folder, f'{self.name}__{self.update_steps_count}')
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.agent.save(folder)
        pickle.dump(self.update_steps_count, open(os.path.join(folder, 'update_steps_count.pkl'), 'wb'))
        if need_dump_replay_buffer:
            self.buffer.save(os.path.join(folder, 'replay_buffer'))

    def load(self, folder):
        import os
        self.agent.load(folder)
        if os.path.exists(os.path.join(folder, 'update_steps_count.pkl')):
            self.update_steps_count = pickle.load(open(os.path.join(folder, 'update_steps_count.pkl'), 'rb'))
        if os.path.exists(os.path.join(folder, 'replay_buffer')):
            self.buffer.load(os.path.join(folder, 'replay_buffer'))


def main(args):
    print('start...')
    print('creat holder...')
    holder = Holder(
        agent_type=args.agent,
        name=args.name,
        env_num=args.env_num,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        buffer_size=args.buffer_size,
        learning_rate=3e-4,
        device=args.device,
        args=args,
    )
    if args.holder_update_steps_num is not None:
        print(f'set update step num to {args.holder_update_steps_num}')
        holder.update_steps_count = args.holder_update_steps_num

    if args.load_folder is not None:
        print(f'load holder and agent from {args.load_folder}')
        holder.load(args.load_folder)

    if not args.no_video:
        print('launch test visualization')
        ims = holder.visualize()
        Process(target=plot_sequence_images, args=(ims, False, True)).start()

    if args.video_only:
        print('exit cause of flag \'video_only = True\'')
        return

    if args.eval:
        NUM_EVALS = 10
        sm, sm_goal = 0, 0
        for i in range(NUM_EVALS):
            cur, goal = holder.get_test_game_mean_reward()
            print(f'step {i} / {NUM_EVALS}')
            print(f'mean reward by 10 runs : {cur}')
            print(f'mean goal achieve by 10: {goal}')
            sm += cur
            sm_goal += goal
        print('-----')
        print(f'Mean Reward : {sm / NUM_EVALS}')
        print(f'Mean goal   : {sm_goal / NUM_EVALS}')
        return

    print(f'init replay buffer with first {args.start_buffer_size} elements')
    holder.insert_N_sample_to_replay_memory(args.start_buffer_size, temperature=50)
    print(f'buffer finished')
    # holder.update_agent(update_step_num=2 * 10**3, temperature=2.0, gamma=0.5)

    print('start training...')
    for i in range(10000):
        # gamma = float(np.clip(0.99 - 200 / (200 + 3*i), 0.1, 0.99))
        gamma = 0.90
        temperature = (50 - (i + 1) ** 0.2) / (i + 1) ** 0.6
        # if i % 16 == 0:
        #     temperature = 20.0
        temperature = float(np.clip(temperature, 0.2, 50.0))

        print(f'step: {i}')
        print(f'temp: {temperature}')

        holder.insert_N_sample_to_replay_memory(300, temperature=temperature)
        holder.update_agent(update_step_num=10, temperature=temperature, gamma=gamma)
        # if i % 16 == 0:
        #     holder.update_agent(update_step_num=100, temperature=temperature, gamma=gamma)

        if i % 20 == 1 and not args.no_eval:
            holder.get_test_game_mean_reward()

        if i % 100 == 1 and not args.no_video:
            ims = holder.visualize()
            Process(target=plot_sequence_images, args=(ims, False, True)).start()

        if i % 500 == 0 and i > 200:
            holder.save(f'./models_saves/', need_dump_replay_buffer=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_folder', type=str, default=None, help='folder to preload weights')
    parser.add_argument('--video_only', type=bool, default=False,
                        help='flag to just record animation from saved weights')
    parser.add_argument('--start_step', type=int, default=0, help='start step')
    parser.add_argument('--name', type=str, default='test_5', help='name for saves')
    parser.add_argument('--env_num', type=int, default=8, help='env num to train process')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--buffer_size', type=int, default=10**5, help='buffer size')
    parser.add_argument('--num_steps', type=int, default=10**6, help='number of steps')
    parser.add_argument('--holder_update_steps_num', type=int, default=None, help='set the number of update steps')
    parser.add_argument('--start_buffer_size', type=int, default=10**5, help='initial size of replay buffer')
    parser.add_argument('--agent', type=str, default='V', help="'V' or 'noV' ot 'torch', two agents to use")
    parser.add_argument('--no-video', action='store_true', default=False, help='use animation records')
    parser.add_argument('--device', type=str, default='cpu', help='use animation records')
    parser.add_argument('--no-eval', action='store_true', default=False, help='do not eval runs')
    parser.add_argument('--eval', action='store_true', default=False, help='do not eval runs')
    parser.add_argument('--env-type', type=str, default='old', help='old or new')
    parser.add_argument(
        '--settings-path',
        type=str,
        default='./envs/gym_car_intersect_fixed/reward_settings_default.json',
        help='path to reward settings'
    )

    # parser.add_argument("--bots_number", type=int, default=0, help="Number of bot cars_full in environment.")
    args = parser.parse_args()

    if args.agent not in {'V', 'noV', 'torch'}:
        raise ValueError('agent set incorrectly')

    if args.env_type not in {'old', 'new'}:
        raise ValueError("set env type to 'new' or 'old'")

    main(args)
