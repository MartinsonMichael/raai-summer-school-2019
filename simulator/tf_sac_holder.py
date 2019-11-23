import chainerrl
from sac_tf import SAC__Agent
from envs.common_envs_utils.env_wrappers import *
from envs.gym_car_intersect import CarRacingHackatonContinuous2
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

    def __init__(self, name, env_num=32, batch_size=32, hidden_size=256, buffer_size=10 * 1000):
        self.batch_size = batch_size
        self.env_num = env_num

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

        # init environment and agent
        def _make_env():
            env = CarRacingHackatonContinuous2(num_bots=0, start_file=None)
            env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=500)
            env = MaxAndSkipEnv(env, skip=4)
            env = DiscreteWrapper(env)
            env = WarpFrame(env, channel_order='hwc')
            return env
        self.env = SubprocVecEnv_tf2([_make_env for _ in range(self.env_num)])
        self.env_test = _make_env()

        self.agent = SAC__Agent(
            picture_shape=(84, 84, 3),
            extra_size=12,
            action_size=5,
            hidden_size=hidden_size,
        )
        self.env_state = self.env.reset()
        self._dones = [False for _ in range(self.env_num)]

    def log(self, test_game_rewards):
        with self.log_summary_writer.as_default():
            tf.summary.scalar(name='mean_reward', data=test_game_rewards.mean(), step=self.update_steps_count)
            # tf.summary.histogram(name='rewards', data=test_game_rewards, step=self.update_steps_count)
            # tf.summary.scalar(name='update_steps', data=self.update_steps_count, step=self.update_steps_count)
            for loss_name, loss_values in self._losses.items():
                tf.summary.scalar(
                    name=f'loss {loss_name}',
                    data=np.array(loss_values).mean(),
                    step=self.update_steps_count
                )
                loss_values = []

    def insert_N_sample_to_replay_memory(self, N, temperature=0.5):
        for _ in range(N // self.env_num):

            action = self.agent.get_batch_actions(
                self.env_state,
                need_argmax=False,
                temperature=temperature,
            )
            new_state, reward, done, info = self.env.step(np.argmax(action, axis=1))

            for s, a, r, d, ns, was_prev_done in zip(self.env_state, action, reward, done, new_state, self._dones):
                if was_prev_done:
                    continue
                self.buffer.append(
                    state=(s * 255).astype(np.uint8),
                    action=a,
                    reward=r,
                    is_state_terminal=d,
                    next_state=(ns * 255).astype(np.uint8),
                )
            self._dones = done
            self.env_state = new_state

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
            print(batch_new[-1])
            print(batch_new[0][0].shape)
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
            print(f'update step {index}')
            self.update_steps_count += 1
            loss_q1, loss_q2, loss_v, loss_policy = self.agent.update_step(
                batch,
                temperature=temperature,
                gamma=gamma,
                v_exp_smooth_factor=v_exp_smooth_factor,
            )
            self._losses['q1'].append(loss_q1)
            self._losses['q2'].append(loss_q2)
            self._losses['v'].append(loss_v)
            self._losses['policy'].append(loss_policy)

    def iterate_over_test_game(self, max_steps=50 * 1000, return_true_frame=False, temperature=1.0):
        state = self.env_test.reset()
        for _ in range(max_steps):
            action = self.agent.get_single_action(
                state,
                need_argmax=False,
                temperature=temperature,
            )
            state, reward, done, info = self.env_test.step(np.argmax(action))

            if not return_true_frame:
                yield state, action, reward, False
            else:
                yield self.env_test.state, action, reward, False

            if done:
                print('\ntest_game_done\n')
                return None, None, None, True
        print('\ntest_game_iteration_limit\n')
        return None, None, None, True

    def get_test_game_reward(
            self,
            max_steps=1000,
    ):
        total_reward = 0
        for _, _, reward, done in self.iterate_over_test_game(max_steps=max_steps):
            if done:
                break
            total_reward += reward
        return total_reward

    def get_test_game_mean_reward(
            self,
            n_games=10,
            max_steps=1000,
    ):
        sm = []
        for _ in range(n_games):
            sm.append(self.get_test_game_reward(max_steps))
        sm = np.array(sm)

        self.log(sm)

    def visualize(self, temperature=1.0):
        ims = []
        for state, action, reward, done in self.iterate_over_test_game(
                max_steps=1000,
                return_true_frame=True,
                temperature=temperature
        ):
            if done:
                break
            ims.append(state)
        return np.array(ims)

    def save(self, folder, need_dump_replay_buffer):
        import os
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
        name=args.name,
        env_num=args.env_num,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        buffer_size=args.buffer_size,
    )
    if args.holder_update_steps_num is not None:
        print(f'set update step num to {args.holder_update_steps_num}')
        holder.update_steps_count = args.holder_update_steps_num

    if args.load_folder is not None:
        print(f'load holder and agent from {args.load_folder}')
        holder.load(args.load_folder)

    print('launch test visualization')
    ims = holder.visualize()
    Process(target=plot_sequence_images, args=(ims, False, True)).start()

    if args.video_only:
        print('exit cause of flag \'video_only = True\'')
        return

    print(f'init replay buffer with first {args.start_buffer_size} elements')
    holder.insert_N_sample_to_replay_memory(args.start_buffer_size, temperature=50)

    print('start training...')
    for i in range(args.start_step, args.num_steps):
        print(f'step: {i}')
        gamma = 0.99
        temperature = 50 / (i + 1)**0.4
        temperature = float(np.clip(temperature, 0.2, 50.0))

        holder.insert_N_sample_to_replay_memory(10**3, temperature=temperature)
        holder.update_agent(update_step_num=10, temperature=temperature, gamma=gamma)

        if i % 10 == 9:
            holder.get_test_game_mean_reward()

        if i % 20 == 19:
            ims = holder.visualize()
            Process(target=plot_sequence_images, args=(ims, False, True)).start()
            holder.save(f'./models_saves/{holder.name}_{i}', need_dump_replay_buffer=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_folder', type=str, default=None, help='folder to preload weights')
    parser.add_argument('--video_only', type=bool, default=False,
                        help='flag to just record animation from saved weights')
    parser.add_argument('--start_step', type=int, default=0, help='start step')
    parser.add_argument('--name', type=str, default='test_5', help='name for saves')
    parser.add_argument('--env_num', type=int, default=8, help='env num to train process')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--buffer_size', type=int, default=15 * 10**4, help='batch size')
    parser.add_argument('--num_steps', type=int, default=10**4, help='number of steps')
    parser.add_argument('--holder_update_steps_num', type=int, default=None, help='set the number of update steps')
    parser.add_argument('--start_buffer_size', type=int, default=15 * 10**4, help='initial size of replay buffer')

    # parser.add_argument("--bots_number", type=int, default=0, help="Number of bot cars in environment.")
    args = parser.parse_args()
    main(args)
