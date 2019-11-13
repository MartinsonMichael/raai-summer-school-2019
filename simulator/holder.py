import chainerrl
from sac_tf import SAC__Agent
from env_wrappers import *
from gym_car_intersect.envs import CarRacingHackatonContinuous2
import argparse

import matplotlib.pyplot as plt

import tensorflow as tf
from matplotlib import animation
from IPython.display import display, HTML
import numpy as np
import datetime
from chainerrl import replay_buffer

from multiprocessing import Process

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

    def __init__(self, name, batch_size=32, hidden_size=256, buffer_size=10 * 1000):
        self.batch_size = batch_size

        # for reward history
        self.update_steps_count = 0
        self.game_count = 0
        self.history = []
        self.name = name

        log_dir = 'logs/' + name
        self.log_summary_writer = tf.summary.create_file_writer(log_dir)

        # init replay buffer
        self.cur_write_index = 0
        self.buffer = replay_buffer.ReplayBuffer(capacity=buffer_size, num_steps=1)

        # init environment and agent
        env = CarRacingHackatonContinuous2(num_bots=0, start_file=None, is_discrete=True)
        env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=5000)
        env = MaxAndSkipEnv(env, skip=4)
        #         env = DiscreteWrapper(env)
        env = WarpFrame(env, channel_order='hwc')
        self.env = env

        self.agent = SAC__Agent(
            picture_shape=(84, 84, 3),
            extra_size=12,
            action_size=5,
            hidden_size=hidden_size
        )
        self.env_state = None
        self.reset_env()

    def log(self, test_game_rewards):
        with self.log_summary_writer.as_default():
            tf.summary.scalar(name='mean_reward', data=test_game_rewards.mean(), step=self.game_count)
            tf.summary.histogram(name='rewards', data=test_game_rewards, step=self.game_count)

            tf.summary.scalar(name='update_steps', data=self.update_steps_count, step=self.game_count)


    def reset_env(self, inc_counter=True):
        self.env_state = self.env.reset()
        if inc_counter:
            self.game_count += 1

    def insert_N_sample_to_replay_memory(self, N, temperature=0.5):
        for _ in range(N):

            if self.env_state is None:
                self.reset_env()

            action = self.agent.get_single_action(
                self.env_state,
                need_argmax=False,
                temperature=temperature,
            )
            new_state, reward, done, info = self.env.step(np.argmax(action))

            self.buffer.append(
                state=self.env_state,
                action=action,
                reward=[reward],
                is_state_terminal=done,
                next_state=new_state,
            )
            # # state
            # self.buffer[0][self.cur_write_index] = self.env_state
            # # action
            # self.buffer[1][self.cur_write_index] = action
            # # reward
            # self.buffer[2][self.cur_write_index] = np.array([reward])
            # # new state
            # self.buffer[3][self.cur_write_index] = new_state
            # # done flag
            # self.buffer[4][self.cur_write_index] =
            self.env_state = new_state

            # reset env if done
            if done or ('needs_reset' in info.keys() and info['needs_reset']):
                self.reset_env()

        self.buffer.stop_current_episode()

    def iterate_over_buffer(self, steps):
        for _ in range(steps):
            batch = self.buffer.sample(self.batch_size)
            batch_new = [
                [item[0]['state'] for item in batch],
                [item[0]['action'] for item in batch],
                [[item[0]['reward']] for item in batch],
                [item[0]['next_state'] for item in batch],
                [[1.0 if item[0]['is_state_terminal'] else 0.0] for item in batch],
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
            print(f'update step {index}')
            self.update_steps_count += 1
            self.agent.update_step(
                batch,
                temperature=temperature,
                gamma=gamma,
                v_exp_smooth_factor=v_exp_smooth_factor,
            )

    def iterate_over_test_game(self, max_steps=1000, return_true_frame=False):
        self.reset_env(inc_counter=True)
        for _ in range(max_steps):
            action = self.agent.get_single_action(
                self.env_state,
                need_argmax=False,
                temperature=1,
            )
            self.env_state, reward, done, info = self.env.step(np.argmax(action))

            if not return_true_frame:
                yield self.env_state, action, reward, done
            else:
                yield self.env.state, action, reward, done

            if done:
                break
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

    def visualize(self):
        ims = []
        for state, action, reward, done in self.iterate_over_test_game(max_steps=40 * 2500, return_true_frame=True):
            if done:
                break
            ims.append(state[0])
        return np.array(ims)


def main(args):
    print('start...')
    holder = Holder(
        name='test_4',
        batch_size=32,
        hidden_size=64,
        buffer_size=5 * 10 ** 4,
    )
    if args.load_folder is not None:
        print(f'load weights from {args.load_folder}')
        holder.agent.load(args.load_folder)
    print('created holder')

    ims = holder.visualize()
    Process(target=plot_sequence_images, args=(ims, False, True)).start()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    if args.video_only:
        return

    holder.insert_N_sample_to_replay_memory(1000)
    print('inserted first 1000 steps')

    print('start training...')
    for i in range(10 * 1000):
        print(f'step: {i}')
        gamma = 0.99
        temperature = 5

        holder.insert_N_sample_to_replay_memory(2000, temperature=temperature - 0.1)
        holder.update_agent(update_step_num=20, temperature=temperature, gamma=gamma)

        if i % 5 == 4:
            holder.get_test_game_mean_reward()

        # clear_output(wait=True)
        # ax.plot(holder.get_history()[:, 0], holder.get_history()[:, 1])
        # display(fig)
        # plt.pause(0.5)


        if i % 20 == 19:
            ims = holder.visualize()
            Process(target=plot_sequence_images, args=(ims, False, True)).start()
            holder.agent.save(f'./models_saves/{holder.name}_{i}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_folder', type=str, default=None, help='folder to preload weights')
    parser.add_argument('--video_only', type=bool, default=False, help='flag to just record animation from saved weights')
    # parser.add_argument("--bots_number", type=int, default=0, help="Number of bot cars in environment.")
    args = parser.parse_args()
    main(args)
