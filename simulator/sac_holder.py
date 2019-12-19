from typing import Dict, Any

import gym
import numpy as np
import tensorflow as tf
import argparse

from envs import SubprocVecEnv_tf2
from envs import Replay_Buffer
from envs.common_envs_utils.env_wrappers import *
from envs.gym_car_intersect_fixed.environment import CarRacingHackatonContinuousFixed

from sac import SAC_Agent_Torch_NoPic
from SAC_github import SAC_Discrete


class Holder:
    ENV_DONE_FLAGS = {'need_reset'}
    def __init__(
        self,
        name,
        learning_rate=3e-4,
        device='cpu',
        args=None,
    ):
        self.batch_size = args.batch_size
        self.env_num = args.num_env
        self.args = args

        # init environment and agent
        _make_env = None
        if args.env_type == 'new':
            def f():
                env = CarRacingHackatonContinuousFixed(settings_file_path=args.settings)
                env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=250)
                env = MaxAndSkipEnv(env, skip=4)
                # env = DiscreteWrapper(env)
                env = DiscreteOnlyLRWrapper(env)
                env = WarpFrame(env, channel_order='chw')
                return env
            _make_env = f
        if args.env_type == 'lun':
            def f():
                env = gym.make('LunarLander-v2')
                env = RewardClipperWrapper(env)
                return env
            _make_env = f
        if args.env_type == 'hopper':
            def f():
                env = gym.make('Hopper-v2')
                env = MaxAndSkipEnv(env, skip=4)
                return env
            _make_env = f

        self.single_test_env = _make_env()

        # if args.agent_type == 'torch-nopic':
        #     self.agent = SAC_Agent_Torch_NoPic(
        #         state_size=self.single_test_env.observation_space.shape[0],
        #         action_size=self.single_test_env.action_space.n,
        #         hidden_size=args.hidden_size,
        #         start_lr=learning_rate,
        #         device=device,
        #     )

        self.agent = SAC_Discrete(
            state_size=self.single_test_env.observation_space.shape[0],
            action_size=self.single_test_env.action_space.n,
            hidden_size=args.hidden_size,
            device=device,
        )

        # for reward history
        self.episode_number = 0
        self.name = name
        self._stats = {}

        log_dir = 'logs/' + name
        self.log_summary_writer = tf.summary.create_file_writer(log_dir)

        # init replay buffer
        self.buffer = Replay_Buffer(buffer_size=args.buffer_size, batch_size=args.batch_size, seed=42, device='cpu')

        self.env = SubprocVecEnv_tf2([_make_env for _ in range(self.env_num)])
        self.env_test = SubprocVecEnv_tf2([_make_env for _ in range(10)])

    def publish_log(self):
        with self.log_summary_writer.as_default():
            for loss_name, loss_values in self._stats.items():
                tf.summary.scalar(
                    name=f'{loss_name}',
                    data=np.array(loss_values).mean(),
                    step=self.episode_number,
                )
            del self._stats
            self._stats = {}

    def store_logs(self, log_dict: Dict[str, Any]):
        for name, value in log_dict.items():
            if name not in self._stats.keys():
                self._stats[name] = []
            self._stats[name].append(value)

    def run_episode(self, is_eval: bool):
        state_batch = self.env.reset()
        done_flags = np.zeros(self.env_num, dtype=np.float32)
        total_rewards = np.zeros(self.env_num, dtype=np.float32)
        if not is_eval:
            self.episode_number += 1

        while done_flags.sum() < self.env_num:
            print('step')
            action_batch = self.agent.batch_action(state_batch, need_argmax=True)
            print(f'action: {action_batch}')
            next_state_batch, reward_batch, done_batch, info_batch = self.env.step(action_batch)

            total_rewards += reward_batch.reshape((self.env_num, )) * done_flags
            done_flags *= done_batch.reshape((self.env_num, ))

            if not is_eval:
                self.buffer.add_batch_experience(
                    state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    done_batch,
                )
                if self.buffer.size() >= self.args.start_buffer_size:
                    for _ in range(2):
                        q1_loss, q2_loss, v_loss, policy_loss, temperature = self.agent.update_step(self.buffer.sample())
                        self.store_logs({
                            'q1 loss': q1_loss,
                            'q2 loss': q2_loss,
                            'v loss': v_loss,
                            'policy loss': policy_loss,
                            'temperature': temperature,
                        })

            state_batch = next_state_batch

            for index, (done, info) in enumerate(zip(done_batch, info_batch)):
                if done or ('need_reset' in info.keys() and info['need_reset']):
                    done_flags[index] = 1.0
                    state_batch[index] = self.env.force_reset([index])[0]

        self.store_logs({'reward': total_rewards.mean()})
        return total_rewards.mean()

    def train(self):
        for step_index in range(10000):
            print(f'step : {step_index}')
            mean_reward = self.run_episode(is_eval=False)
            print(f'mean reward : {mean_reward}')
            if step_index % 10 == 0:
                self.publish_log()


def main(args):
    print('start...')
    print('creat holder...')
    holder = Holder(
        name=args.name,
        learning_rate=3e-4,
        args=args,
    )

    print('start training...')
    holder.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_folder', type=str, default=None, help='folder to preload weights')
    parser.add_argument('--video_only', type=bool, default=False,
                        help='flag to just record animation from saved weights')
    parser.add_argument('--name', type=str, default='test_5', help='name for saves')
    parser.add_argument('--num-env', type=int, default=32, help='env num to train process')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--hidden-size', type=int, default=256, help='hidden size')
    parser.add_argument('--buffer-size', type=int, default=3 * 10**5, help='buffer size')
    parser.add_argument('--start-buffer-size', type=int, default=5 * 10**5, help='initial size of replay buffer')
    parser.add_argument('--agent', type=str, default='torch-nopic', help="'V' or 'noV' ot 'torch', two agents to use")
    parser.add_argument('--device', type=str, default='cpu', help='use animation records')
    parser.add_argument('--eval', action='store_true', default=False, help='do not eval runs')
    parser.add_argument('--env-type', type=str, default='lun', help='old or new')
    parser.add_argument(
        '--settings',
        type=str,
        default='./envs/gym_car_intersect_fixed/settings_v2.json',
        help='path to reward settings'
    )

    args = parser.parse_args()
    main(args)
