import functools
import random
import string
from time import sleep

import chainerrl
import cv2
import gym
import numpy as np
import collections

from chainerrl.wrappers import atari_wrappers
from chainerrl.wrappers.atari_wrappers import LazyFrames
from gym import ActionWrapper
# noinspection PyUnresolvedReferences
from gym_car_intersect.envs import CarRacingHackatonContinuous
from gym import spaces


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class DiscreteWrapper(ActionWrapper):
    def reverse_action(self, action):
        raise NotImplemented

    def __init__(self, env):
        super().__init__(env)
        self.action_space.n = 3
        self.action_space = gym.spaces.discrete.Discrete(self.action_space.n)

    def action(self, action):
        steer = 0.6
        speed = 0.3
        if action == 0:
            return [0, 0, 0]
        if action == 1:
            return [-steer, speed, 0]
        if action == 2:
            return [+steer, speed, 0]
        if action == 3:
            return [0.0, speed, 0]
        if action == 4:
            return [0.0, 0.0, 1.0]
        raise KeyError


class CompressWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return cv2.resize(observation[0], dsize=(84, 84), interpolation=cv2.INTER_CUBIC), observation[1]


class SkRewardWrapper(gym.RewardWrapper):

    def reward(self, reward):
        if reward < 0.0:
            return -1.0
        else:
            return reward


def random_string(len_=10):
    letters = 'rewiutopafsdghjklzcxvbnm123565479'
    return ''.join(random.choice(letters) for _ in range(len_))


class SaveWrapper(gym.ActionWrapper, ):

    def __init__(self, env, random_suffix=None):
        super().__init__(env)
        self.recording = []
        self.number = 0
        self.total_reward = float("-inf")
        self.best_total_reward = float("-inf")
        if random_suffix is None:
            self.random_suffix = random_string(len_=5)
        else:
            self.random_suffix = random_suffix

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        self.recording.append(obs)
        # draw borders
        # self.env.training_status()
        return obs, reward, done, _

    def reset(self):

        obs = self.env.reset()

        self.number += 1  #
        if self.total_reward > self.best_total_reward and self.total_reward > 170:
            self.best_total_reward = self.total_reward
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter("train/" + str(self.number).zfill(7) + "_" + str(
                self.total_reward) + "_" + 'recording' + self.random_suffix + '.mp4',
                                  fourcc, 20.0,
                                  (obs.shape[1], obs.shape[0]))
            for image in self.recording:
                out.write(image)
            out.release()
            cv2.destroyAllWindows()
        self.recording = []
        self.total_reward = 0
        return obs


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.

        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        # assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, info = self.env.step(self.noop_action)
            if done or info.get('needs_reset', False):
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, stack_size):
        super().__init__(env)
        self.stack_size = stack_size
        self.queue = collections.deque(maxlen=self.stack_size)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.stack_size):
            self.queue.append(obs)
        obs = cv2.hconcat(list(self.queue))
        return obs

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.queue.append(obs)
        obs = cv2.hconcat(list(self.queue))
        return obs, reward, done, _


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, channel_order='chw'):
        """Warp frames to 84x84 as done in the Nature paper and later work.

        To use this wrapper, OpenCV-Python is required.
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        shape = {
            'hwc': (self.height, self.width, 3),
            'chw': (3, self.height, self.width),
        }

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=shape[channel_order], dtype=np.uint8)

        # print("inner", self.observation_space.shape)

    def observation(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(np.float32(frame) / 255, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        # print("frame_shape", frame.shape)
        # print("obs_shape", self.observation_space.low.shape)
        return frame


def make_car_env_discrete(env_name, max_frames=30 * 30, env_seed=42):
    print(f'env name: {env_name}')
    env = gym.make(env_name)
    env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=max_frames)
    env = MaxAndSkipEnv(env, skip=4)
    env = DiscreteWrapper(env)
    # env = SaveWrapper(env, random_suffix=random_suffix)
    env = WarpFrame(env)
    env.seed(env_seed)
    return env


def make_bach_env(env_name, max_frames=30 * 30, env_seed=42):
    pass


def main():
    env = make_car_env_discrete()
    state = env.reset()
    print(state.shape)


if __name__ == '__main__':
    main()
