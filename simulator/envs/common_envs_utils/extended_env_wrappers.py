import cv2
import gym
import numpy as np
import collections

import torch


class OriginalStateKeeper(gym.ObservationWrapper):
    """save state"""
    def __init__(self, env, state_save_name='original_state'):
        super().__init__(env)
        self._state_save_name = state_save_name
        self.__setattr__(state_save_name, None)

    def observation(self, observation):
        self.__setattr__(self._state_save_name, observation)
        return observation


class TorchTensorCaster(gym.ObservationWrapper):
    def observation(self, obs):
        if isinstance(obs, dict):
            return {
                name: torch.from_numpy(value)
                if value is not None
                else None
                for name, value in obs.items()
            }
        if isinstance(obs, np.ndarray):
            torch.from_numpy(obs)
        return obs


class ImageWithVectorCombiner(gym.ObservationWrapper):
    """Take 'pov' value (current game display) and concatenate compass angle information with it, as a new channel of image;
    resulting image has RGB+compass (or K+compass for gray-scaled image) channels.
    """
    def __init__(self, env, image_dict_name='picture', vector_dict_name='vector', vector_pre_scale=255.0):
        super().__init__(env)
        self._image_name = image_dict_name
        self._vector_name = vector_dict_name
        self._vector_pre_scale = vector_pre_scale

        image_space = self.env.observation_space.spaces[self._image_name]
        vector_space = self.env.observation_space.spaces[self._vector_name]

        low = self.observation({self._image_name: image_space.low, self._vector_name: vector_space.low})
        high = self.observation({self._image_name: image_space.high, self._vector_name: vector_space.high})

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):

        # print(f"combiner input image shape : {observation['picture'].shape}")

        image = observation[self._image_name]
        vector = observation[self._vector_name] * self._vector_pre_scale
        vector_channel = np.ones(shape=list(image.shape[:-1]) + [len(vector)], dtype=np.float32) * vector
        res = np.concatenate([image.astype(np.float32), vector_channel], axis=-1)

        # print(f"combiner output image shape : {res.shape}")

        return res


class ChannelSwapper(gym.ObservationWrapper):

    def __init__(self, env, image_dict_name='picture'):
        gym.ObservationWrapper.__init__(self, env)
        self._image_dict_name = image_dict_name

        if isinstance(self.observation_space, gym.spaces.Dict):
            # print(f"swap dict axis from : {self.observation_space.spaces['picture'].shape}", end=' ')
            self.observation_space.spaces[self._image_dict_name] = gym.spaces.Box(
                low=ChannelSwapper._image_channel_transpose(
                    self.observation_space.spaces[self._image_dict_name].low,
                ),
                high=ChannelSwapper._image_channel_transpose(
                    self.observation_space.spaces[self._image_dict_name].high,
                ),
                dtype=self.observation_space.spaces[self._image_dict_name].dtype,
            )
            # print(f"to : {self.observation_space.spaces['picture'].shape}")
        elif isinstance(self.observation_space, gym.spaces.Box):
            # print(f"swap box axis from {self.observation_space.shape}", end=' ')
            self.observation_space = gym.spaces.Box(
                low=ChannelSwapper._image_channel_transpose(self.observation_space.low),
                high=ChannelSwapper._image_channel_transpose(self.observation_space.high),
                dtype=self.observation_space.dtype,
            )
            # print(f"to : {self.observation_space.shape}")

    @staticmethod
    def _image_channel_transpose(image):
        return np.transpose(image, (2, 1, 0))

    def observation(self, observation):
        if isinstance(observation, dict):
            # print(f"swapper input image shape : {observation['picture'].shape}")
            observation.update({
                self._image_dict_name:
                    ChannelSwapper._image_channel_transpose(observation[self._image_dict_name])
            })
            return observation
        # print(f"swapper input image shape : {observation.shape}")
        return ChannelSwapper._image_channel_transpose(observation)


class ExtendedMaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4, image_dict_name='picture'):
        """Return only every `skip`-th frame"""
        super(ExtendedMaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
        self._image_dict_name = image_dict_name

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        obs = None
        self._obs_buffer = []
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs[self._image_dict_name])
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        obs.update({self._image_dict_name: max_frame})
        return obs, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class FrameCompressor(gym.ObservationWrapper):
    def __init__(self, env, image_dict_name='picture'):
        """Warp frames to 84x84 as done in the Nature paper and later work.

        To use this wrapper, OpenCV-Python is required.
        """
        gym.ObservationWrapper.__init__(self, env)
        self._image_dict_name = image_dict_name
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.observation_space.spaces[image_dict_name] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(84, 84, 3),
                dtype=np.uint8,
            )
        else:
            raise ValueError('ExtendedWarpFrame should wrap dict observations')

    def observation(self, obs: dict):
        frame = obs[self._image_dict_name]

        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.uint8)
        obs.update({self._image_dict_name: frame})

        # print(f"compressor, output image shape : {obs['picture'].shape}")

        return obs
