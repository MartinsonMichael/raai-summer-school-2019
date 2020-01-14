import cv2
import gym
import numpy as np
import collections


class ExtendedMaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(ExtendedMaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        obs = None
        self._obs_buffer = []
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if obs['picture'] is not None:
                self._obs_buffer.append(obs['picture'])
            total_reward += reward
            if done:
                break
        if len(self._obs_buffer) != 0:
            max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        else:
            max_frame = None
        return {'picture': max_frame, 'vector': obs['vector']}, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ExtendedWarpFrame(gym.ObservationWrapper):
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
        self.shape = shape[channel_order]

    def observation(self, obs):
        frame = obs['picture']
        if frame is not None:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            if frame.shape != self.shape:
                frame = np.transpose(frame, (2, 1, 0))
            frame.astype(np.uint8)

        return {'picture': frame, 'vector': obs['vector']}
