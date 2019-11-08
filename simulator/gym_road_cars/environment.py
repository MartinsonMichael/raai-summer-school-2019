from typing import NamedTuple, Dict, List, Any, Union, Tuple
from shapely import geometry
import Box2D
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import numpy as np

import argparse

from . import reward_constants
from .data_utils import DataSupporter, CarImage
from .contactListener import ContactListener
from .car import DummyCar


class CarRacingSettings(NamedTuple):
    is_discrete: bool
    numbers_bots: int


class CarRacing(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': 30,
    }

    def __init__(self,):
        EzPickle.__init__(self)
        self.np_random = None
        self._settings: CarRacingSettings

        self._data_loader = DataSupporter(
            '/data/Hack/CDS_Lab/sac_branch/simulator/gym_road_cars/env_data_test/cars/',
            '/data/Hack/CDS_Lab/sac_branch/simulator/gym_road_cars/env_data_test/tracks/1_background_segmentation.xml',
            '/data/Hack/CDS_Lab/sac_branch/simulator/gym_road_cars/env_data_test/tracks/background_image.jpg',
        )
        self._b2world = Box2D.b2World(
            gravity=(0, 0),
            contactListener=ContactListener(self),
        )
        self._restricted_world: Dict[str, List[geometry.Polygon]]
        self._init_world()
        self._agent_car: DummyCar

        # self.action_space = spaces.Box(
        #     np.array([-1, -1, -1]),
        #     np.array([+1, +1, +1]),
        #     dtype=np.float32
        # )
        # self.observation_space = spaces.Box(low=low_val, high=high_val, dtype=np.float32)

    def _init_world(self):
        self._restricted_world = {
            'not_road': [],
            'road1': [],
            'road2': [],
            'road_cross': [],
        }
        for polygon in self._data_loader.data.get_polygons(0):
            polygon_name = polygon['label']
            polygon_points = polygon['points']
            if polygon_name in {'not_road', 'road1', 'road2', 'road_cross'}:
                self._restricted_world[polygon_name].append(geometry.Polygon(polygon_points))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        pass

    def reset(self):
        self._agent_car = DummyCar(
            world=self._b2world,
            restricted_world=self._restricted_world,
            track=self._data_loader.peek_track(),
            car_image=self._data_loader.peek_car_image(),
            bot=False,
        )

    def _restriction_reward(self) -> float:
        reward = 0.0
        for not_road_polygon in self._restricted_world['not_road']:
            # not_road_polygon: geometry.Polygon
            if not_road_polygon.hausdorff_distance(self._agent_car.get_wheels_positions()) == 0:
                reward += reward_constants.REWARD_OUT
                break

        for not_road_polygon in self._restricted_world['not_road']:
            # not_road_polygon: geometry.Polygon
            if not_road_polygon.hausdorff_distance(self._agent_car.get_wheels_positions()) == 0:
                reward += reward_constants.REWARD_OUT
                break

        return reward

    def step(self, action) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        pass

    def render(self, mode='human'):
        raise NotImplemented

    def close(self):
        raise NotImplemented

#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--bots_number", type=int, default=4, help="Number of bot cars in environment.")
#     parser.add_argument("--write", default=False, action="store_true", help="Whether write cars' coord to file.")
#     parser.add_argument("--dir", default='car_racing_positions.csv', help="Dir of csv file with car's coord.")
#     parser.add_argument("--no_agent", default=True, action="store_false", help="Wether show an agent or not")
#     parser.add_argument("--using_start_file", default=False, action="store_true",
#                         help="Wether start position is in file")
#     parser.add_argument("--training_epoch", type=int, default=0, help="Wether record end positons")
#     args = parser.parse_args()
#
#     from pyglet.window import key
#
#     a = np.array([0.0, 0.0, 0.0])
#
#
#     def key_press(k, mod):
#         global restart
#         if k == 0xff0d: restart = True
#         if k == key.LEFT:  a[0] = -1.0
#         if k == key.RIGHT: a[0] = +1.0
#         if k == key.UP:    a[1] = +1.0
#         if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
#
#
#     def key_release(k, mod):
#         if k == key.LEFT and a[0] == -1.0: a[0] = 0
#         if k == key.RIGHT and a[0] == +1.0: a[0] = 0
#         if k == key.UP:    a[1] = 0
#         if k == key.DOWN:  a[2] = 0
#
#
#     if args.using_start_file:
#         env = CarRacing(agent=args.no_agent, write=args.write, data_path=args.dir,
#                         start_file=args.using_start_file,
#                         training_epoch=1)
#     else:
#         env = CarRacing(agent=args.no_agent, num_bots=args.bots_number,
#                         write=args.write, data_path=args.dir)
#     env.render()
#     record_video = False
#     if record_video:
#         env.monitor.start('/tmp/video-test', force=True)
#     env.viewer.window.on_key_press = key_press
#     env.viewer.window.on_key_release = key_release
#     while True:
#         env.reset()
#         total_reward = 0.0
#         steps = 0
#         restart = False
#         while True:
#             s, r, done, info = env.step(a)
#             total_reward += r
#             if steps % 200 == 0 or done:
#                 print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
#                 print("step {} total_reward {:+0.2f}".format(steps, total_reward))
#                 # import matplotlib.pyplot as plt
#                 # plt.imshow(s)
#                 # plt.savefig("test.jpeg")
#             steps += 1
#             if not record_video:  # Faster, but you can as well call env.render() every time to play full window.
#                 env.render()
#             if done or restart: break
#     env.close()
