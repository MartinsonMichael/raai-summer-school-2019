from typing import NamedTuple, Dict, List, Any, Union, Tuple

from cv2 import cv2
from shapely import geometry
import Box2D
import gym
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding, EzPickle
import numpy as np


from .reward_constants import *
from .utils import DataSupporter
from .car import DummyCar, RoadCarState


class CarRacing(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': 30,
    }

    def __init__(self,):
        EzPickle.__init__(self)
        self.np_random = None

        self._data_loader = DataSupporter(
            '/data/Hack/CDS_Lab/sac_branch/simulator/gym_road_cars/env_data_test/cars/',
            '/data/Hack/CDS_Lab/sac_branch/simulator/gym_road_cars/env_data_test/tracks/1_background_segmentation.xml',
            '/data/Hack/CDS_Lab/sac_branch/simulator/gym_road_cars/env_data_test/tracks/background_image.jpg',
        )
        self._b2world = Box2D.b2World(
            gravity=(0, 0),
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

    def reset(self, car_track_index=None):
        # car_track_index - to choose track not randomly

        try:
            self._agent_car.destroy()
        except:
            pass

        self._agent_car = DummyCar(
            world=self._b2world,
            restricted_world=self._restricted_world,
            track=self._data_loader.peek_track(index=car_track_index),
            car_image=self._data_loader.peek_car_image(),
            bot=False,
        )
        state, _, _, _ = self.step(0)
        return state

    def _restriction_reward_and_done(self) -> Tuple[float, bool, Dict[str, Any]]:
        reward = 0.0
        done = False
        info = {}

        cur_road_state = self._agent_car.get_road_position_state()
        if cur_road_state == RoadCarState.NOT_ROAD:
            reward += REWARD_OUT
            info['not_road'] = True
            done = True
            info['done'] = True

        if cur_road_state == RoadCarState.OTHER_SIDE:
            reward += REWARD_ROAD_CHANGE
            info['change_road'] = True
            done = True
            info['done'] = True

        return reward, done, info

    def step(self, action) -> Tuple[Tuple[Any, Any], float, bool, dict]:

        if action == 0:
            pass
        if action == 1:
            self._agent_car.gas(1.0)
        if action == 2:
            self._agent_car.brake(1.0)
        if action == 3:
            self._agent_car.steer(1.0)
        if action == 4:
            self._agent_car.steer(-1.0)

        self._agent_car.step(0.04)

        # compute reward and done base on road state
        reward, done, info = self._restriction_reward_and_done()

        # reward for finishing
        if self._agent_car.is_on_finish:
            reward += REWARD_FINISH
            info['finish'] = True
            done = True
            info['done'] = True

        # reward for progress on track
        self._agent_car.update_track_point()
        info['new_track_points'] = self._agent_car.count_of_new_track_point
        if self._agent_car.is_achieve_new_track_point:
            reward += REWARD_TILES * self._agent_car.count_of_new_track_point

        self._b2world.Step(0.04, 6 * 30, 2 * 30)

        return (self.render(), self._agent_car.get_extra_info()), reward, done, info

    def render(self, mode='human') -> np.array:
        background_image = self._data_loader.get_background()
        background_mask = np.zeros(
            shape=(background_image.shape[0], background_image.shape[1]),
            dtype='uint8'
        )

        background_image, background_mask = self.draw_car(
            background_image,
            background_mask,
            self._agent_car,
        )

        return background_image


    @staticmethod
    def draw_car(background_image, background_mask, car: DummyCar) -> Tuple[np.array, np.array]:
        # check dimensions
        if background_image.shape[0] != background_mask.shape[0]:
            raise ValueError('background image and mask have different shape')
        if background_image.shape[1] != background_mask.shape[1]:
            raise ValueError('background image and mask have different shape')
        if car.car_image.mask.shape[0] != car.car_image.image.shape[0]:
            raise ValueError('car image and mask have different shape')
        if car.car_image.mask.shape[1] != car.car_image.image.shape[1]:
            raise ValueError('car image and mask have different shape')

        rotation_mat, (bound_x, bound_y) = car.calc_rotation_matrix()

        print('bounds : ', bound_x, bound_y)

        masked_image = cv2.warpAffine(car.car_image.image, rotation_mat, (bound_x, bound_y))
        car_mask_image = cv2.warpAffine(car.car_image.mask, rotation_mat, (bound_x, bound_y))

        car_x, car_y = car.get_center_point()
        start_x = min(
            max(
                int(car_x - bound_x / 2),
                0,
            ),
            background_image.shape[1],
        )
        start_y = min(
            max(
                int(car_y - bound_y / 2),
                0,
            ),
            background_image.shape[0],
        )
        end_x = max(
            min(
                int(car_x + bound_x / 2),
                background_image.shape[1]
            ),
            0,
        )
        end_y = max(
            min(
                int(car_y + bound_y / 2),
                background_image.shape[0],
            ),
            0,
        )

        if start_x == end_x or start_y == end_y:
            return background_image, background_mask

        mask_start_x = start_x - int(car_x - bound_x / 2)
        mask_start_y = start_y - int(car_y - bound_y / 2)
        mask_end_x = mask_start_x + end_x - start_x
        mask_end_y = mask_start_y + end_y - start_y

        print(mask_start_x, mask_end_x, mask_start_y, mask_end_y)

        cropped_mask = car_mask_image[
           mask_start_y: mask_end_y,
           mask_start_x: mask_end_x,
        ] == 255

        cropped_image = (
            masked_image[
                mask_start_y: mask_end_y,
                mask_start_x: mask_end_x,
                :,
            ]
        )
        background_image[start_y:end_y, start_x:end_x, :][cropped_mask] = cropped_image[cropped_mask]
        return background_image, background_mask

    def close(self):
        raise NotImplemented
