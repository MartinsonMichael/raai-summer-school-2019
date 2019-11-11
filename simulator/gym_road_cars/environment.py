from typing import NamedTuple, Dict, List, Any, Union, Tuple

from cv2 import cv2
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
        self._agent_car = DummyCar(
            world=self._b2world,
            restricted_world=self._restricted_world,
            track=self._data_loader.peek_track(index=car_track_index),
            car_image=self._data_loader.peek_car_image(),
            bot=False,
        )
        state, _, _, _ = self.step(0)
        return state

    def _restriction_reward_and_done(self) -> Tuple[float, bool]:
        reward = 0.0
        done = False

        cur_road_state = self._agent_car.get_road_position_state()
        if cur_road_state == RoadCarState.NOT_ROAD:
            reward += reward_constants.REWARD_OUT
            done = True
        if cur_road_state == RoadCarState.OTHER_SIDE:
            reward += reward_constants.REWARD_ROAD_CHANGE
            done = True

        return reward, done

    @staticmethod
    def calc_rotation_matrix(image, angle, scale=1.0) -> Tuple[Any, Tuple[float, float]]:
        center_rot = (int(image.shape[1] / 2), int(image.shape[0] / 2))
        rotation_mat = cv2.getRotationMatrix2D(center_rot, angle, scale)

        height = image.shape[0]
        width = image.shape[1]

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - center_rot[0]
        rotation_mat[1, 2] += bound_h / 2 - center_rot[1]

        return rotation_mat, (bound_w, bound_h)

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

        # compute reward and done base on road state
        reward, done = self._restriction_reward_and_done()

        # reward for finishing
        if self._agent_car.is_on_finish:
            reward += reward_constants.REWARD_FINISH
            done = True

        # reward for progress on track
        self._agent_car.update_track_point()
        if self._agent_car.is_achieve_new_track_point:
            reward += reward_constants.REWARD_TILES

        return (self.render(), self._agent_car.get_extra_info()), reward, done, {}

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

        rotation_mat, (bound_w, bound_h) = CarRacing.calc_rotation_matrix(
            car.car_image.image,
            car.angle,
        )

        print(f'rotation_mat: {rotation_mat}')

        masked_image = cv2.warpAffine(car.car_image.image, rotation_mat, (bound_w, bound_h))
        car_mask_image = cv2.warpAffine(car.car_image.mask, rotation_mat, (bound_w, bound_h))

        car_x, car_y = car.get_center_point()
        start_x = min(max(int(car_x - bound_w / 2), 0), background_image.shape[1])
        start_y = min(max(int(car_y - bound_h / 2), 0), background_image.shape[0])
        end_x = max(min(int(car_x + bound_w / 2), background_image.shape[1]), 0)
        end_y = max(min(int(car_y + bound_h / 2), background_image.shape[0]), 0)

        if start_x == end_x or start_y == end_y:
            return background_image, background_mask

        mask_start_x = start_x - int(car_x - bound_w / 2)
        mask_start_y = start_y - int(car_y - bound_h / 2)
        mask_end_x = mask_start_x + end_x - start_x
        mask_end_y = mask_start_y + end_y - start_y

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

        print(f'cropped_mask shape {cropped_mask.shape}')
        print(f'cropped_image shape {cropped_image.shape}')
        print(f'cropped back shape {(background_image[start_y:end_y, start_x:end_x, :]).shape}')

        background_image[start_y:end_y, start_x:end_x, :][cropped_mask] = cropped_image[cropped_mask]

        # background_crop = background_image[start_y:end_y, start_x:end_x]
        #

        #
        # mask_bool = (car_mask_image[mask_start_y:mask_end_y, mask_start_x:mask_end_x, 0]).astype('uint8')  # /255
        # mask_inv = cv2.bitwise_not(mask_bool)  # -254
        #
        #
        # masked_image = masked_image[mask_start_y:mask_end_y, mask_start_x:mask_end_x]
        # # Now black-out the area of logo in ROI
        # img1_bg = cv2.bitwise_and(background_crop, background_crop, mask=mask_inv)
        # # Take only region of logo from logo image.
        # img2_fg = cv2.bitwise_and(masked_image, masked_image, mask=mask_bool)
        #
        # # Put logo in ROI and modify the main image
        # dst = cv2.add(img1_bg, img2_fg)
        # background_image[start_y:end_y, start_x:end_x] = dst
        # background_mask[start_y:end_y, start_x:end_x] = background_mask[start_y:end_y, start_x:end_x] + mask_bool

        return background_image, background_mask

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
