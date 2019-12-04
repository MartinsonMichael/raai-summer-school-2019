import Box2D
import cv2
import gym
import numpy as np

from gym import spaces
from gym.utils import seeding, EzPickle


from envs.gym_car_intersect_fixed.car import DummyCar
from envs.gym_car_intersect_fixed.contact_listner import RefactoredContactListener
from envs.gym_car_intersect_fixed.rewards import Rewarder

from envs.gym_car_intersect_fixed.utils import DataSupporter
from shapely import geometry
from typing import List, Union

FPS = 60

class CarRacingHackatonContinuousFixed(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }
    training_epoch = 1

    def __init__(self, num_bots=0):
        EzPickle.__init__(self)

        # load env resources
        import os
        ABS_PATH_TO_DATA = os.path.join(os.path.abspath(''), 'envs', 'gym_car_intersect_fixed', 'env_data')
        self._data_loader = DataSupporter(
            os.path.join(ABS_PATH_TO_DATA, 'cars'),
            os.path.join(ABS_PATH_TO_DATA, 'tracks', '143_mmd_CarRacing.xml'),
            os.path.join(ABS_PATH_TO_DATA, 'tracks', 'background_image.jpg'),
        )

        # init world
        self.seed()
        self.contactListener_keepref = RefactoredContactListener(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self._was_done: bool = False
        self._init_world()

        # init agent data
        self.car = None
        self.rewarder = Rewarder()
        self._agent_goal = None
        self._agent_tiles = []

        # init bots data
        self.num_bots = num_bots
        self.bots = []

        # init gym properties
        self.state = np.zeros_like(self._data_loader.get_background(), dtype=np.uint8)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([+1, +1, +1]),
            dtype=np.float32
        )  # steer, gas, brake
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self._data_loader.get_background().shape,
            dtype=np.uint8
        )
        self._preseted_agent_track = None

    def set_bot_number(self, bot_number):
        self.num_bots = bot_number

    def set_agent_track(self, index):
        """
        Set agent track.
        :param index: index from 0 to number of tracks (smt like 12)
        :return: void
        """
        if index is None:
            print('agent track set to random')
            self._preseted_agent_track = None
            return
        if index < 0 or index > self._data_loader.track_count:
            raise ValueError(f'index must be from 0 to {self._data_loader.track_count}')
        print(f'agent track set to {index}')
        self._preseted_agent_track = index

    def _init_world(self):
        """
        function to create shapely polygons, which define road zones, not road zone
        :return: void
        """
        self.world.restricted_world = {
            'not_road': [],
            'cross_road': [],
        }
        for polygon in self._data_loader.data.get_polygons(0):
            polygon_name = polygon['label']
            polygon_points = polygon['points']
            if polygon_name in {'not_road', 'cross_road'}:
                self.world.restricted_world[polygon_name].append(geometry.Polygon(
                    self._data_loader.convertIMG2PLAY(polygon_points)
                ))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        """
        destroy b2world
        :return: void
        """
        if self._agent_tiles is not None:
            for tile in self._agent_tiles:
                self.world.DestroyBody(tile)
        self._agent_tiles = []
        if self._agent_goal is not None:
            self.world.DestroyBody(self._agent_goal)
        if self.car is not None:
            self.car.destroy()

    def reset(self):
        """
        recreate agent car and bots cars
        :return: initial state
        """
        self._destroy()
        self.time = 0
        self.create_agent_car()

        self.bot_cars = []
        for bot_index in range(self.num_bots):
            self.create_bot_car()

        return self.step(None)[0]

    def create_agent_car(self):
        self.car = DummyCar(
            world=self.world,
            car_image=self._data_loader.peek_car_image(3),
            track=DataSupporter.do_with_points(
                self._data_loader.peek_track(expand_points=100, index=self._preseted_agent_track),
                self._data_loader.convertIMG2PLAY,
            )
            ,
            data_loader=self._data_loader,
            bot=False,
        )

    def create_bot_car(self):
        attempts = 0
        while True:
            bot_car = DummyCar(
                world=self.world,
                car_image=self._data_loader.peek_car_image(),
                track=DataSupporter.do_with_points(
                    self._data_loader.peek_track(expand_points=100),
                    self._data_loader.convertIMG2PLAY,
                ),
                data_loader=self._data_loader,
                bot=True,
            )
            collided_indexes = self.initial_track_check(bot_car.track)
            if len(collided_indexes) == 0:
                print('bot added')
                self.bot_cars.append(bot_car)
                return True
            else:
                attempts += 1
                print(f'created bot collided existed cars: {collided_indexes}')
                if attempts >= 4:
                    return False

    def initial_track_check(self, track) -> List[int]:
        """
        Check if initial track position intersect some existing car. Return list of bots car indexes, which
        collide with track initial position. For agent car return -1 as index.
        :return: list of integers
        """
        init_pos = DataSupporter.get_track_initial_position(track)
        collided_indexes = []
        for bot_index, bot_car in enumerate(self.bot_cars):
            if DataSupporter.dist(init_pos, bot_car.position_PLAY) < 3:
                collided_indexes.append(bot_index)

        if self.car is not None:
            if DataSupporter.dist(self.car.position_PLAY, init_pos) < 3:
                collided_indexes.append(-1)

        return collided_indexes

    def step(self, action: List[float]):
        if self._was_done:
            self._was_done = False
            return self.reset(), 0.0, False, {}

        info = {}
        if action is not None:
            self.car.steer(action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        delta_time = 1.0 / FPS
        self.car.step(delta_time)
        for bot_car in self.bot_cars:
            bot_car.step(delta_time)

        self.world.Step(delta_time, 6 * 30, 2 * 30)
        self.time += delta_time
        self.car.flush_stats()
        self.car.update_stats()

        for index, bot_car in enumerate(self.bot_cars):
            bot_car.flush_stats()
            bot_car.update_stats()
            if bot_car.stats['is_finish']:
                bot_car.destroy()
                del bot_car
                self.bot_cars.pop(index)

        if len(self.bot_cars) < self.num_bots:
            self.create_bot_car()

        self.state = self.render()

        done = self.rewarder.get_step_done(self.car.stats)
        step_reward = self.rewarder.get_step_reward(self.car.stats)
        info.update(self.car.stats)

        self._was_done = done
        return self.state, step_reward, done, info

    def render(self, mode='human') -> np.array:
        background_image = self._data_loader.get_background().astype(np.uint8)
        background_mask = np.zeros(
            shape=(background_image.shape[0], background_image.shape[1]),
            dtype='uint8'
        )

        CarRacingHackatonContinuousFixed.draw_car(
            background_image,
            background_mask,
            self.car,
        )
        for bot_car in self.bot_cars:
            CarRacingHackatonContinuousFixed.draw_car(
                background_image,
                background_mask,
                bot_car,
            )
            if mode == 'debug':
                self.debug_draw_track(
                    background_image=background_image,
                    car=bot_car,
                    point_size=10,
                    color='green',
                )

        if mode == 'debug':
            self.debug_draw_track(
                background_image,
                car=self.car,
                point_size=10,
                color='red'
            )
            # self.debug_draw_restrictions(background_image)

        return background_image

    @staticmethod
    def rotate_image(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    @staticmethod
    def draw_car(background_image, background_mask, car: DummyCar):
        # check dimensions
        if background_image.shape[0] != background_mask.shape[0]:
            raise ValueError('background image and mask have different shape')
        if background_image.shape[1] != background_mask.shape[1]:
            raise ValueError('background image and mask have different shape')
        if car.car_image.mask.shape[0] != car.car_image.image.shape[0]:
            raise ValueError('car image and mask have different shape')
        if car.car_image.mask.shape[1] != car.car_image.image.shape[1]:
            raise ValueError('car image and mask have different shape')

        # rotate car image and mask of car image, and compute bounds of rotated image
        masked_image = CarRacingHackatonContinuousFixed.rotate_image(car.car_image.image, car.angle_degree+90)
        car_mask_image = CarRacingHackatonContinuousFixed.rotate_image(car.car_image.mask, car.angle_degree+90)
        bound_y, bound_x = masked_image.shape[:2]

        # car position in image coordinates (in pixels)
        car_x, car_y = car.position_IMG

        # bounds of car image on background image, MIN/MAX in a case of position near the background image boarder
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

        # in a case there car image is out of background, just return backgraund image
        if start_x == end_x or start_y == end_y:
            return background_image, background_mask

        # compute bounds of car image, in case then car near the bord of backgraund image,
        #    and so displayed car image
        #    less then real car image
        mask_start_x = start_x - int(car_x - bound_x / 2)
        mask_start_y = start_y - int(car_y - bound_y / 2)
        mask_end_x = mask_start_x + end_x - start_x
        mask_end_y = mask_start_y + end_y - start_y

        # finally crop car mask and car image, and insert them to background
        cropped_mask = car_mask_image[
           mask_start_y: mask_end_y,
           mask_start_x: mask_end_x,
           :,
        ] > 240

        cropped_image = (
            masked_image[
                mask_start_y: mask_end_y,
                mask_start_x: mask_end_x,
                :,
            ]
        )
        background_image[start_y:end_y, start_x:end_x, :][cropped_mask] = cropped_image[cropped_mask]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def debug_draw_hull(self, background_image, car, point_size=10, color='red'):
        for point in car._hull.fixtures[0].shape.vertices:
            pnt = DataSupporter.convert_XY2YX(self._data_loader.convertPLAY2IMG(point) + car.position_IMG)
            # print(f'fhull point: {pnt}')
            CarRacingHackatonContinuousFixed.debug_draw_sized_point(
                background_image,
                pnt,
                point_size,
                color,
            )

    def debug_draw_track(self, background_image, car, point_size=10, color='blue'):
        for point in self._data_loader.convertPLAY2IMG(car.track['line']):
            CarRacingHackatonContinuousFixed.debug_draw_sized_point(
                background_image,
                DataSupporter.convert_XY2YX(point),
                point_size,
                color,
            )
        CarRacingHackatonContinuousFixed.debug_draw_sized_point(
            background_image,
            DataSupporter.convert_XY2YX(self._data_loader.convertPLAY2IMG(car.track['line'][car._track_point])),
            point_size,
            'blue',
        )

    def debug_draw_polygon(self,
            background_image,
            polygon: np.array,
            point_size: int = 10,
            color='red',
    ):
        x, y = polygon.exterior.coords.xy
        for point in zip(x, y):
            CarRacingHackatonContinuousFixed.debug_draw_sized_point(
                background_image,
                DataSupporter.convert_XY2YX(self._data_loader.convertPLAY2IMG(np.array(point))),
                point_size,
                color,
            )

    @staticmethod
    def debug_draw_sized_point(
            background_image,
            coordinate: np.array,
            size: int,
            color: Union[np.array, str]
    ):
        if isinstance(color, str):
            color = {
                'red': np.array([255, 0, 0]),
                'green': np.array([0, 255, 0]),
                'blue': np.array([0, 0, 255]),
                'black': np.array([0, 0, 0]),
                'while': np.array([255, 255, 255]),
            }[color]
        y, x = coordinate
        for dx in range(int(-size / 2), int(size / 2) + 1, 1):
            for dy in range(int(-size / 2), int(size / 2) + 1, 1):
                background_image[
                    int(np.clip(y + dy, 0, background_image.shape[0] - 1)),
                    int(np.clip(x + dx, 0, background_image.shape[1] - 1)),
                    :,
                ] = color
