from enum import Enum

import numpy as np
import math
import Box2D
from typing import List, Dict, Tuple, Union, Optional, Any
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)
from cv2 import cv2
from shapely import geometry

from .data_utils import CarImage

SIZE = 0.2
ENGINE_POWER = 100000000
WHEEL_MOMENT_OF_INERTIA = 4000
FRICTION_LIMIT = 1000000


class RoadCarState(Enum):
    SAME_SIDE = 'SAME_SIDE',
    OTHER_SIDE = 'OTHER_SIDE',
    CROSS_ROAD = 'CROSS_ROAD',
    NOT_ROAD = 'NOT_ROAD',


class DummyCar:
    def __init__(
            self,
            world: Box2D.b2World,
            track: np.ndarray,
            car_image: CarImage,
            restricted_world: Dict[str, List[geometry.Polygon]],
            bot: bool = False
    ):
        self._track: np.ndarray = track
        self._track_point: int = 0
        self._old_track_point: int = 0
        self._car_image: CarImage = car_image
        self._restricted_world: Dict[str, List[geometry.Polygon]] = restricted_world
        self._is_bot: bool = bot

        hull_init_angle = DummyCar._angle_by_2_points(
            self._track[2],
            self._track[3]
        )
        hull_init_position = self._track[3]
        wheel_size = self._car_image.size / 10

        # create hull
        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position=hull_init_position,
            angle=hull_init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            tuple(x) for x in
                            DummyCar.get_four_points_around(hull_init_position, self._car_image.size)
                        ]
                    ),
                    density=1.0,
                ),
            ],
        )
        self.hull.name = 'bot_car' if bot else 'agent_car'
        self.wheels = []
        self.fuel_spent = 0.0

        wheels_positions = DummyCar.get_four_points_around([0, 0], self._car_image.size - wheel_size * 3)
        rotation_matrix, _ = self.calc_rotation_matrix()
        rotation_matrix = rotation_matrix[:2, :2]
        wheels_positions = (rotation_matrix @ wheels_positions.T).T

        print(f'angle : {self.angle}')
        print(f'wheels_positions : {wheels_positions}')


        # create wheels
        for w_index in range(4):
            w_position = wheels_positions[w_index]
            w = self.world.CreateDynamicBody(
                position=w_position + hull_init_position,
                angle=hull_init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            tuple(x + hull_init_position) for x in
                            DummyCar.get_four_points_around(w_position, wheel_size)
                        ]
                    ),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
            )
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.wheel_rad = 15.0
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=w_position,
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=3600,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)

            self.wheels.append(w)

        self._speed: List[float] = [0.0, 0.0]
        self._start_road_side = self.get_current_rode_side()
        self._cur_polygon: Tuple[geometry.Polygon, str] = self.find_cur_polygon()

    @staticmethod
    def get_four_points_around(center, size) -> np.array:
        center_x, center_y = center
        size_x, size_y = size
        return np.array([
            [center_x - size_x / 2, center_y - size_y / 2],
            [center_x - size_x / 2, center_y + size_y / 2],
            [center_x + size_x / 2, center_y + size_y / 2],
            [center_x + size_x / 2, center_y - size_y / 2],
        ])

    @property
    def angle(self, mod: str = 'deg'):
        if mod == 'deg':
            return self.hull.angle * 180 / 3.141592653589
        return self.hull.angle

    @property
    def get_start_rode_size(self):
        return self._start_road_side

    @property
    def is_in_known_polygon(self):
        if self._cur_polygon[0] is not None:
            return self._cur_polygon[0].contains(geometry.Point(self.get_center_point()))
        return False

    def iterate_over_front_wheels(self):
        if self.wheels is None or len(self.wheels) != 4:
            raise ValueError('wheels still do not created')
        yield self.wheels[2]
        yield self.wheels[3]

    def iterate_over_back_wheels(self):
        if self.wheels is None or len(self.wheels) != 4:
            raise ValueError('wheels still do not created')
        yield self.wheels[0]
        yield self.wheels[1]

    def find_cur_polygon(self) -> Tuple[Union[geometry.Polygon, None], str]:
        point = geometry.Point(self.get_center_point())
        for polygon_name in self._restricted_world.keys():
            for polygon in self._restricted_world[polygon_name]:
                if polygon.contains(point):
                    return polygon, polygon_name
        return None, 'not_road'

    def update_cur_polygon(self):
        self._cur_polygon = self.find_cur_polygon()

    def get_current_rode_side(self):
        polygon, polygon_name = self.find_cur_polygon()
        if polygon_name not in ['road1', 'road2', 'cross_road'] or polygon is None:
            raise ValueError('unknown road side')
        return polygon_name

    def _polygon_name_to_enum(self) -> RoadCarState:
        if self._cur_polygon[1] == 'road_cross':
            return RoadCarState.CROSS_ROAD
        if self._cur_polygon[1] == 'not_road':
            return RoadCarState.NOT_ROAD
        if self._cur_polygon[1] == self.get_start_rode_size:
            return RoadCarState.SAME_SIDE
        return RoadCarState.OTHER_SIDE

    def get_road_position_state(self) -> RoadCarState:
        if self.is_in_known_polygon:
            return self._polygon_name_to_enum()
        self.update_cur_polygon()
        return self._polygon_name_to_enum()

    def get_center_point(self) -> np.array:
        center_point = np.mean(self.get_wheels_positions(), axis=0)
        if center_point.shape != (2,):
            raise ValueError('incorrect center point shape')
        return center_point

    def get_wheels_positions(self) -> np.array:
        return np.array([
            [wheel.position.x, wheel.position.y] for wheel in self.wheels
        ])

    @staticmethod
    def _angle_by_2_points(
            pointA: np.array,
            pointB: np.array,
    ) -> float:
        return DummyCar._angle_by_3_points(
            pointB,
            pointA,
            pointA + np.array([1, 0])
        )

    @staticmethod
    def _angle_by_3_points(
            pointA: np.array,
            pointB: np.array,
            pointC: np.array) -> float:
        """
        compute angle
        :param pointA: np.array of shape (2, )
        :param pointB: np.array of shape (2, )
        :param pointC: np.array of shape (2, )
        :return: angle in radians between AB and BC
        """
        if pointA.shape != (2,) or pointB.shape != (2,) or pointC.shape != (2,):
            raise ValueError('incorrect points shape')

        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arctan2(np.cross(v1_u, v2_u), np.dot(v1_u, v2_u))

        return angle_between(pointA - pointB, pointC - pointB)

    def get_extra_info(self) -> np.array:
        return np.array([
            self.hull.position.x,
            self.hull.position.y,
            self.hull.angle,
            self.fuel_spent,
            *self.speed,
        ],
            dtype=np.float32,
        )

    def calc_rotation_matrix(self, scale=1.0) -> Tuple[Any, Tuple[float, float]]:
        center_rot = self._car_image.size / 2
        rotation_mat = cv2.getRotationMatrix2D(tuple(center_rot), self.angle, scale)
        bounds = (rotation_mat[:2, :2] @ (center_rot * 2 + 5.0).T).T
        return rotation_mat, bounds.astype(np.int32)

    @staticmethod
    def dist(pointA: np.array, pointB: np.array) -> float:
        return np.sqrt(np.sum((pointA - pointB) ** 2))

    def update_track_point(self):
        car_point = self.get_center_point()
        self._old_track_point = self._track_point
        for track_index in range(self._track_point, len(self._track), 1):
            if self.dist(self._track[track_index], car_point) < 6:
                continue
            self._track_point = track_index
            break

    @property
    def is_achieve_new_track_point(self) -> bool:
        return self._old_track_point == self._track_point

    @property
    def count_of_new_track_point(self) -> float:
        return max(0, self._track_point - self._old_track_point)

    @property
    def is_on_finish(self) -> bool:
        return self._track_point >= len(self._track) - 3

    def gas(self, value):
        """control: rear wheel drive"""
        value = np.clip(value, 0, 1) / 10
        for back_wheel in self.iterate_over_back_wheels():
            diff = value - back_wheel.gas
            diff = np.clip(diff, -0.1, 0.01)
            back_wheel.gas += diff

    def brake(self, b):
        """control: brake b=0..1, more than 0.9 blocks wheels to zero rotation"""
        b = np.clip(b, 0, 1)
        for w in self.wheels:
            w.brake = b

    def steer(self, value):
        """control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position"""
        value = np.clip(value, -1, 1)
        for front_wheel in self.iterate_over_front_wheels():
            front_wheel.steer = value

    def step(self, dt):
        _speed = []
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 2.0)

            friction_limit = FRICTION_LIMIT * 0.6

            # Force
            forw = np.array(w.GetWorldVector((0, 1)))
            side = np.array(w.GetWorldVector((1, 0)))
            v = np.array(w.linearVelocity)
            vf = (forw * v).sum()  # forward speed
            vs = (side * v).sum()  # side speed
            _speed.append([vf, vs])

            w.omega += dt * ENGINE_POWER * w.gas / WHEEL_MOMENT_OF_INERTIA / (abs(w.omega) + 5.0)
            self.fuel_spent += dt * ENGINE_POWER * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.
            f_force *= 205000 * SIZE * SIZE  # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter((
                p_force * side[0] + f_force * forw[0],
                p_force * side[1] + f_force * forw[1]), True)
        self._speed = np.mean(_speed, axis=0)

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []

    @property
    def speed(self):
        speed = np.array(self._speed)
        if speed.shape != (2,):
            raise ValueError(f'incorrect speed shape, need (2, ) find {speed.shape}')
        return speed

    @property
    def car_image(self):
        return self._car_image
