from enum import Enum

import numpy as np
import math
import Box2D
from typing import List, Dict, Tuple, Union, Optional, Any
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)
from cv2 import cv2
from shapely import geometry

from envs.gym_road_cars.utils import CarImage

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

        print(f'hull init angle : {hull_init_angle}')

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
                            DummyCar.get_four_points_around(hull_init_position, self._car_image.size, 0)
                        ]
                    ),
                    density=1.0,
                ),
            ],
        )
        self.hull.name = 'bot_car' if bot else 'agent_car'
        self.wheels = []
        self.fuel_spent = 0.0

        wheels_positions = DummyCar.get_four_points_around(
            [0, 0],
            self._car_image.size - wheel_size * 3,
            0.0,
        )

        # car property
        self.gas: float = 0.0
        self.brake: float = 0.0

        # create wheels
        for is_front, w_index in [(False, 0), (False, 1), (True, 2), (True, 3)]:
            w_position = wheels_positions[w_index]

            print(w_position)

            w = self.world.CreateDynamicBody(
                position=w_position + hull_init_position,
                angle=hull_init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            tuple(x + hull_init_position) for x in
                            DummyCar.get_four_points_around(w_position, wheel_size, hull_init_angle)
                        ]
                    ),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
            )
            w.is_front = is_front
            w.is_back = not is_front
            w.steer = 0.0
            w.speed = np.array([0, 0], dtype=np.float32)
            w.acceleration = np.array([0, 0], dtype=np.float32)

            w.wheel_rad = int(is_front) * 270 * SIZE
            w.phase = 0.0  # wheel angle
            w.omega = 0.0

            w.is_front = False
            w.is_back = False
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
    def get_four_points_around(center, size, angle: float = 0) -> np.array:
        center_x, center_y = center
        size_x, size_y = size
        positions = np.array([
            [center_x - size_x / 2, center_y - size_y / 2],
            [center_x - size_x / 2, center_y + size_y / 2],
            [center_x + size_x / 2, center_y + size_y / 2],
            [center_x + size_x / 2, center_y - size_y / 2],
        ])
        rotation_matrix = DummyCar.get_simple_rotation_matrix(angle)
        positions = (rotation_matrix @ (positions - center).T).T + center
        return positions

    @property
    def angle_radian(self):
        return self.hull.angle

    @property
    def angle_degree(self):
        return self.hull.angle * 180 / np.pi

    @property
    def get_wheels_vectors(self) -> np.array:
        vectors = []
        for wheel in self.wheels:
            vectors.append(
                (DummyCar.get_simple_rotation_matrix(wheel.angle) @ np.array([1, 0]).T).T
            )
        return np.array(vectors)

    @property
    def get_car_vector(self) -> np.array:
        return (DummyCar.get_simple_rotation_matrix(self.angle_radian) @ np.array([1, 0]).T).T

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
        for wheel in self.wheels:
            if wheel.is_front:
                yield wheel

    def iterate_over_back_wheels(self):
        if self.wheels is None or len(self.wheels) != 4:
            raise ValueError('wheels still do not created')
        for wheel in self.wheels:
            if wheel.is_back:
                yield wheel

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
        center_point = np.array([self.hull.position.x, self.hull.position.y])
        if center_point.shape != (2,):
            raise ValueError('incorrect center point shape')
        return center_point

    def get_wheels_positions(self) -> np.array:
        return np.array([
            [wheel.position.x, wheel.position.y] for wheel in self.wheels
        ])

    @staticmethod
    def test__angle_by_2_points():
        for test in [
            ([0, 0], [1, 0], 0.0),
            ([0, 0], [0, 1], np.pi / 2),
            ([0, 0], [-1, 0], -np.pi),
            ([0, 0], [0, -1], -np.pi / 2),
            ([-1, -1], [1, 1], np.pi / 4)
        ]:
            pointA, pointB, ans = test
            if DummyCar._angle_by_2_points(pointA, pointB) % np.pi != ans % np.pi:
                print('test failed')
                print(pointA, pointB)
                print(f'get : {DummyCar._angle_by_2_points(pointA, pointB)}')
                print(f'should be : {ans}')
                print()

    @staticmethod
    def _angle_by_2_points(
            pointA: np.array,
            pointB: np.array,
    ) -> float:
        return DummyCar._angle_by_3_points(
            np.array(pointA) + np.array([1.0, 0.0]),
            np.array(pointA),
            np.array(pointB),
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
                *self.get_center_point(),
                self.angle_radian,
                self.fuel_spent,
                *self.speed,
            ],
            dtype=np.float32,
        )

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

    def gas(self, gas_value):
        """control: rear wheel drive"""
        gas_value = np.clip(gas_value, 0.0, 1.0) / 10.0
        self.gas += gas_value
        self.gas = np.clip(self.gas, 0.0, 1.0)

    def brake(self, brake_value):
        """control: brake b=0..1, more than 0.9 blocks wheels to zero rotation"""
        brake_value = np.clip(brake_value, 0, 1) / 10.0
        self.brake += brake_value
        self.brake = np.clip(self.brake, 0.0, 1.0)

    def steer(self, steer_value):
        """control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position"""
        # angle in radian
        value = np.clip(steer_value, -0.05, 0.05)
        for front_wheel in self.iterate_over_front_wheels():
            front_wheel.steer += value
            front_wheel.steer = np.clip(front_wheel.steer, -0.4, 0.4)
            front_wheel.angle = self.angle_radian + front_wheel.steer

    @staticmethod
    def get_simple_rotation_matrix(angle, convert_to_degree=False, convert_to_radian=False):
        if convert_to_degree:
            angle = angle * 180 / np.pi
        if convert_to_radian:
            angle = angle * np.pi / 180
        sin = np.sin(angle)
        cos = np.cos(angle)
        return np.array([
            [cos, -sin],
            [sin, cos],
        ])

    def step(self, dt):
        _speed = []

        CAR_MASS = 10
        CAR_FORCE = 10

        force_value =

        for wheel_index, wheel in enumerate(self.wheels):

            sing = np.sign(wheel.steer - wheel.joint.angle)
            val = abs(wheel.steer - wheel.joint.angle)
            wheel.joint.motorSpeed = sing * min(50.0 * val, 2.0)

            forw = wheel.GetWorldVector((1, 0))
            side = wheel.GetWorldVector((0, 1))

            v = wheel.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            if wheel_index == 0:
                print()
                print(f'forw : {forw}')
                print(f'side : {side}')
                print(f'speed : {v}')
                print(f'forw speed : {vf}')
                print(f'side speed : {vs}')


            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega
            wheel.omega += dt * ENGINE_POWER * wheel.gas / WHEEL_MOMENT_OF_INERTIA / (
                    abs(wheel.omega) + 5.0)  # small coef not to divide by zero
            self.fuel_spent += dt * ENGINE_POWER * wheel.gas

            if wheel.brake >= 0.9:
                wheel.omega = 0
            elif wheel.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(wheel.omega)
                val = BRAKE_FORCE * wheel.brake
                if abs(val) > abs(wheel.omega):
                    val = abs(wheel.omega)  # low speed => same as = 0
                wheel.omega += dir * val
            wheel.phase += wheel.omega * dt

            vr = wheel.omega * wheel.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.
            f_force *= 205000 * SIZE * SIZE  # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            friction_limit = FRICTION_LIMIT * 0.6
            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            wheel.omega -= dt * f_force * wheel.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            final_force = np.array([
                p_force * side[0] + f_force * forw[0],
                p_force * side[1] + f_force * forw[1],
            ])

            if wheel_index == 0:
                print(f'final_force: {final_force}')

            wheel.ApplyForceToCenter((
                    final_force[0],
                    final_force[1],
                ),
                True
            )

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
