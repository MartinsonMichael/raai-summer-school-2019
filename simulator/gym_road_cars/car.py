from enum import Enum

import numpy as np
import math
import Box2D
from typing import List, Dict, Tuple, Union, Optional
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)
from shapely import geometry

# from hack_env_discrete import SHOW_SCALE

# Top-down car dynamics simulation.
#
# Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
# This simulation is a bit more detailed, with wheels rotation.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.


from .data_utils import CarImage

SIZE = 80 / 1378.0  # SHOW_SCALE #0.02
MC = SIZE / 0.02
ENGINE_POWER = 100000000 * SIZE * SIZE / MC / MC
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE / MC / MC
FRICTION_LIMIT = 1000000 * SIZE * SIZE / MC / MC / 2  # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R = 27 / MC
WHEEL_W = 14 / MC
CENTROID = 220  # that is point which follows target in car...
WHEELPOS = [
    (-45, +60 - CENTROID), (+45, +60 - CENTROID),
    (-45, -70 - CENTROID), (+45, -70 - CENTROID)
]
HULL_POLY4 = [
    (-45, -105 - CENTROID), (+45, -105 - CENTROID),
    (-45, +105 - CENTROID), (+45, +105 - CENTROID)
]
SENSOR_SHAPE = [
    (-45, -105 - CENTROID), (+45, -105 - CENTROID),
    (-45, +105 - CENTROID), (+45, +105 - CENTROID)
]
## Point sensor:
# SENSOR_BOT = [
#     (-10,350-CENTROID), (+10,350-CENTROID),
#     (-10,+360-CENTROID),  (+10,+360-CENTROID)
# ]
SENSOR_BOT = [
    (-50, +110 - CENTROID), (+50, +110 - CENTROID),
    (-10, +300 - CENTROID), (+10, +300 - CENTROID)
]
# SENSOR_ADD = [
#     (-1,+110-CENTROID), (+1,+110-CENTROID),
#     (-50,+200-CENTROID),  (+50,+200-CENTROID)
# ]
WHEEL_COLOR = (0.0, 0.0, 0.0)
WHEEL_WHITE = (0.3, 0.3, 0.3)


class RoadCarState(Enum):
    SAME_SIDE = 0,
    OTHER_SIDE = 1,
    CROSS_ROAD = 2,
    NOT_ROAD = 3,



class DummyCar:
    def __init__(
            self,
            world: Box2D.b2World,
            track: np.ndarray,
            car_image: CarImage,
            restricted_world: Dict[str, List[geometry.Polygon]],
            bot: bool = False
    ):
        """ Constructor to define Car.
        Parameters
        ----------
        world : Box2D World
        init_coord : tuple
            (angle, x, y)
        color : tuple
            Selfexplanatory
        """
        self._track: np.ndarray = track
        self._track_point: int = 0
        self._old_track_point: int = 0
        self._car_image: CarImage = car_image
        self._restricted_world: Dict[str, List[geometry.Polygon]] = restricted_world
        self._is_bot: bool = bot

        init_x, init_y = self._track[0]
        init_angle = DummyCar._angle_by_2_points(
            self._track[0],
            self._track[1]
        )
        width_y, height_x = self._car_image.size

        CAR_HULL_POLY4 = [
            (-height_x / 2, -width_y / 2), (+height_x / 2, -width_y / 2),
            (-height_x / 2, +width_y / 2), (+height_x / 2, +width_y / 2),
        ]
        N_SENSOR_BOT = [
            (-height_x / 2 * 1.11, +width_y / 2 * 1.0), (+height_x / 2 * 1.11, +width_y / 2 * 1.0),
            (-height_x / 2 * 0.8, +width_y / 2 * 3), (+height_x / 2 * 0.8, +width_y / 2 * 3),
        ]
        WHEELPOS = [
            (-height_x / 2, +width_y / 2 / 2), (+height_x / 2, +width_y / 2 / 2),
            (-height_x / 2, -width_y / 2 / 2), (+height_x / 2, -width_y / 2 / 2),
        ]

        N_SENSOR_SHAPE = CAR_HULL_POLY4

        SENSOR = N_SENSOR_BOT if bot else N_SENSOR_SHAPE
        self.world = world

        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(vertices=[(x * SIZE, y * SIZE) for x, y in CAR_HULL_POLY4]),
                    density=1.0,
                    userData='body'
                ),
                fixtureDef(
                    shape=polygonShape(vertices=[(x * SIZE, y * SIZE) for x, y in SENSOR]),
                    isSensor=True,
                    userData='sensor'
                ),
            ],
        )
        self.hull.color = ((0.2, 0.8, 1) if bot else (0.8, 0.0, 0.0))
        self.hull.name = 'bot_car' if bot else 'car'
        self.hull.cross_time = float('inf')
        self.hull.stop = False
        self.hull.path = ''
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R), (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R), (-WHEEL_W, -WHEEL_R)
        ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * SIZE, init_y + wy * SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x * front_k * SIZE, y * front_k * SIZE) for x, y in WHEEL_POLY]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
            )
            w.wheel_rad = front_k * WHEEL_R * SIZE
            w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx * SIZE, wy * SIZE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * SIZE * SIZE,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]

        self._speed: List[float] = [0.0, 0.0]
        self._start_road_side = self.get_current_rode_side()
        self._cur_polygon: Tuple[geometry.Polygon, str] = self.find_cur_polygon()

    @property
    def angle(self):
        return self.hull.angle * 180 / 3.141592653589

    @property
    def get_start_rode_size(self):
        return self._start_road_side

    @property
    def is_in_known_polygon(self):
        if self._cur_polygon[0] is not None:
            return self._cur_polygon[0].contains(geometry.Point(self.get_center_point()))
        return False

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

    def get_center_point(self):
        center_point = np.mean(self.get_wheels_positions(), axis=0)
        if center_point.shape != (2, ):
            raise ValueError('incorrect center point shape')
        return center_point

    def get_wheels_positions(self):
        return np.array([
            [wheel.position.x, wheel.position.y] for wheel in self.wheels
        ])

    @staticmethod
    def _angle_by_2_points(
            pointA: np.array,
            pointB: np.array,
    ):
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
        if pointA.shape != (2,) or pointB.shape != (2,) or pointC.shape != (2, ):
            raise ValueError('incorrect points shape')

        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arctan2(np.cross(v1_u, v2_u), np.dot(v1_u, v2_u))

        print(f'points A: {pointA}')
        print(f'points B: {pointB}')
        print(f'points C: {pointC}')

        return angle_between(pointA - pointB, pointC - pointB)

    def get_extra_info(self):
        return np.array([
                self.hull.position.x,
                self.hull.position.y,
                self.hull.angle,
                self.fuel_spent,
                *self.speed,
            ],
            dtype=np.float32,
        )

    @ staticmethod
    def dist(pointA: np.array, pointB: np.array) -> float:
        return np.sqrt(np.sum((pointA - pointB)**2))

    def update_track_point(self):
        car_point = self.get_center_point()
        self._old_track_point = self._track_point
        for track_index in range(self._track_point, len(self._track), 1):
            if self.dist(self._track[track_index], car_point) < 6:
                continue
            self._track_point = track_index
            break

    @property
    def is_achieve_new_track_point(self):
        return self._old_track_point == self._track_point

    @property
    def is_on_finish(self):
        return self._track_point >= len(self._track) - 3

    def gas(self, gas):
        'control: rear wheel drive'
        gas = np.clip(gas, 0, 1)
        gas /= 10
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.01: diff = 0.01  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        'control: brake b=0..1, more than 0.9 blocks wheels to zero rotation'
        b = np.clip(b, 0, 1)
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        'control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position'
        s = np.clip(s, -1, 1)
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        _speed = []
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 2.0)

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT * 0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(friction_limit, FRICTION_LIMIT * tile.road_friction)
                grass = False

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega
            w.omega += dt * ENGINE_POWER * w.gas / WHEEL_MOMENT_OF_INERTIA / (
                        abs(w.omega) + 5.0)  # small coef not to divide by zero
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

            _speed.append(np.sqrt(
                (p_force * side[0] + f_force * forw[0]) ** 2
                +
                (p_force * side[1] + f_force * forw[1]) ** 2
            ))
            w.ApplyForceToCenter((
                p_force * side[0] + f_force * forw[0],
                p_force * side[1] + f_force * forw[1]), True)
        self._speed = np.mean(_speed, axis=1)


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
            raise ValueError('incorrect speed shape')
        return speed

    @property
    def car_image(self):
        return self._car_image
