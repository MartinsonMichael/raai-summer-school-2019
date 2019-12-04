import numpy as np
import math
import Box2D
from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef

from envs.gym_car_intersect_fixed.utils import DataSupporter
from shapely.geometry import Point


SIZE = 80 / 1378.0 * 0.5
MC = SIZE / 0.02
ENGINE_POWER = 100000000 * SIZE * SIZE / MC / MC
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE / MC / MC
FRICTION_LIMIT = 1000000 * SIZE * SIZE / MC / MC / 2
WHEEL_R = 27 / MC
WHEEL_W = 14 / MC
CENTROID = 220
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


class DummyCar:
    """
    Class of car for Carracing fixed environment.
    Unfortunately there are a lot of legacy magic constants,
        but if you remove or change them, everything will simply don't work correctly.
    """

    def __init__(
            self,
            world: Box2D,
            bot: bool = False,
            car_image=None,
            track=None,
            data_loader=None,
    ):
        """ Constructor to define Car.
        Parameters
        ----------
        world : Box2D World

        """
        self.car_image = car_image
        self.track = track
        self.data_loader = data_loader
        self.is_bot = bot
        self._bot_state = {
            'was_break': False,
        }

        # all coordinates in XY format, not in IMAGE coordinates
        init_x, init_y = DataSupporter.get_track_initial_position(self.track['line'])
        init_angle = DataSupporter.get_track_angle(track) - np.pi / 2
        width_y, height_x = self.car_image.size

        CAR_HULL_POLY4 = [
            (-height_x / 2, -width_y / 2), (+height_x / 2, -width_y / 2),
            (-height_x / 2, +width_y / 2), (+height_x / 2, +width_y / 2)
        ]
        N_SENSOR_BOT = [
            (-height_x / 2 * 1.11, +width_y / 2 * 1.0), (+height_x / 2 * 1.11, +width_y / 2 * 1.0),
            # (-height_x/2*1.11, +width_y/2*1.11), (+height_x/2*1.11, +width_y/2*1.11),
            (-height_x / 2 * 0.8, +width_y / 2 * 3), (+height_x / 2 * 0.8, +width_y / 2 * 3)
            # (-height_x/2*0.22, +width_y/2*2), (+height_x/2*0.22, +width_y/2*2)
        ]
        WHEELPOS = [
            (-height_x / 2, +width_y / 2 / 2), (+height_x / 2, +width_y / 2 / 2),
            (-height_x / 2, -width_y / 2 / 2), (+height_x / 2, -width_y / 2 / 2)
        ]

        N_SENSOR_SHAPE = CAR_HULL_POLY4

        SENSOR = N_SENSOR_BOT if bot else N_SENSOR_SHAPE
        self.world = world
        self._hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[fixtureDef(shape=polygonShape(vertices=[(x * SIZE, y * SIZE) for x, y in CAR_HULL_POLY4]),
                                 density=1.0, userData='body'),
                      fixtureDef(shape=polygonShape(vertices=[(x * SIZE, y * SIZE) for x, y in SENSOR]),
                                 isSensor=True, userData='sensor')])
        self._hull.name = 'bot_car' if bot else 'car'
        self._hull.cross_time = float('inf')
        self._hull.stop = False
        self._hull.collision = False
        self._hull.userData = self._hull
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
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            rjd = revoluteJointDef(
                bodyA=self._hull,
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
            w.tiles = set()
            w.name = 'wheel'
            w.collision = False
            w.penalty = False
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self._hull]
        self.target = (0, 0)

        self._time: int = 0
        self.userData = self
        self._track_point: int = 0
        self._old_track_point: int = 0
        self._state_data = None
        self.flush_stats()

    @property
    def angle_degree(self):
        return self._hull.angle * 180 / np.pi

    @property
    def position_PLAY(self) -> np.array:
        return np.array([self._hull.position.x, self._hull.position.y])

    @property
    def position_IMG(self) -> np.array:
        return self.data_loader.convertPLAY2IMG(self.position_PLAY)

    @property
    def stats(self):
        return self._state_data

    def flush_stats(self):
        """
        Set car statistic data to initial state.
        """
        self._state_data = {
            'new_tiles_count': 0,
            'is_finish': False,
            'is_collided': False,
            'is_on_cross_road': False,
            'is_out_of_track': False,
            'is_out_of_map': False,
            'is_out_of_road': False,

            'speed': 0.0,
            'time': 0,
        }

    def update_stats(self):
        """
        Update car statistic with current car state.
        """
        cur_points = [
            np.array([wheel.position.x, wheel.position.y])
            for wheel in self.wheels
        ]
        # self._state_data['car_position'] = np.mean(cur_points, axis=0)

        for wheel_position in cur_points:
            if np.any(wheel_position < 0):
                self._state_data['is_out_of_map'] = True
            if wheel_position[0] > self.data_loader.playfield_size[0]:
                self._state_data['is_out_of_map'] = True
            if wheel_position[1] > self.data_loader.playfield_size[1]:
                self._state_data['is_out_of_map'] = True

        cur_points = [Point(x) for x in cur_points]

        for wheel_position in cur_points:
            for polygon in self.world.restricted_world['not_road']:
                if polygon.contains(wheel_position):
                    self._state_data['is_out_of_road'] = True
                    break

            for polygon in self.world.restricted_world['cross_road']:
                if polygon.contains(wheel_position):
                    self._state_data['is_on_cross_road'] = True
                    break

            if not self.track['polygon'].contains(wheel_position):
                self._state_data['is_out_of_track'] = True

        # update track progress
        self._update_track_point()

        if ((self.track['line'][self._track_point] - self.track['line'][-1])**2).sum() < 0.5:
            self._state_data['is_finish'] = True

        # update collision from contact listner
        self._state_data['is_collided'] = self._hull.collision

        # add extra info to data:
        self._state_data['speed'] = np.sum(np.sqrt(
            np.array([
                self._hull.linearVelocity.x,
                self._hull.linearVelocity.y,
            ])**2
        ))
        self._state_data['time'] = self._time

    def _update_track_point(self):
        """
        Move car goal point in accordance with car track progress.
        """
        car_point = self.position_PLAY
        self._old_track_point = self._track_point
        for track_index in range(self._track_point, len(self.track['line']), 1):
            if ((self.track['line'][track_index] - car_point)**2).sum() < 2.5:
                continue
            self._track_point = track_index
            break
        self._state_data['new_tiles_count'] = self._track_point - self._old_track_point

    def gas(self, gas):
        """
        Car control: rear wheel drive
        """
        gas = np.clip(gas, 0, 1)
        gas /= 10
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.01: diff = 0.01  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        """
        Car control: brake b=0..1, more than 0.9 blocks wheels to zero rotation
        """
        b = np.clip(b, 0, 1)
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """
        Car control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position
        """
        s = np.clip(s, -1, 1)
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def go_to_target(self):
        """
        Set car params to move one step to current goal. Used for bot cars.
        """
        self._update_track_point()
        x, y = round(self._hull.position.x, 2), round(self._hull.position.y, 2)

        x_pos, y_pos = self.track['line'][self._track_point]
        target_angle = -math.atan2(x_pos - x, y_pos - y)

        x_pos_next, y_pos_next = self.track['line'][self._track_point + 1]
        target_angle_next = -math.atan2(x_pos_next - x, y_pos_next - y)

        direction = -math.pi / 2 + target_angle - self._hull.angle
        direction_next = -math.pi / 2 + target_angle_next - self._hull.angle

        steer_value = math.cos(direction * 0.6 + direction_next * 0.4)
        self.steer(steer_value)

        if abs(steer_value) >= 0.2 and not self._bot_state['was_break']:
            self.brake(0.8)
            self._bot_state['was_break'] = True
        else:
            self.brake(0.0)
            self.gas(0.1)

        if abs(steer_value) < 0.1:
            self._bot_state['was_break'] = False
            self.gas(0.3)


    def step(self, dt):
        """
        Compute forces and apply them to car wheels in accordance with gas/brake/steer state.
        This function must be called once in pyBox2D step.
        """
        self._time += 1

        if self.is_bot:
            self.go_to_target()

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
                if abs(val) > abs(w.omega): val = abs(w.omega)  # low speed => same as = 0
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

    def destroy(self):
        """
        Remove car property from pyBox2D world.
        """
        self.world.DestroyBody(self._hull)
        self._hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []
