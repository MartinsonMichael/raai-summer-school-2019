import typing
import time

from Box2D.b2 import contactListener
from .env_constants import *


class MyContactListener(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    @staticmethod
    def _priority_check(path1, path2):
        target1, target2 = path1[0], path2[0]

        if target1 == '3' and target2 in {'3', '5', '7'}:
            return True
        elif target1 == '3':
            return False

        if target1 == '5' and target2 in {'5', '7', '9'}:
            return True
        elif target1 == '5':
            return False

        if target1 == '7' and target2 in {'7', '9'}:
            return True
        elif target1 == '7':
            return False

        if target1 == '9' and target2 in {'9', '3'}:
            return True
        elif target1 == '9':
            return False

    def BeginContact(self, contact):
        # Data to define sensor data:
        sensA = contact.fixtureA.sensor
        sensB = contact.fixtureB.sensor

        # Data to define collisions:
        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData

        # Check data we have for fixtures:
        fixA = contact.fixtureA.userData
        fixB = contact.fixtureB.userData

        # Processing sensors:
        if sensA and bodyA.name == 'car' and bodyB.name == 'road':
            if bodyB.road_section in bodyA.penalty_sec:
                bodyA.penalty = True
        if sensB and bodyB.name == 'car' and bodyA.name == 'road':
            if bodyA.road_section in bodyB.penalty_sec:
                bodyB.penalty = True

        # Behaviour on crossroads:
        if sensA and bodyA.name == 'bot_car' and bodyB.name == 'road':
            if bodyB.road_section == 1:
                bodyA.cross_time = time.time()
        if sensB and bodyB.name == 'bot_car' and bodyA.name == 'road':
            if bodyA.road_section == 1:
                bodyB.cross_time = time.time()

        if sensA and bodyA.name == 'bot_car' and (bodyB.name in {'car', 'bot_car'}):
            # if self._priority_check(bodyA.path, bodyB.path):
            if fixB == 'body':
                bodyA.stop = True
        if sensB and bodyB.name == 'bot_car' and (bodyA.name in {'car', 'bot_car'}):
            # if self._priority_check(bodyB.path, bodyA.path):
            if fixA == 'body':
                bodyB.stop = True

        # Processing Collision:
        if (bodyA.name in {'car', 'wheel'}) and (bodyB.name in {'car', 'bot_car', 'sidewalk'}):
            if fixB != 'sensor':
                bodyA.collision = True
        if (bodyA.name in {'car', 'bot_car', 'sidewalk'}) and (bodyB.name in {'car', 'wheel'}):
            if fixA != 'sensor':
                bodyB.collision = True

        # processing tiles:
        if (bodyA.name in {'car'}) and (bodyB.name in {'tile'}):
            if not bodyB.road_visited:
                self.env.reward += REWARD_TILES
                bodyB.road_visited = True
        if (bodyA.name in {'tile'}) and (bodyB.name in {'car'}):
            if not bodyA.road_visited:
                self.env.reward += REWARD_TILES
                bodyA.road_visited = True

        # processing targets:
        if (bodyA.name in {'car'}) and (bodyB.name in {'goal'}):
            bodyB.finish = True
        if (bodyA.name in {'goal'}) and (bodyB.name in {'car'}):
            bodyA.finish = True

    def EndContact(self, contact):
        sensA = contact.fixtureA.sensor
        sensB = contact.fixtureB.sensor

        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData

        fixA = contact.fixtureA.userData
        fixB = contact.fixtureB.userData

        if sensA and bodyA.name == 'car' and bodyB.name == 'road':
            if bodyB.road_section in bodyA.penalty_sec:
                bodyA.penalty = False
        if sensB and bodyB.name == 'car' and bodyA.name == 'road':
            if bodyA.road_section in bodyB.penalty_sec:
                bodyB.penalty = False

        # Behaviour on crossroads:
        if sensA and bodyA.name == 'bot_car' and bodyB.name == 'road':
            if bodyB.road_section == 1:
                bodyA.cross_time = float('inf')
        if sensB and bodyB.name == 'bot_car' and bodyA.name == 'road':
            if bodyA.road_section == 1:
                bodyB.cross_time = float('inf')

        if sensA and bodyA.name == 'bot_car' and (bodyB.name in {'car', 'bot_car'}):
            if fixB == 'body':
                bodyA.stop = False
        if sensB and bodyB.name == 'bot_car' and (bodyA.name in {'car', 'bot_car'}):
            if fixA == 'body':
                bodyB.stop = False

        # Processing Collision:
        if (bodyA.name in {'car', 'wheel'}) and (bodyB.name in {'car', 'bot_car', 'sidewalk'}):
            if fixB != 'sensor':
                bodyA.collision = False
        if (bodyA.name in {'car', 'bot_car', 'sidewalk'}) and (bodyB.name in {'car', 'wheel'}):
            if fixA != 'sensor':
                bodyB.collision = False
