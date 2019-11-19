import typing
import time

from Box2D.b2 import contactListener
from .reward_constants import *


class ContactListener(contactListener):
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
        print('begin contact')
        print(contact)

    def EndContact(self, contact):
        print('edn contact')
        print(contact)
