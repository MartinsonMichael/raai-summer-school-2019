from gym.envs.registration import register
from .environment import CarRacing
from .car import DummyCar

register(
    id='CarIntersect-v5',
    entry_point='CarRacing',
)
