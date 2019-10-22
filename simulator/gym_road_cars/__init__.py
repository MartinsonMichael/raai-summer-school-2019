from gym.envs.registration import register
from .environment import CarRacing

register(
    id='CarIntersect-v5',
    entry_point='CarRacing',
)
