from gym.envs.registration import register
from gym_road_cars.environment import CarRacing
from gym_road_cars.car import DummyCar
from gym_road_cars.utils import DataSupporter
from gym_road_cars.cvat import CvatDataset


ENV_NAME = 'CarIntersect-v5'
try:
    register(
        id=ENV_NAME,
        entry_point='CarRacing',
    )
    print(f'successfully register gym env \'{ENV_NAME}\'')
except:
    print(f'fail to register gym env \'{ENV_NAME}\'')
