from gym.envs.registration import register
from envs.gym_road_cars.environment import CarRacing
from envs.gym_road_cars.car import DummyCar
from envs.gym_road_cars.utils import DataSupporter
from envs.gym_road_cars.cvat import CvatDataset

try:
    register(
        id='CarIntersect-v5',
        entry_point='envs.gym_road_cars:CarRacing',
    )
    print(f'successfully register gym env \'CarIntersect-v5\'')
except:
    print(f'fail to register gym env \'CarIntersect-v5\'')
