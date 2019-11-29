from gym.envs.registration import register
from envs.gym_car_intersect_fixed.hack_env__latest import CarRacingHackatonContinuousFixed

try:
    register(
        id='CarIntersect-v52',
        entry_point='envs.gym_car_intersect_fixed:CarRacingHackatonContinuousFixed',
    )
    print('successfully register gym env \'CarIntersect-v52\'')
except:
    print('fail to register gym env \'CarIntersect-v52\'')
