from gym.envs.registration import register
from gym_car_itersect.hack_env__latest import CarRacingHackatonContinuous2

try:
    register(
        id='CarIntersect-v3',
        entry_point='gym_car_intersect:CarRacingHackatonContinuous2',
    )
    print('register CarIntersect-v3')
except:
    print('fail to register CarIntersect-v3')

