from gym.envs.registration import register

register(
    id='CarIntersect-v0',
    entry_point='gym_car_intersect.envs:CarRacing', #this is function which you want to code to behave
)