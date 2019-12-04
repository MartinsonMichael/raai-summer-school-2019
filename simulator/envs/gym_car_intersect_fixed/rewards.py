class Rewarder:
    """
    Class to define reward policy.
    """
    def __init__(self):
        """
        maybe you want to store some data from step to step
        """
        pass

    def get_step_reward(self, car_stats) -> float:
        """
        function to compute reward for current step
        :param car_stats: dist with car stats
        keys are:
        'new_tiles_count': integer, how many track point achieve agent at last step
        'is_finish': bool
        'is_collided': bool
        'is_out_of_track': bool, car not on chosen track
        'is_on_cross_road': bool, car is on cross road
        'is_out_of_map': bool
        'is_out_of_road': bool, car not on any road
        'speed': float, car linear velocity
        'time': integer, steps from car creating
        :return: reward for current step
        """
        step_reward = 0.0

        step_reward += car_stats['new_tiles_count'] * 0.5
        step_reward += car_stats['speed'] * 0.075
        step_reward += car_stats['time'] * -0.001

        if car_stats['is_collided']:
            step_reward += -1.0

        if car_stats['is_finish']:
            step_reward += 5.0

        if car_stats['is_out_of_track']:
            step_reward += -2.0

        if car_stats['is_out_of_map']:
            step_reward += -5.0

        if car_stats['is_out_of_road']:
            step_reward += -5.0

        return step_reward

    def get_step_done(self, car_stats) -> bool:
        """
        function to compute done flag for current step
        :param car_stats: dist with car stats
        keys are:
        'new_tiles_count': integer, how many track point achieve agent at last step
        'is_finish': bool
        'is_collided': bool
        'is_out_of_track': bool, car not on chosen track
        'is_on_cross_road': bool, car is on cross road
        'is_out_of_map': bool
        'is_out_of_road': bool, car not on any road
        'speed': float, car linear velocity
        'time': integer, steps from car creating
        :return: bool, done flag for current step
        """
        done = False

        if car_stats['is_out_of_map']:
            done = True

        if car_stats['is_out_of_road']:
            done = True

        return done