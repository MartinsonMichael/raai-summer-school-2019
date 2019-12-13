import json

class Rewarder:
    """
    Class to define reward policy.
    """
    def __init__(self, settings_file):
        """
        maybe you want to store some data from step to step
        """
        self._settings_reward = None
        self._settings_done = None
        self._load_settings(settings_file)

    def _load_settings(self, file_path):
        with open(file_path, "r") as read_file:
            _settings = json.load(read_file)
        print(f"Use reward settings: {_settings['name']}")
        self._settings_reward = _settings['reward']
        self._settings_done = _settings['done']

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

        step_reward += car_stats['new_tiles_count'] * self._settings_reward['new_tiles_count']
        step_reward += car_stats['speed'] * self._settings_reward['speed_per_point']
        step_reward += car_stats['time'] * self._settings_reward['time_per_point']

        for is_item in ['is_collided', 'is_finish', 'is_out_of_track', 'is_out_of_map', 'is_out_of_road']:
            if car_stats[is_item]:
                step_reward += self._settings_reward[is_item]

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

        for item in self._settings_done['true_flags_to_done']:
            if car_stats[item]:
                done = True
                break

        return done
