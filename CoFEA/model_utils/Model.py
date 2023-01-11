# Model.py
# Kordel France
########################################################################################################################
# This file establishes a class for an object called Model, which can be used to store hyperparameters of a ML model.
########################################################################################################################


class Model():
    def __init__(self,
                 function,
                 algorithm: str,
                 track_file: str,
                 lap_count: int,
                 velocity_min: int,
                 velocity_max: int,
                 gamma: float,
                 nu: float,
                 acceleration_rate: float,
                 acceleration_misfire_rate: float,
                 reset_after_crash: bool,
                 epochs: int,
                 episodes: int,
                 q_stability_error: float,
                 epoch_threshold: int,
                 update_steps: int,
                 policy: list):

        self.function = function
        self.algorithm = algorithm
        self.track_file = track_file
        self.lap_count = lap_count
        self.velocity_min = velocity_min
        self.velocity_max = velocity_max
        self.gamma = gamma
        self.nu = nu
        self.acceleration_rate = acceleration_rate
        self.acceleration_misfire_rate = acceleration_misfire_rate
        self.reset_after_crash = reset_after_crash
        self.epochs = epochs
        self.episodes = episodes
        self.q_stability_error = q_stability_error
        self.epoch_threshold = epoch_threshold
        self.update_steps = update_steps
        self.policy = policy


    def train_agent(self):
        # epochs, steps = self.function(False)
        # return epochs, steps
        policy = self.function()
        self.policy = policy
        return policy


    def print_hyperparameters(self):
        output_string: str = ''
        output_string += '_________________________________________________________________\n'
        output_string += f'Hyperparameters for {self.algorithm} Algorithm on {self.track_file}\n'
        output_string += f'\t# epochs: {self.epochs}\n'
        output_string += f'\t# episodes: {self.episodes}\n'
        output_string += f'\tgamma (discount rate): {self.gamma}\n'
        output_string += f'\tnu (learning rate): {self.nu}\n'
        output_string += f'\t# of laps: {self.lap_count}\n'
        output_string += f'\tmin velocity: {self.velocity_min}\n'
        output_string += f'\tmax velocity: {self.velocity_max}\n'
        output_string += f'\treset to initial position after crash: {self.reset_after_crash}'
        output_string += f'\tP(acceleration misfire) = {self.acceleration_misfire_rate}\n\n'
        print(output_string)
        return output_string
