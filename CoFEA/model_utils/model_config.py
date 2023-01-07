# model_config.py
# Kordel France
########################################################################################################################
# This file contains hyperparameters to establish a reinforcement learning agent
########################################################################################################################


TRACK_NAME: str = 'L-track.txt'
LAP_COUNT: int = 10
VELOCITY_MIN: int = -5          # velocity lower bound
VELOCITY_MAX: int = 5           # velocity upper bound
GAMMA: float = 0.9              # discount rate
NU: float = 0.25                # learning rate
ACCELERATION_RATE: float = 0.8  # probability that the acceleration control succeeds
ACCELERATION_MISFIRE_RATE: float = 1 - ACCELERATION_RATE  # probability that the acceleration control fails
EPOCHS: int = 5
RESET_AFTER_CRASH: bool = True
EPISODES: int = 5
Q_STABILITY_ERROR: float = 0.001
EPOCH_THRESHOLD: int = 100
UPDATE_STEPS: int = 500
STATE_SPACE_VELOCITY = range(VELOCITY_MIN, VELOCITY_MAX + 1)
ACTION_SPACE = [(-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 0),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1)]
