import numpy as np
from random import random, shuffle

import CoFEA.config as config
import CoFEA.environments.track_files.track_config as track_config


def categorical_sample(prob_n, np_random: np.random.Generator):
    """Sample from categorical distribution where each row specifies class probabilities."""
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return np.argmax(csprob_n > np_random.random())


def construct_environment(input_file):
    """
    Constructs the agent environment (the racetrack) from the input file.
    :param input_file: str - the name of the file to build env from
    """
    env_space = []

    # configured specifically for demonstration purposes
    if config.DEMO_MODE and config.DEBUG_MODE:
        # with open(f'io_files/{input_file}', 'r') as track_file:
        with open(f'{config.IO_DIRECTORY}/{input_file}', 'r') as track_file:
            env_space_data = track_file.readlines()
        track_file.close()

        for index, track_line in enumerate(env_space_data):
            if index > 0:
                track_line = track_line.strip()
                if track_line == '':
                    continue
                env_line = [x for x in track_line]
                # env_space.append(env_line)
                env_space.append([x for x in track_line])

    # for monte carlo simulation or other analysis
    else:
        with open(f'{config.IO_DIRECTORY}/{input_file}', 'r') as track_file:
            env_space_data = track_file.readlines()
        track_file.close()

        for index, track_line in enumerate(env_space_data):
            if index > 0:
                track_line = track_line.strip()
                if track_line == '':
                    continue
                env_line = [x for x in track_line]
                # env_space.append(env_line)
                env_space.append([x for x in track_line])

    return env_space


def get_track_shape(input_file) -> tuple:
    """
    Gets shape of the racetrack from the input file.
    :param input_file: str - the name of the file to build env from
    """
    env_space = []
    y_shape: int = 1
    # y_shape: int = 0

    # configured specifically for demonstration purposes
    if config.DEMO_MODE and config.DEBUG_MODE:
        with open(f'{config.IO_DIRECTORY}/{input_file}', 'r') as track_file:
        # with open(f'io_files/{input_file}', 'r') as track_file:
            env_space_data = track_file.readlines()
        track_file.close()

        for index, track_line in enumerate(env_space_data):
            if index > 0:
                track_line = track_line.strip()
                if track_line == '':
                    continue
                env_line = [x for x in track_line]
                # env_space.append(env_line)
                env_space.append([x for x in track_line])
                x_shape = len(track_line)
                y_shape += 1

    # for monte carlo simulation or other analysis
    else:
        with open(f'{config.IO_DIRECTORY}/{input_file}', 'r') as track_file:
            env_space_data = track_file.readlines()
        track_file.close()

        for index, track_line in enumerate(env_space_data):
            if index > 0:
                track_line = track_line.strip()
                if track_line == '':
                    continue
                env_line = [x for x in track_line]
                # env_space.append(env_line)
                env_space.append([x for x in track_line])
                x_shape = len(track_line)
                y_shape += 1

    return (x_shape, y_shape)


def get_new_initial_state(input_file):
    """
    Gets a new starting position on the track.
    :param env_data: list - the racetrack
    """
    env_data = construct_environment(input_file)
    initial_states: list = []
    for y, row in enumerate(env_data):
        for x, col in enumerate(row):
            if col == track_config.STATE_INITIAL:
                initial_states += [(y, x)]

    # select a random position on the starting line
    shuffle(initial_states)
    return initial_states[0]


def get_track_boundaries(input_file):
    """
    Gets a new starting position on the track.
    :param env_data: list - the racetrack
    """
    env_data = construct_environment(input_file)
    initial_states: list = []
    for y, row in enumerate(env_data):
        for x, col in enumerate(row):
            if col == track_config.STATE_COLLISION:
                initial_states += [(y, x)]

    # select a random position on the starting line
    print(initial_states)
    return initial_states
