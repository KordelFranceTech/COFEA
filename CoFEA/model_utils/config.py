# config.py
# Kordel France
########################################################################################################################
# This file contains hyperparameters to control debugging features
########################################################################################################################


# set true if you want to see a verbose data log to diagnose issues
DEBUG_MODE = True
DEMO_MODE = True
# directory name storing all of the i/o files
IO_DIRECTORY = '../../Lab5/io_files'

# parameters for reading the track file and building the track sim
AGENT_SYMBOL: str = 'O'
STATE_INITIAL: str = 'S'
STATE_TERMINAL: str = 'F'
STATE_COLLISION: str = '#'
STATE_TRACK: str = '.'
