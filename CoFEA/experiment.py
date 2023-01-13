from CoFEA.environments import env_frozen_lake, env_cliff_walking, env_racetrack, environment


def get_map_size():
    if MAP_SIZE == "L":
        initial = [9, 1]
    elif MAP_SIZE == "R":
        initial = [26, 1]
    elif MAP_SIZE == "P":
        initial = [28, 1]
    elif MAP_SIZE == "small":
        initial = [0, 0]
    elif MAP_SIZE == "large":
        initial = [0, 0]
    elif MAP_SIZE == "mega":
        initial = [0, 0]
    elif MAP_SIZE == "giga":
        initial = [0, 0]
    return initial


def get_bounds(map_size: str):
    bounds = []
    if map_size == "small":
        bounds = [(0,3), (0,11)]
    elif map_size == "large":
        bounds = [(0, 7), (0, 23)]
    elif map_size == "mega":
        bounds = [(0, 15), (0, 47)]
    elif map_size == "giga":
        bounds = [(0, 31), (0, 95)]
    elif map_size == "L":
        bounds = [(0, 11), (0, 37)]
    elif map_size == "R":
        bounds = [(0, 28), (0, 30)]
    elif map_size == "P":
        bounds = [(0, 30), (0, 30)]
    return bounds


def test_func(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total


def compute_reward(state: int):
    reward = 48 - (((4 - int(state / 12)) ** 2) + ((12 - (state % 12)) ** 2)) ** 0.5
    if state in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
        reward = -100
    return reward


MAP_SIZE: str = "R"
BOUNDS = get_bounds(MAP_SIZE)
NUM_PARTICLES: int = 8
MAX_ITER: int = 100
FEA_RUNS: int = 5
MAP_FILE = env_racetrack
MAP = env_racetrack.Racetrack()
ENV = {"map":MAP, "type": MAP_SIZE}
environment.set_environment(ENV)
INITIAL: list = get_map_size()
TERMINAL_STATE: list = MAP_FILE.TERMINAL_STATE
EPSILON = 0.1
TOTAL_EPISODES = 1
MAX_STEPS = 10
ALPHA = 0.5
GAMMA = 1
EPISODE_REWARD = 0
REWARDS_TRACKER: dict  = {
	'SarsaAgent': [],
	'QLearningAgent': [],
	'ExpectedSarsaAgent': [],
	'SarsaFeaAgent': [],
	'QLearningFeaAgent': [],
	'ExpectedSarsaFeaAgent': []
}

TRAJECTORIES: int = 3
COTRAIN_ITERS: int = 3
COTRAIN_GAMMA: float = 0.8

COUNTER: int = 0
SWARM_UPDATE_COUNTER: int = 0





