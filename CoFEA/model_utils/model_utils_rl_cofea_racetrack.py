import copy
from CoFEA.model_utils.Model import Model
from CoFEA.model_utils import model_config
from CoFEA.model_utils import demo_config_q_learning
from CoFEA.model_utils import config
from FEA.FEA.factorarchitecture import FactorArchitecture

from benchmark import get_best_policy, get_best_policy_osi, get_benchmark_policy, print_policy_string, k
import numpy as np
import random
from random import shuffle
import os
import time
from copy import deepcopy

model_config = model_config

"""
State space:
    - all values of lateral position, x
    - all values of longitudinal position, y
    - all values of lateral velocity, xx
    - all values of longitudinal velocity, yy

Action space:
    - all values of lateral acceleration, xxx
    - all values of longitudinal acceleration, yyy

Model parameters:
    - s = initial state
    - a = initial action
    - s_prime = next state
    - a_prime = next action   
"""

TRACK_NAME: str = model_config.TRACK_NAME
AGENT_SYMBOL: str = 'O'
STATE_INITIAL: str = 'S'
STATE_TERMINAL: str = 'F'
STATE_COLLISION: str = '#'
STATE_TRACK: str = '.'
LAP_COUNT: int = model_config.LAP_COUNT
VELOCITY_MIN: int = model_config.VELOCITY_MIN               # velocity lower bound
VELOCITY_MAX: int = model_config.VELOCITY_MAX               # velocity upper bound
GAMMA: float = model_config.GAMMA                           # discount rate
NU: float = model_config.NU                                 # learning rate
ACCELERATION_RATE: float = model_config.ACCELERATION_RATE   # probability that the acceleration control succeeds
ACCELERATION_MISFIRE_RATE: float = 1 - ACCELERATION_RATE    # probability that the acceleration control fails
EPOCHS: int = model_config.EPOCHS
RESET_AFTER_CRASH: bool = model_config.RESET_AFTER_CRASH
EPISODES: int = model_config.EPISODES
Q_STABILITY_ERROR: float = model_config.Q_STABILITY_ERROR
EPOCH_THRESHOLD: int = model_config.EPOCH_THRESHOLD
UPDATE_STEPS: int = model_config.UPDATE_STEPS
state_space_velocity = model_config.STATE_SPACE_VELOCITY
action_space = model_config.ACTION_SPACE




def reset_hyperparameters():
    """
    Resets configuration and hyperparameters so that a new configuration may be passed immediately after.
    """
    global TRACK_NAME
    global AGENT_SYMBOL
    global STATE_INITIAL
    global STATE_TERMINAL
    global STATE_COLLISION
    global STATE_TRACK
    global LAP_COUNT
    global VELOCITY_MIN
    global VELOCITY_MAX
    global GAMMA
    global NU
    global ACCELERATION_RATE
    global ACCELERATION_MISFIRE_RATE
    global EPOCHS
    global RESET_AFTER_CRASH
    global EPISODES
    global Q_STABILITY_ERROR
    global EPOCH_THRESHOLD
    global UPDATE_STEPS
    global state_space_velocity
    global action_space

    TRACK_NAME = model_config.TRACK_NAME
    AGENT_SYMBOL = config.AGENT_SYMBOL
    STATE_INITIAL = config.STATE_INITIAL
    STATE_TERMINAL = config.STATE_TERMINAL
    STATE_COLLISION = config.STATE_COLLISION
    STATE_TRACK = config.STATE_TRACK
    LAP_COUNT = model_config.LAP_COUNT
    VELOCITY_MIN = model_config.VELOCITY_MIN  # velocity lower bound
    VELOCITY_MAX = model_config.VELOCITY_MAX  # velocity upper bound
    GAMMA = model_config.GAMMA  # discount rate
    NU = model_config.NU  # learning rate
    ACCELERATION_RATE = model_config.ACCELERATION_RATE  # probability that the acceleration control succeeds
    ACCELERATION_MISFIRE_RATE = 1 - ACCELERATION_RATE  # probability that the acceleration control fails
    EPOCHS = model_config.EPOCHS
    RESET_AFTER_CRASH = model_config.RESET_AFTER_CRASH
    EPISODES = model_config.EPISODES
    Q_STABILITY_ERROR = model_config.Q_STABILITY_ERROR
    EPOCH_THRESHOLD = model_config.EPOCH_THRESHOLD
    UPDATE_STEPS = model_config.UPDATE_STEPS
    state_space_velocity = model_config.STATE_SPACE_VELOCITY
    action_space = model_config.ACTION_SPACE


def construct_environment(input_file):
    """
    Constructs the agent environment (the racetrack) from the input file.
    :param input_file: str - the name of the file to build env from
    """
    env_space = []

    # configured specifically for demonstration purposes
    if config.DEMO_MODE and config.DEBUG_MODE:
        with open(f'./environments/track_files/{input_file}', 'r') as track_file:
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


def display_current_state(env_data, state=[0, 0]):
    """
    Prints the current state of the agent to the console - basically the whole track with the car position.
    :param env_data: list - the racetrack
    :param state: list - an initializing state
    """
    # get current state
    current_state = env_data[state[0]][state[1]]
    env_data[state[0]][state[1]] = AGENT_SYMBOL

    # pause console printing for readability
    if config.DEMO_MODE:
        time.sleep(0.3)
    os.system('cls')

    # build the ractrack and place the car
    for index_i in range(0, len(env_data)):
        env_str: str = ''
        env_line = env_data[index_i]
        for index_j in range(0, len(env_line)):
            env_str += env_line[index_j]
        if config.DEMO_MODE:
            print(env_str)
    env_data[state[0]][state[1]] = current_state


def get_new_initial_state(env_data):
    """
    Gets a new starting position on the track.
    :param env_data: list - the racetrack
    """
    initial_states: list = []
    for y, row in enumerate(env_data):
        for x, col in enumerate(row):
            if col == STATE_INITIAL:
                initial_states += [(y, x)]

    # select a random position on the starting line
    shuffle(initial_states)
    return initial_states[0]


def update_state_position(position, velocity_vector):
    """
    Update x- and y- components of the agent position for new state given velocity vector.
    :param position: list - the x- and y- coords of position
    :param velocity_vector: list = the velocity vector to move position by
    """
    y, x = position[0], position[1]
    yy, xx = velocity_vector[0], velocity_vector[1]
    x_prime = x + xx
    y_prime = y + yy
    return y_prime, x_prime


def update_state_velocity(velocity_vector, acc_vector, velocity_min, velocity_max):
    """
    Update x- and y- components of the agent velocity for new state given acceleration vector.
    :param velocity_vector: list - the current velocity
    :param acc_vector: list - the acceleration vector to alter velocity by
    :param velocity_min: int - minimum allowed velocity
    :param velocity_max: int - maximum allowed velocity
    """
    yy = velocity_vector[0] + acc_vector[0]
    xx = velocity_vector[1] + acc_vector[1]
    if xx < velocity_min:
        xx = velocity_min
    if xx > velocity_max:
        xx = velocity_max
    if yy < velocity_min:
        yy = velocity_min
    if yy > velocity_max:
        yy = velocity_max

    return yy, xx


def check_collision_with_bresenham_algorithm(y, x, y_prime, x_prime, env_data):
    """
    Use Bresenham algorithm as suggested by project documentation to determine collision
    :param y: int -current y position
    :param x: int - current x position
    :param y_prime: int - y pos of next state
    :param x_prime: int - x pos of next state
    :env_data: list - the racetrack
    """
    # init a points list with a slope and slope error
    points_list: list = []
    m_prime = 2 * (y_prime - y)
    m_prime_error = m_prime - (x_prime - x)
    y_line = y_prime

    # step among the points and calculate slope
    for x_line in range(x, x_prime + 1):
        points_list.append((x_line, y_line))
        m_prime_error += m_prime
        if m_prime_error >= 0:
            y_line += 1
            m_prime_error = m_prime_error - 2 * (x_prime - x)

    # if slope determines a collision, return and update q-value
    if env_data[y][x] == STATE_COLLISION:
        if config.DEBUG_MODE and config.DEMO_MODE:
            print(f'COLLISION')
        return



def search_for_actionable_track_spaces(env_data, y_crash, x_crash, yy=0, xx=(0),
                                       open_positions=[STATE_TRACK, STATE_INITIAL, STATE_TERMINAL]):
    """
    Local search for positions that are nearby and not boundaries
    :param env_data: list - the racetrack
    :param y_crash: int - y pos of crash
    :param x_crash: int - x pos of crash
    :param yy: int - y velocity
    :param xx: int - x velocity
    :param open_positions: list - list of states that indicate open position types
    """
    # define boundaries for radius
    env_lat = len(env_data)
    env_lon = len(env_data[0])
    max_radius = max(env_lat, env_lon)

    # do local search over radius
    for radius in range(0, max_radius):
        if yy == 0:
            y_span = range(-radius, radius + 1)
        elif yy < 0:
            y_span = range(0, radius + 1)
        else:
            y_span = range(-radius, 1)

        # check each candidate within y-component radius and make sure close enough
        for dy in y_span:
            y = y_crash + dy
            x_span = radius - abs(dy)

            if xx == 0:
                x_span = range(x_crash - x_span, x_crash + x_span + 1)
            elif xx < 0:
                x_span = range(x_crash, x_crash + x_span + 1)
            else:
                x_span = range(x_crash - x_span, x_crash + 1)

            # check each candidate within x-component radius and make sure close enough
            # if so, go to that position
            for x in x_span:
                if y < 0 or y >= env_lat:
                    continue
                if x < 0 or x >= env_lon:
                    continue
                if env_data[y][x] in open_positions:
                    return (y, x)
    return


def act(y_prev, x_prev, yy_prev, xx_prev, acc_vector, env_data, is_deterministic=False,
        reset_after_crash: bool = False):
    """
    Allows agent to act within its environment, showing some exploration ability as well as policy adherence.
    :param y_prev: int - previous y pos
    :param x_prev: int - prvious x pos
    :param yy_prev: int - previous y velocity
    :param xx_prev: int - previous x velocity
    :param acc_vector: list - x- and y- components of acceleration for action
    :param env_data: list - the racetrack
    :param is_deterministic: bool - flag that allows me to turn on and off explorability
    :param reset_after_crash: bool - flat that allows me to indicate what happens after the crash occurs
    """

    # this induces some explorability for the agent
    # explorability randomly occurs
    # get a random number and if greater than acceleration rate, randomly accelerate
    # otherwise, adhere to policy and select action from there
    # definitely exists opportunity to optimize
    if not is_deterministic:
        if random() > ACCELERATION_RATE:
            #### BREAKPOINT ###
            acc_vector = (0, 0)
            # random acceleration vector has issues at times
            # opportunity to optimize
            # xxx = randint(-1, 1)
            # yyy = randint(-1, 1)
            # acc_vector(xxx, yyy)

    # the previous velocity vector
    velocity_vector = (yy_prev, xx_prev)

    # position of the nex state
    yy_prime, xx_prime = update_state_velocity(velocity_vector, acc_vector, VELOCITY_MIN, VELOCITY_MAX)

    # position of previous state
    position = (y_prev, x_prev)
    y, x = update_state_position(position, velocity_vector)

    # find actionable position within proposed position
    y_prime, x_prime = search_for_actionable_track_spaces(env_data, y, x, yy_prime, xx_prime)

    # check to make sure we aren't selecting to travel to same point we are on
    if y_prime != y or x_prime != x:
        if reset_after_crash and env_data[y_prime][x_prime] != STATE_TERMINAL:

            # either we crashed and crash policy indicates reset, or we reached the finish line
            y_prime, x_prime = get_new_initial_state(env_data)
        yy_prime, xx_prime = 0, 0

    return y_prime, x_prime, yy_prime, xx_prime


def update_policy_given_q(cols, rows, state_space_velocity, q, action_space):
    """
    Perform policy update and adjust the console output to reflect the policy update visually.
    :param cols: int - column count for track file
    :param rows: int - row count for track file
    :param state_space_velocity: list - the list of possible state values for velocity
    :param q: float - the current q-value (reward)
    :param action_space: list - a list of possible actions to take
    """
    # init empty policy
    policy = {}
    for y in range(rows):
        for x in range(cols):
            for yy in state_space_velocity:
                for xx in state_space_velocity:
                    # set the policy value at this scalar location
                    policy[(y, x, yy, xx)] = action_space[np.argmax(q[y][x][yy][xx])]
                    # state_vector = (y, x, yy, xx)
                    # q_value = np.argmax(q[y][x][yy][xx])
                    # policy[state_vector] = action_space[q_value]

    # return the updated policy
    return policy


########################################################################################################################
########################################################################################################################
# MARK: -  MAIN LEARNING ALGORITHM BEGIN
########################################################################################################################
########################################################################################################################

def f(states, reset_after_crash: bool = False, reward: float = 0.0, epochs=(EPOCHS), episodes=EPISODES):
    rewards = 0
    total_episodes = 1
    max_steps = 10
    totalReward = {
        'SarsaFeaAgent': [],
        'QLearningFeaAgent': [],
        'ExpectedSarsaFeaAgent': [],
        'Model': []
    }

    rows = len(ENV)
    cols = len(ENV[0])

    # initialize all Q(x, a) arbitrarily
    q_value = [[[[[random() for _ in action_space] for _ in state_space_velocity] for _ in (state_space_velocity)] for _ in line] for line in ENV]
    for y in range(0, rows):
        for x in range(0, cols):
            if ENV[y][x] == STATE_TERMINAL:
                for yy in state_space_velocity:
                    for xx in state_space_velocity:
                        for action_i, action in enumerate(action_space):
                            q_value[y][x][yy][xx][action_i] = reward

    for y in range(0, rows):
        for x in range(0, cols):
            if ENV[y][x] == STATE_TERMINAL:
                q_value[y][x] = [[[reward for _ in action_space] for _ in (state_space_velocity)] for _ in state_space_velocity]

    # initialize all states
    y = np.random.choice(range(0, rows))
    x = np.random.choice(range(0, cols))
    yy = np.random.choice(state_space_velocity)
    xx = np.random.choice(state_space_velocity)

    for i in range(len(states)):
        # perform updates for each episode
        for episode in range(total_episodes):
            episodeReward = 0
            t = 0
            if ENV[y][x] == STATE_TERMINAL:
                break
            if ENV[y][x] == STATE_COLLISION:
                break
            check_collision_with_bresenham_algorithm(y, x, 0, 0, ENV)

            #### BREAKPOINT 1 ###
            # choose action a using policy given from greedy Q
            action = np.argmax(q_value[y][x][yy][xx])

            while t < max_steps:

                #### BREAKPOINT 2 ###
                # take action a, observe r and s'
                y_prime, x_prime, yy_prime, xx_prime = act(y, x, yy, xx, action_space[action], ENV, reset_after_crash=reset_after_crash)
                reward = -1

                # update Q(s, a)
                q_value_prev = q_value[y][x][yy][xx][action]
                q_value_max = max(q_value[y_prime][x_prime][yy_prime][xx_prime])

                #### BREAKPOINT 3 ###
                # generation of <state, action, reward, state, action> tuple
                q_value[y][x][yy][xx][action] = ((1 - NU) * q_value_prev + NU * (reward + GAMMA * q_value_max))
                update_policy_given_q(cols, rows, state_space_velocity, q_value, action_space)

                if config.DEMO_MODE:
                    print(f'\tprevious q-value: {q_value_prev}\n\tupdated q-value: {q_value}')

                #### BREAKPOINT 4 ###
                # new states become current states
                y, x, yy, xx = y_prime, x_prime, yy_prime, xx_prime

                t += 1
                episodeReward += q_value[y][x][yy][xx][action]

            totalReward[type(AGENT).__name__].append(episodeReward)
        rewards += np.mean(totalReward[type(AGENT).__name__])
    reward_error = -float(rewards / float(len(states)) ** 2)
    return reward_error


########################################################################################################################
########################################################################################################################
# MARK: -  MAIN LEARNING ALGORITHM END
########################################################################################################################
########################################################################################################################


def initialize_race(env_data, policy, reset_after_crash, update_steps=UPDATE_STEPS):
    """
    Sets up the race environment and configures the algorithm for learning based on hyperparameters and config file.
    :param env_data: list - the racetrack
    :param reset_after_crash: bool - the crash policy
    :param update_steps: bool - the number of update steps to allow
    :return update_steps: int - the number of update steps left to complete
    """
    # create a new copy of the console output for alteration and reprint
    env_view = deepcopy(env_data)

    # initialize car at starting line
    position_init = get_new_initial_state(env_view)

    # define kinematics
    y, x = position_init
    yy, xx = 0, 0
    clock_count: int = 0

    # build the UI and get current action to take from policy
    for index_i in range(0, update_steps):
        display_current_state(env_view, state=[y, x])
        action = policy[(y, x, yy, xx)]

        if env_data[y][x] == STATE_TERMINAL:
            return index_i

        y, x, yy, xx = act(y, x, yy, xx, action, env_data, reset_after_crash=reset_after_crash)

        if xx == 0 and yy == 0:
            clock_count += 1
        else:
            clock_count = 0

        if clock_count == 5:
            return update_steps

    return update_steps


def build_trajectories(agent, e, config):
    model = copy.deepcopy(agent)
    env = copy.deepcopy(e)
    q_table = model.Q
    trajectories: list = []
    n, max_steps = config.epochs, config.max_steps
    rewards = []
    num_steps = []
    state1 = env.reset()
    action1 = model.choose_action(state1)
    for episode in range(n):
        total_reward = 0
        for i in range(max_steps):
            state2, reward, done, info = env.step(action1)
            action2 = model.choose_action(state2)
            # trajectory.append([state1, action1, state2, action2])
            trajectories.append([state1, action1])

            state1 = state2
            action1 = action2

            total_reward += reward
            if done:
                rewards.append(total_reward)
                num_steps.append(i + 1)
                break
    env.close()
    del env
    del model
    return trajectories


def train_q_learning_algorithm():
    epochs = EPOCHS
    epoch_list: list = []
    step_list: list = []
    racetrack_name = f'{TRACK_NAME}'
    racetrack = construct_environment(racetrack_name)
    track_name = TRACK_NAME

    step_count: int = 0
    # current_policy = get_best_policy(q_table=model.Q)
    current_policy = perform_q_learning(racetrack, reset_after_crash=RESET_AFTER_CRASH, epochs=epochs)
    # for lap in range(0, LAP_COUNT):
    #     step_count += initialize_race(racetrack, current_policy, reset_after_crash=(RESET_AFTER_CRASH))
    return current_policy



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
    else:
        bounds = [(0, 11), (0, 37)]
    return bounds


def compute_reward(state: int):
    reward = 48 - (((4 - int(state / 12)) ** 2) + ((12 - (state % 12)) ** 2)) ** 0.5
    if state in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
        reward = -100
    return reward



# --- MAIN ---------------------------------------------------------------------+
class Particle(object):
    def __init__(self, f, size, position=None, factor=None, global_solution=None, lbest_pos=None):
        self.f = f
        self.lbest_fitness = float('inf')
        self.dim = size
        self.factor = factor
        if position is None:
            self.position = np.random.uniform(BOUNDS[0][0], BOUNDS[1][1], size=size)
            self.lbest_position = np.array([x for x in self.position])
        elif position is not None:
            self.position = position
            self.lbest_position = lbest_pos
            self.lbest_fitness = self.calculate_fitness(global_solution, lbest_pos)
        self.velocity = np.zeros(size)
        self.fitness = self.calculate_fitness(global_solution)

    def __le__(self, other):
        if self.fitness is float:
            return self.fitness <= other.fitness

    def __lt__(self, other):
        if self.fitness is float:
            return self.fitness < other.fitness

    def __gt__(self, other):
        if self.fitness is float:
            return self.fitness > other.fitness

    def __eq__(self, other):
        return (self.position == other.position).all()

    def __str__(self):
        return ' '.join(
            ['Particle with current fitness:', str(self.fitness), 'and best fitness:', str(self.lbest_fitness)])

    def set_fitness(self, fit):
        self.fitness = fit
        if fit < self.lbest_fitness:
            self.lbest_fitness = deepcopy(fit)
            self.lbest_position = np.array([x for x in self.position])

    def set_position(self, position):
        self.position = np.array(position)

    def update_individual_after_compete(self, global_solution=None):
        fitness = self.calculate_fitness(global_solution)
        if fitness < self.lbest_fitness:
            self.lbest_fitness = deepcopy(fitness)
        self.fitness = fitness
        return self

    def calculate_fitness(self, glob_solution, position=None):
        # print(self.position)
        if glob_solution is None:
            fitness = self.f(self.position)
        else:
            solution = [x for x in glob_solution]
            if position is None:
                for i, x in zip(self.factor, self.position):
                    solution[i] = x
            else:
                for i, x in zip(self.factor, position):
                    solution[i] = x
            fitness = self.f(np.array(solution))
        return fitness

    def update_particle(self, omega, phi, global_best_position, v_max, global_solution=None):
        self.update_velocity(omega, phi, global_best_position, v_max)
        self.update_position(global_solution)

    def update_velocity(self, omega, phi, global_best_position, v_max):
        velocity = [x for x in self.velocity]
        n = self.dim

        inertia = np.multiply(omega, velocity)
        phi_1 = np.array([random.random() * phi for i in range(n)])  # exploration
        personal_exploitation = self.lbest_position - self.position  # exploitation
        personal = phi_1 * personal_exploitation
        phi_2 = np.array([random.random() * phi for i in range(n)])  # exploration
        social_exploitation = global_best_position - self.position  # exploitation
        social = phi_2 * social_exploitation
        new_velocity = inertia + personal + social
        self.velocity = np.array([self.clamp_value(v, -v_max, v_max) for v in new_velocity])

    # def update_position(self, global_solution=None):
    #     lo, hi = BOUNDS[0][0], BOUNDS[1][1]
    #     position = self.velocity + self.position
    #     self.position = np.array([self.clamp_value(p, lo, hi) for p in position])
    #     self.fitness = self.calculate_fitness(global_solution)

    # update the particle position based off new velocity updates
    def update_position(self, global_solution=None):
        for i in range(0, self.dim):
            # print(f"position pre: {self.position_i[i]}")
            # self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            # self.position_i[i] = self.position_i[i] + 1
            self.position[i], reward, done, info = ENV.step(np.argmax(AGENT.Q.tolist()[round(self.position[i])]))
            self.position[i] += self.velocity[i]
            # if done: break
            # print(f"position post: {self.position_i[i]}")

            # adjust maximum position if necessary
            if self.position[i] > BOUNDS[i][1]:
                self.position[i] = BOUNDS[i][1]

            # adjust minimum position if neseccary
            if self.position[i] < BOUNDS[i][0]:
                self.position[i] = BOUNDS[i][0]
        self.fitness = self.calculate_fitness(global_solution)


    def clamp_value(self, to_clamp, lo, hi):
        if lo < to_clamp < hi:
            return to_clamp
        if to_clamp < lo:
            return to_clamp
        return hi


class PSO(object):
    def __init__(self, generations, population_size, function, dim, factor=None, global_solution=None, omega=0.729, phi=1.49618):
        self.pop_size = population_size
        self.pop = [Particle(function, dim, factor=factor, global_solution=global_solution) for x in range(population_size)]
        pos = [p.position for p in self.pop]
        with open('pso2.o', 'a') as file:
            file.write(str(pos))
            file.write('\n')

        self.omega = omega
        self.phi = phi
        self.f = function
        self.dim = dim
        pbest_particle = Particle(function, dim, factor=factor, global_solution=global_solution)
        pbest_particle.set_fitness(float('inf'))
        self.pbest_history = [pbest_particle]
        self.gbest = pbest_particle
        self.v_max = abs((BOUNDS[1][1] - BOUNDS[0][0]))
        self.generations = generations
        self.current_loop = 0
        self.factor = np.array(factor)
        self.global_solution = global_solution

    def find_current_best(self):
        sorted_ = sorted(np.array(self.pop), key=attrgetter('fitness'))
        return Particle(self.f, self.dim, position=sorted_[0].position, factor=self.factor,
                 global_solution=self.global_solution, lbest_pos=sorted_[0].lbest_position)

    def find_local_best(self):
        pass

    def update_swarm(self):
        if self.global_solution is not None:
            global_solution = [x for x in self.global_solution]
        else:
            global_solution = None
        omega, phi, v_max = self.omega, self.phi, self.v_max
        global_best_position = [x for x in self.gbest.position]
        for p in self.pop:
            p.update_particle(omega, phi, global_best_position, v_max, global_solution)
            for k in range(0, len(p.position)):
                state1 = round(p.position[k])
                action1 = AGENT.choose_action(state1)
                state2, reward, done, info = ENV.step(action1)
                action2 = AGENT.choose_action(state2)
                AGENT.update(state1, state2, reward, action1, action2)
                if done: break
        curr_best = self.find_current_best()
        self.pbest_history.append(curr_best)
        if curr_best.fitness < self.gbest.fitness:
            self.gbest = curr_best

    def replace_worst_solution(self, global_solution):
        # find worst particle
        self.global_solution = np.array([x for x in global_solution])
        self.pop.sort(key=attrgetter('fitness'))
        print('replacing')
        print(self.pop[-1], self.pop[0])
        partial_solution = [x for i, x in enumerate(global_solution) if i in self.factor] # if i in self.factor
        self.pop[-1].set_position(partial_solution)
        self.pop[-1].set_fitness(self.f(self.global_solution))
        curr_best = Particle(self.f, self.dim, position=self.pop[0].position, factor=self.factor,
                 global_solution=self.global_solution, lbest_pos=self.pop[0].lbest_position)
        random.shuffle(self.pop)
        if curr_best.fitness < self.gbest.fitness:
            self.gbest = curr_best

    def run(self):
        for i in range(self.generations):
            self.update_swarm()
            self.current_loop += 1
            # print(self.gbest)
        return self.gbest.position


class FEA:
    def __init__(self, function, fea_runs, generations, pop_size, factor_architecture, base_algorithm, continuous=True,
                 seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.function = function
        self.fea_runs = fea_runs
        self.base_alg_iterations = generations
        self.pop_size = pop_size
        self.factor_architecture = factor_architecture
        self.dim = factor_architecture.dim
        self.base_algorithm = base_algorithm
        self.global_solution = None
        self.global_fitness = np.inf
        self.solution_history = []
        self.set_global_solution(continuous)
        self.subpopulations = self.initialize_factored_subpopulations()

    def run(self):
        for fea_run in range(self.fea_runs):
            for alg in self.subpopulations:
                # print('NEW SUBPOPULATION\n---------------------------------------------------------------')
                # alg.run(fea_run)
                alg.run()
            self.compete()
            self.share_solution()
            print('fea run ', fea_run, self.global_fitness)

    def set_global_solution(self, continuous):
        if continuous:
            self.global_solution = np.random.uniform(BOUNDS[0][0], BOUNDS[1][1], size=self.factor_architecture.dim)
            self.global_fitness = self.function(self.global_solution)
            self.solution_history = [self.global_solution]

    def initialize_factored_subpopulations(self):
        fa = self.factor_architecture
        alg = self.base_algorithm
        return [
            alg(function=self.function, dim=len(factor), generations=self.base_alg_iterations, population_size=self.pop_size, factor=factor, global_solution=self.global_solution)
            for factor in fa.factors]

    def share_solution(self):
        """
        Construct new global solution based on best shared variables from all swarms
        """
        gs = [x for x in self.global_solution]
        print('global fitness found: ', self.global_fitness)
        print('===================================================')
        for alg in self.subpopulations:
            # update fitnesses
            alg.pop = [individual.update_individual_after_compete(gs) for individual in alg.pop]
            # set best solution and replace worst solution with global solution across FEA
            alg.replace_worst_solution(gs)

    def compete(self):
        """
        For each variable:
            - gather subpopulations with said variable
            - replace variable value in global solution with corresponding subpop value
            - check if it improves fitness for said solution
            - replace variable if fitness improves
        Set new global solution after all variables have been checked
        """
        sol = [x for x in self.global_solution]
        f = self.function
        curr_fitness = f(self.global_solution)
        for var_idx in range(self.dim):
            best_value_for_var = sol[var_idx]
            for pop_idx in self.factor_architecture.optimizers[var_idx]:
                curr_pop = self.subpopulations[pop_idx]
                pop_var_idx = np.where(curr_pop.factor == var_idx)
                position = [x for x in curr_pop.gbest.position]
                var_candidate_value = position[pop_var_idx[0][0]]
                sol[var_idx] = var_candidate_value
                new_fitness = f(sol)
                if new_fitness < curr_fitness:
                    print('smaller fitness found')
                    curr_fitness = new_fitness
                    best_value_for_var = var_candidate_value
            sol[var_idx] = best_value_for_var
        self.global_solution = sol
        self.global_fitness = f(sol)
        self.solution_history.append(sol)


def print_map(map_size: str):
    size: int = 48
    row_size: int = 12
    if map_size == "small":
        size = 48
        row_size = 12
    elif map_size == "large":
        size = 4*12*4
        row_size = 24
    elif map_size == "mega":
        size = 16*12*4
        row_size = 48
    elif map_size == "giga":
        size = 64*12*4
        row_size = 96
    else:
        size = 407
        row_size = 37
    map_str: str = ""
    reward_map_str: str = ""
    for i in range(len(AGENT.Q)):
        actions = AGENT.Q[i]
        action = np.argmax(actions)
        if action == 0:
            map_str += "^"
        elif action == 1:
            map_str += ">"
        elif action == 2:
            map_str += "v"
        elif action == 3:
            map_str += "<"
        else:
            map_str += "-"
        if (i + 1) % row_size == 0:
            map_str += "\n"
    for i in range(0, size):
        reward_map_str += f"{int(compute_reward(i))} "
        if (i + 1) % size == 0:
            reward_map_str += "\n"
    print(map_str)
    print("----")
    print(reward_map_str)


def train_fea_racetrack_model(model, env, env_type, config, initial=[0, 0], num_particles=8, generations=100, fea_iter=5, debug=False):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training config
        criterion
    """
    global AGENT, ENV, BOUNDS
    bounds = ((0,11), (0,37), (model_config.VELOCITY_MIN, model_config.VELOCITY_MAX), (model_config.VELOCITY_MIN, model_config.VELOCITY_MAX))
    BOUNDS = bounds
    totalReward = {
        type(model).__name__: [],
    }

    if debug:
        print(f"model name: {type(model).__name__}")
        print(f"reward: {totalReward}\n")

    reset_hyperparameters()
    print('starting racetrack RL agent for Q-learning algorithm')
    racetrack_name = f'{TRACK_NAME}'
    ENV = construct_environment(racetrack_name)

    agent: Model = Model(function=train_q_learning_algorithm,
                         algorithm='q-learning',
                         track_file=model_config.TRACK_NAME,
                         lap_count=model_config.LAP_COUNT,
                         velocity_min=model_config.VELOCITY_MIN,
                         velocity_max=model_config.VELOCITY_MAX,
                         gamma=model_config.GAMMA,
                         nu=model_config.NU,
                         reset_after_crash=model_config.RESET_AFTER_CRASH,
                         acceleration_rate=model_config.ACCELERATION_RATE,
                         acceleration_misfire_rate=model_config.ACCELERATION_MISFIRE_RATE,
                         epochs=model_config.EPOCHS,
                         episodes=model_config.EPISODES,
                         q_stability_error=model_config.Q_STABILITY_ERROR,
                         epoch_threshold=model_config.EPOCH_THRESHOLD,
                         update_steps=model_config.UPDATE_STEPS,
                         policy=[])
    AGENT = agent

    # agent.train_agent()

    fa = FactorArchitecture(dim=len(bounds))
    fa.diff_grouping(f, 0.1)
    # fa.overlapping_diff_grouping(_function=f, epsilon=0.1, m=0)
    # fa.factors = fa.random_grouping(min_groups=5, max_groups=15, overlap=True)
    # fa.factors = fa.linear_grouping(group_size=7, offset=5)
    # fa.ring_grouping(group_size=2)
    fa.get_factor_topology_elements()
    fea = FEA(f, fea_runs=fea_iter, generations=generations, pop_size=num_particles, factor_architecture=fa, base_algorithm=PSO)
    fea.run()

    current_policy = agent.policy
    agent.print_hyperparameters()
    benchmark_policy = get_best_policy(get_benchmark_policy(type(model).__name__))
    if debug:
        print(f"accuracy: {get_policy_accuracy(current_policy, benchmark_policy)}")

    trajectories: list = build_trajectories(AGENT, ENV, config)

    # print(f"model name: {type(model).__name__}")
    # print(f"reward: {totalReward}\n")
    return trajectories, current_policy, benchmark_policy


def train(model, env, env_type, config):
    #  model = models.create(config.model_name)
    #  model = nn.DataParallel(model).cuda()
    # dataloader = dp.get_dataloader(train_data, config, is_training=True)
    trajectory, current_policy, benchmark_policy = train_fea_racetrack_model(model, env, env_type, config)
    #  return model
    return trajectory, current_policy, benchmark_policy


def get_policy_accuracy(current: list, benchmark: list):
    count = 0
    for i in range(0, len(current)):
        if current[i] == benchmark[i]:
            count += 1
    return count / len(current)


def predict_prob(model, env, config, device):
    q_table = model.Q
    n, max_steps = config.epochs, config.max_steps
    rewards = []
    num_steps = []
    probs = []
    for episode in range(n):
        s = env.reset()
        total_reward = 0
        for i in range(max_steps):
            a = np.argmax(q_table[s, :])
            prob = softmax(q_table[s, :])
            probs += [prob]
            s, r, done, info = env.step(a)
            total_reward += r
            if done:
                # rewards.append([total_reward])
                num_steps.append(i + 1)
                break
            rewards.append([total_reward])
    env.close()
    # print(rewards)
    # print(probs)
    # return np.concatenate(probs)
    return probs



def evaluate(model, env, config, device):
    q_table = model.Q
    n, max_steps = config.epochs, config.max_steps
    rewards = []
    num_steps = []
    for episode in range(n):
        s = env.reset()
        total_reward = 0
        for i in range(max_steps):
            a = np.argmax(q_table[s, :])
            # a = np.argmax(q_table.best_individual.predict(np.identity(env.observation_space.n)[s, :]))
            s, r, done, info = env.step(a)
            total_reward += r
            if done:
                rewards.append(total_reward)
                num_steps.append(i + 1)
                break
    env.close()
    # return 100*np.sum(rewards)/len(rewards)
    current_policy = get_best_policy(q_table=model.Q)
    benchmark_policy = get_best_policy(get_benchmark_policy(type(model).__name__))
    accuracy = get_policy_accuracy(current_policy, benchmark_policy)

    curr_policy_str: str = print_policy_string(benchmark_policy)
    print(f"\n\n\tOPTIMAL POLICY:\n{curr_policy_str}")
    policy_str: str = print_policy_string(current_policy)
    print(f"\n\n\tCURRENT POLICY:\n{policy_str}")
    return accuracy, policy_str




def get_state_action_table(q_table):
    table: list = []
    for i in range(len(q_table)):
        for j in range(len(q_table[0])):
            table.append([i, j])
    return table


def get_randomized_q_table(q_table, env):
    table: list = []
    for i in range(len(q_table)):
        for j in range(len(q_table[0])):
            table.append(random.randrange(env.action_space.n))
    return table


def get_zeroed_q_table(q_table, env):
    return np.zeros([env.observation_space.n, env.action_space.n])


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)




