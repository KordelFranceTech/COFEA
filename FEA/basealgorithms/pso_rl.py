import numpy as np
from CoFEA.environments import env_frozen_lake, env_cliff_walking, environment
import random
import gym
import numpy as np
import random
from operator import attrgetter
from copy import deepcopy, copy

# Agent.py

class Agent:
	"""
	The Base class that is implemented by
	other classes to avoid the duplicate 'choose_action'
	method
	"""
	def choose_action(self, state):
		action = 0
		if np.random.uniform(0, 1) < self.epsilon:
			action = self.action_space.sample()
		else:
			action = np.argmax(self.Q[state, :])
		return action


# SarsaAgent.py

class SarsaAgent(Agent):
	"""
	The Agent that uses SARSA update to improve it's behaviour
	"""
	def __init__(self, epsilon, alpha, gamma, num_state, num_actions, action_space):
		"""
		Constructor
		Args:
			epsilon: The degree of exploration
			gamma: The discount factor
			num_state: The number of states
			num_actions: The number of actions
			action_space: To call the random action
		"""
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.num_state = num_state
		self.num_actions = num_actions

		self.Q = np.zeros((self.num_state, self.num_actions))
		self.action_space = action_space


	def update(self, prev_state, next_state, reward, prev_action, next_action):
		"""
		Update the action value function using the SARSA update.
		Q(S, A) = Q(S, A) + alpha(reward + (gamma * Q(S_, A_) - Q(S, A))
		Args:
			prev_state: The previous state
			next_state: The next state
			reward: The reward for taking the respective action
			prev_action: The previous action
			next_action: The next action
		Returns:
			None
		"""
		predict = self.Q[prev_state, prev_action]
		target = reward + self.gamma * self.Q[next_state, next_action]
		self.Q[prev_state, prev_action] += self.alpha * (target - predict)



# QLearningAgent.py

class QLearningAgent(Agent):
	def __init__(self, epsilon, alpha, gamma, num_state, num_actions, action_space):
		"""
		Constructor
		Args:
			epsilon: The degree of exploration
			gamma: The discount factor
			num_state: The number of states
			num_actions: The number of actions
			action_space: To call the max reward action
		"""
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.num_state = num_state
		self.num_actions = num_actions

		self.Q = np.zeros((self.num_state, self.num_actions))
		self.action_space = action_space


	def update(self, state, state2, reward, action, action2):
		"""
		Update the action value function using the Q-Learning update.
		Q(S, A) = Q(S, A) + alpha(reward + (gamma * Q(S_, A_) - Q(S, A))
		Args:
			prev_state: The previous state
			next_state: The next state
			reward: The reward for taking the respective action
			prev_action: The previous action
			next_action: The next action
		Returns:
			None
		"""
		predict = self.Q[state, action]
		target = reward + self.gamma * np.max(self.Q[state2, :])
		self.Q[state, action] += self.alpha * (target - predict)


# ExpectedSarsaAgent.py

class ExpectedSarsaAgent(Agent):
    def __init__(self, epsilon, alpha, gamma, num_state, num_actions, action_space):
        """
        Constructor
        Args:
            epsilon: The degree of exploration
            gamma: The discount factor
            num_state: The number of states
            num_actions: The number of actions
            action_space: To call the expected action
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = num_state
        self.num_actions = num_actions

        self.Q = np.zeros((self.num_state, self.num_actions))
        self.action_space = action_space

    def update(self, prev_state, next_state, reward, prev_action, next_action):
        """
        Update the action value function using the Expected SARSA update.
        Q(S, A) = Q(S, A) + alpha(reward + (pi * Q(S_, A_) - Q(S, A))
        Args:
            prev_state: The previous state
            next_state: The next state
            reward: The reward for taking the respective action
            prev_action: The previous action
            next_action: The next action
        Returns:
            None
        """
        predict = self.Q[prev_state, prev_action]

        expected_q = 0
        q_max = np.max(self.Q[next_state, :])
        greedy_actions = 0
        for i in range(self.num_actions):
            if self.Q[next_state][i] == q_max:
                greedy_actions += 1

        non_greedy_action_probability = self.epsilon / self.num_actions
        greedy_action_probability = ((1 - self.epsilon) / greedy_actions) + non_greedy_action_probability

        for i in range(self.num_actions):
            if self.Q[next_state][i] == q_max:
                expected_q += self.Q[next_state][i] * greedy_action_probability
            else:
                expected_q += self.Q[next_state][i] * non_greedy_action_probability

        target = reward + self.gamma * expected_q
        self.Q[prev_state, prev_action] += self.alpha * (target - predict)


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
    return bounds


MAP_SIZE: str = "large"
INITIAL = [0, 0]  # initial starting location [x1,x2...]
# overlapping intervals to emulate OSI
# bounds = [(0, 47), (0, 23), (24, 47)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
BOUNDS = get_bounds(MAP_SIZE)
NUM_PARTICLES: int = 8
MAX_ITER: int = 100
env = {"map": env_cliff_walking.CliffWalkingEnv(), "type": MAP_SIZE}
environment.set_environment(env)
env = env["map"]
# env = gym.make('CliffWalking-v0')
# env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, new_step_api=False)
# print(env.observation_space)

# Defining all the required parameters
totalReward = {
	'SarsaAgent': [],
	'QLearningAgent': [],
	'ExpectedSarsaAgent': []
}

# # Defining all the three agents
# expectedSarsaAgent = ExpectedSarsaAgent(
# 	epsilon, alpha, gamma, env.observation_space.n,
# 	env.action_space.n, env.action_space)
# qLearningAgent = QLearningAgent(
# 	epsilon, alpha, gamma, env.observation_space.n,
# 	env.action_space.n, env.action_space)
# sarsaAgent = SarsaAgent(
# 	epsilon, alpha, gamma, env.observation_space.n,
# 	env.action_space.n, env.action_space)


# Defining all the required parameters
epsilon = 0.1
total_episodes = 1
max_steps = 10
alpha = 0.5
gamma = 1
episodeReward = 0

# agent = ExpectedSarsaAgent(
#     epsilon, alpha, gamma, env.observation_space.n,
#     env.action_space.n, env.action_space)

agent = ExpectedSarsaAgent(
    epsilon=epsilon,
    alpha=alpha,
    gamma=gamma,
    num_state=env.observation_space.n,
    num_actions=env.action_space.n,
    action_space=env.action_space
)
# agent = QLearningAgent(
# 	epsilon,
#     alpha,
#     gamma,
#     env.observation_space.n,
# 	env.action_space.n,
#     env.action_space
# )


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


def f(states):
    rewards = 0
    totalReward = {
        'SarsaAgent': [],
        'QLearningAgent': [],
        'ExpectedSarsaAgent': []
    }
    for i in range(len(states)):
        for episode in range(total_episodes):
            # Initialize the necessary parameters before
            # the start of the episode
            t = 0
            # state1 = env.reset()
            state1 = round(states[i])
            action1 = agent.choose_action(state1)
            episodeReward = 0
            while t < max_steps:

                # Getting the next state, reward, and other parameters
                state2, reward, done, info = env.step(action1)
                # reward = compute_reward(state2)

                # Choosing the next action
                action2 = agent.choose_action(state2)

                # Learning the Q-value
                agent.update(state1, state2, reward, action1, action2)
                # print(f'episode: {episode}\n\tstate: {state1}\taction: {action2}\treward: {reward}\tnew state: {state2}\ttarget: {target}')

                state1 = state2
                action1 = action2

                # Updating the respective vaLues
                t += 1
                episodeReward += reward

                # If at the end of learning process
                if done:
                    break
            # Append the sum of reward at the end of the episode
            totalReward[type(agent).__name__].append(episodeReward)
        rewards += np.mean(totalReward[type(agent).__name__])
    reward_error = float(rewards / float(len(states)) ** 2)
    return reward_error


def f0(states):
    "Objective function"
    t = 0
    episodeReward = 0

    rewards = 0
    rewards_list: list = []
    actions_list: list = []
    for i in range(len(states)):
        env.reset()
        state1 = int(states[i])
        action1 = agent.choose_action(state1)
        state2, reward, done, info = env.step(action1)
        reward = compute_reward(state2)
        # action2 = agent.choose_action(state2)
        # agent.update(state1, state2, reward, action1, action2)
        rewards += reward

    # s_index = np.argmax(rewards_list)
    # print(f"states: {states}")
    # print(f"s_index: {s_index}")
    # print(f"rewards_list: {rewards_list}")
    # print(f"actions_list: {actions_list}")

    # state1 = int(states[i])
    # action1 = actions_list[s_index]
    # state2, reward, done, info = env.step(action1)
    # action2 = model.choose_action(state2)
    # model.update(state1, state2, rewards_list[s_index], action1, action2)
    # print(model.Q)
    # # reward_error = abs(compute_reward(int(states[s_index])) - reward)
    # # reward_error = 47 - compute_reward(int(states[s_index]))
    reward_error = -rewards
    print(f"rewards: {rewards}")
    print(f"reward_error: {reward_error}")
    return reward_error


# --- MAIN ---------------------------------------------------------------------+

class ParticleA:
    def __init__(self, x0):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognitive constant
        c2 = 2  # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            # print(f"position pre: {self.position_i[i]}")
            # self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            # self.position_i[i] = self.position_i[i] + 1
            self.position_i[i], reward, done, info = env.step(np.argmax(agent.Q.tolist()[self.position_i[i]]))
            # if done: break
            # print(f"position post: {self.position_i[i]}")

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


class PSOA():
    def __init__(self, costFunc, x0, bounds, num_particles, maxiter):
        global num_dimensions

        num_dimensions = len(x0)
        err_best_g = -1000  # best error for group
        pos_best_g = []  # best position for group

        # establish the swarm
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i = 0
        while i < maxiter:
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1000:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
                for k in range(0, len(swarm[j].position_i)):
                    state1 = swarm[j].position_i[k]
                    action1 = agent.choose_action(state1)
                    state2, reward, done, info = env.step(action1)
                    action2 = agent.choose_action(state2)
                    agent.update(state1, state2, reward, action1, action2)
                    if done: break
            i += 1

        # print final results
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)


def print_map():
    size: int = 48
    row_size: int = 12
    if MAP_SIZE == "small":
        size = 48
        row_size = 12
    elif MAP_SIZE == "large":
        size = 4*12*4
        row_size = 24
    elif MAP_SIZE == "mega":
        size = 16*12*4
        row_size = 48
    elif MAP_SIZE == "giga":
        size = 64*12*4
        row_size = 96
    map_str: str = ""
    reward_map_str: str = ""
    for i in range(len(agent.Q)):
        actions = agent.Q[i]
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
        print(self.position)
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
            self.position[i], reward, done, info = env.step(np.argmax(agent.Q.tolist()[round(self.position[i])]))
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
                action1 = agent.choose_action(state1)
                state2, reward, done, info = env.step(action1)
                action2 = agent.choose_action(state2)
                agent.update(state1, state2, reward, action1, action2)
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




if __name__ == "__main__":
    env.reset()
    pso = PSO(generations=10, population_size=15, function=f, dim=len(BOUNDS))
    # pso.run()
    # print_map()
    from FEA.FEA.factorarchitecture import FactorArchitecture

    fa = FactorArchitecture(dim=2)
    # fa.factors = fa.diff_grouping(f, 0.1)
    # fa.factors = fa.random_grouping(min_groups=5, max_groups=15, overlap=True)
    # fa.factors = fa.linear_grouping(group_size=7, offset=5)
    fa.ring_grouping(group_size=2)
    # print(fa.factors)
    fa.get_factor_topology_elements()

    # fa.load_csv_architecture(file="../../results/factors/F1_m4_diff_grouping.csv", dim=50)
    # func = Function(function_number=1, shift_data_file="f01_o.txt")
    fea = FEA(f, fea_runs=10, generations=10, pop_size=15, factor_architecture=fa, base_algorithm=PSO)
    fea.run()
