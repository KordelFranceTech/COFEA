import numpy as np
from CoFEA.environments import env_frozen_lake, env_cliff_walking, env_racetrack, env_racetrack_v2, environment
import random
import gym


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
    elif map_size == "L":
        bounds = [(0, 11), (0, 37)]
    elif map_size == "R":
        bounds = [(0, 28), (0, 30)]
    elif map_size == "P":
        bounds = [(0, 30), (0, 30)]
    return bounds


MAP_SIZE: str = "L"
INITIAL = [9, 1] # L
# INITIAL = [26, 1] # R
# INITIAL = [28, 1] # P
# overlapping intervals to emulate OSI
# bounds = [(0, 47), (0, 23), (24, 47)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
BOUNDS = get_bounds(MAP_SIZE)
NUM_PARTICLES: int = 8
MAX_ITER: int = 100
# env = {"map": env_cliff_walking.CliffWalkingEnv(), "type": MAP_SIZE}
env = {"map":env_racetrack.Racetrack(), "type": MAP_SIZE}
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
max_steps = 100
alpha = 0.9
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
            state1 = states[i]
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
    reward_error = -float(rewards / float(len(states)) ** 2)
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

class Particle:
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


class PSO():
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
    elif map_size == "L":
        size = 407
        row_size = 37
    elif map_size == "R":
        size = 840
        row_size = 30
    elif map_size == "P":
        size = 900
        row_size = 30
    map_str: str = ""
    reward_map_str: str = ""
    for i in range(len(agent.Q)):
        actions = agent.Q[i]
        action = np.argmax(actions)
        if i == (int(env_racetrack.TERMINAL_STATE[0]*row_size) + env_racetrack.TERMINAL_STATE[1]):
            map_str += "G"
        elif i == (int(INITIAL[0]*row_size) + INITIAL[1]):
            map_str += "S"
        elif action == 0:
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


if __name__ == "__main__":
    env.reset()
    PSO(f, INITIAL, BOUNDS, num_particles=NUM_PARTICLES, maxiter=MAX_ITER)
    print_map(map_size=MAP_SIZE)
