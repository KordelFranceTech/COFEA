import numpy as np



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

