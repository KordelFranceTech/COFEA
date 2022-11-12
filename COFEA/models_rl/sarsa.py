# SarsaAgent.py


import numpy as np
from .params import *
from .agent import Agent


def sarsa_agent():
    sarsaAgent = SarsaAgent(
        epsilon, alpha, gamma, env.observation_space.n,
        env.action_space.n, env.action_space)
    return sarsaAgent


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
