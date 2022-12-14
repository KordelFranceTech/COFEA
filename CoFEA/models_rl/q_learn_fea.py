# QLearningFeaAgent.py

import numpy as np
from .params import *
from .agent import Agent
from CoFEA.models_rl.neural_net_arch import build_nn


def q_learn_fea_agent(env):
	qLearningAgent = QLearningFeaAgent(
		epsilon,
		alpha,
		gamma,
		# env,
		env.observation_space.n,
		env.action_space.n,
		env.action_space,)
	return qLearningAgent


class QLearningFeaAgent(Agent):
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

