# QLearningAgent.py

import numpy as np
from .params import *
from .agent import Agent
from CoFEA.models_rl.neural_net_arch import build_nn


def q_learn_osi_agent(env):
	qLearningAgent = QLearningOsiAgent(
		epsilon,
		alpha,
		gamma,
		env,
		env.observation_space.n,
		env.action_space.n,
		env.action_space,
		population_size=3,
		n_generations=3,
		inertia_weight=0.8,
		cognitive_weight=0.8,
		social_weight=0.8)
	return qLearningAgent


class QLearningOsiAgent(Agent):
	def __init__(self,
				 epsilon,
				 alpha,
				 gamma,
				 env,
				 num_state,
				 num_actions,
				 action_space,
                 population_size=3,
                 n_generations = 3,
                 inertia_weight = 0.8,
                 cognitive_weight = 0.8,
                 social_weight = 0.8):
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
		self.env = env
		self.num_state = num_state
		self.num_actions = num_actions
		self.action_space = action_space
		pso_nn = build_nn(self.num_state,
						   self.num_actions,
						   env,
						   population_size,
						   n_generations,
						   inertia_weight,
						   cognitive_weight,
						   social_weight)
		self.Q = pso_nn
		# self.Q = np.zeros((self.num_state, self.num_actions))


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
		# predict = self.Q[state, action]
		# target = reward + self.gamma * np.max(self.Q[state2, :])
		# self.Q[state, action] += self.alpha * (target - predict)

		predict = self.Q.best_individual.predict(np.identity(self.env.observation_space.n)[state, :])
		# target = reward + self.alpha * (self.gamma * np.max(self.Q.best_individual.predict(np.identity(env.observation_space.n)[state2, :])))
		target = reward + (self.gamma * np.max(self.Q.best_individual.predict(np.identity(self.env.observation_space.n)[state2, :])))

		target_vector = self.Q.best_individual.predict(np.identity(self.env.observation_space.n)[state:state + 1])[0]
		# target_vector = np.random.randint(0, env.action_space.n)
		target_vector[action] = target
		self.Q.cofea_evolve(np.identity(self.env.observation_space.n)[state:state + 1],
							target_vector.reshape(-1, self.env.action_space.n),
							n_generations=3)
		return target
