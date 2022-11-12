# QLearningAgent.py


import numpy as np
from .params import *
from .agent import Agent
from MaciNet.supervised_learning import ParticleSwarmOptimizedNN
from MaciNet.deep_learning import NeuralNetwork
from MaciNet.deep_learning.layers import Activation, Dense
from MaciNet.deep_learning.loss_functions import CrossEntropy
from MaciNet.deep_learning.optimizers import Adam


def q_learn_osi_agent():
    qLearningAgent = QLearningAgentOsi(
        epsilon, alpha, gamma, env.observation_space.n,
        env.action_space.n, env.action_space)
    return qLearningAgent


class QLearningAgentOsi(Agent):
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
		self.action_space = action_space
		pso_nn = self.build_nn(self.num_state, self.num_actions)
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

		target = reward + self.gamma * np.max(self.Q.best_individual.predict(np.identity(env.observation_space.n)[state2, :]))
		target_vector = self.Q.best_individual.predict(np.identity(env.observation_space.n)[state:state + 1])[0]
		# target_vector = np.random.randint(0, env.action_space.n)
		target_vector[action] = target
		self.Q.cofea_evolve(np.identity(env.observation_space.n)[state:state + 1],
							target_vector.reshape(-1, env.action_space.n),
							n_generations=10)


	def build_nn(self, n_inputs, n_outputs):
		# Model builder
		def model_builder(n_inputs, n_outputs):
			model = NeuralNetwork(optimizer=Adam(), loss=CrossEntropy)
			model.add(Dense(16, input_shape=(n_inputs,)))
			model.add(Activation('relu'))
			model.add(Dense(n_outputs))
			model.add(Activation('softmax'))
			return model

		# Print the model summary of a individual in the population
		model_builder(n_inputs=env.observation_space.n, n_outputs=env.action_space.n).summary()
		population_size = 100
		n_generations = 10
		inertia_weight = 0.8
		cognitive_weight = 0.8
		social_weight = 0.8

		print("\nPopulation Size: %d" % population_size)
		print("Generations: %d" % n_generations)
		print("")
		print("\nInertia Weight: %.2f" % inertia_weight)
		print("Cognitive Weight: %.2f" % cognitive_weight)
		print("Social Weight: %.2f" % social_weight)

		model = ParticleSwarmOptimizedNN(population_size=population_size,
										 inertia_weight=inertia_weight,
										 cognitive_weight=cognitive_weight,
										 social_weight=social_weight,
										 max_velocity=5,
										 model_builder=model_builder)

		state = env.reset()
		target_vector = np.array([0, 0, 0, 0])
		model = model.cofea_evolve(np.identity(env.observation_space.n)[state:state + 1],
								   target_vector.reshape(-1, env.action_space.n),
								   n_generations=n_generations)
		return model
