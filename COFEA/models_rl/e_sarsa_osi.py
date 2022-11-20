# ExpectedSarsaAgent.py

import numpy as np
from .params import *
from .agent import Agent
from CoFEA.models_rl.neural_net_arch import build_nn


def expected_sarsa_osi_agent(env):
    expectedSarsaAgent = ExpectedSarsaOsiAgent(
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
    return expectedSarsaAgent


class ExpectedSarsaOsiAgent(Agent):
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
            action_space: To call the expected action
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
        # predict = self.Q[prev_state, prev_action]
        # a = self.Q.best_individual.predict(np.identity(env.observation_space.n)[next_state])[0]
        # print(a)
        # print(a[0])
        # a0 = a[0]
        # print(a0[0])
        expected_q = 0
        # q_max = np.max(self.Q[next_state, :])
        q_max = np.max(self.Q.best_individual.predict(np.identity(self.env.observation_space.n)[next_state, :]))
        greedy_actions = 0
        for i in range(self.num_actions):
            # if self.Q[next_state][i] == q_max:
            preds = self.Q.best_individual.predict(np.identity(self.env.observation_space.n)[next_state])[0]
            if preds[i] == q_max:
                greedy_actions += 1

        non_greedy_action_probability = self.epsilon / self.num_actions
        greedy_action_probability = ((1 - self.epsilon) / greedy_actions) + non_greedy_action_probability

        for i in range(self.num_actions):
            # if self.Q[next_state][i] == q_max:
            preds = self.Q.best_individual.predict(np.identity(self.env.observation_space.n)[next_state])[0]
            if preds[i] == q_max:
                # expected_q += self.Q[next_state][i] * greedy_action_probability
                expected_q += preds[i] * greedy_action_probability
            else:
                # expected_q += self.Q[next_state][i] * non_greedy_action_probability
                expected_q += preds[i] * non_greedy_action_probability

        target = reward + self.gamma * expected_q
        # self.Q[prev_state, prev_action] += self.alpha * (target - predict)

        # target = reward + self.gamma * np.max(self.Q.best_individual.predict(np.identity(env.observation_space.n)[next_state:next_state + 1]))
        target_vector = self.Q.best_individual.predict(np.identity(self.env.observation_space.n)[prev_state:prev_state + 1])[0]
        # target_vector = np.random.randint(0, env.action_space.n)
        target_vector[prev_action] = target
        self.Q.cofea_evolve(np.identity(self.env.observation_space.n)[prev_state:prev_state + 1],
                            target_vector.reshape(-1, self.env.action_space.n),
                            n_generations=3)
        return target

