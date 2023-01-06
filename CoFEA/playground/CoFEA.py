# pso_neural_network.py
# Theta Technologies
########################################################################################################################
# Main driver for training a feedforward neural network optimized by particle swarm optimizers.
########################################################################################################################

from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import gym
import numpy as np

from MaciNet.supervised_learning import ParticleSwarmOptimizedNN
from MaciNet.utils import train_test_split, to_categorical, normalize, Plot
from MaciNet.deep_learning import NeuralNetwork
from MaciNet.deep_learning.layers import Activation, Dense
from MaciNet.deep_learning.loss_functions import CrossEntropy
from MaciNet.deep_learning.optimizers import Adam



def build_osi_network():
    env = gym.make('FrozenLake-v0')
    # print(gym.envs.toy_text.frozen_lake.generate_random_map(size=8, p=0.8))
    '''
        4 x 4 map
        S F F F       (S: starting point, safe)
        F H F H       (F: frozen surface, safe)
        F F F H       (H: hole, fall to your doom)
        H F F G       (G: goal, where the frisbee is located)
    
        8 x 8 map
        "SFFFFFFF"
        "FFFFFFFF"
        "FFFHFFFF"
        "FFFFFHFF"
        "FFFHFFFF"
        "FHHFFFHF"
        "FHFFHFHF"
        "FFFHFFFG"
    '''

    discount_factor = 0.95
    eps = 0.5
    eps_decay_factor = 0.999
    learning_rate = 0.1
    num_episodes = 3
    plot_results = True

    # actions = [0: left, 1: down, 2: right, 3: up]
    actions = ['left', 'down', 'right', 'up']


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
    target_vector = np.array([0, 0, 0, 0 ])
    model = model.cofea_evolve(np.identity(env.observation_space.n)[state:state + 1],
                 target_vector.reshape(-1, env.action_space.n),
                 n_generations=n_generations)


    for i in range(num_episodes):
        print('####################################################################')
        print(f'episode #: {i}')
        state = env.reset()
        eps *= eps_decay_factor
        done = False
        count = 0
        while not done:
            print(count)
            # action = np.random.randint(0, env.action_space.n)
            if np.random.random() < eps:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(model.best_individual.predict(np.identity(env.observation_space.n)[state:state + 1]))
            new_state, reward, done, _ = env.step(action)
            target = reward + discount_factor * np.max(model.best_individual.predict(np.identity(env.observation_space.n)[new_state:new_state + 1]))
            target_vector = model.best_individual.predict(np.identity(env.observation_space.n)[state:state + 1])[0]
            target_vector[action] = target
            model.cofea_evolve(np.identity(env.observation_space.n)[state:state + 1],
                      target_vector.reshape(-1, env.action_space.n),
                      n_generations=n_generations)
            print(f'state: {state}\taction: {actions[action]}\treward: {reward}\tnew state: {new_state}\ttarget: {target}')
            state = new_state
            count += 1

        loss, accuracy = model.best_individual.test_on_batch(np.identity(env.observation_space.n)[state:state + 1],
                      target_vector.reshape(-1, env.action_space.n))
        print(f"Accuracy:  {float(100 * accuracy)}\nLoss: {loss}")

    # print(model.best_individual)
    # y_pred = np.argmax(model.best_individual.predict(env.observation_space), axis=1)
    # if plot_results:
    #     Plot().plot_in_2d(env.observation_space,
    #                       y_pred,
    #                       title="Particle Swarm Optimized Neural Network",
    #                       accuracy=accuracy,
    #                       legend_labels=range(y.shape[1]))
    return model.best_individual

build_osi_network()