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
learning_rate = 0.8
num_episodes = 500

# actions = [0: left, 1: down, 2: right, 3: up]
actions = ['left', 'down', 'right', 'up']


# Dummy Data for testing
X, y = datasets.make_classification(n_samples=1000,
                                    n_features=10,
                                    n_classes=4,
                                    n_clusters_per_class=1,
                                    n_informative=2)

data = datasets.load_iris()
plot_results: bool = True
X = normalize(data.data)
y = data.target
y = to_categorical(y.astype("int"))

# Model builder
def model_builder(n_inputs, n_outputs):

    model = NeuralNetwork(optimizer=Adam(), loss=CrossEntropy)
    model.add(Dense(16, input_shape=(n_inputs,)))
    model.add(Activation('relu'))
    model.add(Dense(n_outputs))
    model.add(Activation('softmax'))

    # DEEP Q-LEARNING
    # print('#######################################################################')
    # print('#######################################################################')
    # print('#######################################################################')
    # print('TRAINING DEEP Q-LEARNING NEURAL NETWORK')
    # model = Sequential()
    # model.add(InputLayer(batch_input_shape=(1, env.observation_space.n)))
    # model.add(Dense(20, activation='relu'))
    # model.add(Dense(env.action_space.n, activation='linear'))
    # model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # print(model.summary())

    # print(f'Q-table after training:\n   action\n{q_table}\n\n\n')

    return model


# Print the model summary of a individual in the population
model_builder(n_inputs=X.shape[1], n_outputs=y.shape[1]).summary()
population_size = 100
n_generations = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

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

# print(f"model: {type(model)}")
# model = model.evolve(X_train, y_train, n_generations=n_generations)
# print(f"model: {type(model)}")
# loss, accuracy = model.test_on_batch(X_test, y_test)
# print(f"model: {type(model)}")
# print("Accuracy: %.1f%%" % float(100 * accuracy))
#
# # Reduce dimension to 2D using PCA and plot the results
# y_pred = np.argmax(model.predict(X_test), axis=1)



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
            action = np.argmax(
              model.best_individual.predict(np.identity(env.observation_space.n)[state:state + 1]))
        new_state, reward, done, _ = env.step(action)
        target = reward + discount_factor * np.max(model.best_individual.predict(np.identity(env.observation_space.n)[new_state:new_state + 1]))
        # target = reward + discount_factor * np.max(np.random.randint(0, env.action_space.n)[new_state:new_state + 1])
        target_vector = model.best_individual.predict(np.identity(env.observation_space.n)[state:state + 1])[0]
        # target_vector = np.random.randint(0, env.action_space.n)
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


# model = model.best_individual.evolve(X_train, y_train, n_generations=n_generations)
# loss, accuracy = model.best_individual.test_on_batch(X_test, y_test)
# print("Accuracy: %.1f%%" % float(100 * accuracy))
#
# # Reduce dimension to 2D using PCA and plot the results
# y_pred = np.argmax(model.best_individual.predict(X_test), axis=1)