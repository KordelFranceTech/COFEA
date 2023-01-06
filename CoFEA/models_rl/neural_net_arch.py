from Alchemy.supervised_learning import ParticleSwarmOptimizedNN
from Alchemy.deep_learning import NeuralNetwork
from Alchemy.deep_learning.layers import Activation, Dense
from Alchemy.deep_learning.loss_functions import CrossEntropy, MeanSquaredErrorLoss
from Alchemy.deep_learning.optimizers import Adam
import numpy as np


# Model builder
def neural_net(n_inputs, n_outputs):
    ## q learn
    # model = NeuralNetwork(optimizer=Adam(), loss=CrossEntropy)
    # model.add(Dense(20, input_shape=(n_inputs,)))
    # model.add(Activation('relu'))
    # model.add(Dense(n_outputs))
    # # model.add(Activation('linear'))
    # model.add(Activation('softmax'))

    # model = NeuralNetwork(optimizer=Adam(), loss=MeanSquaredErrorLoss)
    # model.add(Dense(20, input_shape=(n_inputs,)))
    # model.add(Activation('relu'))
    # model.add(Dense(n_outputs))
    # model.add(Activation('softmax'))

    model = NeuralNetwork(optimizer=Adam(), loss=MeanSquaredErrorLoss)
    model.add(Dense(20, input_shape=(n_inputs,)))
    model.add(Activation('relu'))
    # model.add(Dense(20, input_shape=(n_inputs,)))
    # model.add(Activation('relu'))
    model.add(Dense(n_outputs))
    # model.add(Activation('linear'))
    model.add(Activation('softmax'))
    return model


def build_nn(n_inputs,
             n_outputs,
             env,
             population_size=10,
             n_generations=3,
             inertia_weight=0.8,
             cognitive_weight=0.8,
             social_weight=0.8,
             debug: bool = False):

    # Print the model summary of a individual in the population
    neural_net(n_inputs=n_inputs, n_outputs=n_outputs).summary()

    if debug:
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
                                     model_builder=neural_net)

    state = env.reset()
    target_vector = np.array([1, 0, 0, 0])
    model = model.cofea_evolve(np.identity(env.observation_space.n)[state:state + 1],
                               target_vector.reshape(-1, env.action_space.n),
                               n_generations=n_generations)
    return model
