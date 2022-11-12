# pso_neural_network.py
# Theta Technologies
########################################################################################################################
# Main driver for training a feedforward neural network optimized by particle swarm optimizers.
########################################################################################################################

from __future__ import print_function
from sklearn import datasets
import numpy as np
import pandas as pd


from MaciNet.supervised_learning import ParticleSwarmOptimizedNN
from MaciNet.utils import train_test_split, to_categorical, normalize, Plot
from MaciNet.deep_learning import NeuralNetwork
from MaciNet.deep_learning.layers import Activation, Dense
from MaciNet.deep_learning.loss_functions import CrossEntropy
from MaciNet.deep_learning.optimizers import Adam


def build_pso_neural_network(datapath:str, plot_results:bool=True):
    # Dummy Data for testing
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=10,
                                        n_classes=4,
                                        n_clusters_per_class=1,
                                        n_informative=2)

    # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    dataframe: pd.DataFrame = pd.read_csv(datapath, header=None)
    dataframe = dataframe.sample(frac=1)
    y = dataframe[16]
    X = np.array(normalize(dataframe.drop([16], 1)))
    y = np.array(to_categorical(y.astype("int")))
    n_samples, n_features = X.shape
    n_hidden = n_features

    # Model builder
    def model_builder(n_inputs, n_outputs):
        model = NeuralNetwork(optimizer=Adam(), loss=CrossEntropy)
        model.add(Dense(n_hidden, input_shape=(n_features,)))
        model.add(Activation('relu'))
        model.add(Dense(y.shape[1], input_shape=(n_features,)))
        model.add(Activation('softmax'))
        return model

    # Print the model summary of a individual in the population
    model_builder(n_inputs=X.shape[1], n_outputs=y.shape[1]).summary()

    population_size = 1000
    n_generations = 20

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

    model = model.evolve(X_train, y_train, n_generations=n_generations)
    loss, accuracy = model.test_on_batch(X_test, y_test)
    print("Accuracy: %.1f%%" % float(100 * accuracy))

    # Reduce dimension to 2D using PCA and plot the results
    y_pred = np.argmax(model.predict(X_test), axis=1)
    if plot_results:
        Plot().plot_in_2d(X_test,
                          y_pred,
                          title="Particle Swarm Optimized Neural Network",
                          accuracy=accuracy,
                          legend_labels=range(y.shape[1]))
