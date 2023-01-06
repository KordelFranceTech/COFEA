# convolutional_neural_network.py
# Theta Technologies
########################################################################################################################
# Main driver for training a convolutional neural network over multimodal sensor data.
########################################################################################################################

from __future__ import print_function
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

from Alchemy.deep_learning import NeuralNetwork
from Alchemy.utils import train_test_split, to_categorical
from Alchemy.utils import Plot
from Alchemy.deep_learning.optimizers import Adam
from Alchemy.deep_learning.loss_functions import CrossEntropy
from Alchemy.deep_learning.layers import Dense, Dropout, Conv2D, Flatten, Activation
from Alchemy.deep_learning.layers import BatchNormalization



def build_convolutional_neural_network(datapath:str, plot_results:bool=True):

    # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    dataframe: pd.DataFrame = pd.read_csv(datapath, header=None)
    y_vals_nn = dataframe[16]
    x_vals_nn = dataframe.drop([16], 1)

    x_vals_nn = np.array(x_vals_nn)
    y_vals_nn = np.array(y_vals_nn)
    x_vals_nn = x_vals_nn[len(x_vals_nn) - len(y_vals_nn):,:]

    # now declare these vals as np array
    # this ensures they are all of identical data type and math-operable
    x_data_train = np.array(x_vals_nn)
    y_data_train = np.array(y_vals_nn)

    X = x_data_train
    y = y_data_train

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    # n_samples, n_features, depth = X.shape
    n_samples, n_features = X.shape
    n_hidden = n_features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, seed=9)

    # Reshape X to (n_samples, channels, height, width)
    X_train = X_train.reshape(-1,1,n_features, 1)
    X_test = X_test.reshape(-1,1,n_features, 1)


    model = NeuralNetwork(optimizer=Adam(),
                        loss=CrossEntropy,
                        validation_data=(X_test, y_test))

    model.add(Conv2D(n_filters=16, filter_shape=(3,3), stride=1, input_shape=(1, n_hidden, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Conv2D(n_filters=32, filter_shape=(3,3), stride=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(y.shape[1]))
    model.add(Activation('softmax'))
    model.summary(name="ConvNet")

    train_err, val_err = model.fit(X_train, y_train, n_epochs=100, batch_size=32)

    # Training and validation error plot
    n = len(train_err)
    if plot_results:
        training, = plt.plot(range(n), train_err, label="Training Error")
        validation, = plt.plot(range(n), val_err, label="Validation Error")
        plt.legend(handles=[training, validation])
        plt.title("Error Plot")
        plt.ylabel('Error')
        plt.xlabel('Iterations')
        plt.show()

    final_loss, accuracy = model.test_on_batch(X_test, y_test)
    print("Accuracy:", accuracy)
    print(f'loss: {final_loss}')

    y_pred = np.argmax(model.predict(X_test), axis=1)

    if plot_results:
        # Reduce dimension to 2D using PCA and plot the results
        Plot().plot_in_2d(X_test,
                          y_pred,
                          title="Convolutional Neural Network",
                          accuracy=accuracy,
                          legend_labels=["negative","positive"])

        Plot().plot_in_3d(X_test, y_pred)


