# maci_net.py
# Theta Technologies
########################################################################################################################
# Playground file training a feedforward neural network (with backprop) over multimodal sensor data.
########################################################################################################################


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

from Alchemy.deep_learning import NeuralNetwork
from Alchemy.utils import train_test_split, to_categorical, Plot
from Alchemy.deep_learning.optimizers import Adam
from Alchemy.deep_learning.loss_functions import CrossEntropy
from Alchemy.deep_learning.layers import Dense, Activation
import pandas as pd


def main():

    filename: str = 'Sheet 1-SotechDataset_Sensor1_rev2.csv'
    # filename: str = "SotechDataset_Sensor1.csv"

    # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    dataframe: pd.DataFrame = pd.read_csv(filename, header=None)
    dataframe = dataframe.sample(frac=1)
    y_vals_nn = dataframe[16]
    x_vals_nn = dataframe.drop([16], 1)

    # now declare these vals as np array
    # this ensures they are all of identical data type and math-operable
    x_data_train = np.array(x_vals_nn)
    y_data_train = np.array(y_vals_nn)

    X = x_data_train
    y = y_data_train

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))
    n_samples, n_features = X.shape
    n_hidden = n_features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, seed=37)

    model = NeuralNetwork(optimizer=Adam(),
                          loss=CrossEntropy,
                          validation_data=(X_test, y_test))
    model.add(Dense(n_hidden, input_shape=(n_features,)))
    model.add(Activation('relu'))
    model.add(Dense(n_hidden, input_shape=(n_features,)))
    model.add(Activation('relu'))
    model.add(Dense(n_hidden, input_shape=(n_features,)))
    model.add(Activation('relu'))
    ## MARK: - note that at least 3 hidden layers are needed for accurate detection on sotech datasets
    ## The increase in accuracy from each additional layer therafter is marginal
    model.add(Dense(n_hidden, input_shape=(n_features,)))
    model.add(Activation('relu'))
    model.add(Dense(n_hidden, input_shape=(n_features,)))
    model.add(Activation('relu'))
    model.add(Dense(n_hidden, input_shape=(n_features,)))
    model.add(Activation('relu'))
    model.add(Dense(y.shape[1], input_shape=(n_features,)))
    model.add(Activation('softmax'))
    model.summary(name="MLP")

    train_err, val_err = model.fit(X_train, y_train, n_epochs=700, batch_size=32)

    # Training and validation error plot
    n = len(train_err)
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

    # Reduce dimension to 2D using PCA and plot the results
    y_pred = np.argmax(model.predict(X_test), axis=1)
    Plot().plot_in_2d(X_test,
                      y_pred,
                      title="Theta Breathomics Classifier",
                      accuracy=accuracy,
                      legend_labels=["negative","positive"])

    Plot().plot_in_3d(X_test, y_pred)


if __name__ == "__main__":
    main()
