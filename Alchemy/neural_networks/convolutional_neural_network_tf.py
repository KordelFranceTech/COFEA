import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from Alchemy.utils import train_test_split, to_categorical, Plot


def build_convolutional_neural_network_tf(datapath:str, plot_results:bool=True):

    x_data_train = np.array(x_data)
    y_data_train = np.array(y_data)

    print(f'x data train shape: {x_data_train.shape}')
    print(f'y data train shape: {y_data_train.shape}')

    # Convert to one-hot encoding
    y = to_categorical(y_data_train.astype("int"))

    n_samples, n_features, depth = x_data_train.shape
    n_hidden = n_features
    num_classes = 3
    # input_shape = (10, 14, 1)
    input_shape = (n_hidden, depth, 1)

    x_train, x_test, y_train, y_test = train_test_split(x_data_train, y_data_train, test_size=0.3, shuffle=True, seed=9)
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # print("x_train shape:", x_train.shape)
    # print(f'x_train[0] shape: {x_train[0].shape}')
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            ####
            # keras.Input(shape=input_shape),
            # layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            # layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            # layers.Flatten(),
            # layers.Dropout(0.5),
            # layers.Dense(num_classes, activation="softmax"),
            ####

            keras.Input(shape=input_shape),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    print(model.summary())
    batch_size = 64
    epochs = 25
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", "mse"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    score = model.evaluate(x_test, y_test, verbose=0)
    model.save("goldilox_time_series_cnn.h5")
    # model.save("goldilox_time_series_cnn.model", save_format="h5")
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    probability_model = keras.Sequential([model, keras.layers.Softmax()])
    predictions = probability_model.predict(x_train)
    print(f'predicted: {predictions[-1]}')
    print(f'actual: {y_train[-1]}')
    print(np.argmax(predictions[-1]))

