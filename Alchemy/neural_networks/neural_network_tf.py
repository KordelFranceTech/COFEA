import numpy as np
import pandas as pd
from tensorflow import keras
from Alchemy.utils import train_test_split, to_categorical, Plot


def build_neural_network_tf(datapath:str, plot_results:bool=True):

    # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    dataframe: pd.DataFrame = pd.read_csv(datapath, header=None)
    dataframe = dataframe.sample(frac=1)
    y_vals_nn = dataframe[16]
    x_vals_nn = dataframe.drop([16], 1)

    # now declare these vals as np array
    # this ensures they are all of identical data type and math-operable
    x_data_train = np.array(x_vals_nn)
    y_data_train = np.array(y_vals_nn)
    x_data = x_data_train
    y_data = y_data_train

    num_classes = 2
    x_data_train = np.array(x_data)
    y_data_train = np.array(y_data)
    n_samples, n_features = x_data_train.shape
    n_hidden = n_features
    x_train, x_test, y_train, y_test = train_test_split(x_data_train, y_data_train, test_size=0.3, shuffle=True, seed=9)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    model = keras.Sequential([
        # tf.keras.layers.Flatten(input_shape=(train_images.shape)),
        keras.layers.Flatten(input_shape=(n_features,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        # keras.layers.Dense(n_hidden, activation='relu'),
        keras.layers.Dense(n_hidden, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),

        # keras.layers.Flatten(input_shape=(10,14)),
        # keras.layers.Conv2D(15, kernel_size=(3, 3), activation="relu"),
        # keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # keras.layers.Conv2D(15, kernel_size=(3, 3), activation="relu"),
        # keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # keras.layers.Flatten(),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(num_classes, activation='softmax')

    ])

    # model = keras.Sequential([
    #     # tf.keras.layers.Flatten(input_shape=(train_images.shape)),
    #     keras.layers.Flatten(input_shape=(10, 14)),
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(10)
    # ])

    # model.compile(optimizer='adam',
    #               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])


    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


    model.summary()

    # model.fit(x_train, y_train, epochs=20)
    model.fit(x_train, y_train, batch_size=10, epochs=20, validation_split=0.3)


    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('\nTest accuracy:', test_acc)
    probability_model = keras.Sequential([model, keras.layers.Softmax()])
    predictions = probability_model.predict(x_train)
    print(predictions[-1])
    print(np.argmax(predictions[-1]))


