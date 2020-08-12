"""
Script containing modules for creating models.
"""

import tensorflow.keras as keras
from config import NUM_GENRES


def create_logreg(input_shape, num_genres=NUM_GENRES):
    """
    create a logistic regression model.
    """
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(
                num_genres,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l2(0.01),
            ),
        ]
    )

    return model


def create_CNN(input_shape, num_genres=NUM_GENRES):

    model = keras.Sequential()

    # conv layer 1
    model.add(
        keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            activation="relu",
            input_shape=input_shape,
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=pool_size))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.2))

    # conv layer 2
    model.add(
        keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=pool_size))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.2))

    # conv layer 3
    model.add(
        keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=pool_size))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.2))

    # flatten and dense layer
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # softmax layer
    model.add(keras.layers.Dense(num_genres, activation="softmax"))

    return model

