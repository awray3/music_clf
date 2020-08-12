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
