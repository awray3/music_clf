""" Train a logistic regression model on the spectrograms """
import warnings

import numpy as np
import tensorflow.keras as keras

from config import JSON_PATH, MODEL_DIR
from load_data import load_data, load_mappings
from evaluation_utilities import (
    get_classification_report,
    get_confusion_matrix,
    plot_history,
)

warnings.filterwarnings(action="ignore")


def create_model(input_shape, num_genres=10):
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


if __name__ == "__main__":

    # load data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(JSON_PATH)

    # create the model
    input_shape = X_train[0].shape
    model = create_model(input_shape)

    # compile the model
    optim = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optim, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    # train the model
    history = model.fit(
        X_train, y_train, validation_data=(X_valid, y_valid), epochs=30, batch_size=32
    )

    # evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=1)

    print(f"Test accuracy: {acc}, test loss: {loss}")

    # view reports
    get_confusion_matrix(model, X_test, y_test)
    get_classification_report(
        model, X_test, y_test, target_names=load_mappings(JSON_PATH)
    )

    # plot history
    plot_history(history)
