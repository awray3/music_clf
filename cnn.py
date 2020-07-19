""" Train a logistic regression model on the spectrograms """
import os
import warnings

warnings.filterwarnings(action="ignore")

import numpy as np
import pandas as pd
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Conv2D,
    BatchNormalization,
    MaxPool2D,
    Dropout,
    Input,
)
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split

from utilities import subset_data
from model_wrapper import ModelWrapper


def add_conv_chunk(model, num_filters, kernel_size, pool_size, drop_prob):
    """adds a convolutional chunk to the model.
    num_filters - integer, controls number of channels
    kernel_size - 2-tuple of ints
    pool_size - 2-tuple of ints
    drop_prob - float between 0 and 1
    """
    model.add(
        Conv2D(
            num_filters,
            kernel_size=kernel_size,
            activation="relu",
            data_format="channels_last",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(Dropout(drop_prob))


def create_cnn(
    num_genres=2, num_filters=40,
):
    model = Sequential()
    model.add(Input(shape=input_shape))

    add_conv_chunk(
        model,
        num_filters=num_filters,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        drop_prob=0.2,
    )

    add_conv_chunk(
        model,
        num_filters=num_filters,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        drop_prob=0.2,
    )

    add_conv_chunk(
        model,
        num_filters=num_filters,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        drop_prob=0.2,
    )

    model.add(Flatten())
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu", kernel_regularizer=l2(0.0001)))
    model.add(Dense(num_genres, activation="softmax"))

    return model


if __name__ == "__main__":

    if not os.path.exists(two_genre_path := "./models/two_class/"):
        os.mkdir(two_genre_path)

    meta_path = os.path.join(".", "data", "fma_metadata", "meta_df.csv")

    meta_df = pd.read_csv(os.path.join(meta_path), index_col=0)

    input_shape = np.load(meta_df["mel_path"].iloc[0])["arr_0"].shape + (1,)

    genre_sublist = ["Rock", "Hip-Hop"]

    # Load the melspectrograms
    X, y = subset_data(meta_df, genre_sublist, n_samples=1996)

    # train-valid-test split
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, random_state=1, shuffle=True, test_size=0.1
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, random_state=1, test_size=0.1
    )

    conv_net_path = os.path.join(two_genre_path, "convnet/")
    conv_net = ModelWrapper(
        batch_size=32,
        model_dir=conv_net_path,
        genre_labels=genre_sublist,
        X_train=X_train[:, :, :, np.newaxis],
        y_train=y_train,
        X_valid=X_valid[:, :, :, np.newaxis],
        y_valid=y_valid,
        X_test=X_test[:, :, :, np.newaxis],
        y_test=y_test,
    )

    conv_net_model = create_cnn()

    conv_net.attach_model(conv_net_model)
    conv_net.summary()
    conv_net.fit(num_epochs=50, verbose=1)
    conv_net.plot_history()
