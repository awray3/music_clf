""" Train a logistic regression model on the spectrograms """
import os
import warnings

from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from model_wrapper import ModelWrapper
from utilities import subset_data

warnings.filterwarnings(action="ignore")

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

logreg_path = os.path.join(two_genre_path, "logreg/")
logistic_reg = ModelWrapper(
    batch_size=32,
    model_dir=logreg_path,
    genre_labels=genre_sublist,
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test,
)

logistic_reg_model = Sequential(
    [
        Flatten(input_shape=input_shape),
        Dense(2, activation="softmax", kernel_regularizer=l2(0.01)),
    ]
)

logistic_reg.attach_model(logistic_reg_model)
logistic_reg.summary()
logistic_reg.fit(num_epochs=50, verbose=0)
logistic_reg.plot_history()
