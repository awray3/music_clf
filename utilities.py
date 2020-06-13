import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from librosa.feature import melspectrogram
from librosa import power_to_db, load, get_duration
from librosa.display import specshow

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence 
from sklearn.metrics import confusion_matrix, classification_report

def view_melspec(source, sr):
    plt.figure(figsize=(10, 4))
    S = melspectrogram(source, sr=sr, n_mels=128)
    S_dB = power_to_db(S, ref=np.max)
    specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=sr,
                             fmax=2048)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Melspectrogram')
    plt.tight_layout()
    plt.show()

def plot_sample(genre, meta_df, nrow=3, waveform=False, **kwargs):
    """
    Plots 2*nrow randomly chosen spectrograms
    (or waveforms if set to True) from the given genre.
    """
    # sample 6 randomly chosen songs with given genre
    samples = meta_df.loc[meta_df[genre] == 1, :].sample(n=2 * nrow)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Genre: " + genre)
    fig.subplots_adjust(hspace=0.7, wspace=0.4)

    for i in range(2 * nrow):
        plt.subplot(nrow, 2, i+1)


        if waveform:
            wave = librosa.load(samples["mp3_path"].iloc[i], **kwargs)
            plt.xlabel("Sample Position")
            plt.ylabel("Amplitude")
        else:
            #Load the melspectrogram
            S = np.load(samples["mel_path"].iloc[i])['arr_0']
            #convert from power to db 
            S_dB = power_to_db(S, ref=np.max)
            specshow(S_dB, x_axis="time", y_axis="mel", fmax=2048,
                    **kwargs)
            plt.colorbar(format="%+2.0f dB")
            plt.title("Track id " + str(samples["track_id"].iloc[i]))
    plt.show()


def id_from_path(mp3_path):
    """
    returns the id of the given mp3 path.
    """
    return os.path.split(mp3_path)[1][:-4]

def attach_onehot_encoding(df, column_name):
    """
    Append the onehot representation of `column` onto the right end
    of the array df.
    """

    df = pd.concat([df, pd.get_dummies(df.genre)], axis=1)

    return df

def mp3_to_mel_path(mp3_path, melspec_dir):
    """
    Take the mp3 path and find the melspec path.
    Returns the melspec path.
    """

    melspec_path = os.path.join(melspec_dir, id_from_path(mp3_path) + ".npz")

    if os.path.exists(melspec_path):
        return melspec_path
    else:
        raise FileNotFoundError(
            f"Did not find a melspectrogram with id {id_from_path(mp3_path)}"
        )

def read_metadata_file(path, all_filepaths, bad_filepaths):
    all_metadata = pd.read_csv(path, header=[0,1], index_col=0)
    
    cols_to_keep = [('track', 'genre_top')]

    # This will be the main dataframe for here on out:
    df = all_metadata.loc[
                all_metadata[('set', 'subset')] == 'small',
                cols_to_keep
    ]

    df.reset_index(inplace=True)
    
    df.columns = ['track_id', 'genre']
    
    # add filepaths to the dataframe
    df['mp3_path'] = all_filepaths
    
    # Remove bad mp3s from the dataframe so that we skip them.
    if df.mp3_path.isin(bad_filepaths).sum():
        df.drop(
            df.loc[df.mp3_path.isin(bad_filepaths), :].index,
            inplace=True
        )
        print(f"Dropped {len(bad_filepaths)} rows from the dataframe.")

    df['duration'] = df['mp3_path'].apply(lambda x:
            get_duration(filename=x))

    return df



class MyModel:
    """
    Wrapper class that manages training, loading, and evaluation of a model.
    """

    def __init__(self, batch_size, model_dir, genre_labels, X_train, y_train, X_test, y_test):
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.genre_labels = genre_labels
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


        self.model_path = glob.glob(os.path.join(model_dir, "*.h5"))
        ### Currently this will fail if there isn't already a file there.

        if self.model_path:
            self.model_path = self.model_path[0]
        else:
            self.model_path = os.path.join(self.model_dir, "model.h5")

        self.input_shape = (1,) + self.X_train.shape[1:]

    def attach_model(self, model):
        """ Attach a Keras model. """
        self.model = model

    def load_model(self):
        """
        Loads the model from the model directory. Alternative to passing 
        a new model.
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            raise ValueError("File Not Found. Please train the model.")

    def fit(self, num_epochs):
        checkpoint_callback = ModelCheckpoint(
            self.model_path,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max",
        )
        reducelr_callback = ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_delta=0.01, verbose=1
        )
        early_stop = EarlyStopping(
            monitor="val_accuracy", patience=10, verbose=1, restore_best_weights=True
        )

        callbacks_list = [checkpoint_callback, reducelr_callback, early_stop]

        self.history = self.model.fit(
            x=self.X_train,
            y=self.y_train,
            epochs=num_epochs,
            validation_split=0.1,
            verbose=1,
            callbacks=callbacks_list,
        )

    def summary(self):
        return self.model.summary()

    def _compile(self):
        self.model.compile(
            optimizer=Adam(lr=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def evaluate(self, verbose=0, all=False):

        print("Training set")
        self.model.evaluate(
                x=self.X_train,
                y=self.y_train,
                verbose=verbose)
        print("Testing set")
        self.model.evaluate(
                x=self.X_test,
                y=self.y_test,
                verbose=verbose)

    def plot_history(self):
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Valid"], loc="upper left")
        plt.show()

        plt.savefig(os.path.join(self.model_dir, "last_training_acc.png"))

        # Plot training & validation loss values
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Valid"], loc="upper left")
        plt.show()
        plt.savefig(os.path.join(self.model_dir, "last_training_loss.png"))

    def confusion_matrix(self, print_labels=True):

        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        classes = np.argmax(self.y_test, axis=1)


        print(confusion_matrix(classes, y_pred))

    def classification_report(self):

        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        classes = np.argmax(self.y_test, axis=1)

        print(
            classification_report(
                classes, y_pred,
                target_names=self.genre_labels
            )
        )
