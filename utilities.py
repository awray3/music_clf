"""
Collection of helpful utilities for visualization and processing data
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from librosa.feature import melspectrogram
from librosa import power_to_db
from librosa.display import specshow


def view_melspec(source, sr):
    """quickly plot a melspectrogram."""

    plt.figure(figsize=(10, 4))
    S = melspectrogram(source, sr=sr, n_mels=200)
    S_dB = power_to_db(S, ref=np.max)
    specshow(S_dB, x_axis='time',
             y_axis='mel', sr=sr,
             fmax=4092)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Melspectrogram')
    plt.tight_layout()
    plt.show()


def plot_sample(genre, meta_df, nrow=3, **kwargs):
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

        # Load the melspectrogram
        S = np.load(samples["mel_path"].iloc[i])['arr_0']
        # convert from power to db
        S_dB = power_to_db(S, ref=np.max)
        specshow(S_dB, x_axis="time", y_axis="mel", fmax=4092,
                 **kwargs)
        plt.colorbar(format="%+2.0f dB")
        plt.title("Track id " + str(samples["track_id"].iloc[i]))
    plt.show()


def id_from_path(mp3_path):
    """
    returns the id of the given mp3 path.
    """
    return os.path.split(mp3_path)[1][:-4]


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


def standardize_2d_array(X):
    """ This is a min-max scaler for a 2D numpy array. """
    return (X-X.min())/(X.max()-X.min())


def subset_data(meta_df, genre_list, n_samples=None):
    """
    Loads melspecs from meta_df from the given genres and normalizes them.
    Only returns a random subset of size n_samples.
    """

    if n_samples:
        meta_sub_df = meta_df.loc[meta_df["genre"].isin(genre_list),
                                  :].sample(n=n_samples)
    else:
        meta_sub_df = meta_df.loc[meta_df["genre"].isin(genre_list),
                                  :].sample(frac=1)

    all_melspecs = []
    for path in meta_sub_df.mel_path.values:
        if not np.any(np.isnan(
                loaded_array := standardize_2d_array(np.load(path)["arr_0"])
        )):
            all_melspecs.append(loaded_array)
        else:
            print(
                f"track with id {id_from_path(path)} was non-standardizable. \
                Dropping from dataset.")
            meta_sub_df.drop(
                meta_sub_df.loc[meta_sub_df.mel_path == path, :].index,
                inplace=True
            )

    X = np.stack(all_melspecs)
    y = meta_sub_df[genre_list].to_numpy()

    assert X.shape[0] == len(y)

    return X, y
