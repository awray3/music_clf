"""
Functions for taking raw mp3s and creating the inputs and targets
for training.
"""

# I added a comment right here!

from librosa import load
from typing import List, Set, Tuple
import numpy as np
from audioread.exceptions import NoBackendError

from preprocessing.tracks.mp3 import MP3


def unique_shapes(X: List[np.array]) -> Set[int]:
    """
    Takes a list of numpy arrays, X, and compiles the shapes of the
    components. To be used inside the `prepare_mp3s_and_lables` function
    to get all of the shapes of loaded mp3 files.
    """
    shapes = set([])
    for arr in X:
        shapes.add(arr.shape[1])
    return shapes


def prepare_mp3s_and_labels(mp3_list: List[MP3],
                            sr: int=22050,
                            duration: float=5.0
                            ) -> Tuple[np.array, np.array, List[str]]:
    """
    Takes a list of MP3 objects and imports them as numpy arrays with
    the specified sample rate (kHz) and duration (seconds).
    Returns:
        * numpy array of sources, with shape
        (len(mp3_list), 1, sr * duration)
        * genre array containing the genre info
        * split_label array containing split information
    """
    sources = np.empty((len(mp3_list), 1, int(sr * duration)))
    genres = list()
    split_labels = list()
    count = 0
    num_unprocessed = 0

    for mp3 in mp3_list:
        try:
            count += 1
            src, sr = load(mp3.path, sr=sr, mono=True)

            # trims the src file to be the correct length
            src = src[:int(sr * duration)]

            # adds a new axis to tell the Kapre mel_spectrogram
            # layer that the mp3 is in mono format.
            src = src[np.newaxis, :]

            # add in this source to the sources array
            sources[count - 1, :, :] = src[:, :]

            # append the genre
            genres.append(mp3.genre)

            # append train/valid/test splitting labels
            split_labels.append(mp3.split_label)

            if (count % 100 == 0):
                print("Finished step %s." % count)
        except (RuntimeError, NoBackendError):
            num_unprocessed += 1
            print("Could not process track id %s because of a runtime error."
                  % mp3.track_id())

    genres = np.array(genres)
    split_labels = np.array(split_labels)

    print("Number of files that could not be processed: %s" % num_unprocessed)

    return sources, genres, split_labels


def convert_and_save(audio_dir, track_ids, df, filename):
    """
    Boiler plate for running `prepare_mp3s_and_labels`.
    Frees up memory after processing.
    """
    X, y, split_labels = prepare_mp3s_and_labels(audio_dir, track_ids, df)
    np.savez(
        filename + '.npz', X, y, split_labels)

    # delete after saving to free up memory
    del X
    del y
    del split_labels
