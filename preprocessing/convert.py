"""
Functions for taking raw mp3s and creating the inputs and targets
for training.
"""

# I added a comment right here!

from typing import List, Tuple

import numpy as np
from audioread.exceptions import NoBackendError
from librosa import load

from preprocessing.tracks.mp3 import MP3


def prepare_mp3s_and_labels(mp3_list: List[MP3],
                            sr: int = 22050,
                            duration: float = 5.0
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

    num_mp3s = len(mp3_list)
    num_unprocessed = 0
    count = 0

    sources = np.empty((len(mp3_list), 1, int(sr * duration)))
    genres = list()
    split_labels = list()

    print("Number of mp3s: %s" % num_mp3s)

    for mp3 in mp3_list:
        print('Processing track ' + mp3.track_id() + '.', end='\r')
        try:
            count += 1
            sources[count - 1, 0, :], _ = load(mp3.path,
                                               sr=sr,
                                               duration=duration,
                                               mono=True)

            # append the genre
            genres.append(mp3.genre)

            # append train/valid/test splitting labels
            split_labels.append(mp3.split_label)

        except (RuntimeError, NoBackendError):
            num_unprocessed += 1
            print(
                "Could not process track id %s because of a runtime error or a corrupt audio file."
                % mp3.track_id())
            print(f"number processed: {count+num_unprocessed}")
        except ValueError:
            num_unprocessed += 1
            print(
                "Track id " + mp3.track_id() +
                " appears to be shorter than the selected duration of %s seconds. Skipping track."
                % duration)
            print(load(mp3.path, sr=sr, duration=duration, mono=True)[0].shape)

    genres = np.array(genres)
    split_labels = np.array(split_labels)

    # trim the sources array if some mp3s fail to load.
    if num_unprocessed > 0:
        sources = np.delete(sources, np.s_[-num_unprocessed:], 0)

    print("shape of sources array: ", sources.shape)

    print("Number of files that could not be processed: %s" % num_unprocessed)

    return sources, genres, split_labels


def convert_and_save(mp3_list: List[MP3],
                     sr: int = 22050,
                     duration: float = 5.0,
                     chunk_size: int = 5000):
    """
    Boiler plate for running `prepare_mp3s_and_labels`.
    Frees up memory after processing.
    """
    num_chunks = int(np.ceil(len(mp3_list) / 5000.0))

    X, y, split_labels = prepare_mp3s_and_labels(audio_dir, track_ids, df)
    np.savez(filename + '.npz', X, y, split_labels)

    # delete after saving to free up memory
    del X
    del y
    del split_labels
