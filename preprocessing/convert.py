"""
Functions for taking raw mp3s and creating the inputs and targets
for training.
"""

# I added a comment right here!

from librosa import load
from typing import List, Set
import numpy as np

from preprocessing.tracks.paths import get_audio_path
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
                            split_option: str=False) -> Tuple[np.array, np.array, List[str]]:
    if split_option:
        split_labels = []

    genre = []
    count = 0
    num_unprocessed = 0
    sources = np.empty((len(mp3_list), 1, sr))
    
    for mp3 in mp3_list:
        try:
            count += 1
            src, sr = load(mp3.path, sr=sr, mono=True)

            # trims the src file to be the correct length
            src = src[:int(sr * len_second)]

            # adds a new axis to tell the Kapre mel_spectrogram
            # layer that the mp3 is in mono format.
            src = src[np.newaxis, :]

            # add in this source to the sources array
            sources[count - 1, :, :] = src[:, :]

            # append the genre from metadata
            # genres.append(
            #     meta_df.loc[meta_df['track_id'] == tr_id,
            #                 ('track', 'genre_top')].values[0])

            # append train/valid/test splitting labels
            if split_option:
                split_labels.append(
                    meta_df.loc[meta_df['track_id'] == tr_id,
                                ('set', 'split')].values[0])

            if (count % 100 == 0):
                print("Finished step %s." % count)
        except:
            print("Could not process track id %s because of a runtime error."
                  % tr_id)
            continuehints to annotate a function that returns an Iterable that always yields two values: a

    genres = np.array(genres)

    # trim the songs so that they can be put into a numpy array.
    shapes = unique_shapes(sources)
    min_song_len = min(shapes)
    for idx in range(len(sources)):
        sources[idx] = sources[idx][:, :min_song_len]

    return np.stack(sources), genres, split_labels


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
