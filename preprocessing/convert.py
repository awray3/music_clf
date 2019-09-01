"""
Functions for taking raw mp3s and creating the inputs and targets
for training.
"""

# I added a comment right here!

from librosa import load
import numpy as np

from preprocessing.tracks.paths import get_audio_path


class InvalidFMASizeSpecification(Exception):
    pass


def unique_shapes(X):
    """
    Takes a list of numpy arrays, X, and compiles the shapes of the
    components. To be used inside the `prepare_mp3s_and_lables` function
    to get all of the shapes of loaded mp3 files.
    """
    shapes = set([])
    for arr in X:
        shapes.add(arr.shape[1])
    return shapes


def prepare_mp3s_and_labels(audio_dir: str,
                            meta_df: pd.DataFrame,
                            len_second=1.0,
                            track_id_col: str='track_id'):
    """
    Needs a dataframe where
    * rows are individual tracks
    * columns are 'track_id', 'genre', and (optionally) 'split_label'.

    If the 'split_label' column is omitted (e.g. you wish to create
    a custom train/test split) then the code will process the files
    without the splitting labels.
    """
    sources = []
    genres = []
    split_labels = []
    count = 0
    # for tr_id in track_ids:
    for tr_id in meta_df.loc[track_id_col].tolist():
        try:
            count += 1
            src, sr = load(
                get_audio_path(audio_dir, tr_id),
                sr=None, mono=True
            )

            # trims the src file to be the correct length
            src = src[:int(sr * len_second)]

            # adds a new axis to tell the Kapre mel_spectrogram
            # layer that the mp3 is in mono format.
            src = src[np.newaxis, :]

            sources.append(src)

            # append the genre from metadata
            genres.append(
                meta_df.loc[meta_df['track_id'] == tr_id,
                            ('track', 'genre_top')].values[0])

            # append train/valid/test splitting labels
            split_labels.append(
                meta_df.loc[meta_df['track_id'] == tr_id,
                            ('set', 'split')].values[0])

            if (count % 100 == 0):
                print("Finished step %s." % count)
        except:
            print("Could not process track id %s because of a runtime error."
                  % tr_id)
            continue

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
