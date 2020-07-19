""" processes metadata into a simpler metdata file."""

import os

import pandas as pd
import librosa
from torchaudio import info

from utilities import id_from_path


def read_metadata_file(path, all_filepaths, bad_filepaths):
    """
    read in the raw metadata file and trim it.
    """
    all_metadata = pd.read_csv(path, header=[0, 1], index_col=0)

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
                                          librosa.get_duration(filename=x))

    return df


def attach_onehot_encoding(df):
    """
    Append the onehot representation of `column` onto the right end
    of the array df.
    """

    df = pd.concat([df, pd.get_dummies(df.genre)], axis=1)

    return df


if __name__ == "__main__":
    # setup audio directories
    audio_dir = os.path.join(".", "data", "fma_small/")
    raw_metadata_path = os.path.join(".", "data", "fma_metadata", "tracks.csv")
    target_metadata_path = os.path.join(
        ".", "data", "fma_metadata", "meta_df.csv")

    melspec_dir = os.path.join(".", "data", "fma_small_melspecs")

    #  Make the folders for the given models:
    if not os.path.exists("./models/"):
        os.mkdir("./models/")


    # check if the path exists and exit if it doesn't.
    if not os.path.exists(raw_metadata_path):
        raise FileNotFoundError(
            "Data directory or metadata file not found; check it is in the correct place.")

    # check if the metadata file has already been created
    if os.path.exists(target_metadata_path):
        raise FileExistsError(
            "Metadata file exists. Double check that the \
            preprocessing hasn't already been executed.")

    filepaths = librosa.util.find_files(audio_dir)

    # Extract bad files using torchaudio
    bad_files = []
    for file in filepaths:
        try:
            info_obj = info(file)[0]
        except RuntimeError:
            bad_files.append(file)

    # runs the preprocessing of the data file and conversion to melspecs
    meta_df = read_metadata_file(raw_metadata_path, filepaths, bad_files)

    # delete samples less than 30 seconds
    meta_df.drop(meta_df.loc[(meta_df.duration < 28),
                             :].index, axis=0, inplace=True)

    meta_df["mel_path"] = meta_df["mp3_path"].apply(
        lambda path: os.path.join(melspec_dir, id_from_path(path) + ".npz"))

    meta_df = attach_onehot_encoding(meta_df)

    meta_df.to_csv(target_metadata_path)
