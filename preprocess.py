""" processes metadata into a simpler metdata file."""

import os
import math
import json

import pandas as pd
import librosa

from config import (
    RAW_META_PATH,
    FMA_SIZE,
    FMA_DATA_PATH,
    SAMPLE_RATE,
    DURATION,
    SAMPLES_PER_TRACK,
    JSON_PATH,
)


def read_metadata_file(raw_metadata_path, fma_size):
    """
    read in the raw metadata file from the FMA dataset trims it.
    Output: pandas dataframe with two columns: genre and track id.
    """

    # load in the whole metadata file
    all_metadata = pd.read_csv(raw_metadata_path, header=[0, 1], index_col=0)

    # Genre column
    genre_col = [("track", "genre_top")]

    # select out the tracks for the current dataset along with genre
    df = all_metadata.loc[all_metadata[("set", "subset")] == fma_size, genre_col]

    # make track ID into its own column
    df.reset_index(inplace=True)

    # rename the columns
    df.columns = ["track_id", "genre"]

    # converrt the ID to a string padded to a length of 6 with zeros. This makes it so that
    # the first three characters of the ID string are the subfolder and all 6 form the file name.
    df["track_id"] = df["track_id"].apply(lambda x: str(x).zfill(6))

    # # Remove bad mp3s from the dataframe so that we skip them.
    # if df.mp3_path.isin(bad_filepaths).sum():
    #     df.drop(
    #         df.loc[df.mp3_path.isin(bad_filepaths), :].index,
    #         inplace=True
    #     )
    #     print(f"Dropped {len(bad_filepaths)} rows from the dataframe.")

    # df['duration'] = df['mp3_path'].apply(lambda x:
    #                                       librosa.get_duration(filename=x))

    return df


def save_mfcc(meta_df, json_path, num_chunks=5, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Load waveforms and extract MFCCs. Splits each track into chunks `num_chunks` times. 
    """

    samples_per_chunk = int(SAMPLES_PER_TRACK / num_chunks)
    expected_chunk_length = math.ceil(samples_per_chunk / hop_length)

    # initialize the data dictionary
    data = {"genre": [], "label": [], "mfcc": []}

    # get the list of genres
    data["genre"] = meta_df["genre"].unique().tolist()

    # encode the labels into the dataframe
    meta_df["genre_enc"] = meta_df["genre"].apply(data["genre"].index)

    # loop through the meta_df array
    for idx, row in meta_df.iterrows():

        # get the mp3 file location from the track id
        track_path = os.path.join(
            FMA_DATA_PATH, row["track_id"][:3], row["track_id"] + ".mp3"
        )

        try:  # in case of import errors, e.g. corrupted data files

            # load the waveform for that track
            waveform, _ = librosa.load(track_path)

            print(len(waveform))
            # skip the track if it is too short
            # if len(waveform) < SAMPLES_PER_TRACK:
            #     continue

            for s in range(num_chunks):
                start_idx = s * samples_per_chunk
                end_idx = start_index + samples_per_chunk

                # get the mfccs
                mfcc = librosa.feature.mfcc(
                    waveform[start_idx:end_idx],
                    n_mfcc=n_mfcc,
                    n_fft=n_fft,
                    hop_length=hop_length,
                )
                mfcc = mfcc.T

                print(f"Length of mfcc: {len(mfcc)}, expected: {expected_chunk_length}")

                if len(mfcc) == expected_chunk_length:
                    # convert to list from numpy array
                    data["mfcc"].append(mfcc.tolist())
                    data["label"].append(row["genre_enc"])

                    print(
                        f"Processing track {row['track_id']}, chunk {s+1}, label {row['genre_enc']}"
                    )
        except:
            pass

    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":

    # extract the track id and genre info from the metadata csv
    meta_df = read_metadata_file(RAW_META_PATH, FMA_SIZE)

    # Load in the MFCCs and save the data in a JSON file.
    save_mfcc(meta_df, JSON_PATH)

    # setup audio directories
    # audio_dir = os.path.join(".", "data", "fma_small/")
    # raw_metadata_path = os.path.join(".", "data", "fma_metadata", "tracks.csv")
    # target_metadata_path = os.path.join(".", "data", "fma_metadata", "meta_df.csv")

    # melspec_dir = os.path.join(".", "data", "fma_small_melspecs")

    # #  Make the folders for the given models:
    # if not os.path.exists("./models/"):
    #     os.mkdir("./models/")

    # # check if the path exists and exit if it doesn't.
    # if not os.path.exists(raw_metadata_path):
    #     raise FileNotFoundError(
    #         "Data directory or metadata file not found; check it is in the correct place."
    #     )

    # # check if the metadata file has already been created
    # if os.path.exists(target_metadata_path):
    #     raise FileExistsError(
    #         "Metadata file exists. Double check that the \
    #         preprocessing hasn't already been executed."
    #     )

    # filepaths = librosa.util.find_files(audio_dir)

    # # Extract bad files using torchaudio
    # bad_files = []
    # for file in filepaths:
    #     try:
    #         info_obj = info(file)[0]
    #     except RuntimeError:
    #         bad_files.append(file)

    # # runs the preprocessing of the data file and conversion to melspecs
    # meta_df = read_metadata_file(raw_metadata_path, filepaths, bad_files)

    # # delete samples less than 30 seconds
    # meta_df.drop(meta_df.loc[(meta_df.duration < 28), :].index, axis=0, inplace=True)

    # meta_df["mel_path"] = meta_df["mp3_path"].apply(
    #     lambda path: os.path.join(melspec_dir, id_from_path(path) + ".npz")
    # )

    # meta_df = attach_onehot_encoding(meta_df)

    # meta_df.to_csv(target_metadata_path)
