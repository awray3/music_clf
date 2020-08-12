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


def save_mfcc(
    dataset_path,
    json_path,
    option="mfcc",
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
    num_segments=5,
    n_mels=128,
    fmax=SAMPLE_RATE // 2,
):
    """
    Save the mfccs in a json path. Also split each sample up.

    option: can be either "mfcc" or "melspectrogram".
    """

    # dictionary to store data
    data = {"mappings": [], "labels": [], option: []}

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

    expected_length_of_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all genres
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):

        # ensure not in root level
        if dirpath is not dataset_path:

            # save the semantic (genre) label
            genre = dirpath.split("/")[-1]
            data["mappings"].append(genre)

            print(f"Processing {genre}")
            # process files for specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)

                waveform, _ = librosa.load(file_path)

                # process segments extracting mfcc and storing data

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    end_sample = start_sample + num_samples_per_segment

                    if option is "mfcc":
                        mfcc = librosa.feature.mfcc(
                            waveform[start_sample:end_sample],
                            n_mfcc=n_mfcc,
                            n_fft=n_fft,
                            hop_length=hop_length,
                        )

                        feature_to_export = mfcc.T
                    elif option is "melspectrogram":
                        # option is melspectrogram
                        melspec = librosa.feature.melspectrogram(
                            waveform[start_sample:end_sample],
                            n_mels=n_mels,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            fmax=fmax,
                        )
                        feature_to_export = melspec.T
                    else:
                        raise ValueError(
                            "option needs to be either melspectrogram or mfcc."
                        )

                    # store mfcc for segment if it has expected length
                    if len(feature_to_export) == expected_length_of_segment:
                        data[option].append(feature_to_export.tolist())
                        data["labels"].append(i - 1)

                    print(f"Processed {f}, segment{s+1}, label {i-1}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    # for loading MFCCs
    save_mfcc(DATASET_PATH, MFCC_JSON_PATH, num_segments=10, option="mfcc")

    # for loading melspectrograms
    # save_mfcc(DATASET_PATH, MELSPEC_JSON_PATH,
    #           num_segments=10, option="melspectrogram")
