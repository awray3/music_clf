import os
import warnings

import pandas as pd
from numpy.random import rand
from numpy import savez
from librosa import load, power_to_db
from librosa.feature import melspectrogram

from utilities import id_from_path

warnings.filterwarnings(action='ignore')

melspec_dir = os.path.join(".", "data", "fma_small_melspecs")
meta_path = os.path.join(".","data","fma_metadata","meta_df.csv")

global_dur = 2.0
global_sr = 22050

def save_melspec(
    mp3_path, song_len, duration, mid_pct=0.5, sr=global_sr, output_dir=melspec_dir
):
    """
    Take the mp3 path and make a melspectrogram, then pass
    the melspectrogram through the function log(x+1) and save it.
    Returns the filepath to the melspectrogram.
    mid_pct is the middle percentage (default 50%).
    """

    output_path = os.path.join(output_dir, id_from_path(mp3_path) + ".npz")

    # Choose a random offset uniformly in the middle 50% of the song.
    offset = (rand() - 0.5) * mid_pct * (song_len / 2) + (song_len / 2)

    src, _ = load(mp3_path, sr=sr, duration=duration, offset=offset)

    savez(output_path,
            melspectrogram(y=src, sr=sr, n_mels=128, fmax=2048)
            )

    print(f"Finished {id_from_path(mp3_path)}", end="\r")

    return output_path

if not os.path.exists(melspec_dir):
    os.mkdir(melspec_dir)

meta_df = pd.read_csv(meta_path, index_col=0)

# Make all melspectrograms and save them to a new directory,
# and at the same time put the output filepaths into the meta dataframe.
meta_df["mel_path"] = meta_df.apply(
    lambda row: save_melspec(
        row["mp3_path"], row["duration"], sr=global_sr, duration=global_dur
    ),
    axis=1,
)

# save the modified dataframe
meta_df.to_csv(meta_path)
