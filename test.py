import pandas as pd
import os

from preprocessing.tracks.paths import create_mp3_objects
from preprocessing.convert import prepare_mp3s_and_labels

meta_fp = os.path.join('data', 'fma_metadata', 'small_track_info.csv')

# subdir = str(input('Enter 3-digit subdir: '))

subdir = '099'
audio_dir = os.path.join('data', 'fma_small', subdir)
# audio_dir = os.path.join('data', 'fma_small')

df = pd.read_csv(meta_fp, index_col=0)

mp3_list = create_mp3_objects(audio_dir, df)
print("Finished creating the mp3 objects.", "Beginning Librosa loading.")

X, y, split_labels = prepare_mp3s_and_labels(mp3_list, duration=1.0)
