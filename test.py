"""
Main script for testing preprocessing module.
"""
import os
from time import time
import pandas as pd

from preprocessing.tracks.paths import create_mp3_objects
from preprocessing.convert import prepare_mp3s_and_labels

print("Loading metadata.")
meta_fp = os.path.join('data', 'fma_metadata', 'medium_track_info.csv')

# subdir = str(input('Enter 3-digit subdir: '))
# subdir = '133'
# subdir = '054'
# audio_dir = os.path.join('data', 'fma_small', subdir)
audio_dir = os.path.join('data', 'fma_medium')

df = pd.read_csv(meta_fp, index_col=0)

# 080391
# df.drop(df.loc[df['track_id'] == 80391, :].index, axis=0, inplace=True)
#hopefully the issue won't happen until the "mass issue"
# print(df.head())

mp3_list = create_mp3_objects(audio_dir, df)
print("Finished creating the mp3 objects.", "Beginning Librosa loading.")
t1 = time()
X, y, split_labels = prepare_mp3s_and_labels(mp3_list, duration=2.0)

t2 = time()

print(f"Completed preprocessing in {t2-t1}")
