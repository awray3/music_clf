import pandas as pd
import os

from preprocessing.tracks.paths import create_mp3_objects

meta_fp = os.path.join('data', 'fma_metadata', 'small_track_info.csv')

audio_dir = os.path.join('data', 'fma_small', '000')

df = pd.read_csv(meta_fp, index_col=0)

mp3_list = create_mp3_objects(audio_dir, df)
