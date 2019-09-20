import pandas as pd
import os
from typing import List

from preprocessing.convert import convert_and_save
from preprocessing.tracks.paths import track_id_from_directory
from preprocessing.tracks.mp3 import MP3
from preprocessing.tracks.paths import create_mp3_objects

print("""
Welcome to my FMA preprocessing module.

* Please run create_fma_df.py first to create an mp3 metadata dataframe.
""")


def preprocess(audio_dir: str, fma_size: str, mp3_list: List[MP3]) -> None:
    """
    Main function for running the preprocessing step.
    Takes in the audio directory and a list of MP3 objects
    and produces a .npz file for training.
    """
    """
    FMA-Small Dataset
    """
    if fma_size == 'small':
        print('You have selected to preprocess the small version.')

        print('Beginning conversion of mp3s...')
        convert_and_save(audio_dir_small, small_track_ids, df_small,
                         os.path.join('..', 'small_processed_arr'))
        print('... done.')
    """
    FMA-Medium Dataset
    """
    if fma_size == 'medium':
        print('You have selected to preprocess the medium version.')

        med_track_ids = track_id_from_directory(audio_dir_med)

        chunkSize = 5000

        # convert and save the dataset in chunks.
        for i in range(5):
            convert_and_save(audio_dir_med,
                             med_track_ids[i * chunkSize:(i + 1) * chunkSize],
                             df_med, 'med_processed_arr_' + str(i + 1))


preprocess('small')
# preprocess(input('Select Size (small/med) for preprocessing: '))
