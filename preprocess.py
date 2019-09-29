"""
Main script for testing preprocessing module.
"""
import os
from time import time
import pandas as pd

from preprocessing.tracks.paths import create_mp3_objects
from preprocessing.convert import prepare_mp3s_and_labels
from create_fma_df import create_fma_df 

    

def run_preprocessing(fma_option: str, duration=1.0):

    print("Loading metadata.")

    meta_dir = os.path.join('data', 'fma_metadata')
    meta_fp = os.path.join(meta_dir, fma_option + '_track_info.csv')

    if not os.path.exists(meta_fp):
        create_fma_df(meta_dir, meta_fp, fma_option)

    audio_dir = os.path.join('data', 'fma_' + fma_option)

    df = pd.read_csv(meta_fp, index_col=0)

    mp3_list = create_mp3_objects(audio_dir, df)
    print("Finished creating the mp3 objects.", "Beginning Librosa loading.")

    t1 = time()

    X, y, split_labels = prepare_mp3s_and_labels(mp3_list, duration=duration)

    t2 = time()

    print(f"Completed preprocessing in {t2-t1} seconds.")

if __name == '__main__':

    fma_option = input('Preprocess FMA size small or medium? :')

    if fma_option not in ['small', 'medium']:
        raise ValueError('Please select small or medium.')

    duration = input('Enter float for duration in seconds (default 1.0): ')

    run_preprocessing(fma_option)
