"""
Preprocessing an mp3 dataset.

Inputs: 
    * meta_dir: directory where the FMA metadata is contained.
    * meta_path - file path to the .csv file containing info about
    track id, genre, and optionally the split label.
    * audio_dir - folder containing all mp3 files to be loaded.
    * fma_size (default: None) - string for the size of FMA dataset 
    to process. If left to None, no csv is created.


Note: if using the FMA dataset, each song only lasts up to 30 seconds.

This script performs the following options
"""
import os
import pandas as pd

from preprocessing.clean import clean_mp3_directory
from preprocessing.fma import create_csv


def main(meta_dir: str,
         meta_path: str,
         audio_dir: str,
         fma_size: str=None) -> None:

    print('Beginning preprocessing.')

    # if using FMA dataset, create metadata csv if it doesn't already exist.
    if fma_size:
        print('You have selected to use one of the FMA datasets.')
        if os.path.isfile(meta_path):
            if fma_size not in ['small', 'medium']:
                raise NotImplementedError('Please choose small or medium.')
            create_csv(fma_size, meta_path=meta_path, output_dir=meta_dir)
        else:
            raise ValueError('Track csv path not found.')

    # run cleaning script on the audio directory.
    meta_df = pd.read_csv(meta_path)
    if os.path.isdir(audio_dir):
        meta_df = clean_mp3_directory(audio_dir, meta_df)
        meta_df.to_csv(meta_path)
    else:
        raise ValueError('Audio directory not found.')



    print('Finished preprocessing.')



if __name__ == '__main__':
    TRACKS_PATH = os.path.join('data', 'fma_metadata', 'tracks.csv')
    META_DIR = os.path.join('data', 'fma_metadata')
    AUDIO_DIR = os.path.join('data', 'fma_small', '000')
    main(META_DIR, TRACKS_PATH, AUDIO_DIR, fma_size='small')
