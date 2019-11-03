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
"""
import os
import pandas as pd

from preprocessing.clean import clean_mp3_directory
from preprocessing.fma import create_df


def main(meta_dir: str,
         meta_path: str,
         audio_dir: str,
         fma_size: str = None) -> None:
    """
    Main function for the preprocessing script. This is a
    wrapper around create_df and clean_mp3_directory that manages
    the rows in the resulting dataframe.

    This script, when the fma_size is specified, will
        - create a metadata dataframe appropriate for loading data from the
            metadata csv
        - clean the mp3 directory, asking for permission to delete
        the files that were found to be corrupted
        - remove rows from the metadata dataframe
        - save the csv to the metadata directory.

    When the fma_size is not specified this will delete the corrupted files
    (again with permission) and just print out a list of files.
    """

    print('Beginning preprocessing.')

    # if using FMA dataset, create metadata csv if it doesn't already exist.
    if fma_size:
        print('You have selected to use one of the FMA datasets.')

        # check that the tracks.csv file exists
        if os.path.isfile(meta_path):
            if fma_size not in ['small', 'medium']:
                raise NotImplementedError('Please choose small or medium.')
            meta_df = create_df(fma_size, meta_path=meta_path)
            print(f"number rows: {len(meta_df)}")
        else:
            raise ValueError('Track csv path not found.')

    # run cleaning script on the audio directory.

    if os.path.isdir(audio_dir):
        corrupted_mp3_labels = clean_mp3_directory(audio_dir)

        print(corrupted_mp3_labels)
        if fma_size:

            meta_df = meta_df.loc[
                ~meta_df['track_id'].isin(corrupted_mp3_labels), :]

            print(f"number rows: {len(meta_df)}")

            # output meta_df to csv
            meta_df.to_csv(os.path.join(meta_dir, fma_size +
                                        '_track_info.csv'), index=False)

        else:
            print("Names of deleted files: ")
            print(corrupted_mp3_labels)

    else:
        raise ValueError('Audio directory not found.')



    print('Finished preprocessing.')



if __name__ == '__main__':
    TRACKS_PATH = os.path.join('data', 'fma_metadata', 'tracks.csv')
    META_DIR = os.path.join('data', 'fma_metadata')
    AUDIO_DIR = os.path.join('data', 'fma_small')
    main(META_DIR, TRACKS_PATH, AUDIO_DIR, fma_size='small')
