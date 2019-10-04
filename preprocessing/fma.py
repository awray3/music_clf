"""
takes in an FMA size specification (small, med only currently supported)
as well as the path to tracks.csv and returns a processed
dataframe compatible with preprocessing reqruirements.
"""
import os
import pandas as pd


def create_csv(fma_size: str,
               meta_path: str,
               output_dir: str) -> pd.DataFrame:
    """
    meta_dir - directory to fma_metadata/tracks.csv
    fma_size - "small" or "medium". "large" and "full" not implemented
    """

    tracks = pd.read_csv(meta_path, index_col=0, header=[0, 1])

    keep_cols = [('track', 'genre_top')]

    # slice out genre and split label
    if fma_size == 'small':
        df = tracks.loc[
            tracks[('set', 'subset')] == fma_size,
            keep_cols
        ]
    elif fma_size == 'medium':
        df = tracks.loc[
            tracks[('set', 'subset')].isin(['small', 'medium']),
            keep_cols
        ]

    # create track_id from the index
    df.reset_index(inplace=True)

    # rename columns
    df.columns = ['track_id', 'genre']

    df.to_csv(os.path.join(output_dir, fma_size +
                           '_track_info.csv'), index=False)
