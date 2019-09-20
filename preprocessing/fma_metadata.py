"""
takes in an FMA size specification (small, med only currently supported)
as well as the path to tracks.csv and returns a processed
dataframe compatible with preprocessing reqruirements.
"""
import pandas as pd


def get_fma_csv(meta_dir: str, fma_size: str) -> pd.DataFrame:
    """
    meta_dir - directory to fma_metadata/tracks.csv
    fma_size - "small" or "medium". "large" and "full" not implemented
    """

    if fma_size != 'small' and fma_size != 'medium':
        raise NotImplementedError

    tracks = pd.read_csv(meta_dir, index_col=0, header=[0, 1])

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

    return df
