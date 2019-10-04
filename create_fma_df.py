import os

from preprocessing.fma import get_fma_csv


def create_csv(fma_size: str):
    meta_dir = os.path.join('data', 'fma_metadata')
    meta_fp = os.path.join(meta_dir, 'tracks.csv')

    if fma_size != 'small' and fma_size != 'medium':
        raise ValueError('Need to select small or medium.')

    df = get_fma_csv(meta_fp, fma_size)

    df.to_csv(os.path.join(meta_dir, fma_size + '_track_info.csv'), index=False)


if __name__ == "__main__":
    fma_size = input('Select fma size (small or medium): ')
    create_csv(fma_size)
