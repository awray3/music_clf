import os

from preprocessing.fma_metadata import get_fma_csv

def create_fma_df(meta_dir, meta_fp, fma_option)

if fma_option not in ['small', 'medium']:
	raise ValueError('Need to select small or medium.')

df = get_fma_csv(meta_fp, fma_size)

df.to_csv(os.path.join(meta_dir, fma_size + '_track_info.csv'))

if __name__ == '__main__':

    meta_dir = os.path.join('data', 'fma_metadata')
    meta_fp = os.path.join(meta_dir, 'tracks.csv')

    fma_option = input('Select fma size (small or medium): ')
    create_fma_df(meta_dir, meta_fp, fma_option)
