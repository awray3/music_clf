import os

from preprocessing.fma_metadata import get_fma_csv

meta_dir = os.path.join('data', 'fma_metadata')
meta_fp = os.path.join(meta_dir, 'tracks.csv')

fma_size = input('Select fma size (small or medium): ')

if fma_size != 'small' and fma_size != 'medium':
	raise ValueError('Need to select small or medium.')

df = get_fma_csv(meta_fp, fma_size)

df.to_csv(os.path.join(meta_dir, fma_size + '_track_info.csv'), index=False)
