"""
Main script for training the models.
"""
import os
import torch
from torch.utils import data
import torchaudio
import pandas as pd
import numpy as np

from preprocessing.create_fma_df import create_csv
from preprocessing.data import Mp3Dataset

fma_size = 'small'
meta_path = 'data/fma_metadata/small_track_info.csv'
audio_path = 'data/fma_' + fma_size

# create the metadata csv if not already made
if not os.path.isfile(meta_path):
    create_csv(fma_size)

# load the metadata csv.
meta_df = pd.read_csv(meta_path).sample(frac=1, random_state=1)

# shuffle into 60/20/20 train valid test split
train_df, valid_df, test_df = np.split(
        meta_df.sample(frac=1, random_state=1),
        [int(.6*len(meta_df)), int(.8*len(meta_df))])

# parameters
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 4}

# turn on sox
torchaudio.initialize_sox()

# Data generators
training_set = Mp3Dataset(train_df, audio_path, 1.0)
training_generator = data.DataLoader(training_set, **params)

validation_set = Mp3Dataset(valid_df, audio_path, 1.0)
validation_generator = data.DataLoader(validation_set, **params)

test_set = Mp3Dataset(test_df, audio_path, 1.0)
test_generator = data.DataLoader(test_set, **params)

# Testing ground
print(training_set[0][0].size()) # (1, 81, 128) now!

# Model stuff

# close sox
torchaudio.shutdown_sox()
