"""
Main script for training and analyzing models.
"""
# import os

import numpy as np
import pandas as pd

import torch
import torchaudio
from models import Baseline_cnn
from preprocessing.data import Mp3Dataset
from torch.utils.data import DataLoader

# audio path

# small fma dataset
meta_path = "./data/fma_metadata/small_track_info.csv"
audio_path = "./data/fma_small"

# medium fma dataset
# meta_path = "./data/fma_metadata/medium_track_info.csv"
# audio_path = "./data/fma_medium"

# load the metadata csv and shuffle it.
meta_df = pd.read_csv(meta_path).sample(frac=1, random_state=1)

# shuffle into 60/20/20 train valid test split
train_df, valid_df, test_df = np.split(
        meta_df.sample(frac=1, random_state=1),
        [int(.6 * len(meta_df)), int(.8 * len(meta_df))]
        )

# parameters
params = {'batch_size': 16, 'shuffle': True, 'num_workers': 0}
learning_rate = 1e-2
num_epochs = 2

# turn on sox
torchaudio.initialize_sox()

# Data generators
training_set = Mp3Dataset(train_df, audio_path, 1.0)
training_generator = DataLoader(training_set, **params)

# validation_set = Mp3Dataset(valid_df, audio_path, 1.0)
# validation_generator = DataLoader(validation_set, **params)

# test_set = Mp3Dataset(test_df, audio_path, 1.0)
# test_generator = DataLoader(test_set, **params)

# Model stuff
model = Baseline_cnn()

# create model instance

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# two epochs for now
for t in range(num_epochs):
    for idx, (batch_mel, batch_genre) in enumerate(training_generator):
        # forward pass:
        pred_genre = model(batch_mel)

        # calculate loss
        loss = loss_fn(pred_genre, batch_genre)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        print(f'Finished step {idx} of {6400//16}.', end='\r')
    print(loss)

# close sox
torchaudio.shutdown_sox()
