"""
Main script for training the models.
"""
import os
import torch
import torch.utils.data as data
import torchaudio
import pandas as pd
import numpy as np

from create_fma_df import create_csv
from preprocessing.data import Mp3Dataset
from models import Baseline_cnn




def main(fma_size, meta_path, audio_path, duration):
    # create the metadata csv if not already made
    if not os.path.isfile(meta_path):
        create_csv(fma_size)

    # load the metadata csv.
    meta_df = pd.read_csv(meta_path).sample(frac=1, random_state=1)

    # shuffle into 60/20/20 train valid test split
    train_df, valid_df, test_df = np.split(
        meta_df.sample(frac=1, random_state=1),
        [int(.6 * len(meta_df)), int(.8 * len(meta_df))])

    # parameters
    params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}

    # turn on sox
    torchaudio.initialize_sox()

    # Data generators
    training_set = Mp3Dataset(train_df, audio_path, 1.0)
    training_generator = data.DataLoader(training_set, **params)

    # validation_set = Mp3Dataset(valid_df, audio_path, 1.0)
    # validation_generator = data.DataLoader(validation_set, **params)

    # test_set = Mp3Dataset(test_df, audio_path, 1.0)
    # test_generator = data.DataLoader(test_set, **params)

    batch_mel, batch_genre = next(iter(training_generator))

    print(batch_mel)
    print(batch_mel.shape)

    ### Model stuff

    # create model instance

    # loss_fn = torch.nn.CrossEntropyLoss()

    # learning_rate = 1e-2
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # # two epochs for now
    # i = 1
    # for t in range(2):
    #     for batch_mel, batch_genre in training_generator:
    #         # # # forward pass:
    #         pred_genre = model(batch_mel)

    #         # # # calculate loss
    #         loss = loss_fn(pred_genre, batch_genre)
    #         optimizer.zero_grad()

    #         loss.backward()
    #         optimizer.step()
    #         print(f'Finished step {i} of {6400//16}.', end='\r')
    #         i += 1
    #     print(loss)


    # close sox
    torchaudio.shutdown_sox()


if __name__ == '__main__':
    fma_size = 'small'
    meta_path = 'data/fma_metadata/small_track_info.csv'
    audio_path = 'data/fma_' + fma_size
    duration = 1.0  # 1 second long spectrograms
    main(fma_size, meta_path, audio_path, duration)
