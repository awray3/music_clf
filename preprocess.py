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
    if os.path.isdir(audio_dir):
        clean_mp3_directory(audio_dir)
    else:
        raise ValueError('Audio directory not found.')

    print('Finished preprocessing.')

    # load the metadata csv and shuffle it.
    # meta_df = pd.read_csv(meta_path).sample(frac=1, random_state=1)

    # shuffle into 60/20/20 train valid test split
    # train_df, valid_df, test_df = np.split(
    # meta_df.sample(frac=1, random_state=1),
    # [int(.6 * len(meta_df)), int(.8 * len(meta_df))])

    # parameters
    # params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}

    # turn on sox
    # torchaudio.initialize_sox()

    # Data generators
    # training_set = Mp3Dataset(train_df, audio_path, 1.0)
    # training_generator = data.DataLoader(training_set, **params)

    # validation_set = Mp3Dataset(valid_df, audio_path, 1.0)
    # validation_generator = data.DataLoader(validation_set, **params)

    # test_set = Mp3Dataset(test_df, audio_path, 1.0)
    # test_generator = data.DataLoader(test_set, **params)

    # Model stuff

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


if __name__ == '__main__':
    TRACKS_PATH = os.path.join('data', 'fma_metadata', 'tracks.csv')
    META_DIR = os.path.join('data', 'fma_metadata')
    AUDIO_DIR = os.path.join('data', 'fma_small', '000')
    main(META_DIR, TRACKS_PATH, AUDIO_DIR, fma_size='small')
