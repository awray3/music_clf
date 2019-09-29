import os
import torch
from torch.utils.data import Dataset
from torch.utils import data
import torchaudio
import pandas as pd

sr = 22050

class Mp3Dataset(Dataset):
    """
    Mp3 dataset class to work with the FMA dataset.
    Input:
    df - pandas dataframe containing track_id and genre.
    audio_path - directory with mp3 files
    duration - how much of the songs to sample
    """

    def __init__(self, df: pd.DataFrame, audio_path: str, duration: float):

        self.audio_path = audio_path
        self.IDs = df['track_id'].astype(str).to_list()
        self.genre_list = df.genre.to_list()
        self.duration = duration

        self.E = torchaudio.sox_effects.SoxEffectsChain()
        self.E.append_effect_to_chain("trim", [0, self.duration])
        self.E.append_effect_to_chain("rate", [sr])
        self.E.append_effect_to_chain("channels", ["1"])

        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        genre = self.genre_list[index]

        # sox: set input file
        self.E.set_input_file(self.get_path_from_ID(ID))

        # use sox to read in the file using my effects
        waveform, _ = self.E.sox_build_flow_effects()  # size: [1, len * sr]

        melspec = self.mel(waveform)

        return melspec, genre

    def get_path_from_ID(self, ID):
        """
        Gets the audio path from the ID using the FMA dataset format
        """
        track_id = ID.zfill(6)

        return os.path.join(self.audio_path, track_id[:3], track_id + '.mp3')

if __name__ == '__main__':

    # my path to audio files
    audio_path = os.path.join('data', 'fma_small')

    # my dataframe that has track_id and genre info
    df = pd.read_csv('data/fma_metadata/small_track_info.csv')

    torchaudio.initialize_sox()

    dataset = Mp3Dataset(df, audio_path, 1.0)

    params = {'batch_size': 8, 'shuffle': True, 'num_workers': 2}

    dataset_loader = data.DataLoader(dataset, **params)

    print(next(iter(dataset_loader)))

    torchaudio.shutdown_sox()
