import os
import pandas as pd

from torch.utils.data import Dataset
import torchaudio
import torch

class Mp3Dataset(Dataset):

    def __init__(self, csv_path, audio_path):

        metadata = pd.read_csv(csv_path)
        self.audio_path = audio_path
        self.IDs = metadata.track_id.astype(str).to_list()
        self.genre_list = metadata.genre.to_list()

    def __len__(self):
        return len(self.IDs)


    def __getitem__(self, index):
        ID = self.IDs[index]
        genre = self.genre_list[index]

        waveform, sr = torchaudio.load(self.get_path_from_ID(ID))

        return waveform, sr, genre

    def get_path_from_ID(self, ID):
        """
        Gets
        """
        track_id = ID.zfill(6)

        return os.path.join(self.audio_path, track_id[:3], track_id + '.mp3')
