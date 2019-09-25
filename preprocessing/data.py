"""
The dataset class used to load mp3 data.
"""
import os
import pandas as pd

from torch.utils.data import Dataset
import torch
import torchaudio

class Mp3Dataset(Dataset):
    """
    The dataset class used to load mp3 data.
    """

    def __init__(self, csv_path, audio_path, duration):

        metadata = pd.read_csv(csv_path)
        self.audio_path = audio_path
        self.IDs = metadata.track_id.astype(str).to_list()
        self.genre_list = metadata.genre.to_list()
        self.duration = duration

        # create the chain of preprocessing
        self.E = torchaudio.sox_effects.SoxEffectsChain()
        self.E.append_effect_to_chain("trim", [0, self.duration])
        self.E.append_effect_to_chain("rate", [16000])
        self.E.append_effect_to_chain("channels", ["1"])

    def __len__(self):
        return len(self.IDs)


    def __getitem__(self, index):
        ID = self.IDs[index]
        genre = self.genre_list[index]

        self.E.set_input_file(self.get_path_from_ID(ID))

        waveform, _ = self.E.sox_build_flow_effects() # size: [1, len * sr]

        # padding in case the waveform is too short
        if waveform.size()[1] < self.duration * 16000:
            # on small: only 98567 does this.
            new_waveform = torch.zeros(1, int(self.duration * 16000))
            new_waveform[:, :waveform.size()[1]] = waveform
            waveform = new_waveform
        # convert to melspec
        melspec = torchaudio.transforms.MelSpectrogram()(waveform)

        # return waveform
        return melspec, genre

    def get_path_from_ID(self, ID):
        """
        Gets the audio path from the ID.
        """
        track_id = ID.zfill(6)

        return os.path.join(self.audio_path, track_id[:3], track_id + '.mp3')
