"""
The dataset class used to load mp3 data.
"""
import os

from torch.utils.data import Dataset
import torch
import torchaudio


sr = 22050

n_mels = 64
fft_window_pts = 512
fft_window_dur = fft_window_pts * 1.0 / sr # 23 ms window length
hop_size = fft_window_pts // 2 # 50% overlap between consecutive frames


class Mp3Dataset(Dataset):
    """
    The dataset class used to load mp3 data.
    Specify train/validation/test splits by altering the input
    for df.
    """

    def __init__(self, df, audio_path, duration):

        self.audio_path = audio_path
        self.IDs = df.track_id.astype(str).to_list()
        self.genre_list = df.genre.to_list()
        self.genre_dict = dict(zip(set(self.genre_list),
                                   range(len(set(self.genre_list)))))
        self.duration = duration

        # create the chain of preprocessing
        self.E = torchaudio.sox_effects.SoxEffectsChain()
        self.E.append_effect_to_chain("trim", [0, self.duration])
        self.E.append_effect_to_chain("rate", [sr])
        self.E.append_effect_to_chain("channels", ["1"])

    def __len__(self):
        return len(self.IDs)


    def __getitem__(self, index):
        ID = self.IDs[index]
        # genre = self.one_hot(self.genre_list[index])
        genre = self.genre_dict[self.genre_list[index]]

        self.E.set_input_file(self.get_path_from_ID(ID))

        waveform, _ = self.E.sox_build_flow_effects() # size: [1, len * sr]

        # padding in case the waveform is too short
        if waveform.size()[1] < self.duration * sr:
            # on small: only 98567 does this.
            new_waveform = torch.zeros(1, int(self.duration * sr))
            new_waveform[:, :waveform.size()[1]] = waveform
            waveform = new_waveform
        # convert to melspec
        melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=fft_window_pts,
            hop_length=hop_size,
            n_mels=n_mels)(waveform)

        # transpose the last two coordinates so that time is interpreted
        # as channels
        # melspec = melspec.permute(0, 2, 1)
        # return waveform
        return melspec, genre

    def get_path_from_ID(self, ID):
        """
        Gets the audio path from the ID.
        """
        track_id = ID.zfill(6)

        return os.path.join(self.audio_path, track_id[:3], track_id + '.mp3')

