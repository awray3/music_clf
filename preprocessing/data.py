"""
The dataset class used to load mp3 data.
"""
import os

import pandas as pd
import torch.utils.data as data
import torch
import torchaudio

sr = 22050

n_mels = 64
fft_window_pts = 512
fft_window_dur = fft_window_pts * 1.0 / sr  # 23 ms window length
hop_size = fft_window_pts // 2  # 50% overlap between consecutive frames


class Mp3Dataset(data.Dataset):
    """
    The dataset class used to load mp3 data.
    Specify train/validation/test splits by altering the input
    for df.
    """

    def __init__(self, df, audio_path, duration):

        self.audio_path = audio_path
        self.IDs = df.track_id.astype(str).to_list()
        self.genre_list = df.genre.to_list()
        self.genre_dict = dict(
            zip(set(self.genre_list), range(len(set(self.genre_list)))))
        self.duration = duration

        # create the chain of preprocessing
        self.E = torchaudio.sox_effects.SoxEffectsChain()
        self.E.append_effect_to_chain("trim", [0, self.duration])
        self.E.append_effect_to_chain("rate", [sr])
        self.E.append_effect_to_chain("channels", ["1"])

        self.clean_IDs()

        self.melspecf = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                       n_fft=fft_window_pts,
                                                       hop_length=hop_size,
                                                       n_mels=n_mels)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        # genre = self.one_hot(self.genre_list[index])
        genre = self.genre_dict[self.genre_list[index]]

        self.E.set_input_file(self.get_path_from_ID(ID))

        waveform, _ = self.E.sox_build_flow_effects()  # size: [1, len * sr]

        if waveform.size()[1] < self.duration * sr:
            new_waveform = torch.zeros(1, int(self.duration * sr))
            new_waveform[:, :waveform.size()[1]] = waveform
            waveform = new_waveform

        # convert to melspec
        melspec = self.melspecf(waveform).detach()

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

    def clean_IDs(self):
        """
        Checks if each of the files in the ID list exists or not.
        Removes these elements from the index if not found.
        """

        files_not_found = []
        for ID in self.IDs:
            if os.path.exists(self.get_path_from_ID(ID)):
                continue
            else:
                files_not_found.append(ID)

        if files_not_found != []:
            print('The following IDs will be removed from the ID list.')
            for ID in files_not_found:
                print(ID)

            self.IDs = [ID for ID in self.IDs if ID not in files_not_found]


if __name__ == '__main__':
    audio_path = os.path.join('data', 'fma_small')
    df = pd.read_csv('data/fma_metadata/small_track_info.csv')

    torchaudio.initialize_sox()

    dataset = Mp3Dataset(df, audio_path, 1.0)

    params = {'batch_size': 8, 'shuffle': True, 'num_workers': 2}

    dataset_loader = data.DataLoader(dataset, **params)

    print(next(iter(dataset_loader)))

    torchaudio.shutdown_sox()
