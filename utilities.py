import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from librosa.feature import melspectrogram
from librosa import power_to_db, load
from librosa.display import specshow
from tensorflow.keras.utils import Sequence

def view_melspec(source, sr):
    plt.figure(figsize=(10, 4))
    S = melspectrogram(source, sr=sr)
    S_dB = power_to_db(S, ref=np.max)
    specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=sr,
                             fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()


def read_metadata_file(path, all_filepaths, bad_filepaths):
    all_metadata = pd.read_csv(path, header=[0,1], index_col=0)
    
    cols_to_keep = [('track', 'genre_top')]

    # This will be the main dataframe for here on out:
    df = all_metadata.loc[
                all_metadata[('set', 'subset')] == 'small',
                cols_to_keep
    ]

    df.reset_index(inplace=True)
    
    df.columns = ['track_id', 'genre']
    
    # encode the columns
    enc = LabelEncoder()
    
    df['genre_enc'] = enc.fit_transform(df['genre'].to_numpy().reshape(-1,1))
    
    # add filepaths to the dataframe
    df['path'] = all_filepaths
    
    # Remove bad mp3s from the dataframe so that we skip them.
    if df.path.isin(bad_filepaths).sum():
        df.drop(
            df.loc[df.path.isin(bad_filepaths), :].index,
            inplace=True
        )
        print(f"Dropped {len(bad_filepaths)} rows from the dataframe.")
        
    return df


class Batch_generator(Sequence) :
    """
    Data generator class. Takes in the meta dataframe (or a train/test/split) 
    and peels off the paths and the encoded genres.
    """
  
    def __init__(self, meta_df, batch_size, sr, duration):
        self.music_filepaths = meta_df['path'].to_list()
        self.labels = meta_df['genre_enc'].to_numpy()
        self.batch_size = batch_size
        self.sr = sr
        self.duration = duration
    
    def __len__(self):
        """
        Return number of batches.
        """
        return (np.ceil(len(self.music_filepaths) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx):
        
        batch_x = self.music_filepaths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        return self._stack_melspecs(batch_x), batch_y
    
    def _stack_melspecs(self, filepath_list):
        """
        A helper function for loading batches of melspectrograms.
        Stack the melspectrograms of the files in the list.
        Extends by zeros if needed.
        """
        sources = [load(file, sr=self.sr, duration=self.sr)[0] for file in filepath_list]

        melspecs = [melspectrogram(src, sr=self.sr) for src in sources]
        
        
        stacked_arr = np.zeros((len(filepath_list),
                                max(
                                    [melspec.shape[0] for melspec in melspecs]
                                ),
                                max(
                                    [melspec.shape[1] for melspec in melspecs]
                                )
                               )
                              )

        for i in range(len(filepath_list)):
            stacked_arr[i,
                    :melspecs[i].shape[0],
                    :melspecs[i].shape[1]] = melspecs[i]
        
        return stacked_arr
