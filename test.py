"""
Tutorial for pytorch and torchaudio
"""
import os
import torchaudio
from preprocessing.data import Mp3Dataset

audio_path = os.path.join('data', 'fma_small')

csv_path = os.path.join('data', 'fma_metadata', 'small_track_info.csv')


torchaudio.initialize_sox()

mydata = Mp3Dataset(csv_path, audio_path, 5.0)

sizes = []

for i in range(len(mydata)):
    try:
        sizes.append(mydata[i].size()[1])
    except RuntimeError as e:
        print("Got error:" + repr(e))

print(set(sizes))

