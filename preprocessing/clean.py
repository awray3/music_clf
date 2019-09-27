"""
The purpose of this script is to clean the mp3 directory of 
damaged mp3 files that don't load.
"""
import os

import torchaudio

def clean_mp3_directory(audio_dir):
    for root, dirs, files in os.walk(audio_dir):
        if dirs == []:
            for f in files:
                try:
                    print('Loading ' + f, end='\r')
                    torchaudio.load(os.path.join(root, f))
                except RuntimeError as e:
                    print("Unable to load file" + f + ", " + repr(e))
                    delete_option = input('Delete this file?')
                    if delete_option == 'yes':
                        os.remove(os.path.join(root,f))





if __name__ == '__main__':
    audio_dir = 'data/fma_small'
    clean_mp3_directory(audio_dir)
