"""
This module has functions to create an mp3 object for each
mp3 in a directory and assigning it its path and track id.
"""
from typing import List
import os
import pandas as pd

from preprocessing.tracks.mp3 import MP3


def create_mp3_objects(audio_dir: str, genre_df: pd.DataFrame) -> List[MP3]:
    """
    Walks the given directory looking for MP3 files
    and creates a list of them.

    genre_df should be a dataframe with at least the following
    column names:
    * track_id, with type str
    * genre, with type str
    * split_label, with type str. This column should only have
      "training", "validation", or "test" values.
    """

    # if 'split_label' feature is not provided (e.g. custom
    # splitting is desired), this will give every MP3 a blank splitting label.
    if "split_label" not in genre_df.columns:
        genre_df["split_label"] = ''

    mp3_list = []

    for path, dirnames, files in os.walk(audio_dir):
        for file in files:
            if dirnames == []:
                mp3_list.append(
                    MP3(
                        os.path.join(path, file),
                        genre_df.loc[
                            genre_df['track_id'] == int(file[:-4].lstrip('0')),
                            'genre'].values[0],
                        genre_df.loc[
                            genre_df['track_id'] == int(file[:-4].lstrip('0')),
                            'split_label'].values[0],
                    )
                )
    return mp3_list

# -----------------------------------------------------------------

# def track_id_from_directory(audio_dir):
#     """
#     Loops through the given directory and extracts the track ids.
#     Returns a list of track ids.
#     """
#     track_ids = []
#     for _, dirnames, files in os.walk(audio_dir):
#         if dirnames == []:
#             print('_')
#             # strip off the file extension '.mp3'
#             # track_ids.extend(int(file[:-4]) for file in files)
#     return track_ids


# def get_audio_path(audio_dir, track_id):
#     """
#     Return the path to the mp3 given the directory where the audio
#     is stored and the track ID.
#     Examples
#     --------
#     >>> import utils
#     >>> small_audio_dir = os.environ.get('fma_small')
#     >>> utils.get_audio_path(AUDIO_DIR, 2)
#     '../data/fma_small/000/000002.mp3'
#     """
#     track_id_str = '{:06d}'.format(track_id)
#     return os.path.join(audio_dir, track_id_str[:3], track_id_str + '.mp3')
