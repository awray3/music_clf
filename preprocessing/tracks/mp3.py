"""
An MP3 object contains the path to the actual mp3
file in the given directory and the genre of the mp3.

Class has the method track_id, which returns only the
file name (without extension).
"""
import os


class MP3():

    def __init__(self, path: str, genre: str) -> None:
        self.path = path
        self.genre = genre

    def track_id(self) -> str:
        return os.path.split(self.path)[1][:-4]
