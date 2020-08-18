"""
Class file for the genre recognition service to be used by flask.
"""

import numpy as np
import tensorflow.keras as keras

from preprocess import process_file

SAVED_MODEL = "model.h5"


class _Genre_Recognition_Service:

    model = None
    _instance = None

    _mappings = [
        "pop",
        "metal",
        "disco",
        "blues",
        "reggae",
        "classical",
        "rock",
        "hiphop",
        "country",
        "jazz",
    ]

    def predict(self, file_path):
        """
        Predict the genre of the wav file at file_path.
        """

        # extract the mfccs
        mfcc = self.preprocess(file_path)

        # convert from a 3d array to a 4d array
        mfcc = mfcc[..., np.newaxis]

        # predict on each chunk, extract the class with highest probability
        prediction_indices = np.argmax(self.model.predict(mfcc), axis=1)

        # ideally, each chunk will have the same prediction.
        # Some might be different, so take the most common element.
        prediction_idx = int(np.bincount(prediction_indices).argmax())

        prediction = self._mappings[prediction_idx]

        return prediction

    def preprocess(self, file_path):
        """
        Preprocesses the file.
        This will load the sample, split it into chunks,
        and return a 3D array. The batch dimension indexes the chunks.
        """

        mfcc = np.array(process_file(file_path))

        return mfcc


def Genre_Recognition_Service():
    """
    Ensure only one service is running.
    """
    if _Genre_Recognition_Service._instance is None:
        _Genre_Recognition_Service._instance = _Genre_Recognition_Service()
        _Genre_Recognition_Service.model = keras.models.load_model(SAVED_MODEL)

    return _Genre_Recognition_Service._instance


if __name__ == "__main__":

    grs = Genre_Recognition_Service()

    genre_1 = grs.predict("./data/jazz/jazz.00001.wav")
    genre_2 = grs.predict("./data/rock/rock.00001.wav")

    print(f"Predicted Genres: {genre_1}, {genre_2}")
