"""
Client for the Flask server.
"""

import requests

URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_PATH = "./data/jazz/jazz.00005.wav"

if __name__ == "__main__":

    # get the audio stream
    audio_stream = open(TEST_AUDIO_PATH, "rb")

    # send the audio file and type of file
    values = {"file": (TEST_AUDIO_PATH, audio_stream, "audio/wav")}

    response = requests.post(URL, files=values)

    data = response.json()

    print(f"Predicted Keyword is {data['keyword']}")

