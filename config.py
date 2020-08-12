"""
Holds all of the global variables.
"""
import os

# data directory folders
DATA_DIR = "./data"

# mfcc or melspectrogram?
DATA_OPTION = "mfcc"

# processed data location
JSON_PATH = os.path.join(DATA_DIR, DATA_OPION + ".json")


# waveform properties
SAMPLE_RATE = 22050
DURATION = 25  # seconds; some files are not quite 30 seconds.
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
