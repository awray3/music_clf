import os

from librosa.util.files import find_files
from torchaudio import info

from utilities import read_metadata_file

# setup audio directories
audio_dir = os.path.join(".", "data", "fma_small/")
raw_metadata_path = os.path.join(".","data", "fma_metadata", "tracks.csv")
target_metadata_path = os.path.join(".","data", "fma_metadata", "meta_df.csv")

# check if the path exists and exit if it doesn't.
if not os.path.exists(raw_metadata_path):
    raise FileNotFoundError("Data directory or metadata file not found; check it is in the correct place.")

# check if the metadata file has already been created
if os.path.exists(target_metadata_path):
    raise FileExistsError("Metadata file exists. Double check that the preprocessing hasn't already been executed.")

filepaths = find_files(audio_dir)

# Extract bad files using torchaudio
bad_files = []
for file in filepaths:
    try:
        info_obj = info(file)[0]
    except RuntimeError:
        bad_files.append(file)

# runs the preprocessing of the data file and conversion to melspecs
meta_df = read_metadata_file(raw_metadata_path, filepaths, bad_files)
meta_df.to_csv(target_metadata_path)
