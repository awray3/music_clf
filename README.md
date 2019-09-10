# Music CLF
Preprocessing and deep learning models for music genre classification.

# Disclaimer

This project is a work-in-progress. My code is being updated when possible, so the 
instructions below may not be quite up to speed yet. 

## Description

This project was inspired by the [FMA dataset](https://github.com/mdeff/fma) and analysis done by [Priya Dwivedi](https://towardsdatascience.com/using-cnns-and-rnns-for-music-genre-recognition-2435fb2ed6af). The goals of the project
include
- easier and improved preprocessing of mp3 data
- extending analysis to larger mp3 datasets

### Preprocessing

The `preprocess.py` module takes as input:

- a file directory containing mp3 files, assuming all mp3s are on the
same directory level.
- a `.csv` file describing the data. This file needs to have at least two columns called `track_id` and `genre`.
	- Optionally, you can have a third column, `split_label`, which contains the labels `training`, `valid`, and `test`, to perform the train/valid/test split automatically.

The result of `preprocessing.py` is a numpy `.npz` file to be read into the model notebook.

### Using the FMA dataset
If you wish to use the FMA dataset for these models, follow these additional steps before running `preprocess.py`.

1. Download the FMA metadata and the FMA dataset of your choice. Currently, only small and medium sizes are supported by my preprocessing library.
2. Place this library in the directory one level above the FMA directories.
3. Run `create_fma_df.csv` to turn `fma_metadata/tracks.csv` into a csv file named `[your size]_track_info.csv`, where `[your_size]` will either be `small` or `medium`.
4. Specify the FMA option when running `preprocess.py`.


## Explanation of `Librosa.load()`

The core of this project is in the `librosa` package. When we call `librosa.load()`, we can
specify the duration with the option `duration=...`. 
The output of this is a 1D numpy array which has length equal to `duration * sr`. 

Let's take an example to understand this clearly. Suppose we keep the sample rate `sr = 22050`, which is a sampling rate of 22kHz (which I believe means the computer pulls a sample from the song at that frequency). Then, if we set `duration=5.0`, this will take a 5 second duration of the song and will produce a numpy array of length `22050 * 5 = 110,250`.


# To-Do

- [ ] Get `preprocessing.py` working 
- [ ] Modify `convert.py` to work with an arbitrary list of `MP3` objects.
	- [x] Works with lists of mp3 objects
	- [x] include a `split_label` attribute in MP3 class
	- [x] finish modifying the `convert.py` script
	- [ ] implement specific exception classes
	- [ ] trim resulting numpy array in case of unreadable mp3s
- [ ] Write descriptions
	- [ ] for `preprocessing` library
	- [ ] for models
