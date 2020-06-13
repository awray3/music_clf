# Music CLF
Preprocessing and deep learning models for music genre classification.

This project was inspired by the [FMA dataset](https://github.com/mdeff/fma) and analysis done by [Priya Dwivedi](https://towardsdatascience.com/using-cnns-and-rnns-for-music-genre-recognition-2435fb2ed6af).

# Installation

Follow these steps if you wish to work with the notebook on your own machine.

## Directory Management

Within a clone of this repository create a new folder called `data`.
Obtain the FMA dataset (I used the "small" version) and metadata zip files
from their repository linked above. Unzip them in this folder.
You should now have folders `data/fma_small/` and `data/fma_metadata`.

## Environment setup

Create a new anaconda environment:

	conda create --name genre_recognition python=3.8
	conda activate genre_recognition

Then install the prerequisites:

	conda install -c conda-forge ffmpeg
	pip install -r requirements.txt

Note: `ffmpeg` is a tool for working with audio data that is used by the python package `librosa`, which will be used for audio processing.
This will allow us to work with `.mp3` files, `.wav` files, and more 
within python.

## Preprocessing

There are two preprocessing steps. The script `metadata_preprocess.py`
processes the metadata files given from the FMA dataset, and 
`mp3_to_melspec.py` converts the mp3 files into melspectrograms saved
as numpy files (`.npz`).

	python metadata_preprocess.py
	python mp3_to_melspec.py

# Roadmap
	
* Study working models on small datasets

