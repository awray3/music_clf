# Music CLF
Genre classification using the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset.

This project was inspired by the [FMA dataset](https://github.com/mdeff/fma), though due to 
technical issues with this dataset I decided to instead use the GTZAN dataset.

# Installation

Follow these steps if you wish to use these scripts on your own machine.

## Training

### Environment setup

Create a new anaconda environment:

	conda create --name genre_recognition python=3.8
	conda activate genre_recognition

Then install the prerequisites:

	conda install -c conda-forge ffmpeg
	pip install -r requirements.txt

Note: `ffmpeg` is a tool for working with audio data that is used by the python package `librosa`, which will be used for audio processing.
This will allow us to load `.mp3` and `.wav` files.

### Preprocessing

Once you have downloaded the GTZAN dataset, run the preprocessing script

	python preprocess.py

### Training

You can view available models to train in the `models.py` file. Currently there is logistic regression and a convolutional neural network avialable to train. Modify the model creation section in `train.py` and run

	python train.py

which will give you a model summary, training information, and evaluation diagnostics.



