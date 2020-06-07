# Music CLF
Preprocessing and deep learning models for music genre classification.

This project was inspired by the [FMA dataset](https://github.com/mdeff/fma) and analysis done by [Priya Dwivedi](https://towardsdatascience.com/using-cnns-and-rnns-for-music-genre-recognition-2435fb2ed6af).

# Installation

First create a new anaconda environment:

	conda create --name genre_recognition python=3.8
	conda activate genre_recognition

Then install the prerequisites:

	conda install -c conda-forge ffmpeg
	pip install -r requirements.txt

`ffmpeg` is a tool for working with audio data that is used by the python
package `librosa`, which will be used for audio processing.
This will allow us to work with `.mp3` files, `.wav` files, and more 
within python.
