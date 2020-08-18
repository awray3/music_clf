# Music CLF
Genre classification using the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset.

This project was inspired by the [FMA dataset](https://github.com/mdeff/fma), though due to 
technical issues with this dataset I decided to instead use the GTZAN dataset.

# Installation

Follow these steps if you wish to use these scripts on your own machine.

## Testing the server

If you want to test the server functionality, follow these steps. Run the server:

	cd server/flask
	python server.py

Then in a new terminal run 

	python local/client.py

This can be run on any audio file. To change the audio file adjust the path variable `TEST_AUDIO_FILE` inside of `client.py`. You should then see the prediction.

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

Once you have downloaded the GTZAN dataset, run the preprocessing script:

	python classifier/preprocess.py

This script will extract MFCCs (mel-frequency cepstral coefficients) from the `.wav` files and store the 
data and labels in a `.json` file.

### Training

You can view available models to train in the `models.py` file.
Currently there is logistic regression and a convolutional neural network avialable to train.
Modify the model creation section in `train.py` and run

	python classifier/train.py

which will give you a model summary, training information, and evaluation diagnostics. 

# Roadmap

- [x] Refactor code into scripts
- [x] Get Flask server working
- [x] Ping server with client
- [ ] Add uwsgi layer
- [ ] Add nginx layer
- [ ] Build docker files and images

