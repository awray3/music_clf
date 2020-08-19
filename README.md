# Music CLF
Genre classification using the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset.

This project was inspired by the [FMA dataset](https://github.com/mdeff/fma), though due to 
technical issues with this dataset I decided to instead use the GTZAN dataset.

# Usage

Follow these steps if you wish to use these scripts on your own machine.

## Testing the Flask Server

If you want to test the server functionality with just a local flask server, follow these steps. Run the server:

	cd server/flask
	python server.py

Then in a new terminal run 

	python local/client.py

This can be run on any audio file. To change the audio file adjust the path variable `TEST_AUDIO_FILE` inside of `client.py`. You should then see the prediction.

## Training

### Environment setup

Install the prerequisites by creating a new anaconda environment:

	conda env create -f environment.yml
	conda activate genre_rec

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

