# Music CLF
Genre classification using the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset.

This project was inspired by the [FMA dataset](https://github.com/mdeff/fma), though due to 
technical issues with this dataset I decided to instead use the GTZAN dataset.

You can find the deployed web app on Heroku: [https://music-clf.herokuapp.com/](https://music-clf.herokuapp.com/)

# Usage

Follow these steps if you wish to try out the code on your own machine.

## Environment Setup

Install the prerequisites by creating a new anaconda environment:

	conda env create -f environment.yml
	conda activate genre_rec

## Start the Flask server

If you want to test the server functionality with just a local flask server, follow these steps. Run the server:

	python app.py

Then visit `localhost:5000` in your web browser.

## Model Creation

If you wish to recreate the training process, first download the GTZAN dataset and refer to the steps below.

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
- [x] Deploy to Heroku
