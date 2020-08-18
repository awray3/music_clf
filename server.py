"""
Flask server 
"""

import os
import random

from flask import Flask, request, jsonify

from genre_rec_service import Genre_Recognition_Service

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():

    # get audio file
    audio_file = request.files["file"]

    # random string of digits for file name
    file_name = str(random.randint(0, 100000))

    # save the file
    audio_file.save(file_name)

    # invoke the genre recognition service
    grs = Genre_Recognition_Service()

    # make prediction
    prediction = grs.predict(file_name)

    # remove the file
    os.remove(file_name)

    # send back the data in json format

    data = {"keyword": prediction}

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False)
