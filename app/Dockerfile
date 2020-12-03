FROM python:3.7

RUN apt-get update -qq & apt-get install -y libsndfile1 ffmpeg
COPY app.py .
COPY genre_rec_service.py .
COPY config.py .
COPY model.h5 .
COPY templates/ .
COPY requirements.txt .

# install conda environment

RUN pip install -r requirements.txt

RUN python app.py