# Playlist context prediction

Materials supporting the ECIR '21 paper: "Predicting the Listening Contexts of Music Playlists Using Knowledge Graphs", by [Giovanni Gabbolini](https://giovannigabbolini.github.io) and [Derek Bridge](http://www.cs.ucc.ie/~dgb/).

## Setup
- Create conda env from `environment.yml`, and activate it.

## Prepare the data:
- Download the MPD dataset from [here](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge), and place it in `res/r/spotify_recsys2018`
- Run `src/data.py`

## Prepare the models:
- Run `src/experiments/tune.py` to tune the models;
- Run `src/experiments/run.py` to save the model with best parameters;

## Replicate the results:
- Run `src/experiments/tables.py` for the tables;
- Run `src/experiments/sign_test.py` to run significance tests.