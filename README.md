# music_clf
Deep learning models for music genre classification. 




## Description

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



# To-Do

- [ ] Get `preprocessing.py` working 
- [ ] Modify `convert.py` to work with an arbitrary list of `MP3` objects.
	- [x] Works with lists of mp3 objects
	- [x] include a `split_label` attribute in MP3 class.
	- [ ] finish modifying the `convert.py` script.
- [ ] Write descriptions
	- [ ] for `preprocessing` library
	- [ ] for models
