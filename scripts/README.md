# Usage of scripts.

`train_cnn_model.py` and `train_cnn_data_augmentation_model.py` need to be `scp` to a GPU virtual instance and be run from the command line as for example:

```
python train_cnn_model.py 25 50
```

and:

```
python train_cnn_data_augmentation_model.py 25 50
```

where the **first argument** is the image size (25 in the examples above) and the **second argument** is the number of epochs to train the models (50 in the examples above).

NOTE that the image size available are 25, 50 and 100, which corresponds to the available pickle files of preprocessed images.

Example of `scp`:

```
scp -i ~/.ssh/google_compute_engine ~/Downloads/train_cnn_model.py 35.185.49.209:
scp -i ~/.ssh/google_compute_engine ~/Downloads/train_cnn_data_augmentation_model.py 35.185.49.209:
```

The scripts create .png files (ROC curves and confusion matrix) that can be `scp` to the local Mac as:

```
scp -i ~/.ssh/google_compute_engine 35.185.49.209:~/*.png /Users/carles/
```

The scripts also create text files with summary metrics and saves the models and the models weights.

FINALLY, the script `dl_functions.py` is used by the notebooks.
