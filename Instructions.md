# Instructions.

## 1 - EDA.
Notebook `01-tasty-images-EDA.ipynb` is intended to explore data and perform some initial analysis to gain insights as well as present some visualizations.

## 2 - Getting the data.
After EDA, it was decided that the best images are from `epicurious.com` as they have high social media scores and they are taken by professionals, so the good images will be curated from that site. Also, the worse images are coming from `food.com` as they have low social scores and they are taken by amateurs by their smartphones. Even though a given food recipe may high reviews, the associated photos can be of really bad quality, not appealing at all.

Also, most of the food recipes acquired by Wellio are sourced from `food.com` and `epicurious.com`.

We will get 10,000 images from each site, totaling a balanced data set of 20,000 images.

Images will be copied over a bucket in Google Cloud Storage as the storage in Datalab is ephemeral.

This can be found in notebook `02-tasty-images-download-images.ipynb`.

## 3 - Splitting images into a training and testing data sets.
This functionality has not been used for the project, but it can be useful and handy if required.

This can be found in the notebook `03-tasty-images-create-train-test-split-images.ipynb`.

## 4 - Images pre-processing.
Images come on different sizes, mostly 640 pixels wide and 480 pixels tall. We resize them to a desired size and use `keras.preprocessing.image` to convert them into tensors using a custom function dl_functions.`normalize_images_array`
found in the script `dl_functions.py` in the `scripts` folder.

The sizes used are 25, 50 and 100. The created data sets, ready to train a model, will be saved as pickle files and then stored into Google Cloud Storage as well.

Note that due to lack of enough computing resources, I was not able to pre-process the images to a bigger size (i.e 150 and bigger).

This can be found in the notebook `4-tasty-images-pre-processing-images.ipynb`.

## 5 - Training convolutional neural network models.

No GPUs were used as they are not available in Google Datalab. For data augmentation only 10 epochs were performed as it can take more than a day to train a model, and even 5 epochs for images of bigger size.

The following two notebooks were intended to train some convolutional neural network architectures with few images to check their performance.

The architectures used can be found on `dl_functions.py` file under the `scripts` folder.

`05-tasty-images-CNN-model-initial-1.ipynb`

`06-tasty-images-CNN-model-initial-2.ipynb`

The following notebook train models with images resized to 25.

`07-tasty-images-CNN-model-image-size-25.ipynb`

The following two notebooks train models with images resized to 50.

`08-tasty-images-CNN-model-image-size-50-1.ipynb`

`09-tasty-images-CNN-model-image-size-50-2.ipynb`

The following notebook train models with images resized to 100.

`10-tasty-images-CNN-model-image-size-100.ipynb`

The following notebook train models with images resized to 25 using data augmentation.

`11-tasty-images-CNN-data-augmentation-model-image-size-25.ipynb`

The following notebook train models with images resized to 100 using data augmentation.

`13-tasty-images-CNN-data-augmentation-model-image-100.ipynb`

## 6 - Transfer learning.

`14-transfer-learning-InceptionV3-data-prep-1.ipynb`

`15-transfer-learning-InceptionV3-models-2.ipynb`
