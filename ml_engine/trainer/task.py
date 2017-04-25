from __future__ import print_function

import numpy as np
import pandas as pd
import os
import glob
import pickle
import gzip
import h5py
#import dl_functions
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.cross_validation import train_test_split
#from matplotlib import pyplot as plt
#%matplotlib inline
from tensorflow.python.lib.io import file_io


# Defining an architecture.
def cnn_model_v_0(IMG_SIZE):
    global NUM_CLASSES
    NUM_CLASSES = 2
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            activation='relu'))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model


def cnn_model_v_1(IMG_SIZE):
    global NUM_CLASSES
    NUM_CLASSES = 2
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model


# Load data from pickle file located on a bucket.
with file_io.FileIO("gs://wellio-kadaif-tasty-images-project-pre-processed-images/pre_processed_images/image_data_20000_100.txt", 'r') as f:
    X, y = pickle.load(f)

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.4,
                             zoom_range=0.1,
                             horizontal_flip=False,
                             fill_mode='nearest')

# Creating a train, test split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)


# Creating a validation split out of the training set.
X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train,
                                                          test_size=0.1,
                                                          random_state=42)

# The labels need to be converted to categorical.
# Note that we have 2 categories: good (1) and bad (0) images.
y_train_fit_sparse = np_utils.to_categorical(y_train_fit, 2)

y_val_sparse = np_utils.to_categorical(y_val, 2)

y_test_sparse = np_utils.to_categorical(y_test, 2)

datagen.fit(X_train)


# Creating an instance of a CNN model.
# The image size is 100.
IMG_SIZE = 100

model_1 = model_1 = cnn_model_v_1(IMG_SIZE)

model_1.compile(loss='binary_crossentropy', optimizer='rmsprop',
                metrics=['accuracy'])

model_1.summary()

model_1.fit_generator(datagen.flow(X_train_fit, y_train_fit_sparse,
                                   batch_size=128), steps_per_epoch=len(X_train_fit),
                                   epochs=5, validation_data=(X_val, y_val_sparse))

score = model_1.evaluate(X_test, y_test_sparse, verbose=1)

print('Test loss: {:0,.4f}'.format(score[0]))
print('Test accuracy: {:.2%}'.format(score[1]))

# List of predictions.
predicted_images = []
for i in model_1.predict(X_test):
    predicted_images.append(np.where(np.max(i) == i)[0])

print("AUC: {:.2%}\n".format(roc_auc_score(y_test, predicted_images)))

# Creating a confusion matrix.
# plt.figure(figsize=(8, 8))
# cf = dl_functions.show_confusion_matrix(confusion_matrix(y_test, predicted_images), ['Class 0', 'Class 1'])
# plt.savefig('confusion_matrix.png')

# List of probabilities
predictions_probability = model_1.predict_proba(X_test)

# Creating ROC curve.
# plt.figure(figsize=(7, 7))
# rc = dl_functions.plot_roc(y_test, predictions_probability[:,1], "CNN - " + str(len(model_1.layers)) + " layers | # images: " + str(len(X)) + " | image size: " + str(IMG_SIZE), "Tasty Food Images")
# plt.savefig('roc_curve.png')
# model.save('model.h5')
# job_dir='gs://kadaif.getwellio.com'
#
#     # Save model.h5 on to google storage
# with file_io.FileIO('model.h5', mode='r') as input_f:
#     with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
#         output_f.write(input_f.read())
