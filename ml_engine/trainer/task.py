from __future__ import print_function
# example5-keras.py

import numpy as np
#np.random.seed(42)
#import tensorflow as tf
#tf.set_random_seed(42)
import pandas as pd
import os
import glob
import pickle
import gzip
#import dl_functions
#from IPython.display import display
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.cross_validation import train_test_split
#from matplotlib import pyplot as plt
from tensorflow.python.lib.io import file_io
#from datetime import datetime
#import time

# import cPickle as pickle
#import argparse
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.optimizers import RMSprop
#import tensorflow
#import pandas as pd
#import json
#import seaborn as sns
#import datalab.storage as storage
#import datalab.bigquery as bq
#import pandas as pd
#from collections import Counter
#import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
#from sklearn.cross_validation import  cross_val_score
#import numpy as np
#from time import time
#from operator import itemgetter
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import f1_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.cross_validation import cross_val_predict
#from sklearn.metrics import accuracy_score
#from sklearn import metrics
#import pickle
#import scipy
#from keras.datasets import reuters
#from keras.models import Sequential
import pickle
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

#import numpy as np
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D



from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
#import keras
batch_size = 128
epochs = 1
# reset everything to rerun in jupyter
#tf.reset_default_graph()


with file_io.FileIO("gs://wellio-kadaif-tasty-images-project-pre-processed-images/pre_processed_images/image_data_20000_100.txt", 'r') as f:
  X, y = pickle.load(f)
# with file_io.FileIO("gs://wellio-kadaif-smart-spyder-model-data/label.txt", 'r') as f:
#   labels = pickle.load(f)


#with open("gs://wellio-kadaif-smart-spyder-model-data/data.txt", "rb") as fp:
    #data = pickle.load(fp)
#with open("gs://wellio-kadaif-smart-spyder-model-data/label.txt", "rb") as fp:
    #labels = pickle.load(fp)
#with open("gs://wellio-kadaif-smart-spyder-model-data/data.txt", "rb") as fp:
    #data = pickle.load(fp)
#with open("gs://wellio-kadaif-smart-spyder-model-data/label.txt", "rb") as fp:
    #labels = pickle.load(fp)



# with open('pre_processed_images/image_data_20000_100.txt', 'rb') as f:
#   X, y = pickle.load(f)


# In[4]:

datagen = ImageDataGenerator(rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.4,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest')

def cnn_model_v_0(IMG_SIZE):
    global NUM_CLASSES
    NUM_CLASSES = 2
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))
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
    model.add(Convolution2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))
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


# #### Creating a train, test split.

# In[5]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# #### Creating a validation split out of the training set.

# In[7]:

X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# #### The labels need to be converted to categorical. Note that we have 2 categories: good (1) and bad (0) images.

# In[8]:

y_train_fit_sparse = np_utils.to_categorical(y_train_fit, 2)


# In[9]:

y_val_sparse = np_utils.to_categorical(y_val, 2)


# In[10]:

y_test_sparse = np_utils.to_categorical(y_test, 2)


# In[11]:

datagen.fit(X_train)


# #### Creating an instance of a CNN model.

# ##### The image size is 100.

# In[12]:

IMG_SIZE = 100


# In[13]:

model_1 = cnn_model_v_1(IMG_SIZE)


# In[14]:

model_1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[15]:

model_1.summary()


# In[ ]:

model_1.fit_generator(datagen.flow(X_train_fit, y_train_fit_sparse, batch_size=128), steps_per_epoch=len(X_train_fit), epochs=5, validation_data=(X_val, y_val_sparse))


# In[17]:

score = model_1.evaluate(X_test, y_test_sparse, verbose=1)


# In[18]:

print('Test loss: {:0,.4f}'.format(score[0]))
print('Test accuracy: {:.2%}'.format(score[1]))

# model.save('model.h5')
# job_dir='gs://kadaif.getwellio.com'
#
#     # Save model.h5 on to google storage
# with file_io.FileIO('model.h5', mode='r') as input_f:
#     with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
#         output_f.write(input_f.read())
