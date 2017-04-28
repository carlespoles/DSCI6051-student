#!/usr/bin/python
from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import os
import glob
import pickle
import gzip
import h5py
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
from tensorflow.python.lib.io import file_io
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def show_confusion_matrix(C, class_labels=['0', '1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.
    Draws confusion matrix with associated metrics.

    Minimum required imports:

    import matplotlib.pyplot as plt
    import numpy as np
    Source: http://notmatthancock.github.io/2015/10/28/confusion-matrix.html

    """
    assert C.shape == (2, 2), "Confusion matrix should be from binary classification only."

    # true negative, false positive, etc...
    tn = C[0, 0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.plot([-0.5, 2.5], [0.5, 0.5], '-k', lw=2)
    ax.plot([-0.5, 2.5], [1.5, 1.5], '-k', lw=2)
    ax.plot([0.5, 0.5], [-0.5, 2.5], '-k', lw=2)
    ax.plot([1.5, 1.5], [-0.5, 2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34, 1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''], rotation=90)
    ax.set_yticks([0, 1, 2])
    ax.yaxis.set_label_coords(-0.09, 0.65)

    # Fill in initial metrics: tp, tn, etc...
    ax.text(0, 0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 1,
            'True Pos: %d\n(Num Pos: %d)'%(tp, NP),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2, 0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0, 2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.show();


def plot_confusion_matrix(y_true, y_pred):
    """
    The function takes the labels and predictions to create a colorful
    confusion matrix. This is more indicated to multi-class problems.
    """
    cm_array = confusion_matrix(y_true, y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks, pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size


def plot_roc(y_test, y_prediction, model_label, plot_title):
    """
    The function takes as parameters the test labels (observations),
    the test predictions and two strings to customize the plot: the
    name of the model, and a title for the plot.
    It relies on sklearn.metrics.roc_curve to extract fpr and tpr
    to make the plot.
    IMPORTANT: the predictions must be probabilities in order to plot
    the curve: not .predict(X_test), but .predict_proba(X_test)
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_prediction)
    plt.plot(fpr, tpr, label=model_label)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.plot([0, 1], [0, 1], 'k--', color="red", label="Random guess")
    plt.legend(loc='best')
    plt.grid()
    plt.title("ROC Curve - " + plot_title + "");


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


def train_models(IMG_SIZE, NUM_EPOCHS):
    # Load data from pickle file located on a bucket.
    with file_io.FileIO('gs://wellio-kadaif-tasty-images-project-pre-processed-images/pre_processed_images/image_data_20000_' + IMG_SIZE + '.txt', 'r') as f:
        X, y = pickle.load(f)

    # Creating a train, test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # Creating a validation split out of the training set.
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train,
                                                              test_size=0.1,
                                                              random_state=42)

    # The labels need to be converted to categorical.
    # Note that we have 2 categories: good (1) and bad (0) images.
    y_train_sparse = np_utils.to_categorical(y_train, 2)

    y_test_sparse = np_utils.to_categorical(y_test, 2)

    # Creating an instance of a CNN model.
    IMG_SIZE = int(IMG_SIZE)
    NUM_EPOCHS = int(NUM_EPOCHS)
    model_1 = cnn_model_v_1(IMG_SIZE)

    model_1.compile(loss='binary_crossentropy', optimizer='rmsprop',
                    metrics=['accuracy'])

    model_1.summary()

    model_1.fit(X_train, y_train_sparse, batch_size=128, epochs=NUM_EPOCHS, verbose=1, validation_split=0.3)

    score = model_1.evaluate(X_test, y_test_sparse, verbose=1)

    print('Test loss: {:0,.4f}'.format(score[0]))
    print('Test accuracy: {:.2%}'.format(score[1]))

    # List of predictions.
    predicted_images = []
    for i in model_1.predict(X_test):
        predicted_images.append(np.where(np.max(i) == i)[0])

    print("AUC: {:.2%}\n".format(roc_auc_score(y_test, predicted_images)))

    results_dir = "gs://wellio-kadaif-tasty-images-project-images/script_results"

    file_name_1 = "CNN_" + str(len(model_1.layers)) + "_layers_number_images_" + str(len(X)) + "_image_size_" + str(IMG_SIZE) + "_metrics.txt"

    with open(file_name_1, "w") as f:
        f.write('CNN number of layers: ' + str(len(model_1.layers)))
        # f.write('\nNumber of images: ' + str(len(X)))
        f.write('\nNumber of images: {:0,.0f}'.format(len(X)))
        f.write('\nImage size: ' + str(IMG_SIZE))
        f.write('\nNumber of epochs: ' + str(NUM_EPOCHS))
        f.write('\nTest loss: {:0,.4f}'.format(score[0]))
        f.write('\nTest accuracy: {:.2%}'.format(score[1]))
        f.write("\nAUC: {:.2%}\n".format(roc_auc_score(y_test, predicted_images)))

    # with file_io.FileIO(file_name, mode='r') as input_f:
    #     with file_io.FileIO(results_dir + '/' + file_name, mode='w+') as output_f:
    #         output_f.write(input_f.read())

    model_name_1 = file_name = "CNN_" + str(len(model_1.layers)) + "_layers_number_images_" + str(len(X)) + "_image_size_" + str(IMG_SIZE)

    # Creating a confusion matrix.
    plt.figure(figsize=(9, 9))
    cf = show_confusion_matrix(confusion_matrix(y_test, predicted_images), ['Class 0', 'Class 1'])
    plt.savefig(model_name_1 + '_confusion_matrix.png')

    # List of probabilities.
    predictions_probability = model_1.predict_proba(X_test)

    # Creating ROC curve.
    plt.figure(figsize=(9, 9))
    rc = plot_roc(y_test, predictions_probability[:,1], "CNN - " + str(len(model_1.layers)) + " layers | # images: " + str(len(X)) + " | image size: " + str(IMG_SIZE), "Tasty Food Images")
    plt.savefig(model_name_1 + '_roc_curve.png')

    # Saving model and model weights.
    model_1.save(model_name_1 + '.h5')
    model_1.save_weights(model_name_1 + '_weights.h5')

    # Train model with different architecture.
    # Creating a train, test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # Creating a validation split out of the training set.
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train,
                                                              test_size=0.1,
                                                              random_state=42)

    # The labels need to be converted to categorical.
    # Note that we have 2 categories: good (1) and bad (0) images.
    y_train_sparse = np_utils.to_categorical(y_train, 2)

    y_test_sparse = np_utils.to_categorical(y_test, 2)

    model_2 = cnn_model_v_0(IMG_SIZE)

    model_2.compile(loss='binary_crossentropy', optimizer='rmsprop',
                    metrics=['accuracy'])

    model_2.summary()

    model_2.fit(X_train, y_train_sparse, batch_size=128, epochs=NUM_EPOCHS, verbose=1, validation_split=0.3)

    score = model_2.evaluate(X_test, y_test_sparse, verbose=1)

    print('Test loss: {:0,.4f}'.format(score[0]))
    print('Test accuracy: {:.2%}'.format(score[1]))

    # List of predictions.
    predicted_images = []
    for i in model_2.predict(X_test):
        predicted_images.append(np.where(np.max(i) == i)[0])

    print("AUC: {:.2%}\n".format(roc_auc_score(y_test, predicted_images)))

    results_dir = "gs://wellio-kadaif-tasty-images-project-images/script_results"

    file_name_2 = "CNN_" + str(len(model_2.layers)) + "_layers_number_images_" + str(len(X)) + "_image_size_" + str(IMG_SIZE) + "_metrics.txt"

    with open(file_name_2, "w") as f:
        f.write('CNN number of layers: ' + str(len(model_2.layers)))
        #f.write('\nNumber of images: ' + str(len(X)))
        f.write('\nNumber of images: {:0,.0f}'.format(len(X)))
        f.write('\nImage size: ' + str(IMG_SIZE))
        f.write('\nNumber of epochs: ' + str(NUM_EPOCHS))
        f.write('\nTest loss: {:0,.4f}'.format(score[0]))
        f.write('\nTest accuracy: {:.2%}'.format(score[1]))
        f.write("\nAUC: {:.2%}\n".format(roc_auc_score(y_test, predicted_images)))

    # with file_io.FileIO(file_name, mode='r') as input_f:
    #     with file_io.FileIO(results_dir + '/' + file_name, mode='w+') as output_f:
    #         output_f.write(input_f.read())

    model_name_2 = file_name = "CNN_" + str(len(model_2.layers)) + "_layers_number_images_" + str(len(X)) + "_image_size_" + str(IMG_SIZE)

    # Creating a confusion matrix.
    plt.figure(figsize=(9, 9))
    cf = show_confusion_matrix(confusion_matrix(y_test, predicted_images), ['Class 0', 'Class 1'])
    plt.savefig(model_name_2 + '_confusion_matrix.png')

    # List of probabilities.
    predictions_probability = model_2.predict_proba(X_test)

    # Creating ROC curve.
    plt.figure(figsize=(9, 9))
    rc = plot_roc(y_test, predictions_probability[:,1], "CNN - " + str(len(model_2.layers)) + " layers | # images: " + str(len(X)) + " | image size: " + str(IMG_SIZE), "Tasty Food Images")
    plt.savefig(model_name_2 + '_roc_curve.png')

    # Saving model and model weights.
    model_2.save(model_name_2 + '.h5')
    model_2.save_weights(model_name_2 + '_weights.h5')


if __name__ == '__main__':
    try:
        image_size = sys.argv[1]
        number_epochs = sys.argv[2]
        train_models(image_size, number_epochs)
    except Exception as e:
        print("Error:", str(e))

# example usage: $python train_cnn_model.py 25 50 => image size is 25 and 50 epochs.
