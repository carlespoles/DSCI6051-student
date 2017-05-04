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
from keras.models import Sequential, model_from_json, load_model
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
    #plt.plot([0, 1], [0, 1], 'k--', color="red", label="Random guess")
    plt.plot([0, 1], [0, 1], 'k--', color="red", label="")
    plt.legend(loc='best')
    plt.grid()
    plt.title("ROC Curve - " + plot_title + "");


def plot_models():
    # Load data from pickle file located on a bucket.
    with file_io.FileIO('gs://wellio-kadaif-tasty-images-project-pre-processed-images/pre_processed_images/image_data_20000_25.txt', 'r') as f:
        X, y = pickle.load(f)

    # Creating a train, test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    model_1 = load_model('CNN_14_layers_number_images_20000_image_size_25.h5')
    model_2 = load_model('CNN_14_layers_number_images_20000_image_size_50.h5')
    model_3 = load_model('CNN_14_layers_number_images_20000_image_size_100.h5')
    model_4 = load_model('CNN_10_layers_number_images_20000_image_size_25.h5')
    model_5 = load_model('CNN_10_layers_number_images_20000_image_size_50.h5')
    model_6 = load_model('CNN_10_layers_number_images_20000_image_size_100.h5')
    model_7 = load_model('CNN_10_layers_number_images_20000_image_size_25_data_augmentation.h5')
    model_8 = load_model('CNN_10_layers_number_images_20000_image_size_50_data_augmentation.h5')
    model_9 = load_model('CNN_10_layers_number_images_20000_image_size_100_data_augmentation.h5')

    # List of probabilities.
    predictions_probability_1 = model_1.predict_proba(X_test)
    predictions_probability_4 = model_4.predict_proba(X_test)
    predictions_probability_7 = model_7.predict_proba(X_test)

    with file_io.FileIO('gs://wellio-kadaif-tasty-images-project-pre-processed-images/pre_processed_images/image_data_20000_50.txt', 'r') as f:
        X, y = pickle.load(f)

    # Creating a train, test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    predictions_probability_2 = model_2.predict_proba(X_test)
    predictions_probability_5 = model_5.predict_proba(X_test)
    predictions_probability_8 = model_8.predict_proba(X_test)

    with file_io.FileIO('gs://wellio-kadaif-tasty-images-project-pre-processed-images/pre_processed_images/image_data_20000_100.txt', 'r') as f:
        X, y = pickle.load(f)

    # Creating a train, test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    predictions_probability_3 = model_3.predict_proba(X_test)
    predictions_probability_6 = model_6.predict_proba(X_test)
    predictions_probability_9 = model_9.predict_proba(X_test)

    # Creating ROC curve.
    plt.figure(figsize=(9, 9))
    plot_roc(y_test, predictions_probability_1[:, 1], "CNN - " + str(len(model_1.layers)) + " layers | image size: 25", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_2[:, 1], "CNN - " + str(len(model_2.layers)) + " layers | image size: 50", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_3[:, 1], "CNN - " + str(len(model_3.layers)) + " layers | image size: 100", "Tasty Food Images | # images: " + str(len(X)))

    plt.savefig('CNN_14_all_roc_curve.png')

    results_dir = "gs://wellio-kadaif-tasty-images-project-images/script_results"

    # Sending file to bucket.
    with file_io.FileIO('CNN_14_all_roc_curve.png', mode='r') as input_f:
        with file_io.FileIO(results_dir + '/' + 'CNN_14_all_roc_curve.png', mode='w+') as output_f:
            output_f.write(input_f.read())

    # Creating ROC curve.
    plt.figure(figsize=(9, 9))
    plot_roc(y_test, predictions_probability_4[:, 1], "CNN - " + str(len(model_4.layers)) + " layers | image size: 25", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_5[:, 1], "CNN - " + str(len(model_5.layers)) + " layers | image size: 50", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_6[:, 1], "CNN - " + str(len(model_6.layers)) + " layers | image size: 100", "Tasty Food Images | # images: " + str(len(X)))

    plt.savefig('CNN_10_all_roc_curve.png')

    # Sending file to bucket.
    with file_io.FileIO('CNN_10_all_roc_curve.png', mode='r') as input_f:
        with file_io.FileIO(results_dir + '/' + 'CNN_10_all_roc_curve.png', mode='w+') as output_f:
            output_f.write(input_f.read())

    # Creating ROC curve.
    plt.figure(figsize=(9, 9))
    plot_roc(y_test, predictions_probability_7[:, 1], "CNN Data Augmen. - " + str(len(model_7.layers)) + " layers | image size: 25", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_8[:, 1], "CNN Data Augmen. - " + str(len(model_8.layers)) + " layers | image size: 50", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_9[:, 1], "CNN Data Augmen. - " + str(len(model_9.layers)) + " layers | image size: 100", "Tasty Food Images | # images: " + str(len(X)))

    plt.savefig('CNN_10_all_data_augmentation_roc_curve.png')

    # Sending file to bucket.
    with file_io.FileIO('CNN_10_all_data_augmentation_roc_curve.png', mode='r') as input_f:
        with file_io.FileIO(results_dir + '/' + 'CNN_10_all_data_augmentation_roc_curve.png', mode='w+') as output_f:
            output_f.write(input_f.read())

    # Creating ROC curve of all curves.
    plt.figure(figsize=(9, 9))
    plot_roc(y_test, predictions_probability_1[:, 1], "CNN - " + str(len(model_1.layers)) + " layers | image size: 25", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_2[:, 1], "CNN - " + str(len(model_2.layers)) + " layers | image size: 50", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_3[:, 1], "CNN - " + str(len(model_3.layers)) + " layers | image size: 100", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_4[:, 1], "CNN - " + str(len(model_4.layers)) + " layers | image size: 25", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_5[:, 1], "CNN - " + str(len(model_5.layers)) + " layers | image size: 50", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_6[:, 1], "CNN - " + str(len(model_6.layers)) + " layers | image size: 100", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_7[:, 1], "CNN Data Augmen. - " + str(len(model_7.layers)) + " layers | image size: 25", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_8[:, 1], "CNN Data Augmen. - " + str(len(model_8.layers)) + " layers | image size: 50", "Tasty Food Images | # images: " + str(len(X)))
    plot_roc(y_test, predictions_probability_9[:, 1], "CNN Data Augmen. - " + str(len(model_9.layers)) + " layers | image size: 100", "Tasty Food Images | # images: " + str(len(X)))

    plt.savefig('CNN_14_10_all_roc_curve.png')

    # Sending file to bucket.
    with file_io.FileIO('CNN_14_10_all_roc_curve.png', mode='r') as input_f:
        with file_io.FileIO(results_dir + '/' + 'CNN_14_10_all_roc_curve.png', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    try:
        plot_models()
    except Exception as e:
        print("Error:", str(e))

# example usage: $python plot_all_cnn_models.py
