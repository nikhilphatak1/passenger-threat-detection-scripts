'''
This file parsing and partitioning is designed based on
Brian Farrar's and William Cukierski's kernels.

In order to train the model using the gcloud command line system,
this script was packaged into the appropriate format, but is more legible
in this format.
'''

import numpy as np
import pandas as pd
import os
import re

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import random
from timeit import default_timer as timer
import argparse

INPUT_FOLDER = 'stage1/aps'
PREPROCESSED_DATA_FOLDER = 'preprocessed'
STAGE1_LABELS = 'stage1_labels.csv'
STAGE1_TEST_SET = 'stage1_sample_submission.csv'
THREAT_ZONE = 1 # this value is altered depending on the zone one wants to train on
BATCH_SIZE = 16
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = 'tsa_logs/train/'
MODEL_PATH = 'tsa_logs/model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('neuralnetmodel-v0.1', LEARNING_RATE, IMAGE_DIM,
                                                IMAGE_DIM, THREAT_ZONE ))

def preprocess_tsa_data():


    SUBJECT_LIST = []

    batch_num = 1
    threat_zone_examples = []
    start_time = timer()

    for subject in SUBJECT_LIST:

        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')

        images = images.transpose()

        for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list,
                                                             tsa.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(tsa.get_subject_zone_label(tz_num,
                             tsa.get_subject_labels(STAGE1_LABELS, subject)))

            for img_num, img in enumerate(images):

                print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                print('Threat Zone Label -> {}'.format(label))

                if threat_zone[img_num] is not None:

                    base_img = np.flipud(img)

                    rescaled_img = tsa.convert_to_grayscale(base_img)

                    high_contrast_img = tsa.spread_spectrum(rescaled_img)

                    masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                    cropped_img = tsa.crop(masked_img, crop_dims[img_num])

                    normalized_img = tsa.normalize(cropped_img)

                    zero_centered_img = tsa.zero_center(normalized_img)
                    threat_zone_examples.append([[tz_num], zero_centered_img, label])

        if ((len(threat_zone_examples) % (BATCH_SIZE * EXAMPLES_PER_SUBJECT)) == 0):
            for tz_num, tz in enumerate(tsa.zone_slice_list):

                tz_examples_to_save = []
                tz_examples = [example for example in threat_zone_examples if example[0] ==
                               [tz_num]]

                tz_examples_to_save.append([[features_label[1], features_label[2]]
                                            for features_label in tz_examples])

                np.save(PREPROCESSED_DATA_FOLDER +
                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1,
                                                         len(threat_zone_examples[0][1][0]),
                                                         len(threat_zone_examples[0][1][1]),
                                                         batch_num),
                                                         tz_examples_to_save)
                del tz_examples_to_save

            del threat_zone_examples
            threat_zone_examples = []
            batch_num += 1

    if (len(threat_zone_examples) > 0):
        for tz_num, tz in enumerate(tsa.zone_slice_list):

            tz_examples_to_save = []


            tz_examples = [example for example in threat_zone_examples if example[0] ==
                           [tz_num]]

            tz_examples_to_save.append([[features_label[1], features_label[2]]
                                        for features_label in tz_examples])

            np.save(PREPROCESSED_DATA_FOLDER +
                    'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1,
                                                     len(threat_zone_examples[0][1][0]),
                                                     len(threat_zone_examples[0][1][1]),
                                                     batch_num),
                                                     tz_examples_to_save)


def get_train_test_file_list():

    global FILE_LIST
    global TRAIN_SET_FILE_LIST
    global TEST_SET_FILE_LIST

    if os.listdir(PREPROCESSED_DATA_FOLDER) == []:
        print ('No preprocessed data available.  Skipping preprocessed data setup..')
    else:
        FILE_LIST = [f for f in os.listdir(PREPROCESSED_DATA_FOLDER)
                     if re.search(re.compile('-tz' + str(THREAT_ZONE) + '-'), f)]
        train_test_split = len(FILE_LIST) - \
                           max(int(len(FILE_LIST)*TRAIN_TEST_SPLIT_RATIO),1)
        TRAIN_SET_FILE_LIST = FILE_LIST[:train_test_split]
        TEST_SET_FILE_LIST = FILE_LIST[train_test_split:]
        print('Train/Test Split -> {} file(s) of {} used for testing'.format(
              len(FILE_LIST) - train_test_split, len(FILE_LIST)))



def pipeline(filename, path):

    preprocessed_tz_scans = []
    feature_batch = []
    label_batch = []

    preprocessed_tz_scans = np.load(os.path.join(path, filename))

    np.random.shuffle(preprocessed_tz_scans)

    for example_list in preprocessed_tz_scans:
        for example in example_list:
            feature_batch.append(example[0])
            label_batch.append(example[1])

    feature_batch = np.asarray(feature_batch, dtype=np.float32)
    label_batch = np.asarray(label_batch, dtype=np.float32)

    return feature_batch, label_batch


def shuffle_train_set(train_set):
    sorted_file_list = random.shuffle(train_set)
    TRAIN_SET_FILE_LIST = sorted_file_list



def neuralnetmodel(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy',
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + MODEL_NAME,
                        tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)

    return model

def create_model():
    parser = argparse.ArgumentParser()
    args, task_args = parser.parse_known_args()
    model = neuralnetmodel(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)
    return model,task_args

def training():

    val_features = []
    val_labels = []

    get_train_test_file_list()

    model = neuralnetmodel(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)

    for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
        if j == 0:
            val_features, val_labels = pipeline(test_f_in, PREPROCESSED_DATA_FOLDER)
        else:
            tmp_feature_batch, tmp_label_batch = pipeline(test_f_in,
                                                                PREPROCESSED_DATA_FOLDER)
            val_features = np.concatenate((tmp_feature_batch, val_features), axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)

    val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)



    for i in range(N_TRAIN_STEPS):

        shuffle_train_set(TRAIN_SET_FILE_LIST)

        for f_in in TRAIN_SET_FILE_LIST:

            feature_batch, label_batch = pipeline(f_in, PREPROCESSED_DATA_FOLDER)
            feature_batch = feature_batch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)

            model.fit({'features': feature_batch}, {'labels': label_batch}, n_epoch=1,
                      validation_set=({'features': val_features}, {'labels': val_labels}),
                      shuffle=True, snapshot_step=None, show_metric=True,
                      run_id=MODEL_NAME)
