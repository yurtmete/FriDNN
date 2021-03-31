import datetime
import os
import sys
import time
from argparse import ArgumentParser
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import (EarlyStopping, History, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)

from keras.optimizers import Adam


from train_test.helpers import sorted_alphanumeric
from train_test.lists import core_test_set_speakers, fricative_list, val_audio_list
from train_test.model import FriDNN
from train_test.test_utils import prediction_whole_core_test, calculate_performance
from train_test.train_utils import DataGenerator

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# training parameters

BATCH_SIZE = 256  # bath size
NUMBER_OF_AUDIOS = 16  # number of utterance selected for a single batch
EPOCHS = 1000
WINDOW_SIZE = 320  # input size of the network


def train_and_test_model(path_to_TIMIT, pre_delay, target_dir, only_test):

    # creating relatives paths for test data set, validation data set and train data set
    test_audio_list = []
    for dialect in os.listdir(os.path.join(path_to_TIMIT, "TIMIT/TEST/")):
        for person in os.listdir(os.path.join(path_to_TIMIT, "TIMIT/TEST", dialect)):
            if person in core_test_set_speakers:
                temp_list = os.listdir(os.path.join(
                    path_to_TIMIT, "TIMIT/TEST", dialect, person))
                temp_list = [x for x in temp_list if x[-3:]
                             == 'WAV' and 'SA' not in x]
                for sentence in temp_list:
                    test_audio_list.append(os.path.join(
                        path_to_TIMIT, "TIMIT/TEST", dialect, person, sentence))

    val_audio_full_paths = [os.path.join(
        path_to_TIMIT, "TIMIT/TEST", item) for item in val_audio_list]

    train_audio_list = []
    for dialect in os.listdir(os.path.join(path_to_TIMIT, "TIMIT/TRAIN")):
        for person in os.listdir(os.path.join(path_to_TIMIT, "TIMIT/TRAIN", dialect)):
            temp_list = os.listdir(os.path.join(
                path_to_TIMIT, "TIMIT/TRAIN", dialect, person))
            temp_list = [x for x in temp_list if x[-3:]
                         == 'WAV' and 'SA' not in x]
            for sentence in temp_list:
                train_audio_list.append(os.path.join(
                    path_to_TIMIT, "TIMIT/TRAIN", dialect, person, sentence))

    assert WINDOW_SIZE / pre_delay == 2, "Window size and prediction delay are set wrong!"

    if not only_test:

        os.makedirs(target_dir, exist_ok=False)

        # Network Creation
        name = "{}-{}-{}".format(os.path.basename(target_dir), pre_delay,
                                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        model = FriDNN(input_size=WINDOW_SIZE)

        model.summary()

        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])

        with open(os.path.join(target_dir, name + '_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()

        os.makedirs(os.path.join(target_dir, 'logs'))

        tensorboard = TensorBoard(log_dir=os.path.join(target_dir, 'logs'))
        es = EarlyStopping(monitor='val_loss', min_delta=0,
                           mode='min', verbose=2, patience=40)

        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                               patience=10, verbose=1, mode='auto', cooldown=0, min_lr=0)

        os.makedirs(os.path.join(target_dir, 'checkpoints'))

        filepath = os.path.join(target_dir, 'checkpoints', name +
                                "_" + "epoch_{epoch:02d}_valloss_{val_loss:.3f}.hdf5")

        mc1 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False,
                              mode='auto', period=1)

        # Initialization of data generators for training and validation
        train_gen = DataGenerator(train_audio_list,
                                      BATCH_SIZE,
                                      NUMBER_OF_AUDIOS,
                                      WINDOW_SIZE,
                                      pre_delay,
                                      'training')

        val_gen = DataGenerator(val_audio_full_paths,
                                    BATCH_SIZE,
                                    NUMBER_OF_AUDIOS,
                                    WINDOW_SIZE,
                                    pre_delay,
                                    'validation')

        hist = model.fit_generator(train_gen, epochs=EPOCHS, verbose=1,
                                   callbacks=[tensorboard, es, mc1, lr],
                                   validation_data=val_gen,
                                   max_queue_size=4, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)

    if only_test:

        # testing the trained model
        different_epoch_paths = os.listdir(
            os.path.join(target_dir, 'checkpoints'))

        different_epoch_paths = sorted_alphanumeric(different_epoch_paths)

        last_checkpoint_path = os.path.join(
            target_dir, 'checkpoints', different_epoch_paths[-1])

    else:

        # testing the trained model
        different_epoch_paths = os.listdir(
            os.path.join(target_dir, 'checkpoints'))

        different_epoch_paths = list(
            filter(lambda x_: x_[-4:] == "hdf5" and x_.split('_epoch_')[0] == name.split('_epoch_')[0], different_epoch_paths))

        different_epoch_paths = sorted_alphanumeric(different_epoch_paths)

        last_checkpoint_path = os.path.join(
            target_dir, 'checkpoints', different_epoch_paths[-1])

    prediction_whole_core_test(target_dir,
                               test_audio_list,
                               last_checkpoint_path,
                               WINDOW_SIZE,
                               pre_delay)

    calculate_performance(WINDOW_SIZE,
                          pre_delay,
                          target_dir)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('-D', '--timit_directory',
                        type=str,
                        required=True,
                        help='Directory containing TIMIT dataset.',
                        metavar='<TimitDirectory>')

    parser.add_argument('-d', '--delay',
                        type=int,
                        help='Detection delay in samples',
                        metavar='<Delay>',
                        default=160)

    parser.add_argument('-t', '--target_dir',
                        type=str,
                        required=True,
                        help='Target directory for the experiment',
                        metavar='<TargetDirectory>')

    parser.add_argument('--test_only',
                        default=False,
                        action='store_true',
                        help='Test already trained model')

    args = parser.parse_args()

    train_and_test_model(args.timit_directory,
                         args.delay,
                         args.target_dir,
                         args.test_only)
