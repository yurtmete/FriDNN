import os
import sys
from collections import Counter
from pathlib import Path

import keras
import numpy as np
from keras.models import load_model
from soundfile import read


from train_test.lists import (fricative_list, silence_list,
                              unvoiced_fricative_list, voiced_fricative_list)


def windowing(audio, label, ws, flag, delay):
    """ Sample extractor for training and validation for binary training (classes: fricative phonemes,
    vs rest of the phonemes)

    #Arguments
        audio (numpy array): Numpy array that contains the wav file of the utterance
        label (numpy array): Numpy array that contains the label file for the corresponding utterance in binary,
        0 means non-fricative, 1 means fricative sample
        ws (int): Size of the extracted sample in samples
        flag (int): Flag for sample generation, if it is 1, then a fricative sample is generated, and if it is 0,
        non-fricative sample is generated
        delay(int) (in samples): Decides the prediction and also labeling point of the extracted label. If it is 0, then
        prediction and labeling point is the last index of the extracted sample
        if it is a positive value, then the prediction and labeling point is in the left side of the last index of the
        extracted sample, meaning prediction and labeling use future samples,
        finally if it is a negative value, then the prediction and labeling point is in the right side of the last index
        of the extracted sample,
        meaning labeling uses future samples for labeling therefore prediction extrapolates the upcoming samples
        during prediction.
        spectral_tilt_array (numpy array): Spectral tilt array corresponding to the utterance (pre-calculated) 
    #Returns
        sample (numpy array): Numpy array that contains the extracted sample wav file
        sample_label (int): Integer indicating the label for the extracted sample, 1 for fricative sample,
        0 for non-fricative sample

    """
    if flag == 1:  # fricative sample generation
        try:  # some utterances in the TIMIT dataset do not have fricative phoneme at all. To avoid this issue,
            # this error handling is applied
            label_point = np.random.choice(
                np.where(label[ws - 1 - delay:len(audio) - np.max([0, delay])] == 1)[0]) + ws - 1 - delay
        except ValueError:  # some of the utterances does not have a single fricative
            label_point = np.random.randint(
                ws - 1 - delay, len(audio) - np.max([0, delay]))
    else:  # non-fricative sample generation
        label_point = np.random.choice(
            np.where(label[ws - 1 - delay:len(audio) - np.max([0, delay])] == flag)[0]) + ws - 1 - delay

    sample = audio[label_point - ws + 1 + delay:label_point + 1 + delay]

    sample_label = label[label_point]

    return sample, sample_label


def binary_label(data_phn, data):
    """Function for creating a label file for an utterance using its PHN file for binary training

    #Arguments
        data_phn (numpy array): Numpy array that contains the phn file of the utterance
        data (numpy array): Numpy array that contains the wav file for the corresponding utterance
    #Returns
        label_array (numpy array): Numpy array that contains the labels for the corresponding utterance for each sample
        in time. It has same length with the
            corresponding wav file of the utterance (data) and indicates the label for each sample in the utterance. It
            contains 0's (non-fricative) and 1's (for fricative)
    """
    data_phn[0][0] = 0  # some silence parts do not start at 0 index in the TIMIT dataset
    data_phn[-1][1] = len(
        data)  # in some phn files, end index of the last phoneme does not
    # have the same length with the corresponding utterance
    label_array = np.zeros(len(data))

    for phoneme in data_phn:

        if phoneme[2] in fricative_list:
            label_array[int(phoneme[0]):int(phoneme[1])] = 1
        elif phoneme[2] in silence_list:
            label_array[int(phoneme[0]):int(phoneme[1])] = 2
        else:
            label_array[int(phoneme[0]):int(phoneme[1])] = 0

    return label_array


def phoneme_level_label(data_phn, data):  # change the name later to phoneme level
    """Function for creating a label file for an utterance using its PHN file for multi-class training

    #Arguments
        data_phn (numpy array): Numpy array that contains the phn file of the utterance
        data (numpy array): Numpy array that contains the wav file for the corresponding utterance
    #Returns
        label_array (numpy array): Numpy array that contains the phoneme codes for the corresponding utterance for each
        sample in time. It has same length with the
            corresponding wav file of the utterance (data)
    """
    data_phn[0][0] = 0  # some silence parts do not start at 0 index in the TIMIT dataset
    data_phn[-1][1] = len(
        data)  # in some phn files, end index of the last phoneme
    # does not have the same length with the corresponding utterance
    label_array = np.chararray((len(data)), itemsize=5, unicode=True)

    for phoneme in data_phn:
        label_array[phoneme[0]:phoneme[1]] = str(phoneme[2])

    return label_array


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self,
                 list_paths,
                 batch_size,
                 noa,
                 window_size,
                 delay,
                 use_case,
                 shuffle=True,
                 ):
        """Initialization"""
        self.batch_size = batch_size  # number of samples per batch
        self.noa = noa  # number of utterance per batch
        # number of samples is extracted from one utterance in a batch
        self.nos = self.batch_size // self.noa
        # the lists contains the relative paths for the utterance in the TIMIT dataset
        self.list_paths = list_paths
        self.indexes = np.arange(len(self.list_paths))
        # sample lengths that is samples from utterances in the TIMIT dataset
        self.window_size = window_size
        self.delay = delay  # delay in the detection
        self.shuffle = shuffle  # decides if the list that contains relative
        # paths of utterance should be shuffled after one epoch
        # decides where to use the generator, for training or validation
        self.use_case = use_case
        self.flag = 0

        if self.use_case == 'validation':
            np.random.seed(1011)
            self.shuffle = False

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.list_paths) / self.noa))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        if index != self.__len__() - 1:
            indexes = self.indexes[index * self.noa:(index + 1) * self.noa]
        else:
            indexes = self.indexes[index * self.noa:]
        # Find list of IDs
        list_paths_batch = [self.list_paths[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_paths_batch)
        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths_batch):
        """Generates data containing batch_size samples"""

        count = 0
        # Load data
        raw_samples = []
        raw_labels = []
        for i, temp_path in enumerate(list_paths_batch):
            # Store samples and labels
            temp_sample = read(temp_path)[0]
            raw_samples.append(temp_sample)
            temp_label = binary_label(np.genfromtxt(temp_path[:-4] + '.PHN',
                                                    dtype=[('myint', 'i4'), ('myint2', 'i4'),
                                                           ('mystring', 'U25')], comments='*'), temp_sample)
            raw_labels.append(temp_label)

        # Initialization
        x = np.zeros((len(list_paths_batch) * self.nos, self.window_size))
        
        y = np.zeros((len(list_paths_batch) * self.nos, 3))

        # Create Samples and Labels from loaded raw samples and raw labels

        for _ in range(self.nos):
             
            desired_label = self.flag % 3
            for k in range(len(list_paths_batch)):
                single_temp_sample, single_temp_label = windowing(raw_samples[k], raw_labels[k], self.window_size,
                                                                  desired_label, self.delay)
                
                single_temp_label_np = np.array([0, 0, 0])
                single_temp_label_np[int(single_temp_label)] = 1
                y[count] = single_temp_label_np

                single_temp_sample = single_temp_sample / \
                    np.std(single_temp_sample, axis=0)
                x[count] = single_temp_sample
                count += 1
            self.flag += 1

        x = np.expand_dims(x, axis=2)

        return x, y
