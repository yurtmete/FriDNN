import os

from sklearn.metrics import classification_report
import numpy as np
from soundfile import read

from train_test.train_utils import phoneme_level_label
from keras.models import load_model

from train_test.lists import (fricative_list, silence_list,
                              unvoiced_fricative_list, voiced_fricative_list)

def calculate_performance(input_size_model, delay_model, experiment_folder):

    prediction = np.load(os.path.join(
        experiment_folder, 'results', 'prediction.npy'))

    ground_truth = np.load(os.path.join(
        experiment_folder, 'results', 'phonemegroundtruth.npy'))

    nua = np.where(ground_truth == 'nua')[0]

    ground_truth = np.delete(ground_truth, nua)
    prediction = np.delete(prediction, nua, axis=0)

    ground_truth_binary = np.isin(ground_truth, fricative_list).astype(int)

    total_prediction_binary = np.argmax(prediction, axis=1)
    total_prediction_binary_ = total_prediction_binary == 1

    with open(os.path.join(experiment_folder, 'performance.txt'), 'w') as f:
        f.writelines(classification_report(ground_truth_binary, total_prediction_binary_,
                     target_names=['non-fricative', 'fricative'], digits=4))

    calculate_performance_to_compare_state_of_the_art(experiment_folder, ground_truth, total_prediction_binary_)

def calculate_performance_to_compare_state_of_the_art(experiment_folder, ground_truth_phoneme_level, binary_prediction):

    masked_voiced_fricatives = np.where(np.isin(ground_truth_phoneme_level,voiced_fricative_list)==True)[0]

    ground_truth_phoneme_level = np.delete(ground_truth_phoneme_level, masked_voiced_fricatives)

    binary_prediction = np.delete(binary_prediction, masked_voiced_fricatives)

    borders = np.where((ground_truth_phoneme_level[:-1] != ground_truth_phoneme_level[1:])==True)[0]

    borders = np.hstack((borders, np.array([ground_truth_phoneme_level.shape[0]-1])))

    mv_gt = []
    mv_pre = []

    for i, border in enumerate(borders):
        if i == 0:
            start = 0
            end = border
        else:
            start = borders[i-1]+1
            end = border
            
        mv_gt.append(ground_truth_phoneme_level[border])
        mv_pre.append(np.argmax(np.bincount(binary_prediction[start:end+1])))

    mv_gt = np.array(mv_gt)
    mv_pre = np.array(mv_pre)

    mv_gt_binary = np.isin(mv_gt, unvoiced_fricative_list)*1

    with open(os.path.join(experiment_folder, 'performance_to_compare_state_of_the_art.txt'), 'w') as f:
        f.writelines(classification_report(mv_gt_binary, mv_pre,
                     target_names=['non-fricative', 'fricative'], digits=4))


def prediction_one_utterance(model, utterance_path, window_size, delay=0):
    """Generates the output of a trained model for an utterance

    #Arguments
        model (Keras model): Keras model to be used for output calculation
        utterance_path (str): Relative path for the utterance
        window_size (int): Window size indicates the input size of the network
        delay (int): same delay in the function windowing

    #Returns

        prediction (numpy array): Output of the network for the corresponding utterance (output of the single sigmoid
        neuron at the end, takes values between [0,1])
        phoneme_ground_truth (numpy array): Ground truth for the corresponding utterance in phoneme label
    """
    loaded_utterance = read(utterance_path)[0]
    len_utterance = len(loaded_utterance)
    phoneme_ground_truth = phoneme_level_label(
        np.genfromtxt(utterance_path[:-4] + ".PHN", dtype=[('myint', 'i4'), ('myint2', 'i4'), ('mystring', 'U25')],
                      comments='*'), loaded_utterance)

    input_to_prediction = []

    for i in range(len_utterance):  # in each sample prediction

        temp_sample = np.copy(loaded_utterance)

        if i + delay + 1 - window_size < 0:

            temp_sample = temp_sample[0:i + delay + 1]

            temp_sample = np.hstack(
                (np.zeros(window_size-len(temp_sample)), temp_sample))

        elif i + delay > len_utterance - 1:

            temp_sample = temp_sample[i + delay +
                                      1 - window_size:len_utterance]

            temp_sample = np.hstack(
                (temp_sample, np.zeros(window_size-len(temp_sample))))

        else:
            temp_sample = temp_sample[i + delay +
                                      1 - window_size:i + delay + 1]

        assert len(
            temp_sample) == window_size, 'Data generation is wrong for testing!'

        input_to_prediction.append(
            (temp_sample / np.std(temp_sample, axis=0)).reshape(window_size, 1))

    input_to_prediction = np.array(input_to_prediction)

    prediction = model.predict(input_to_prediction, batch_size=32).squeeze()

    assert len(prediction) == len(
        phoneme_ground_truth), 'Prediction and ground truth have different lengths!'

    return prediction, phoneme_ground_truth


def prediction_whole_core_test(dir_to_save, test_audio_list, model_path, window_size, delay=0):
    """Generates the output of a trained model for all utterances in the core test set of the TIMIT Dataset

    #Arguments
        model_path (str): path for the saved Keras model
        window_size (int): Window size indicates the input size of the network
        binary_evaluation (boolean): Binary or Multi-class
        noisy_evaluation (boolean): Noisy or clean
        snr (list of integers): The SNR values that are applied on the corresponding utterance.

    #Returns
        The function does not return anything, however saves certain numpy arrays

        prediction (numpy array): Concatenation of the output of the model for all utterances in the core test set of
        the TIMIT Dataset
            Between each utterance output, 0 is inserted to separate the prediction of each utterance
        phoneme_ground_truth (numpy array):
        Concatenation of the phoneme code ground truths of all utterances in the core
        test set of the TIMIT Dataset
            Between each utterance phoneme code ground truth, 'nua' (New Utterance Alarm) is inserted to separate the
            phoneme code ground truth of each utterance

    """

    dir_to_save = '{}/results'.format( 
        dir_to_save)

    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    model = load_model(model_path)

    prediction = np.array([]).reshape(0, 3)
    phoneme_ground_truth = np.array([])

    # Testing for each utterance in the Core Test Set
    for i, utterance in enumerate(test_audio_list):

        print('evaluating {} for utterance {}'.format(model_path, i + 1))

        temp = prediction_one_utterance(
            model, utterance, window_size, delay)

        prediction = np.vstack((prediction, temp[0]))
        # to be able to separate predictions
        prediction = np.vstack((prediction, np.array([3, 3, 3])))

        phoneme_ground_truth = np.hstack((phoneme_ground_truth, temp[1]))
        # to be able to separate utterances
        phoneme_ground_truth = np.hstack(
            (phoneme_ground_truth, np.array(['nua'])))

    prediction = prediction[:-1]
    phoneme_ground_truth = phoneme_ground_truth[:-1]
    np.save(os.path.join(dir_to_save, 'prediction'), prediction)
    np.save(os.path.join(dir_to_save, 'phonemegroundtruth'), phoneme_ground_truth)
