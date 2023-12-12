# FriDNN
Fricative Detection Using 1D CNNs. 

This repository contains the implementation of the model we described in our paper, "Fricative Phoneme Detection Using Deep Neural Networks and its Comparison to Traditional Methods". 

You can access the paper here: https://www.isca-speech.org/archive/pdfs/interspeech_2021/yurt21_interspeech.pdf

## Installation

1- Install CUDA and CUDNN, recommended versions

* CUDA Version 10.0

* CUDNN Version 7.4.1

2- Install python (Python >= 3.6 recommended)

3- Setup the required dependencies

```
$ git clone https://github.com/yurtmete/FriDNN.git 
$ cd FriDNN
$ virtualenv env -p python3.6
$ source env/bin/activate
$ pip install -e .
```
## To train and test a new model

Use the main script to train a model:

```
$ python train_test/main.py -h
usage: main.py [-h] -D <TimitDirectory> [-d <Delay>] -t <TargetDirectory>
               [--test_only]

optional arguments:
  -h, --help            show this help message and exit
  -D <TimitDirectory>, --timit_directory <TimitDirectory>
                        Directory containing TIMIT dataset.
  -d <Delay>, --delay <Delay>
                        Detection delay in samples
  -t <TargetDirectory>, --target_dir <TargetDirectory>
                        Target directory for the experiment
  --test_only           Test already trained model
```

### Example:
```
python train_test.py -D /home/user/workspace -t experiment_1
```
**Note:** /home/user/workspace directory should hold TIMIT folder. 

**Note:** Checkpoints, performance results, tensorboard logs, model summary will be saved in *experiment_1* folder.

## To test and calculate the performance results stated in the paper

Run the following command to test the model that is used in the paper to report the performance:

```
python train_test.py -D /home/user/workspace -t model_in_paper --test_only
```

**Note:** Target directory should be set to model_in_paper. It holds the checkpoint of the model we used in our paper.

**Note:** The flag **--test_only** should be passed, as the training is already performed.  
