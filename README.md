# FriDNN
Fricative Detection Using 1D CNNs

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
## To train the model

Use the main script to train a model:

```
$ python train_test/main.py -h
usage: main.py [-h] [-D <TimitDirectory>] [-d <Delay>] [-t <TargetDirectory>]
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
python train_test.py -D /home/user/workspace 
```
**Note** /home/user/workspace directory should hold TIMIT folder. 

## To test and calculate the performance results stated in the paper 