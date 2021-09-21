# Introduction
This PyTorch template helps you to start a new machine learning project quickly. It has the following structure:
```
pytorch_template
├── config
│     ├─ASR
│     │     ├─experiment.yaml
│     │     ├─model
│     │     │   ├─ simpleLSTM.yaml
│     │     │   └─ simpleLinear.yaml
│     │     ├─spec_layer
│     │     │   ├─ Mel.yaml
│     │     │   └─ STFT.yaml
│     └─AMT
├── models
│     ├─Models.py
│     └─Tasks.py
├── utils
```

`Tasks.py` contains the tasks you want to work on, such as Automatic Speech Recognition (ASR).

`Models.py` contains the model architectures you want to experiment with.

`config` contains the `.yaml` configuration files.

If you want to try a new model on the existing ASR task, simply add it to `models/Models.py`, and then add a new configuration file for this model as `config/ASR/model/YourNewModel.yaml`.

`python train_ASR.py model=YourNewModel` will train an ASR using your newly defined model `YourNewModel`.



# Usage
## A. Quick Start
### Step1: Download the source code
`git clone https://github.com/KinWaiCheuk/pytorch_template.git`

### Step2: Install the dependencies
`pip install -r requirements.txt`

### Step3: Train a model
`python train_ASR.py`

This command trains a simple LSTM model using the [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) dataset.

By default, train_ASR.py uses the configuration file at `config/ASR/experiment.yaml`. i.e. This model is using Melspectrograms (`spec_layer: Mel`) as the inputs of the LSTM model (`model: simpleLSTM`).

## B. Configuration changes
Due to the usage of `hydra` in this framework, there are two ways to change the configurations.

1. Modifying `config/ASR/experiment.yaml` directly
1. Overwritting the arguments on a command line interface (CLI)

The following section explains how to change the configurations using CLI.
### Change model type
`python train_ASR.py model=simpleLinear`
This line overwrites `model: simpleLSTM` with `model: simpleLinear` and then runs `train_ASR.py` using the overwritten configuration

### Change input type
`python train_ASR.py spec_layer=STFT`
This line overwrites `spec_layer: Mel` with `spec_layer: STFT` and then runs `train_ASR.py` using the overwritten configuration.