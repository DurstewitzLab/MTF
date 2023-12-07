# Dynamical Systems Reconstruction using MVAE-TF

## Setup
Install your anaconda distribution of choice, e.g. miniconda via the bash
script ```miniconda.sh```:
```
$ ./miniconda.sh
```
Create the local environment:
```
$ conda env create -f environment.yml
```
Activate the environment and install the package
```
$ conda activate BPTT_TF-torch1.10
$ pip install -e .
```

## Running Code

Before running any example make sure to activate the environment!

The code is executed running the main.py and ubermain.py files. 

The default arguments are defined in the main file. The ubermain file lets you select multiple settings for efficient hyperparameter grid searches.
At the bottom of the ubermain file, you can select the number of runs for each experiment, the number of CPUs for multiprocessing, and the number of processes run parallel on a single GPU.

## Example Run and Evaluation

With the default settings, a dendPLRNN is trained with MTF on multimodal data from the chaotic Lorenz-63 system. Results and example visualizations can be accessed by running tensorboard in the respective logdir. An example evaluation for a Lorenz system trained on Gaussian, ordinal and count data, which was used in Figure 2a) in the paper can be found in the Jupyter Notebook "Example Evaluation Lorenz".In this notebook you can sample trajectories from the trained system and compute evaluation metrics such as the maximum Lyapunov exponent and state space agreement.

## Model Structure

The SAE folder contains all model architectures relating to the encoder/decoder part of the model. The different encoder architectures (CNN, RNN, Transformer etc.) are in the encoder.py foloder, while the decoder/observation models (Gaussian, ordinal, Poisson, ZIP, Negative Binomial, and categorical) are in the decoder.py file.

The bptt folder contains all files relating to MTF training. Most importantly, the RNN_model.py file defines the model architecture of the DSR model, the bptt_algorithm.py file defines the loss function and training algorithm and the models.py file defines the overall model architecture (encoder/decoder and DSR model).
Data loading is via the dataset_multimodal.py file, with data paths and dataset parameters defined in the load_dataset_multimodal function in utils.py.

## Evaluation
Evaluating trained models efficiently can be done by calling ```main_eval.py```,
where the save paths and input data are defined at the bottom of the file, and selected evaluation metrics are output as csv files.

## Software Versions
* Python 3.11
* PyTorch 2.1.1
