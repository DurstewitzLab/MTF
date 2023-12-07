import numpy as np

from dataset import AheadPredictionDataset
from preprocessor import Preprocessor

"""
Example for the usage of the AheadPredictionDataset classe
"""

#Create some data with missing values
continuous_data = np.random.rand(100,2)
missings = np.random.permutation(100)[:25]
continuous_data[missings] = np.nan
ordinal_data = np.random.randint(1, 8, (100,2))

#Dataset parameters
seq_len = 20
batch_size = 3
bpe = 5
ignore_leading_nans = True
return_targets = False
return_tensor = False

#Train-test-split configuration parameters. Choose and define one depending on the
#way you want the train-test split to be defined:
valid_test_len = 10         #determines the number of non-missing data points in the test set
valid_train_len = None      #same for the train set
exact_test_len = None       #determines the exact number of data points in the test set, regardless of whether they're missing
exact_train_len = None      #same for the train set
valid_test_fraction = None  #determines the fraction of all non-missing data points that make up the test set
valid_train_fraction = None #same for the train set

#Create an AheadPredictionDataset instance, and hand over the train-test-split argument of your choice.
ds = AheadPredictionDataset(seq_len, batch_size, bpe, ignore_leading_nans, return_targets,
                            return_tensor, valid_test_len = valid_test_len)
#Determine how to preprocess the data. Note that they are carried out in the order that you define.
pp_steps_continuous_train = ['boxcox', 'zscore', 'smooth']
pp_steps_continuous_test = ['boxcox', 'zscore']
#For some preprocessing steps (shift, smooth, const_imputation) you have to pass additional arguments. 
pp_args = {'kernel': 'bartlett', 'kernel_width': 5}
#Add the continuous time series with preprocessing steps
ds.add_timeseries(continuous_data, 'continuous', pp_steps_continuous_train, pp_steps_continuous_test, pp_args)
#Because this was the first timeseries added to ds, train-test split is done according to it

#Note that the valid sequence indices are those where an entire sequence can be drawn that starts with a non-missing value
sequence_indices = ds.valid_sequence_indices
#The length of the dataset is equal to the number of valid sequence indices
dataset_length = len(ds)

#Add the ordinal data as a second timeseries, without preprocessing
ds.add_timeseries(ordinal_data, 'ordinal')

#Printing the Dataset gives an overview over length, timeseries and settings
print(ds)

#Subscribing the dataset will return a sequence of length seq_len for each timeseries added to it
#This sequence comes from the train set
continuous_sequence, ordinal_sequence = ds[0]

#Get a random dataloader from ds to obtain an iterable over batches
dl = ds.get_rand_dataloader()
for i, batch in enumerate(dl):
    #do something
    pass

#Set return_target True to obtain targets (which are the sequences moved one step forward)
ds.return_target = True
(continuous_sequence, ordinal_sequence), (continuos_target, ordinal_target) = ds[0]

#If you prefer to obtain not a list of tensors (one for each timeseries), but a single tensor, set
ds.return_tensor = True
#the different data modalities will be concatenated along axis 1 of the tensor:
ds.return_target = False
combined_tensor = ds[0]

#Access the individual timeseries via the timeseries dict
continuous_train_set = ds.timeseries['continuous'].train_data
ordinal_test_set = ds.timeseries['ordinal'].test_data
continuous_raw_data = ds.timeseries['continuous'].raw_data

#They also have a plot function. It plots train, test and raw data into one plot for each feature.
#To change the appearance of the train, test, and raw data graphs, use the respective kwargs.
axes = ds.timeseries['continuous'].plot(marker='v', linestyle='dashed')
ds.timeseries['continuous'].plot(axes=axes, raw=True, marker='v', linestyle='')


"""
Example for the usage of the Preprocessor class
"""

#Create some data with missings
raw_data = np.random.rand(100,2)
missings = np.random.permutation(100)[:25]
continuous_data[missings] = np.nan
#Determine how to preprocess the data. Note that they are carried out in the order that you define.
pp_steps = ['boxcox', 'zscore', 'smooth', 'zero_imputation']
pp_args = {'kernel': 'bartlett', 'kernel_width': 3}

#Create a Preprocessor instance. Reference it to only part of the data
pp = Preprocessor(raw_data[:50], pp_steps, pp_args, name='example_preprocessor')
#Now all data-dependent parameters of the preprocessing steps have been determined
#according to the first half of raw_data. These include e.g. lambdas for boxcox transform,
#mean and std for zscoring etc. Preprocessing args are required arguments for the
#preprocessing functions that cannot be determined by data, e.g. the smoothing kernel.

#Printing the preprocessor gives an overview over the preprocessing steps and their order
print(pp)
#Calling the preprocessor on data transforms it according to the pre-determined parameters
preprocessed_data = pp(raw_data[50:])
#The preprocessor has an inverse function that carries out the inverses of all invertible preprocessing
#steps in the reverse order. This is handy if you want to map predictions into the original observation space.
inverse_pp_data = pp.inverse(preprocessed_data)