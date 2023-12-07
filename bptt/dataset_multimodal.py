from torch.utils.data import Dataset, DataLoader , Subset
import numpy as np
import torch as tc
from bptt.preprocessor import Preprocessor
from collections import OrderedDict
import matplotlib.pyplot as plt

class TimeSeries:
    """
    Holds one time series and preprocessors for its train/test set respectively.
    Has attributes 'train_data' and 'test_data' which hold the preprocessed train/test set of the time series.
    """
    def __init__(self, name, data, test_index, train_preprocessor, test_preprocessor, feature_names):
        self.raw_data = tc.tensor(data, dtype=tc.float)
        self.train_preprocessor = train_preprocessor
        self.test_preprocessor = test_preprocessor
        self.train_data = tc.tensor(train_preprocessor(data[:test_index+1]), dtype=tc.float)
        self.test_data = tc.tensor(test_preprocessor(data[test_index:]), dtype=tc.float)
        self.T = data.shape[0]
        self.dim = data.shape[1]
        self.test_index = test_index
        
        self.name = name
        self.feature_names = feature_names
        for k in range(len(feature_names), self.dim):
            self.feature_names.append(f'{name}[{k}]')
        
    def __repr__(self):
        return (f'TimeSeries "{self.name}" of length {self.T}\n\ttrain_preprocessor: {self.train_preprocessor}\n\t'
                f'test_preprocessor: {self.test_preprocessor}')
    
    def __len__(self):
        return self.T
    
    def plot(self, axes=None, raw=False, **kwargs):
        """
        Plot the preprocessed (if raw: raw) timeseries. If ax is none, create new figure.
        """
        newfig = (axes is None)
        if newfig:
            fig, axes = plt.subplots(self.dim, 1, figsize=(6.4, 1+1.6*self.dim))       
            fig.suptitle(self.name)
            if self.dim == 1:
                axes = [axes]
        x = tc.arange(self.T)
        for i, ax in enumerate(axes):
            features = slice(i, i+1)
            if raw:
                ax.plot(x, self.raw_data[:, features], **kwargs)
            else:
                ax.plot(x, tc.cat((self.train_data[:self.test_index, features], self.test_data[:, features])), **kwargs)
            if self.test_index < self.T:
                ax.plot([self.test_index, self.test_index], ax.get_ylim(), linestyle='dashed', color='grey')
            if newfig:
                ax.title.set_text(self.feature_names[i])
        
        if newfig:
            fig.tight_layout()        
        return axes
    
    def to(self, device):
        self.raw_data = self.raw_data.to(device)
        self.train_data = self.train_data.to(device)
        self.test_data = self.test_data.to(device)


class AheadPredictionDataset(Dataset):
    """
    Universal Dataset subclass for ahead prediction, where the training is done on the first part
    and testing on the second part of a time series.
    seq_len:    Length of sequences drawn for training (if 0, it is maximized to the complete train length)
    batch_size: Number of sequences per batch (if 0, it is maximized according to #valid data points and bpe)
    bpe:        Number of batches per epoch
    ignore_leading_nans: Remove leading nans from the first timeseries, and
        remove the leading values of the other timeseries accordingly
    return_target: Calling the dataset will return a tuple of data and targets,
        else, data only.
    return_tensor: If true, calling the dataset, or calling dataset.test_item,
        will return a tensor containing all not-none timeseries
        concatenated along axis 1 (the feature axis). If false, it will return a list
        of tensors, one for each timeseries.
    tolerate_partial_missings: Defines how missing values in the reference timeseries are handled.
        If False, only data points with no missing features are considered valid (default)
        If True, data points with not all features missing are considered valid
    TRAIN-TEST-SPLIT Configuration arguments:
        The following arguments are mutually exclusive. Pass one of them to define the way
        the train-test-split is carried out. Passing none results in no test set.
        valid_test_len/valid_train_len:   
            Number of non-missing data points in the test/train set.
            Ensures that the test set starts with a non-missing data point.
        test_len_exact/train_len_exact:   
            Exact number of data points in the test/train set.
            Note: That means that the test set may start with a missing value or even contain only missings.
        valid_test_fraction/valid_train_fraction:
            Fraction of all non-missing data points in the test/train set.
            Ensures that the test set starts with a non-missing data point.
    """
    def __init__(self, seq_len: int, batch_size: int, bpe: int, valid_indices: bool=True,
                 ignore_leading_nans: bool=True, return_target: bool=False, return_index: bool=False, 
                 return_tensor: bool=False, tolerate_partial_missings: bool=False,
                 valid_test_len=None, valid_train_len=None, exact_test_len=None, exact_train_len=None,
                 valid_test_fraction=None, valid_train_fraction=None):

        super().__init__()   
        self.seq_len = seq_len 
        self.batch_size = batch_size
        self.bpe = bpe
        self.valid_indices=valid_indices
        self.ignore_leading_nans = ignore_leading_nans
        self.return_target = return_target
        self.return_index=return_index      
        self.return_tensor = return_tensor
        self.tolerate_partial_missings = tolerate_partial_missings
        self.timeseries = OrderedDict()
        self.train_test_split_config = {'valid_test_len':valid_test_len, 'valid_train_len':valid_train_len, 
                                        'exact_test_len':exact_test_len, 'exact_train_len':exact_train_len,
                                        'valid_test_fraction':valid_test_fraction, 
                                        'valid_train_fraction':valid_train_fraction}
        if sum([x is not None for x in self.train_test_split_config.values()]) > 1:
            raise RuntimeError('Ambiguous train-test-split configuration. Please pass '
                               f'at most one of the arguments {self.train_test_split_config.values()}.')
                              
    def _reference(self, data):
        """
        Set the train-test split, the valid (non-missing) indices, the valid indices for sequences,
        the removal of leading nans and the total length according to the reference time series data. 
        This method is called as soon as the first time series is appended to the dataset.
        """
        #If leading nans are ignored, determine the number of leading data points to be deleted
        self.leading_nans = 0
        if self.ignore_leading_nans:
            while np.all(np.isnan(data[self.leading_nans])):
                self.leading_nans += 1
            data = data[self.leading_nans:]
        #Set total number of time points
        self.T = data.shape[0]
        self.number_of_sequences = int(self.T / self.seq_len)
        
        #Store reference data
        self.reference_data = tc.tensor(data, dtype=tc.float)        
        #Perform train-test split. Valid training/test indices are all non-missing data indices in the
        #train/test set. Test index indicates where the test set starts.
        self.valid_training_indices, self.valid_test_indices, self.test_index \
            = self.train_test_split(self.reference_data, self.train_test_split_config,
                                    self.tolerate_partial_missings)  
        #Sequence Length == 0 means do not split the time series into sequences
        if self.seq_len==0:
            self.seq_len = self.test_index
        elif self.seq_len > self.test_index:  
            raise ValueError(f'Sequence length {self.seq_len} too long. The train set has '
                             f'only length {self.test_index}.')
        #Valid sequence indices are non-missing data points that are early enough
        #to draw a sequence of length seq_len starting from there
        self.valid_sequence_indices = self.valid_training_indices[self.valid_training_indices <= self.test_index - self.seq_len]
        #Automatically maximize batch_size if batch_size==0
        if self.batch_size==0:
            self.batch_size = len(self.valid_sequence_indices) // self.bpe
        #Warn if number of batches * batches per epoch exceeds number of available valid sequence indices
        points_per_epoch = self.batch_size * self.bpe        
        if len(self.valid_sequence_indices) < points_per_epoch:
            print(f'Warning: The train data of the reference time series contain only {len(self)} distinct '
                  f'sequences of length {self.seq_len}, since they must start at a non-missing data point. '
                  f'For an epoch of {self.bpe} batches of {self.batch_size} sequences each, '
                  f'you need {points_per_epoch}. The resulting number of batches per epoch '
                  f' is {np.ceil(len(self)/self.batch_size)}.')
        
    @property
    def data(self):
        return self.raw_data.clone()
    
    def __repr__(self):
        if len(self.timeseries)>0:
            return (f'AheadPredictiondDataset of {self.T} time steps with timeseries'+'\n\t'
                    +"\n\t+ ".join(self.timeseries.keys()) + '\n'
                    +f'Settings: \n\treturn_target: {self.return_target}\n'
                    +f'\treturn_tensor: {self.return_tensor}')
        else:
            return 'Empty AheadPredictionDataset'

    def __len__(self):
        if self.valid_indices:
          return len(self.valid_sequence_indices)
        else:
          return self.number_of_sequences

    def __getitem__(self, idx):
        valid_idx = self.valid_sequence_indices[idx]
        x = []
        y = []
        for key, sts in self.timeseries.items():
            if sts is not None:
                x.append(sts.train_data[valid_idx : valid_idx + self.seq_len])
                y.append(sts.train_data[valid_idx + 1 : valid_idx + self.seq_len + 1])
            elif not self.return_tensor:
                x.append(None)
                y.append(None)
        if self.return_tensor:
            x = tc.cat(x, dim=1)
            y = tc.cat(y, dim=1)
        if self.return_target:
        
          if self.return_index:
              
            return x, y, idx
          else:
            return x,y
        else:
            return x
    
    def add_timeseries(self, data: np.ndarray, name='', train_preprocessing=[], 
                          test_preprocessing=[], preprocessing_args={}, feature_names=[]):
        """
        Add a TimeSeries object to the dataset.timeseries dictionary.
        data:   time series array of shape (time x features). If None, a None object
                will be added to dataset.timeseries instead of a TimeSeries.
        name:   key of the timeseries in dataset.timeseries
        train/test_preprocessing: list of strings, indicating the preprocessing steps
                for train and test set of the timeseries. These must be implemented in
                the preprocessor class
        preprocessing_args: dict of arguments passed to the preprocessing functions        
        feature_names: collection of strings, indicating the names of the time series features.
                Useful for plotting.
        """
        if name=='':
            name = f'timeseries_{len(self.timeseries)+1}'
            
        if data is not None:  
            
            data = data.copy()
            if (data.shape[0] == 1) and len(data.shape)==3:
                data = data.squeeze(0)
            if len(self.timeseries)==0:
                self._reference(data)                
            data = data[self.leading_nans:]
            if self.T > data.shape[0]:
                missing = self.T - data.shape[0]
                data = np.vstack([data, np.zeros((missing, data.shape[1]))])
            else:
                data = data[:self.T]
    
            trainPP = Preprocessor(data[:self.test_index+1], steps=train_preprocessing, 
                                   args=preprocessing_args, name=f'train_preprocessor_{name}')
            testPP = Preprocessor(data[:self.test_index+1], steps=test_preprocessing, 
                                  args=preprocessing_args, name=f'test_preprocessor_{name}')
            self.timeseries[name] = TimeSeries(name, data, self.test_index, trainPP, testPP, feature_names)
            
        else:
            if len(self.timeseries)==0:
                raise ValueError('The first timeseries of a dataset cannot be None.')
            self.timeseries[name] = None
            
            
    def add_timeseries_hier(self, data: np.ndarray, name='', train_preprocessing=[], 
                          test_preprocessing=[], preprocessing_args={}, feature_names=[]):
        """
        Add a TimeSeries object to the dataset.timeseries dictionary.
        data:   time series array of shape (sub_id x time x features). If None, a None object
                will be added to dataset.timeseries instead of a TimeSeries.
        name:   key of the timeseries in dataset.timeseries
        train/test_preprocessing: list of strings, indicating the preprocessing steps
                for train and test set of the timeseries. These must be implemented in
                the preprocessor class
        preprocessing_args: dict of arguments passed to the preprocessing functions        
        feature_names: collection of strings, indicating the names of the time series features.
                Useful for plotting.
        """
        print("test")
        if name=='':
            name = f'timeseries_{len(self.timeseries)+1}'
            
        if data is not None:  
            
            data = data.copy()
                        
            #data = data[self.leading_nans:]
           # if self.T > data.shape[1]:
           #     missing = self.T - data.shape[0]
           #     data = np.vstack([data, np.zeros((missing, data.shape[1]))])
           # else:
           #     data = data[:self.T]
    
            trainPP = Preprocessor(data[:,:self.test_index+1], steps=train_preprocessing, 
                                   args=preprocessing_args, name=f'train_preprocessor_{name}')
            testPP = Preprocessor(data[:,:self.test_index+1], steps=test_preprocessing, 
                                  args=preprocessing_args, name=f'test_preprocessor_{name}')
            self.timeseries[name] = TimeSeries(name, data, self.test_index, trainPP, testPP, feature_names)
            
        else:
            if len(self.timeseries)==0:
                raise ValueError('The first timeseries of a dataset cannot be None.')
            self.timeseries[name] = None
        
    def show_timeseries(self):
        """
        Prints info about the dataset.timeseries
        """
        for name, sts in self.timeseries.items():
            if sts is not None:
                print(f'{name}: {sts}')
            else:
                print(f'{name}: None')
                
    def train_set(self):
        """
        Get a list of the train_sets of all dataset.timeseries.
        """
        res = []
        for name, sts in self.timeseries.items():
            if sts is not None:
                res.append(sts.train_data)
            else:
                res.append(None)
        return res
    
    def test_set(self):
        """
        Get a list of the test_sets of all dataset.timeseries.
        """
        res = []
        for name, sts in self.timeseries.items():
            if sts is not None:
                res.append(sts.test_data)
            else:
                res.append(None)
        return res
        
    def test_tensor(self):
        """
        Get a tensor of the test_sets of all dataset.timeseries.
        """
        res = []
        for name, sts in self.timeseries.items():
            if sts is not None:
                res.append(sts.test_data)
            else:
                res.append(None)
        stack=tc.cat(res, dim=-1)
        return stack
    
    def test_item(self, idx, seq_len=0):
        """
        Similar to subscription/__getitem__ routine, but returns sequence from the test set.
        Idx indicates the index among the valid test indices.
        If seq_len==0, the sequence reaches from idx to the end of the test set.
        """
        if not self.has_test_set():
            raise RuntimeError('Test item requested from dataset which has no test set')
        if seq_len==0:
            seq_len = self.test_len_with_missings() - 1
        valid_idx = self.valid_test_indices[idx] - self.test_index
        x = []
        y = []
        for key, sts in self.timeseries.items():
            if sts is not None:
                x.append(sts.test_data[valid_idx : valid_idx + seq_len])
                y.append(sts.test_data[valid_idx + 1 : valid_idx + seq_len + 1])
            elif not self.return_tensor:
                x.append(None)
                y.append(None)
        if self.return_tensor:
            x = tc.cat(x, dim=1)
            y = tc.cat(y, dim=1)
        if self.return_target:
            return x, y
        else:
            return x
    
    def has_test_set(self):
        """
        True if the test set is not empty
        """
        return self.test_index < self.T - 1
    
    def test_len_with_missings(self):
        """
        Returns the de facto test set length after the train-test split
        """
        return self.test_set()[0].shape[0]
    
    def get_rand_dataloader(self):
        """
        Returns a torch dataloader with sequences in random order
        """        
        def list_collate(batch):
            print(len(batch))
            batch_data = [[] for sts in self.timeseries]
            if self.return_target:
                target_data = [[] for sts in self.timeseries]
                for data_item, target_item in batch:
                    for i in range(len(self.timeseries)):
                        if data_item[i] is not None:
                            batch_data[i].append(data_item[i])
                            target_data[i].append(target_item[i])
                for i in range(len(self.timeseries)):
                    if len(batch_data[i]) > 0:
                        batch_data[i] = tc.stack(batch_data[i])
                        target_data[i] = tc.stack(target_data[i])
                    else:
                        batch_data[i] = None
                        target_data[i] = None                
                return batch_data, target_data
            else:
                for data_item in batch:
                    for i in range(len(self.timeseries)):
                        if data_item[i] is not None:
                            batch_data[i].append(data_item[i])
                for i in range(len(self.timeseries)):
                    if len(batch_data[i]) > 0:
                        batch_data[i] = tc.stack(batch_data[i])
                    else:
                        batch_data[i] = None
                return batch_data
    
        indices = np.random.permutation(len(self))[:self.bpe*self.batch_size]
        subset = Subset(self, indices.tolist())
        if self.return_tensor:
            dataloader = DataLoader(subset, batch_size=self.batch_size)
        else:
            dataloader = DataLoader(subset, batch_size=self.batch_size, collate_fn=list_collate)
        return dataloader
        
    def to(self, device: tc.device):
        self._data = self._data.to(device)

        
    
    def get_rand_batch_indices(self):
        """
        Returns n indices of sequences in random order, where n is self.batch_size 
        """
        indices = np.random.permutation(len(self))[:self.bpe*self.batch_size]
        return indices

    def to(self, device: tc.device):
        self.reference_data = self.reference_data.to(device)
        for sts in self.timeseries.values():
            if sts is not None:
                sts.to(device)
    
    def train_test_split(self, data_tensor, config, tolerate_partial_missings=False):
        """
        The test set are the last m data points of the time series, where
            m = min{n | {x_n, x_{n+1}, ... , x_N} contains at least <test_len> non-missing data points}
        m cannot be greater than <max_test_points>.
        <test_len> cannot be greater than half the total number of non-missing data points.
        exclude_listwise_missing: if true, datapoints are considered invalid if a single feature is missing.
            Else, they are invalid only if all features are missing.
        Note that the first element of the test set is the last target of the train set. Therefore, the last
        data point to predict in trainig will always be non-missing.
        """                
        if tolerate_partial_missings:
            valid_indices = tc.arange(data_tensor.shape[0])[(~data_tensor.isnan()).any(axis=1)]
        else:
            valid_indices = tc.arange(data_tensor.shape[0])[(~data_tensor.isnan()).all(axis=1)]
            
        test_len = 0
        if config['valid_test_len']:
            test_len = config['valid_test_len']
            if len(valid_indices) <= config['valid_test_len']:
                raise ValueError(f'The time series contains only {len(valid_indices)} non-missings, while '
                                   f'you requested {test_len} for the test set.')           
        elif config['valid_train_len']:
            test_len = len(valid_indices) - config['valid_train_len']            
            if len(valid_indices) <= config['valid_train_len']:
                raise ValueError(f'The time series contains only {len(valid_indices)} non-missings, while '
                                   f"you requested {config['valid_train_len']} for the train set.")
        elif config['valid_test_fraction']:
            test_len = int(np.ceil(len(valid_indices) / config['valid_test_fraction']))
        elif config['valid_train_fraction']:
            test_len = int(np.floor(len(valid_indices) / (1-config['valid_train_fraction'])))
        
        if test_len > 0:
            valid_training_indices = valid_indices[:-test_len]
            valid_test_indices = valid_indices[-test_len:]            
            # test_len_incl_nan = data_tensor.shape[0] - valid_test_indices[0]
            # if test_len_incl_nan > max_test_points:
            #     raise ValueError(f'The test set would be of length {test_len_incl_nan}, '
            #                        f'but you requested max {max_test_points}')            
            test_index = valid_test_indices[0].item()
        elif config['exact_test_len']:
            if data_tensor.shape[0] <= config['exact_test_len']:
                raise ValueError('Requested test_len greater than the timeseries length')
            test_index = data_tensor.shape[0] - config['exact_test_len']
            valid_training_indices = valid_indices[valid_indices <= test_index]
            valid_test_indices = valid_indices[valid_indices >= test_index]
        elif config['exact_train_len']:
            if data_tensor.shape[0] < config['exact_train_len']:
                raise ValueError('Requested train_len greater than the timeseries length')
            test_index = config['exact_train_len'] - self.leading_nans
            valid_training_indices = valid_indices[valid_indices <= test_index]
            valid_test_indices = valid_indices[valid_indices >= test_index]
        else:
            valid_training_indices = valid_indices
            valid_test_indices = tc.tensor([])
            test_index = data_tensor.shape[0] - 1
        
        return valid_training_indices, valid_test_indices, test_index
        
