from scipy import stats
import numpy as np
import torch as tc
import warnings



class Preprocessor:
    """
    A Preprocessor combines a selection of preprocessing steps. It is defined for a reference data set and
    for a sequence of preprocessing steps. Calling a Preprocessor with a data argument performs this
    sequence on the data, ignoring missing values. 
    The parameters for the preprocessing steps are determined w.r.t. the reference set, e.g. the lambdas
    for the Boxcox transform, or mean and std for zscoring.
    The time dim must be the 0th dimension of all data passed to the Preprocessor.
    Additional arguments to the processing functions must by passed as dict in the args argument.
    """
    
    def __init__(self, reference_data: np.ndarray, steps: list, args: dict={}, name='without name'):

        super().__init__()
        self.name = name
        self.pipeline = []
        self.steps = steps
        # self.reference_data = reference_data
        self.features = reference_data.shape[1]
        d = reference_data.copy()
        self.args = self.set_optional_args_to_default(args)
        for st in steps:            
            if st == 'shift':
                self.shift_value = args['shift']
                self.pipeline.append(lambda x: self.shift(x, self.shift_value))
                d = self.shift(d, self.shift_value)
                
            elif st == 'boxcox':
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    self.boxcox_lmbdas = np.zeros(reference_data.shape[1])*np.nan
                    for i in range(reference_data.shape[1]):
                        notna_map = ~np.isnan(reference_data[:,i])
                        try:
                            _, self.boxcox_lmbdas[i] = stats.boxcox(reference_data[notna_map, i], lmbda=None)
                        except ValueError as e:
                            pass #This happens if data is constant or nan. Then lmbda remains the default 0.
                        except RuntimeWarning as RW:
                            if 'overflow' in RW.__repr__():
                                pass #This happens if data is almost constant. Then lmbda remains the default 0.
                            else:
                                raise
                self.pipeline.append(lambda x: self.boxcox(x, self.boxcox_lmbdas))
                d = self.boxcox(d, self.boxcox_lmbdas)                    
                
            elif st == 'zscore':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.zscore_mean = np.nanmean(d, 0)
                    self.zscore_std = np.nanstd(d, 0)
                self.pipeline.append(lambda x: self.zscore(x, self.zscore_mean, self.zscore_std))
                d = self.zscore(d, self.zscore_mean, self.zscore_std)                    
                
            elif st == 'smooth':
                self.smooth_kernel = args['kernel']
                self.smooth_kernel_width = args['kernel_width']
                self.pipeline.append(lambda x: self.smooth(x, self.smooth_kernel, self.smooth_kernel_width))
                d = self.smooth(d, self.smooth_kernel, self.smooth_kernel_width)
                
            elif st == 'mean_imputation':
                self.partial_imputation = self.args['partial_imputation']
                self.imputation_mean = np.nanmean(d, 0)
                self.pipeline.append(lambda x: self.const_imputation(x, self.imputation_mean,
                                                                     self.partial_imputation))
                d = self.const_imputation(d, self.imputation_mean, self.partial_imputation)
                
            elif st == 'zero_imputation':
                self.partial_imputation = self.args['partial_imputation']
                self.imputation_const = 0
                self.pipeline.append(lambda x: self.const_imputation(x, 0, self.partial_imputation))
                d = self.const_imputation(d, 0, self.partial_imputation)
                
            elif st == 'const_imputation':
                self.partial_imputation = self.args['partial_imputation']
                self.imputation_const = args['imputation_const']
                self.pipeline.append(lambda x: self.const_imputation(x, self.imputation_const,
                                                                     self.partial_imputation))
                d = self.const_imputation(d, self.imputation_const, self.partial_imputation)
                
            else:
                raise NotImplementedError(f'Unknown preprocessing step {st}')
            
    def __call__(self, data):
        if isinstance(data, np.ndarray):
            d = data.copy()
        elif isinstance(data, tc.Tensor):
            d = data.clone()
        if d.shape[1] != self.features or len(d.shape)>2:
            raise ValueError(f'Preprocessor {self.name} received invalid data (wrong shape)')
        for f in self.pipeline:
            d = f(d)
        return d

    def __repr__(self):
        if len(self.steps)>0:
            steps_string = ' --> '.join([f'{ st }' for st in self.steps])
        else:
            steps_string = 'no preprocessing'
        return f'{self.name}: {steps_string}'
    
    def set_optional_args_to_default(self, args):
        if 'partial_imputation' not in args.keys():
            args['partial_imputation'] = False
        if 'kernel' not in args.keys():
            args['kernel'] = 'bartlett'
        if 'kernel_width' not in args.keys():
            args['kernel_width'] = 5
        return args
        
    def shift(self, data, shift):
        """
        Adds a constant <shift> to the data
        """
        return data + shift
        
    def boxcox(self, data, lmbdas):
        """
        Box cox transformation
        """
        data_ = data.T
        for i in range(data_.shape[0]):
            if lmbdas[i] == 0:
                data_[i] = np.log(data_[i])
            elif ~np.isnan(lmbdas[i]):
                data_[i] = (data_[i]**lmbdas[i] - 1) / lmbdas[i]
            
        return data_.T
    
    def zscore(self, data, mean, std):    
        """
        Calculates z-score
        """
        std[std==0] = 1
        return (data - mean)/std
        
    def smooth(self, data, kernel='bartlett', kernel_width=5):
        """
        Convolves the time series up to test_index with the selected kernel, ignoring nans.
        """
        def nanconvolve(array, kernel):
            """
            Equivalent to numpy.convolve(array, kernel, mode='valid') except:
                - kernel is divided by kernel sum
                - if there are nans in the convolution, the dot product is taken only across the notnan values.
                  only if there is a sequence of nans as long as the kernel, nans will remain in the data.
            """
            if np.isnan(kernel).any():
                raise RuntimeError('convolution kernel must not contain nans')
            array_without_nan = np.nan_to_num(array, nan=0.)
            M = len(kernel)
            N = len(array)
            res = np.zeros(N-M+1)
                
            for i in range(N-M+1):
                local_notnans = ~np.isnan(array[i:i+M])
                local_kernel = kernel[local_notnans]
                if local_kernel.sum() != 0:            
                    res[i] = np.dot(array_without_nan[i:i+M], kernel/(kernel[local_notnans].sum()))
                else:
                    res[i] = np.nan
            return res
        
        if isinstance(kernel, str):
            window = eval(f'np.{kernel}({kernel_width})')
        else:
            window = kernel
            kernel_width = len(kernel)
            
        data = data.T        
        for i in range(data.shape[0]):
            signal = data[i]
            left_pad = int(np.floor((kernel_width - 1)/2))
            right_pad = int(np.ceil((kernel_width - 1)/2))+1
            signal = np.r_[signal[left_pad:0:-1], signal, signal[-1:-right_pad:-1]]
            data[i] = nanconvolve(signal, window)
        return data.T
    
    def const_imputation(self, data, const, partial):
        """
        Imputes nans with constant. If partial, imputes only if not all features are nan.
        """
        if isinstance(const, np.ndarray) and len(const.shape)==1:
            const = np.tile(const, (data.shape[0], 1))
        else:
            const = np.ones(data.shape) * const
        const[~np.isnan(data)] = 0
        if partial:
            data = data.copy()
            data[~np.isnan(data).all(axis=1)] = np.nan_to_num(data[~np.isnan(data).all(axis=1)],
                                                              nan=0)
        else:
            data = np.nan_to_num(data, copy=True, nan=0)
        
        return data + const
    
    def inverse(self, data):
        """
        Inverts all invertible operations of the preprocessor on <data>
        This is handy to embed predictions from a model trained on 
        preprocessed data in the observation space.
        """
        
        def inverse_boxcox(data, lmbdas):
            data_ = data.T
            for i in range(data_.shape[0]):
                if lmbdas[i] == 0:
                    data_[i] = np.exp(data_[i])
                elif ~np.isnan(lmbdas[i]):
                    data_[i] = (lmbdas[i]*data_[i] + 1)**(1/lmbdas[i])
                
            return data_.T
        
        def inverse_zscore(data, mean, std):
            return data*std + mean
        
        for st in reversed(self.steps):
            if st=='zscore':
                data = inverse_zscore(data, self.zscore_mean, self.zscore_std)
            if st=='boxcox':
                data = inverse_boxcox(data, self.boxcox_lmbdas)
            if st=='shift':
                data = self.shift(data, -self.shift_value)
                
        return data
            