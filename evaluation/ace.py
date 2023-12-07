

import scipy.stats
import numpy as np



# function to compute autocorrelation
def get_acf(ts, lag):

    result = np.zeros(lag+1)
    result[0] = 1

    mean  = np.mean(ts)
    variance = np.var(ts)
    
    for i in range(1,lag):
        T_var = len(ts[:-i])
        result[i] = np.dot((ts[:-i]-mean), (ts[i:]-mean))/(variance*T_var)
    return result

# function to compute spearman autocorrelation    
def spearman_autocorrelation(ts,lag):
    result = np.zeros(lag+1)
    result[0]=1
    for i in range(1,lag):
        result[i]=scipy.stats.spearmanr(ts[:-i],ts[i:]).correlation
        
    return result


def autocorrelation_error(generated, ground_truth, lags):
    
    dims=generated.shape[-1]
    error=np.zeros(dims)
    for i in range(dims):
        ac_generated=get_acf(generated[:,i],lags)
        ac_gt=get_acf(ground_truth[:,i],lags)
        
        error[i]=np.mean((ac_generated-ac_gt)**2)
        
        average_error=np.mean(error)
    
    return average_error
    
def spearman_autocorrelation_error(generated, ground_truth, lags):
    
    dims=generated.shape[-1]
    error=np.zeros(dims)
    for i in range(dims):
        ac_generated=np.nan_to_num(spearman_autocorrelation(generated[:,i],lags))
        ac_gt=spearman_autocorrelation(ground_truth[:,i],lags)
    
        error[i]=np.mean((ac_generated-ac_gt)**2)
        
    
    return np.mean(error), error
    
    
