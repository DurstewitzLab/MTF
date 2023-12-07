import scipy.stats
import numpy as np

def spearman_corr(x_gen, x_true):

    n_features=x_true.shape[-1]

    # plot_kl(x_gen, x_true, n_bins)
    spearman_train=np.zeros((n_features,n_features))
    spearman_test=np.zeros((n_features,n_features))
    

    for i in range(n_features):
        for j in range(n_features):
            spearman_train[i,j]=scipy.stats.spearmanr(x_true[:,i],x_true[:,j]).correlation
            spearman_test[i,j]=scipy.stats.spearmanr(x_gen[:,i],x_gen[:,j]).correlation        
    
        
    difference=(np.nan_to_num(spearman_train)-np.nan_to_num(spearman_test))**2
    spearman_diff_matrix=np.sqrt(np.nan_to_num(difference))
    spearman_diff=[]
    for i in range(n_features):
      spearman_diff.append(np.mean(spearman_diff_matrix[i]))
    spearman_mean=np.mean(spearman_diff)
    
                    
    return spearman_diff, spearman_mean
