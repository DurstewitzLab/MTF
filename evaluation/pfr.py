import numpy as np
##zero rate

def poisson_firing_rates(x_gen, x_true):


  T,dp=x_true.shape

  gt_counts=np.zeros((dp))
  gt_zeros=np.zeros((dp))
  
  generated_counts=np.zeros((dp))
  generated_zeros=np.zeros((dp))
  
  
    ##firing rate
  for j in range(dp):
      generated_counts[j]=np.mean(x_gen[:,j])
      gt_counts[j]=np.mean(x_true[:,j])


    ##zero rate
  for i in range(T):
      for j in range(dp):
          if x_gen[i,j]==0:
              generated_zeros[j]+=1
          if x_true[i,j]==0:
              gt_zeros[j]+=1
              
  difference_rate=[]
  difference_zeros=[]
  
  for j in range(dp):
    difference_rate.append(np.abs(generated_counts[j]-gt_counts[j])/T) 
    difference_zeros.append(np.abs(generated_zeros[j]-gt_zeros[j])/T)
 
        
  return difference_rate, difference_zeros
  