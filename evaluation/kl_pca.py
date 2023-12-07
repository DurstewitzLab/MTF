## Metrics
import torch as tc
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
from evaluation.klx_gmm import calc_kl_from_data
import scipy.spatial

#ommented out due to frequent errors with package
#from procrustes import rotational


def marginalize_pdf(pdf, except_dims):
    """
    Marginalize out all except the specified dims
    :param pdf: multidimensional pdf
    :param except_dims: specify dimensions to keep
    :return: marginalized pdf
    """
    if len(pdf.shape) > 2:
        l = list(range(len(pdf.shape)))
        l = [i for i in l if i not in except_dims]
        pdf = pdf.sum(tuple(l))
    return pdf


def plot_kl(x_gen, x_true):
    p_gen, p_true = get_pdf_from_timeseries(x_gen, x_true)
    kl_value = kullback_leibler_divergence(p_true, p_gen)
    p_true = marginalize_pdf(p_true, except_dims=(0, 2))
    if p_gen is not None:
        p_gen = marginalize_pdf(p_gen, except_dims=(0, 2))
    else:
        p_gen = 0 * p_true
    if kl_value is None:
        kl_string = 'None'
    else:
        kl_string = '{:.2f}'.format(kl_value)

    fig, axs = plt.subplots(1, 2)
   # fig.set_title('KLx: {}'.format(kl_string))
    axs[0].imshow(p_gen.numpy().T[::-1])
    axs[0].set_xticks(())
    axs[0].set_yticks(())
    axs[0].set_title('generated data,KLx: {}'.format(kl_string))
  #  axs[0].set_title()
    axs[1].imshow(p_true.numpy().T[::-1])
    axs[1].set_xticks(())
    axs[1].set_yticks(())
    axs[1].set_title('ground truth')
    plt.savefig('KLx_002_20bins.pdf')
    plt.show()
    
    return kl_value


def loss_kl(x1, x2, n_bins=10, symmetric=True):
    """
    Spatial KL-divergence loss function
    :param x1: time series 1
    :param x2: time series 2, reference time series
    :param n_bins: number of histogram bins
    :param symmetric: symmetrized KL-divergence
    :return: loss (scalar)
    """
    p1, p2 = get_pdf_from_timeseries(x1, x2, n_bins)
    kl21 = kullback_leibler_divergence(p2, p1)

    if not symmetric:
        loss = kl21  # assuming p2 is ground truth
    else:
        kl12 = kullback_leibler_divergence(p1, p2)
        loss = (kl12 + kl21) / 2
    return loss


def kullback_leibler_divergence(p1, p2):
    """
    Calculate Kullback-Leibler divergences
    """
    if p1 is None or p2 is None:
        kl = None
    else:
        kl = (p1 * tc.log(p1 / p2)).sum()
    return kl


def calc_histogram(x, n_bins, min_, max_):
    """
    Calculate a multidimensional histogram in the range of min and max
    works by aggregating values in sparse tensor,
    then exploits the fact that sparse matrix indices may contain the same coordinate multiple times,
    the matrix entry is then the sum of all values at the coordinate
    for reference: https://discuss.pytorch.org/t/histogram-function-in-pytorch/5350/9
    Outliers are discarded!
    :param x: multidimensional data: shape (N, D) with N number of entries, D number of dims
    :param n_bins: number of bins in each dimension
    :param min_: minimum value
    :param max_: maximum value to consider for histogram
    :return: histogram
    """
    D = x.shape[1]  # number of dimensions
    # get coordinates
    coord = tc.LongTensor(x.shape)
    for d in range(D):
        span = max_[d] - min_[d]
        xd = (x[:, d] - min_[d]) / span
        xd = xd * n_bins
        xd = xd.long()
        coord[:, d] = xd
    # discard outliers
    cond1 = coord > 0
    cond2 = coord < n_bins
    inlier = cond1.all(1) * cond2.all(1)
    coord = coord[inlier]

    size_ = tuple(n_bins for d in range(D))
    hist = tc.sparse.FloatTensor(coord.t(), tc.ones(coord.shape[0]), size=size_).to_dense()
    return hist


def get_min_max_range(x_true):
    min_ = -2 * x_true.std(0)
    max_ = 2 * x_true.std(0)
    return min_, max_


def normalize_to_pdf_with_laplace_smoothing(histogram, n_bins, smoothing_alpha=10e-6):
    if histogram.sum() == 0:  # if no entries in the range
        pdf = None
    else:
        dim_x = len(histogram.shape)
        pdf = (histogram + smoothing_alpha) / (histogram.sum() + smoothing_alpha * n_bins ** dim_x)
    return pdf

def get_pdf_from_timeseries(x_gen, x_true, n_bins=30):
    """
    Calculate spatial pdf of time series x1 and x2
    :param x_gen: multivariate time series: shape (T, dim)
    :param x_true: multivariate time series, used for choosing range of histogram
    :param n_bins: number of histogram bins
    :return: pdfs
    """
    min_, max_ = get_min_max_range(x_true)
    hist_gen = calc_histogram(x_gen, n_bins=n_bins, min_=min_, max_=max_)
    hist_true = calc_histogram(x_true, n_bins=n_bins, min_=min_, max_=max_)

    p_gen = normalize_to_pdf_with_laplace_smoothing(histogram=hist_gen, n_bins=n_bins)
    p_true = normalize_to_pdf_with_laplace_smoothing(histogram=hist_true, n_bins=n_bins)
    return p_gen, p_true


def klx_metric(x_gen, x_true, n_bins=10):

    p_gen, p_true = get_pdf_from_timeseries(x_gen, x_true, n_bins)
    return kullback_leibler_divergence(p_true, p_gen)
    
    
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
                     
                     
def grid_search_klx(step_size, grid_steps, length, x_gen, x_gt):

    boundary=(grid_steps*step_size)/2
    
    x = np.arange(-boundary, boundary, step_size)
    y = np.arange(-boundary, boundary, step_size)
    z = np.arange(-boundary, boundary, step_size)
    theta=np.arange(-boundary, boundary, step_size)
    
    klx=np.zeros((grid_steps,grid_steps,grid_steps,grid_steps))
    
    print("beginning grid search.")
    
    for i in range(grid_steps):
        for j in range(grid_steps):
            for k in range(grid_steps):
                for l in range(grid_steps):
                    
                    axis=[x[i], y[j], z[k]]
                    angle=theta[l]
                    
                    rotated_data=np.dot(x_gen[:length,:3], rotation_matrix(axis, angle))
                    klx[i,j,k,l]=klx_metric(tc.tensor(rotated_data), x_gt[:length], n_bins=10)
                 #   print("KLx:", klx[i,j,k,l])
    print("finished grid search.")
    
    return klx


###############################################################

def kl_pca(ground_truth, latent_states, n_bins):
    
    latent_states=tc.nan_to_num(latent_states)    
    latent_states[latent_states == float("Inf")] = 0
    
    sc = StandardScaler() # creating a StandardScaler object
    pca = PCA()
    lat_std = sc.fit_transform(latent_states)
   # X_pca = pca.fit(lat_std[:])
    num_components = ground_truth.shape[-1]
    pca = PCA(num_components)
    X_pca = pca.fit_transform(lat_std) # fit and reduce dimension
    X_std = sc.fit_transform(X_pca) #standardize latent states
    T=X_std.shape[0]
    data_gt=tc.tensor(np.copy(ground_truth))
    
    
    grid_search=1
    ######grid search
    if grid_search:
    
      if data_gt.shape[-1]>3:
        num_components = 3
        pca = PCA(num_components)
        X_pca = pca.fit_transform(data_gt)
        X_std = sc.fit_transform(X_pca)
        data_gt=tc.copy(X_std)
    
      grid_steps=10
      step_size=0.2
      boundary=(grid_steps*step_size)/2
      
      kl_grid=grid_search_klx(step_size, grid_steps, 3000, X_std, data_gt)
    
      x = np.arange(-boundary, boundary, step_size)
      y = np.arange(-boundary, boundary, step_size)
      z = np.arange(-boundary, boundary, step_size)
      theta=np.arange(-boundary, boundary, step_size)
  
      min_index=np.where(kl_grid == kl_grid.min())    
      axis = [x[min_index[0][0]], y[min_index[1][0]], z[min_index[2][0]]]
      angle = theta[min_index[3][0]]
      rotated_optimal=np.dot(X_std, rotation_matrix(axis, angle))
    #  minimal_kl=calc_kl_from_data(tc.tensor(transformed).cpu(), data_gt.clone().detach().cpu())
      minimal_kl=klx_metric(tc.tensor(rotated_optimal), data_gt.clone().detach().cpu(), n_bins=10)
    
    #### Compute rotation via procrustes transformation. import commented out due to potential dependency issues
    
    else:
      procrustes=scipy.spatial.procrustes(X_std, data_gt[:T])#
      rotated_procrustes=procrustes[0]
      gt_procrustes=procrustes[1]
      #rescale for KL bin
      sc = StandardScaler()
      scaled_procrustes=sc.fit_transform(rotated_procrustes)
      scaled_gt_procrustes=sc.fit_transform(gt_procrustes)
      minimal_kl=calc_kl_from_data(tc.tensor(scaled_procrustes).cpu(), tc.tensor(scaled_gt_procrustes).cpu())
     
     
      data2=data_gt[:T].numpy()
      result = rotational(X_std, data2, translate=True)
      transformed = np.dot(result.new_a, result.t)
      
      minimal_kl=calc_kl_from_data(tc.tensor(transformed).cpu(), data_gt.clone().detach().cpu())
 
    print("Minimal KL value:", minimal_kl)
    
    return minimal_kl





