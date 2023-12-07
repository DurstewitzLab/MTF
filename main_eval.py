from typing import List


import os
import numpy as np
import torch as tc
import pandas as pd
from glob import glob
from scipy.stats import median_abs_deviation
import utils

#all metrics are defined in the evaluation folder
from evaluation import mse
from evaluation import ope
from evaluation import ppe
from evaluation import klx
from evaluation import kl_pca
from evaluation import sce
from evaluation import pfr
from evaluation import ace
#from evaluation import cat_classifier
from evaluation import lyap
from evaluation.klx_gmm import calc_kl_from_data

from bptt import models
from bptt import preprocessor


from scipy import stats
from evaluation.psh import power_spectrum_error, power_spectrum_error_per_dim

EPOCH = None
SKIP_TRANS = 0
DATA_GENERATED = None
PRINT = True


def get_generated_data_long(model,data):
    """
    Use global variable as a way to draw one long trajectory only once for evaluating several metrics, for speed.
    :param model:
    :return:
    """
    global DATA_GENERATED
    # Problem: if block is only entered once per training,
    # to trajectory is never updated with better models.
    if DATA_GENERATED is None:
        X, Z = model.generate_free_trajectory(data, len(data))
        DATA_GENERATED = X[SKIP_TRANS:]

    return DATA_GENERATED

    
def get_generated_data(model,data):
    """
    Use global variable to draw several short trajectories.
    :param model:
    :return:
    """
    global DATA_GENERATED
    # Problem: if block is only entered once per training,
    # to trajectory is never updated with better models.
    if DATA_GENERATED is None:
    
        experimental_data=True
    
        if experimental_data:
          T_total=360
          trajectories=[]
      #    for i in range(5):
      #      X, Z = model.generate_free_trajectory(data[72*i:], 72)
      #      trajectories.append(X[:])
            
          for i in range(1):
             X, Z = model.generate_free_trajectory(data[:], 360)
             trajectories.append(X[:])
          
        else:
    
          trajectories=[]
          SKIP_TRANS=200
          T_total=100000
          for i in range(100):
            X, Z = model.generate_free_trajectory(data, 1200)
            trajectories.append(X[SKIP_TRANS:])
        
        DATA_GENERATED=tc.stack(trajectories, axis=0).reshape(1, T_total, -1).squeeze(0)


    return DATA_GENERATED


def printf(x):
    if PRINT:
        print(x)
        
def split_dataset(data, dims):

  dg=dims[0]
  do=dims[1]
  dp=dims[2]

  # split input and target according to dimensionality of multimodal observations. For non-observed modalities, dimensions are set to zero
  data_g = data[:, :dg]
  data_o = data[:, dg:dg + do]
  data_p = data[:, dg + do:]

  return (data_g, data_o, data_p)
  
def smooth( data, kernel='bartlett', kernel_width=5):
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
                    res[i] = 0 #np.nan
            return res
    

class Evaluator(object):
    def __init__(self, init_data):
        model_ids, data, save_path, dims = init_data
        self.model_ids = model_ids
        self.save_path = save_path
        self.dims=dims
        
        self.data = tc.tensor(data[SKIP_TRANS:], dtype=tc.float)
       # self.data= self.data_set.test_tensor()
        
        self.name = NotImplementedError
        self.dataframe_columns = NotImplementedError

    def metric(self, model):
        return NotImplementedError

    def evaluate_metric(self):
        metric_dict = dict()
        assert self.model_ids is not None
        for model_id in self.model_ids:
            model = self.load_model(model_id, self.dims)
            metric_dict[model_id] = self.metric(model)
        self.save_dict(metric_dict)
        

    def load_model(self, model_id, dims):
    
        
        model = models.Model()
        model.init_from_model_path(model_id, EPOCH)
        model.eval()
        print(f"# params: {model.get_num_trainable()}")
        return model

    def save_dict(self, metric_dict):
        df = pd.DataFrame.from_dict(data=metric_dict, orient='index')
        df.columns = self.dataframe_columns
        utils.make_dir(self.save_path)
        df.to_csv('{}/{}.csv'.format(self.save_path, self.name), sep='\t')
        
        
class EvaluateKLx(Evaluator):
    def __init__(self, init_data):
        super(EvaluateKLx, self).__init__(init_data)
        self.name = 'klx'
        self.dataframe_columns = ('klx',)

    def metric(self, model):
        self.data = self.data.to(model.device)
        data_gen = get_generated_data(model, self.data)
        
        self.data_g,_,_=split_dataset(self.data, self.dims)
        data_gen_g,_,_=split_dataset(data_gen, self.dims)
        
        T, dx = data_gen_g.size()
        
        T_gmm=10000
        
        print("Time steps KLx:", T)
        if dx < 5:
            print("Computing KLx-BIN")
            klx_value = klx.klx_metric(data_gen_g, self.data_g, n_bins=10).cpu()
        else:
            print("Computing KLx-GMM")
            klx_value = calc_kl_from_data(data_gen_g[:T_gmm].cpu(), self.data_g[:T_gmm].cpu())

        printf('\tKLx {}'.format(klx_value.item()))
        return [np.array(klx_value.numpy())]

    @staticmethod
    def pca(x_gen, x_true):
        '''
        perform pca for to make KLx-Bin feasible for
        high dimensional data. Computes the first 5 principal
        components.
        '''
        U, S, V = tc.pca_lowrank(x_true, q=5, center=False, niter=10)
        x_pca = x_true @ V[:, :5]
        x_gen_pca = x_gen @ V[:, :5]
        return x_gen_pca, x_pca
        
        
class EvaluateLyap(Evaluator):
    def __init__(self, init_data):
        super(EvaluateLyap, self).__init__(init_data)
        self.name = 'lyap'
        dz=30
        self.dataframe_columns = tuple(['Lyap_dim_{}'.format(dim) for dim in range(dz)])

    def metric(self, model):
        self.data = self.data.to(model.device)
        warm_up=10
        initial_state=self.data[warm_up:warm_up+10]
        spectrum=lyap.lyapunov_spectrum(model,initial_state, T=3000, T_trans=200, ons=5) 
        printf('\tmax.Lyap{}'.format(spectrum[0]))
        
        return spectrum
        
        
class EvaluateKL_PCA(Evaluator):
    def __init__(self, init_data, dataset):
        super(EvaluateKL_PCA, self).__init__(init_data)
        self.name = 'kl_pca'
        self.dataframe_columns = ('kl_pca',)

    def metric(self, model):
        self.data = self.data.to(model.device)
        data_gen = get_generated_data(model, self.data)
        
        latent_trajectories=[]
        SKIP_TRANS=300
        for i in range(25):
          X, Z = model.generate_free_trajectory(self.data, 2300)
          DATA_GENERATED = Z[SKIP_TRANS:]
          latent_trajectories.append(DATA_GENERATED)
        
        dz=latent_trajectories[0].shape[-1]
        data_gen_lat=tc.stack(latent_trajectories).reshape(1,-1, dz)
        
        
        #add filepaths for computation of kl_pca, since this does not correspond to test data by default

        if dataset=="lorenz":
          X_gaussian = np.load("")
        
        data_gt=tc.tensor(X_gaussian)
        
        print("Computing KLx-PCA")
        klx_value = kl_pca.kl_pca(data_gt, data_gen_lat[0], n_bins=10).cpu()
        
        return [np.array(klx_value.numpy())]


class EvaluateMSE(Evaluator):
    def __init__(self, init_data):
        super(EvaluateMSE, self).__init__(init_data)
        self.name = 'mse'
        self.n_steps = 10
        self.T=10000
        self.dataframe_columns = tuple(['mse_{}'.format(i) for i in range(1, 1 + self.n_steps)])

    def metric(self, model):
        dg=self.dims[0]
        
        warm_up=False
        warm_up_steps=1
        warm_up_sequence=10
                
        if warm_up:
          mse_results = mse.n_steps_ahead_pred_mse_warm_up(model, self.data[:self.T], n_steps=self.n_steps, dg=dg, warm_up_steps=warm_up_steps, warm_up_sequence=warm_up_sequence)
        else:
          mse_results = mse.n_steps_ahead_pred_mse(model, self.data[:self.T], n_steps=self.n_steps, dg=dg)
        for step in [1, 5, 10]:
            printf('\tMSE-{} {}'.format(step, mse_results[step-1]))
        return mse_results
        
        
class EvaluateMSE_exp(Evaluator):
    def __init__(self, init_data):
        super(EvaluateMSE_exp, self).__init__(init_data)
        self.name = 'mse_exp'
        self.n_steps = 15
        #length of test set during training
        self.T=72
        self.dataframe_columns = tuple(['mse_exp{}'.format(i) for i in range(1, 1 + self.n_steps)])

    def metric(self, model):
        
        dg=self.dims[0]
        
        mse_results = mse.n_steps_ahead_pred_mse(model, self.data[-self.T:], n_steps=self.n_steps, dg=dg)
        for step in [1, 5, 10]:
            printf('\tMSE-{} {}'.format(step, mse_results[step-1]))
        return mse_results
        


class EvaluatePSE(Evaluator):
    def __init__(self, init_data):
        super(EvaluatePSE, self).__init__(init_data)
        self.name = 'pse'
        n_dim = self.dims[0]
        self.dataframe_columns = tuple(['mean PSE'] + ['PSE_dim_{}'.format(dim) for dim in range(n_dim)])

    def metric(self, model):
        self.data = self.data.to(model.device)
        
        T_max=self.data.shape[0]
        T_gen=360
        T=np.minimum(T_max, T_gen)
        
        SKIP_TRANS=0
        X, Z = model.generate_free_trajectory(self.data,T+SKIP_TRANS)
        data_gen = X[SKIP_TRANS:]
        
       # data_gen = get_generated_data(model, self.data)
        
        self.data_g,_,_=split_dataset(self.data, self.dims)
        data_gen_g,_,_=split_dataset(data_gen, self.dims)

        x_gen = data_gen_g.cpu().unsqueeze(0).numpy()
        x_true = self.data_g.cpu().unsqueeze(0).numpy()[:,:T]
        
        print(x_gen.shape)
        print(x_true.shape)
        
        pse_per_dim = power_spectrum_error_per_dim(x_gen=x_gen, x_true=x_true)
        pse = power_spectrum_error(x_gen=x_gen, x_true=x_true)

        printf('\tPSE {}'.format(pse))
        printf('\tPSE per dim {}'.format(pse_per_dim))
       
        return [pse] + pse_per_dim
        
class EvaluateOPE(Evaluator):
    def __init__(self, init_data):
        super(EvaluateOPE, self).__init__(init_data)
        self.name = 'ope'
        self.n_steps = 10
        self.T=10000
        self.dataframe_columns = tuple(['ope_{}'.format(i) for i in range(1, 1 + self.n_steps)])

    def metric(self, model):
        do=self.dims[1]
        dg=self.dims[0]
        ope_results = ope.n_steps_ahead_pred_ope(model, self.data[:self.T], n_steps=self.n_steps, dg=dg, do=do)
        for step in [1, 5, 10]:
            printf('\tOPE-{} {}'.format(step, ope_results[step-1]))
        return ope_results
        
        
class EvaluatePPE(Evaluator):
    def __init__(self, init_data):
        super(EvaluatePPE, self).__init__(init_data)
        self.name = 'ppe'
        self.n_steps = 10
        self.T=10000
        self.dataframe_columns = tuple(['ppe_{}'.format(i) for i in range(1, 1 + self.n_steps)])

    def metric(self, model):
        do=self.dims[1]
        dg=self.dims[0]
        dp=self.dims[2]
  
        
        ppe_results = ppe.n_steps_ahead_pred_ppe(model, self.data[:self.T], n_steps=self.n_steps, dg=dg, do=do,  dp=dp)
        for step in [1, 5, 10]:
            printf('\tPPE-{} {}'.format(step, ppe_results[step-1]))
        return ppe_results
        
        
class EvaluateSCE(Evaluator):
    def __init__(self, init_data):
        super(EvaluateSCE, self).__init__(init_data)
        self.name = 'sce'
        n_dim = self.dims[1]
        self.dataframe_columns = tuple(['mean SCE'] + ['SCE_dim_{}'.format(dim) for dim in range(n_dim)])

    def metric(self, model):
        self.data = self.data.to(model.device)
        data_gen = get_generated_data(model, self.data)
        ####
        _,self.data_o,_=split_dataset(self.data, self.dims)
        _,data_gen_o,_=split_dataset(data_gen, self.dims)
        
        x_gen = data_gen_o.cpu().numpy()
        x_true = self.data_o.cpu().numpy()
        
        sce_per_dim, sce_err = sce.spearman_corr(x_gen=x_gen, x_true=x_true)

        printf('\tSCE {}'.format(sce_err))
        printf('\tSCE per dim {}'.format(sce_per_dim))
        
        return [sce_err] + sce_per_dim
        
        
class EvaluateCatClassifier(Evaluator):
    def __init__(self, init_data):
        super(EvaluateCatClassifier, self).__init__(init_data)
        self.name = 'lcc'
        n_dim = self.dims[1]
        self.dataframe_columns = ('lc_score',)

    def metric(self, model):
        self.data = self.data.to(model.device)
        X, Z = model.generate_free_trajectory(self.data[-72:], 72)
        
        conf_matrix, score=cat_classifier.classifier_testset(model, self.data,Z)

        print(conf_matrix)
        
        return score  

        
        
class EvaluateACE(Evaluator):
    def __init__(self, init_data):
        super(EvaluateACE, self).__init__(init_data)
        self.name = 'ace'
        n_dim = self.dims[1]
        self.dataframe_columns = ('ACE',)
       # self.dataframe_columns = tuple(['mean ACE'] + ['ACE_dim_{}'.format(dim) for dim in range(n_dim)])

    def metric(self, model):
        self.data = self.data.to(model.device)
        data_gen = get_generated_data(model, self.data)
        ####
        _,self.data_o,_=split_dataset(self.data, self.dims)
        _,data_gen_o,_=split_dataset(data_gen, self.dims)
        
        x_gen = data_gen_o.cpu().numpy()
        x_true = self.data_o.cpu().numpy()
        
        T=50000
        
        ace_mean, ace_per_dim = ace.spearman_autocorrelation_error(x_gen[:T], x_true[:T], 200)

        printf('\tACE {}'.format(ace_mean))
        
        return [ace_mean]
        
        
class EvaluateACEP(Evaluator):
    def __init__(self, init_data):
        super(EvaluateACEP, self).__init__(init_data)
        self.name = 'acep'
        n_dim = self.dims[2]
        self.dataframe_columns = ('ACEP',)
       # self.dataframe_columns = tuple(['mean ACE'] + ['ACE_dim_{}'.format(dim) for dim in range(n_dim)])

    def metric(self, model):
        self.data = self.data.to(model.device)
        data_gen = get_generated_data(model, self.data)
        ####
        _,_,self.data_p=split_dataset(self.data, self.dims)
        _,_,data_gen_p=split_dataset(data_gen, self.dims)
        
        x_gen = data_gen_p.cpu().numpy()
        x_true = self.data_p.cpu().numpy()
        
        T=50000

        acep_mean = ace.autocorrelation_error(x_gen[:T], x_true[:T], 200)

        printf('\tACEP {}'.format(acep_mean))
        
        return [acep_mean]
        
        
        
class EvaluatePFR(Evaluator):
    def __init__(self, init_data):
        super(EvaluatePFR, self).__init__(init_data)
        self.name = 'pfr'
        n_dim = self.dims[2]
        self.dataframe_columns = tuple(['pfr_{}'.format(dim) for dim in range(n_dim)]+['pzr_{}'.format(dim) for dim in range(n_dim)])

    def metric(self, model):
        self.data = self.data.to(model.device)
        data_gen = get_generated_data(model, self.data)
        
        _,_,self.data_p=split_dataset(self.data, self.dims)
        _,_,data_gen_p=split_dataset(data_gen, self.dims)

        x_gen = data_gen_p.cpu().numpy()
        x_true = self.data_p.cpu().numpy()
        
        difference_rate, difference_zeros  = pfr.poisson_firing_rates(x_gen=x_gen, x_true=x_true)
        
        printf('\tPoisson Rate {}'.format(difference_rate))
        printf('\tZero Rate {}'.format(difference_zeros))

        return difference_rate+difference_zeros


class SaveArgs(Evaluator):
    def __init__(self, init_data):
        super(SaveArgs, self).__init__(init_data)
        self.name = 'args'
        self.dataframe_columns = ('dim_g','dim_o','dim_p', 'dim_z','dim_f', 'n_bases')
        

    def metric(self, model):
        args = model.args
        return [self.dims[0],self.dims[1],self.dims[2], args['dim_z'],args['dim_forcing'], args['n_bases']]


def gather_eval_results(eval_dir='save', save_path='save_eval', gaussian_metrics=None, ordinal_metrics=None, poisson_metrics=None):
    """Pre-calculated metrics in individual model directories are gathered in one csv file"""

    metrics=[]
    for metric in gaussian_metrics:
      metrics.append(metric)
    for metric in ordinal_metrics:
      metrics.append(metric)
    for metric in poisson_metrics:
      metrics.append(metric)
              
    metrics.append('args')
    model_ids = get_model_ids(eval_dir)
    for metric in metrics:
        paths = [os.path.join(model_id, '{}.csv'.format(metric)) for model_id in model_ids]
        data_frames = []
        for path in paths:
            try:
                data_frames.append(pd.read_csv(path, sep='\t', index_col=0))
            except:
                print('Warning: Missing model at path: {}'.format(path))
        data_gathered = pd.concat(data_frames)
        utils.make_dir(save_path)
        
        #directly output the median values for selected metrics after the evaluation is complete

        if "klx" in metric:
        
          klx_np=np.asarray(data_gathered["klx"][:])
          print("KLx:", np.median(klx_np[:]), median_abs_deviation(klx_np[:]))
          
        if "kl_pca" in metric:
        
          klx_np=np.asarray(data_gathered["kl_pca"][:])
          print("KLx:", np.median(klx_np[:]), median_abs_deviation(klx_np[:]))
          
        if "pse" in metric:
        
          pse_np=np.asarray(data_gathered["mean PSE"])
          print("PSC:", np.median(pse_np[:]), median_abs_deviation(pse_np[:]))
        
        if "mse_exp" in metric:
        
          mse_np=np.asarray(data_gathered["mse_exp10"])
          print("MSE:", np.mean(mse_np[:]), median_abs_deviation(mse_np[:]))
        
        
        metric_save_path = '{}/{}.csv'.format(save_path, metric)
        data_gathered.to_csv(metric_save_path, sep='\t')


def choose_evaluator_from_metric(metric_name, init_data, dataset):
    if metric_name == 'mse':
        EvaluateMetric = EvaluateMSE(init_data)
    elif metric_name == 'mse_exp':
        EvaluateMetric = EvaluateMSE_exp(init_data)
    elif metric_name == 'lcc':    
        EvaluateMetric = EvaluateCatClassifier(init_data)
    elif metric_name == 'klx':
        EvaluateMetric = EvaluateKLx(init_data)
    elif metric_name == 'lyap':
        EvaluateMetric = EvaluateLyap(init_data)
    elif metric_name == 'kl_pca':
        EvaluateMetric = EvaluateKL_PCA(init_data, dataset)
    elif metric_name == 'pse':
        EvaluateMetric = EvaluatePSE(init_data)
    elif metric_name == 'ope':
        EvaluateMetric = EvaluateOPE(init_data)
    elif metric_name == 'sce':
        EvaluateMetric = EvaluateSCE(init_data)
    elif metric_name == 'ace':
        EvaluateMetric = EvaluateACE(init_data)
    elif metric_name == 'acep':
        EvaluateMetric = EvaluateACEP(init_data)
    elif metric_name == 'ppe':
        EvaluateMetric = EvaluatePPE(init_data)
    elif metric_name == 'pfr':
        EvaluateMetric = EvaluatePFR(init_data)
    else:
        raise NotImplementedError
    return EvaluateMetric


def eval_model_on_data_with_metric(model, data, metric, dataset):
    init_data = (None, data, None)
    EvaluateMetric = choose_evaluator_from_metric(metric, init_data, dataset)
    #EvaluateMetric.data = data
    metric_value = EvaluateMetric.metric(model)
    return metric_value


def is_model_id(path):
    """Check if path ends with a three digit, e.g. save/test/001 """
    run_nr = path.split('/')[-1]
    three_digit_numbers = {str(digit).zfill(3) for digit in set(range(1000))}
    return run_nr in three_digit_numbers


def get_model_ids(path):
    """
    Get model ids from a directory by recursively searching all subdirectories for files ending with a number
    """
    assert os.path.exists(path)
    if is_model_id(path):
        model_ids = [path]
    else:
        all_subfolders = glob(os.path.join(path, '**/*'), recursive=True)
        model_ids = [file for file in all_subfolders if is_model_id(file)]
    assert model_ids, 'could not load from path: {}'.format(path)
    return model_ids


def eval_model(args):
    save_path = args.load_model_path
    evaluate_model_path(args, model_path=save_path, metrics=args.metrics)


def evaluate_model_path(model_path=None, gaussian_data_path=None, ordinal_data_path=None,poisson_data_path=None, gaussian_metrics=None, ordinal_metrics=None, poisson_metrics=None, use_gaussian=None,use_ordinal=None,use_poisson=None, preprocessing=False, dataset=None):
    """Evaluate a single model in directory model_path w.r.t. metrics and save results in csv file in model_path"""

    dims=[]
    data=[]
    if use_gaussian: 
      data_g = utils.read_data(gaussian_data_path)
      if len(data_g.shape)==1:
        data_g=data_g.reshape(-1,1)
      dg=data_g.shape[-1]
      data.append(data_g)
      print(data_g.shape)
    else:
      dg=0
    dims.append(dg)
    if use_ordinal:
      data_o = utils.read_data(ordinal_data_path)
      do=data_o.shape[-1]
      data.append(data_o)
      print(data_o.shape)
    else:
      do=0
    dims.append(do)
    if use_poisson:
      data_p = utils.read_data(poisson_data_path)
      dp=data_p.shape[-1]
      data.append(data_p)
    else:
      dp=0
    dims.append(dp)
    
    data=np.concatenate(data, axis=-1)
    
    if preprocessing:
          kernel_width=20
         #box cox
          for i in range(do+dp):
            data[:,dg+i],_=stats.boxcox(np.abs(data[:,dg+i]+0.0001))
          #z score
          data[:,dg:]=(data[:,dg:]-np.mean(data[:,dg:], axis=0))/np.std(data[:,dg:], axis=0)

    
    model_ids = [model_path]
    global DATA_GENERATED
    DATA_GENERATED = None
    
    init_data = (model_ids, data, model_path, dims)
    Save = SaveArgs(init_data)
    Save.evaluate_metric()
    

    for metric_name in gaussian_metrics:
          EvaluateMetric = choose_evaluator_from_metric(metric_name=metric_name, init_data=(model_ids, data, model_path,dims), dataset=dataset)
          EvaluateMetric.evaluate_metric()
    for metric_name in ordinal_metrics:
        EvaluateMetric = choose_evaluator_from_metric(metric_name=metric_name, init_data=(model_ids, data, model_path,dims),  dataset=dataset)
        EvaluateMetric.evaluate_metric()
    for metric_name in poisson_metrics:
        EvaluateMetric = choose_evaluator_from_metric(metric_name=metric_name, init_data=(model_ids, data, model_path,dims),  dataset=dataset)
        EvaluateMetric.evaluate_metric()


def evaluate_all_models(eval_dir, gaussian_data_path, ordinal_data_path,poisson_data_path, gaussian_metrics, ordinal_metrics, poisson_metrics, use_gaussian,use_ordinal,use_poisson, preprocessing, dataset):
    model_paths = get_model_ids(eval_dir)
    n_models = len(model_paths)
    print('Evaluating {} models'.format(n_models))
    for i, model_path in enumerate(model_paths):
        print('{} of {}'.format(i+1, n_models))
        # try:
        evaluate_model_path(model_path=model_path, gaussian_data_path=gaussian_data_path, ordinal_data_path=ordinal_data_path,poisson_data_path=poisson_data_path, gaussian_metrics=gaussian_metrics, ordinal_metrics=ordinal_metrics, poisson_metrics=poisson_metrics, use_gaussian=use_gaussian,use_ordinal=use_ordinal,use_poisson=use_poisson, preprocessing=preprocessing, dataset=dataset)
        # except:
        #     print('Error in model evaluation {}'.format(model_path))
    return


def print_metric_stats(save_path: str, metrics: List):
    # MSE
    path = os.path.join(save_path, 'mse.csv')
    df = pd.read_csv(path, delimiter='\t')
    mse5 = (df.mean(0, numeric_only=True)['5'], df.std(numeric_only=True)['5'])
    mse10 = (df.mean(0, numeric_only=True)['10'], df.std(numeric_only=True)['10'])
    mse20 = (df.mean(0, numeric_only=True)['20'], df.std(numeric_only=True)['20'])

    #PSE
    path = os.path.join(save_path, 'pse.csv')
    df = pd.read_csv(path, delimiter='\t')
    pse = (df.mean(0, numeric_only=True)['mean'], df.std(numeric_only=True)['mean'])

    # Dstsp
    path = os.path.join(save_path, 'klx.csv')
    df = pd.read_csv(path, delimiter='\t')
    df_sub = df['klx']
    df_sub = df_sub[df_sub > 0]
    klx = (df_sub.mean(0), df_sub.std(0))

    new_df = pd.DataFrame({
        '5-MSE': mse5,
        '10-MSE': mse10,
        '20-MSE': mse20,
        'PSC': pse,
        'KLX': klx
    })
    new_df.to_csv(os.path.join(save_path, 'stats1.csv'), sep='\t')



if __name__ == '__main__':

    #determine number of workers, depending on your resources
    tc.set_num_threads(2)
    
    
    eval_dirs = [
    "results/Example_settings/g01o01p01noise0.1n10dz30df30/"
    ]

    #determine whether preprocessing was used for evaluation, calls Box-Cox and z-scoring
    preprocessing=False
    dataset="lorenz"

    for eval_dir in eval_dirs:
    
        ##determine which modalities were used to train nmodel
        use_gaussian=1
        use_ordinal=1
        use_poisson=0
        
        #define which modality-specific metrics are computed
        gaussian_metrics = ["klx"]#'"klx", "pse", "mse", "lyap", "kl_pca"
        ordinal_metrics = [] #'ope', 'ace', 'sce'
        poisson_metrics = [] #'acep'
   
        #add gaussian filepaths for computation of kl_pca directly to the EvaluateKL_PCA class if trained without Gaussian data,
        # since the "ground truth" data does not correspond to test data by default
      
        if dataset=="lorenz":
          gaussian_data_path = "datasets/lorenz/lorenz_multimodal_gaussian_dt0.05_test.npy"
          ordinal_data_path = "datasets/lorenz//lorenz_multimodal_ordinal_dt0.05_test.npy"
          poisson_data_path = "datasets/lorenz//lorenz_multimodal_poisson_dt0.05_test.npy"
          
        
        evaluate_all_models(eval_dir=eval_dir, gaussian_data_path=gaussian_data_path, ordinal_data_path=ordinal_data_path,poisson_data_path=poisson_data_path, gaussian_metrics=gaussian_metrics, ordinal_metrics=ordinal_metrics,poisson_metrics=poisson_metrics, use_gaussian=use_gaussian, use_ordinal=use_ordinal, use_poisson=use_poisson, preprocessing=preprocessing, dataset=dataset)        
        gather_eval_results(eval_dir=eval_dir, save_path=eval_dir, gaussian_metrics=gaussian_metrics, ordinal_metrics=ordinal_metrics, poisson_metrics=poisson_metrics)



