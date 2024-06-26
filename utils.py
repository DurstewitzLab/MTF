import os
import glob
import pickle

import numpy as np
import pandas as pd
import torch as tc
from tensorboardX import SummaryWriter

from bptt import dataset
from argparse import Namespace

from bptt.dataset_multimodal import AheadPredictionDataset

import subprocess


def compute_plrnn_params(dz: int, db: int, dx: int, learnz0: bool,
                         fix_obs_model: bool) -> int:
    '''
    Compute number of Parameters of a specific PLRNN configuartion.
    '''
    awh = dz*dz + dz
    basis = dz*db+db
    z0 = 0
    if learnz0:
        z0 = dx*(dz-dx)
    obs = 0
    if not fix_obs_model:
        obs = dz*dx

    return awh + basis + z0 + obs


def prepare_device(args: Namespace) -> tc.device:
    '''
    Prepare and return a torch.device instance.
    '''
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # check cuda availability
    if args.use_gpu and tc.cuda.is_available():
        id_ = args.device_id
        device = tc.device('cuda', id_)
        name = tc.cuda.get_device_name(id_)
        print(f"Using device {name} for training ({device}).")
    else:
        device = tc.device('cpu')
        num_threads = tc.get_num_threads()
        print(f"Training on {device} with {num_threads} threads.")
    return device


def get_runs(trial_path):
    run_nrs = list(os.walk(trial_path))[0][1]
    run_nrs = [r[0:3] for r in run_nrs]
    return run_nrs


def infer_latest_epoch(run_path: str) -> int:
    '''
    Loop over model_*.pt checkpoints and find the 
    model checkpoint of the latest epoch.
    '''
    # list checkpoints
    chkpts = glob.glob(os.path.join(run_path, '*.pt'))
    assert chkpts, f"No model found in {run_path}"

    # find latest epoch
    latest = 0
    for chkpt in chkpts:
        epoch = int(chkpt.split('_')[-1].strip('.pt'))
        if latest < epoch:
            latest = epoch
    return latest


def create_run_path(trial_path):
    """increase by one each run, if none exists start at '000' """
    run_nrs = get_runs(trial_path)
    if not run_nrs:
        run_nrs = ['000']
    run = str(int(max(run_nrs)) + 1).zfill(3)
    run_dir = os.path.join(trial_path, run)
    return run_dir


def create_savepath(args):
    save_path = os.path.join('results', args.experiment)
    make_dir(save_path)
    trial_path = os.path.join(save_path, args.name)
    make_dir(trial_path)
    if args.run is None:
        run_path = create_run_path(trial_path)
    else:
        run_path = os.path.join(trial_path, str(args.run).zfill(3))
    make_dir(run_path)
    return run_path


def init_writer(args):
    trial_path = create_savepath(args)
    writer = None
    if args.use_tb:
        if not args.no_printing:
            print('initialize tensorboard writer at {}'.format(trial_path))
        writer = SummaryWriter(trial_path)
    return writer, trial_path


def read_data(data_path):
    if data_path is None:
        data = None
    else:
        assert os.path.exists(data_path)
        data = np.load(data_path, allow_pickle=True)
    return data
    
    
def cut_dataset_to_be_divisible_by_sequence_length(data, args):

        number_of_points_to_cut = data.shape[0] % args.seq_len
        data = data[number_of_points_to_cut:]
        assert data.shape[0] % args.seq_len == 0

        args.number_of_sequences = int(data.shape[0] / args.seq_len)
       # data=data.reshape(args.number_of_sequences, args.seq_len, data.shape[-1])

        return data


def load_dataset(args):
    data = read_data(args.data_path)
    data_shape = data.shape
    args.dim_x = data.shape[-1]
    
    if len(data_shape) == 3:
      data=data.reshape(1,-1, args.dim_x).squeeze(0)
    
  #  assert len(data_shape) == 2, ("Data is expected to be of shape (T x N). "
  #      f"Please check your data format! Received shape: {data_shape}.")
    data_set = dataset.GeneralDataset(data, args.seq_len, 
                                      args.batch_size, args.batches_per_epoch)
    
    return args, data_set


def load_dataset_multimodal(args, experimental_dataset):

    # Dataset parameters
    
    if experimental_dataset:
      test_len=args.seq_len
      valid_test_len = args.seq_len
    else:
      test_len=1000
      valid_test_len = 1000
      

    ignore_leading_nans = True
    # Set return_target True to obtain targets (which are the sequences moved one step forward)
    return_targets = True
    return_index = True
    # If you prefer to obtain not a list of tensors (one for each timeseries), but a single tensor, set
    return_tensor = True
    valid_indices=True

    # Create an AheadPredictionDataset instance
    ds = AheadPredictionDataset(args.seq_len, args.batch_size, args.batches_per_epoch,valid_indices, ignore_leading_nans,
                                return_targets,return_index,
                                return_tensor, valid_test_len=valid_test_len)
    datasets=[]
    if args.use_gaussian == 1:

        gaussian_data = read_data(args.gaussian_data_path)
        
        if len(gaussian_data.shape)==1:
          gaussian_data=gaussian_data.reshape(-1,1)

        s = np.random.normal(0, args.gaussian_observation_noise, (gaussian_data.shape))
        gaussian_data=gaussian_data+s
        
        args.dim_g = gaussian_data.shape[1]
        datasets.append(gaussian_data)

        ds.add_timeseries(gaussian_data, 'gaussian')
      #  ds.add_timeseries(gaussian_data, 'gaussian', pp_steps_continuous_train, pp_steps_continuous_test, pp_args)
    else:
        args.dim_g = 0

    if args.use_ordinal == 1:
        ordinal_data = read_data(args.ordinal_data_path)
        
        if len(ordinal_data.shape)==1:
          ordinal_data=ordinal_data.reshape(-1,1)
        
        if args.ordinal_dimension==0:
          args.dim_o = ordinal_data.shape[1]
        else:
          args.dim_o=args.ordinal_dimension
          
        print(args.dim_o)
        ordinal_data=ordinal_data[:,:args.dim_o]
        
        datasets.append(ordinal_data)
        ds.add_timeseries(ordinal_data, 'ordinal')

    else:
        args.dim_o = 0

    if args.use_poisson == 1:
        poisson_data = read_data(args.poisson_data_path)
        datasets.append(poisson_data)
        
        ds.add_timeseries(poisson_data, 'poisson')
        args.dim_p = poisson_data.shape[1]

    else:
        args.dim_p = 0
        
    args.number_of_sequences = int(datasets[0].shape[0] / args.seq_len)  
    
    inputs = read_data(args.inputs_path)
    if inputs is not None:
        ds.add_timeseries(inputs, name='inputs')
        args.dim_s = ds.timeseries['inputs'].dim
    else:
        args.dim_s = 0
        
    return args, ds
    
    
def load_dataset_preprocessing(args, experimental_dataset):

    if experimental_dataset:
      test_len=args.seq_len
      valid_test_len = args.seq_len
    else:
      test_len=1000
      valid_test_len = 1000
    # Dataset parameters
    ignore_leading_nans = True
    # Set return_target True to obtain targets (which are the sequences moved one step forward)
    return_targets = True
    return_index = True
    # If you prefer to obtain not a list of tensors (one for each timeseries), but a single tensor, set
    return_tensor = True
    valid_indices=True

    # Create an AheadPredictionDataset instance
    ds = AheadPredictionDataset(args.seq_len, args.batch_size, args.batches_per_epoch,valid_indices, ignore_leading_nans,
                                return_targets,return_index,
                                return_tensor, valid_test_len=valid_test_len)
                                
                                
    pp_steps_continuous_train = ['boxcox', 'zscore', 'smooth']
    pp_steps_continuous_test = ['boxcox', 'zscore']
    pp_args = {'kernel': 'bartlett', 'kernel_width': args.kernel_width_preprocessing}                            
                                

    if args.use_gaussian == 1:
      gaussian_data = read_data(args.gaussian_data_path)
      args.dim_g = gaussian_data.shape[1]
      ds.add_timeseries(gaussian_data, 'gaussian')
    
    else:
      args.dim_g = 0
    
    
    if args.use_ordinal == 1:
      ordinal_data = read_data(args.ordinal_data_path)
      args.dim_o = ordinal_data.shape[1]
      ds.add_timeseries(ordinal_data, 'ordinal', pp_steps_continuous_train, pp_steps_continuous_test, pp_args)
    else:
      args.dim_o = 0
    
    if args.use_poisson == 1:
      poisson_data = read_data(args.poisson_data_path)
      args.dim_p= poisson_data.shape[1]
      ds.add_timeseries(poisson_data, 'poisson', pp_steps_continuous_train, pp_steps_continuous_test, pp_args)
     
    else:
      args.dim_p = 0    
        
    return args, ds

def load_dataset_multiple_shooting(args,experimental_dataset):

    if experimental_dataset:
      test_len=args.seq_len
      valid_test_len = args.seq_len
      print("experimental dataset")
    # Dataset parameters

    else:
      test_len=1000
      valid_test_len = 1000
    
    ignore_leading_nans = True
    # Set return_target True to obtain targets (which are the sequences moved one step forward)
    return_targets = True
    return_index = True
    # If you prefer to obtain not a list of tensors (one for each timeseries), but a single tensor, set
    return_tensor = True
    # indices are determined through the sequence length (important for batching)
    valid_indices=False

    # Create an AheadPredictionDataset instance
    ds = AheadPredictionDataset(args.seq_len, args.batch_size, args.batches_per_epoch, valid_indices,ignore_leading_nans,
                                return_targets,return_index,
                                return_tensor, valid_test_len=valid_test_len)

    data=[]
    if args.use_gaussian == 1:
        gaussian_data = read_data(args.gaussian_data_path)
        gaussian_data=cut_dataset_to_be_divisible_by_sequence_length(gaussian_data, args)
        
        print(gaussian_data.shape)
        if len(gaussian_data.shape)==1:
          gaussian_data=gaussian_data.reshape(-1,1)
        args.dim_g = gaussian_data.shape[1]
        data.append(gaussian_data)
        
    
        pp_steps_continuous_train = ['boxcox', 'zscore', 'smooth']
        pp_steps_continuous_test = ['boxcox', 'zscore']
        pp_args = {'kernel': 'bartlett', 'kernel_width': 5}

        ds.add_timeseries(gaussian_data, 'gaussian')
      #  ds.add_timeseries(gaussian_data, 'gaussian', pp_steps_continuous_train, pp_steps_continuous_test, pp_args)
    else:
        args.dim_g = 0

    if args.use_ordinal == 1:
        ordinal_data = read_data(args.ordinal_data_path)
        ordinal_data=cut_dataset_to_be_divisible_by_sequence_length(ordinal_data, args)
        args.dim_o = ordinal_data.shape[1]
        ds.add_timeseries(ordinal_data, 'ordinal')
        data.append(ordinal_data)
    else:
        args.dim_o = 0

    if args.use_poisson == 1:
        poisson_data = read_data(args.poisson_data_path)
        poisson_data=cut_dataset_to_be_divisible_by_sequence_length(poisson_data, args)
        ds.add_timeseries(poisson_data, 'poisson')
        args.dim_p = poisson_data.shape[1]
        data.append(poisson_data)
    else:
        args.dim_p = 0
          
     # self.data_point_for_last_shooting_node = tc.tensor(data[-1], dtype=tc.float).unsqueeze(dim=0)


    return args, ds


    

def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass


def make_attribute_dict(args):
    """"make args a dict whose elements can be accessed like: dict.element"""
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self
    args = AttrDict(args)
    return args


def save_args(args, save_path, writer):
    """ add hyperparameters to txt file """
    d = args.__dict__
    txt = ''
    for k in d.keys():
        txt += ('{} {}\n'.format(k, d[k]))
    if writer is not None:
        writer.add_text(tag="""hypers""", text_string=str(txt), global_step=None)
    filename = '{}/hypers.txt'.format(save_path)
    with open(filename, 'w') as f:
        f.write(txt)
    filename = '{}/hypers.pkl'.format(save_path)
    with open(filename, 'wb') as f:
        pickle.dump(d, f)
    if not args.no_printing:
        print(txt)


def check_args(args):
    def assert_positive_or_none(arg):
        if arg is not None:
            assert arg > 0

  #  assert args.data_path[-4:] == '.npy'
  #  assert args.data_path is not None
    assert_positive_or_none(args.clip_range)
    assert args.dim_z > 0
    if args.n_bases is not None:
        assert args.n_bases >= 0
    assert args.learning_rate > 0
    assert args.n_epochs > 0
    # assert args.alpha_reg >= 0
    # assert args.n_states_reg >= 0
    # list entries are tuples of floats
    # first entry is between 0 and 1 and sum of all is not higher than one
    # but higher than 0
    # second entry is > 0


def save_files(save_path):
    curdir = os.path.abspath('.')
    from distutils.dir_util import copy_tree
    save_path = os.path.join(save_path, 'python_files')
    copy_tree(curdir, save_path)


def read_csv_to_df(path):
    with open(path, 'rb') as f:
        df = pd.read_csv(f, sep='\t')
    return df


def save_to_pickle(variable, file_name):
    filename = '{}.pkl'.format(file_name)
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)


def load_from_pickle(file_name):
    filename = '{}.pkl'.format(file_name)
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    return variable


def get_current_gpu_utilization():
    """
    From: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
    Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are GPU utilization in %.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8'
    )

    # Convert lines into a dictionary
    gpu_util = [int(x) for x in result.strip().split('\n')]
    return dict(zip(range(len(gpu_util)), gpu_util))
    

