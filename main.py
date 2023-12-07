import argparse
import torch as tc
import utils
from bptt import bptt_algorithm
from bptt import svae_algorithm
from bptt import ms_algorithm
from bptt.PLRNN_model import PLRNN

tc.set_num_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(description="TF RNN Training")
    parser.add_argument('--experiment', type=str, default="example")
    parser.add_argument('--name', type=str, default='Lorenz63')
    parser.add_argument('--run', type=int, default=None)

    # gpu
    parser.add_argument('--use_gpu',
        type=int,
        choices=[0, 1],
        help="If set to 1 (True), use GPU for training.",
        default=0
    )
    # cuda:0, cuda:1 etc.
    parser.add_argument('--device_id',
        type=int,
        help="Set the GPU device id (as determined by ordering in nvidia-smi) when use_gpu==1.",
        default=0
    )

    # general settings
    parser.add_argument('--no_printing', type=bool, default=True)
    parser.add_argument('--use_tb', type=bool, default=True)
    

   # parser.add_argument('--experimental_dataset', type=int, default=0)

    # define which dataset to train on
    dynsys="lorenz"
    
    if dynsys=="lorenz":
    
      parser.add_argument('--gaussian_data_path', type=str, default="datasets/lorenz/lorenz_multimodal_gaussian_dt0.05_train.npy")
      parser.add_argument('--ordinal_data_path', type=str, default="datasets/lorenz//lorenz_multimodal_ordinal_dt0.05_train.npy")
      parser.add_argument('--poisson_data_path', type=str, default="datasets/lorenz/lorenz_multimodal_poisson_dt0.05_train.npy")
      
      #provide external cues to the model during training
      parser.add_argument('--inputs_path', type=str, default=None)
      #define whether this as an experimental dataset, where definition of test set is different
      parser.add_argument('--experimental_dataset', type=int, default=0)
      
   
    #gaussian
    parser.add_argument('--use_gaussian', type=int, default=1)
    parser.add_argument('--gaussian_observation_noise', type=float, default=0.0)
    
    #ordinal                    
    parser.add_argument('--use_ordinal', type=int, default=1)
    parser.add_argument('--scale_ordinal_data', type=int, default=7)
    parser.add_argument('--ordinal_scaling', type=float, default=1.0)
    #tune how many ordinal dimensions to use for binary roessler study
    parser.add_argument('--ordinal_dimension', type=int, default=0)

    #poisson
    parser.add_argument('--use_poisson', type=int, default=1)
    parser.add_argument('--poisson_scaling', type=float, default=1.0)

    #preprocessing for non-Gaussian modalities
    parser.add_argument('--preprocessing', type=int, default=0)
    parser.add_argument('--kernel_width_preprocessing', type=int, default=10)
    parser.add_argument('--T_max', type=int, default=10000)
    

    #autoencoder
    parser.add_argument('--enc_model', '-em', type=str,choices=['CNN','MLP','Linear', 'Identity', "POG"], default='CNN')
    parser.add_argument('--dec_model_gaussian', '-dmg', type=str, choices=['Identity', 'MLP', 'Linear', 'categorical'], default='Linear')
    parser.add_argument('--dec_model_ordinal', '-dmo', type=str, choices=['cumulative_link', 'linear','Identity', 'categorical'], default='cumulative_link')
    parser.add_argument('--dec_model_poisson', '-dmz', type=str, choices=['poisson', 'zip','negative_binomial','cumulative_link','categorical'], default='poisson')
    
    
    # trade off loss terms
    parser.add_argument('--alpha_latent', '-alat', type=float, default=1.0)
    parser.add_argument('--alpha_rec', '-arec', type=float, default=1.0)
    parser.add_argument('--alpha_obs', '-aobs', type=float, default=1.0)
    parser.add_argument('--estimate_noise_covariances', '-nc', type=int, default=0)
    #use annealing scheme that fades out encoder loss
    parser.add_argument('--rec_annealing', '-reca', type=int, default=0)
    
    # SVAE training
    parser.add_argument('--train_svae', '-svae', type=int, default=0)
    parser.add_argument('--sample_rec', type=int, default=1)
    
    # Multiple Shooting Training
    parser.add_argument('--train_multiple_shooting', '-ms', type=int, default=0)
    parser.add_argument('--number_of_sequences', '-ns', type=int, default=500)
    parser.add_argument('--reg_z0', '-z0reg', type=float, default=1.)

    # resume from a model checkpoint
    parser.add_argument('--load_model_path', type=str, default=None)
    # epoch is inferred if None
    parser.add_argument('--resume_epoch', type=int, default=None)

    # model
    parser.add_argument('--fix_obs_model', '-fo',
        type=int,
        help="Fix B to be of shape used for Id. TF.",
        default=1
    )
    parser.add_argument('--dim_z',
        type=int,
        help="Dimension M of latent state vector of the model", 
        default=15
    )
    parser.add_argument('--n_bases', '-nb',
        type=int,
        help="Number of bases to use in dendr-PLRNN and clipped-PLRNN latent models.",
        default=5
    )

    parser.add_argument('--dim_forcing',
        type=int,
        help="Dimension K of encoder model/teacher forced states", 
        default=5
    )
    parser.add_argument('--clip_range', '-clip',
        type=float,
        help="Clips the latent state vector to the support of [-clip_range, clip_range].",
        default=10
    )
    parser.add_argument('--model', '-m',
        type=str,
        default='PLRNN'
    )
    # specifiy which latent model to choose (only affects model PLRNN)
    parser.add_argument('--latent_model', '-ml',
        type=str,
        help="Latent steps to use, i.e. vanilla PLRNN, dendr-PLRNN, clipped-PLRNN.",
        choices=PLRNN.LATENT_MODELS,
        default='clipped-PLRNN'
    )
    #mean-centering, as described in Brenner, Hess et al. (2022)
    parser.add_argument('--layer_norm', '-ln',
        type=int,
        help="Relaxed LayerNorm for Id. TF to work." \
             " Re-centers latent state vector at each time step.",
        default=1
    )
    parser.add_argument('--learn_z0', '-z0',
        type=int,
        help="If set to 1 (True), jointly learn a mapping from data" \
             "to initial cond. for remaining latent states.",
        default=0
    )

    # Teacher Forcing Hyperparamameters
    parser.add_argument('--use_inv_tf', '-itf',
        type=int,
        help="'Invert' the B matrix to force latent states. (Inv. TF).",
        default=0
    )
    parser.add_argument('--estimate_forcing', '-ef',
        type=int,
        help="If set to 1 (True), estimate a forcing interval" \
             "using Autocorrelation & Mutual Information.",
        default=0
    )
    
    # Sparse teacher forcing interval tau
    parser.add_argument('--n_interleave', '-ni',
        type=int,
        help="Teacher forcing interval (tau).",
        default=15
    )
    parser.add_argument('--batch_size', '-bs',
        type=int,
        help="Sequences are gathered as batches of this size (computed in parallel).",
        default=16
    )
    parser.add_argument('--batches_per_epoch', '-bpi',
        type=int,
        help="Amount of sampled batches that correspond to 1 epoch of training.",
        default=25
    )
    parser.add_argument('--seq_len', '-sl',
        type=int,
        help="Sequence length sampled from the total pool of the data.",
        default=200
    )
    parser.add_argument('--save_step', '-ss',
        type=int,
        help="Interval of computing and saving metrics to be stored to TB.",
        default=2
    )
    parser.add_argument('--save_img_step', '-si',
        type=int,
        help="Interval of saving images to TB and model parameters to storage.",
        default=2
    )

    # optimization
    parser.add_argument('--learning_rate', '-lr',
        type=float,
        help="Global Learning Rate.",
        default=1e-3
    )
    parser.add_argument('--n_epochs', '-n', type=int, default=5000)
    parser.add_argument('--gradient_clipping', '-gc',
        type=int,
        help="Gradient norm clip value for gradient clipping (GC). Value of 0 corresponds to no GC."
             "A value of 1 corresponds to enabling GC, where the optimal clipping value is estimated. This takes some time (training for 15 epochs)."
             "Specifying a value > 1 manually sets the GC value.",
        default=5
    )

    parser.add_argument('--gaussian_noise_level', '-gnl',
        type=float,
        help="Gaussian observation noise level added to observations/teacher signals. Value expected to be fraction of data standard deviation",
        default=0.01
    )

    # regularization
    parser.add_argument('--use_reg', '-r',
        type=int,
        help="If set to 1 (True), use Manifold Attractor Regularization (MAR).",
        default=0
    )
    parser.add_argument('--reg_ratios', '-rr',
        nargs='*',
        help="Ratio of states to regularize. A value of 0.5 corresponds to 50% of dim_z regularized.",
        type=float,
        default=[0.5]
    )
    parser.add_argument('--reg_alphas', '-ra',
        nargs='*',
        help="Regularization weighting, determines the strength of regularization.",
        type=float,
        default=[1e-2]
    )
    parser.add_argument('--reg_norm', '-rn',
        type=str,
        help="Regularization norm. L2 -> standard, L1 -> sparsification of W, h (and pulls A stricter to 1).",
        choices=['l2', 'l1'],
        default='l2'
    )
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def train(args):
    # prepare training device
    device = utils.prepare_device(args)

    writer, save_path = utils.init_writer(args)
    
    if args.preprocessing==1:
      args, data_set = utils.load_dataset_preprocessing(args,args.experimental_dataset)
    else:
      if args.train_multiple_shooting==1:
        args, data_set = utils.load_dataset_multiple_shooting(args,args.experimental_dataset)
      else:
        args, data_set = utils.load_dataset_multimodal(args, args.experimental_dataset)
    
    utils.check_args(args)
    utils.save_args(args, save_path, writer)
    
    if args.train_svae==1:
      training_algorithm = svae_algorithm.SVAE(args, data_set, writer, save_path,
                                             device)                                         
    elif args.train_multiple_shooting==1:
      training_algorithm = ms_algorithm.multiple_shooting(args, data_set, writer, save_path,
                                             device)
    else:
      training_algorithm = bptt_algorithm.BPTT(args, data_set, writer, save_path,
                                               device)
    training_algorithm.train()
    return save_path


def main(args):
    train(args)


if __name__ == '__main__':
    main(get_args())

