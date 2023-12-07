from multitasking import *


def ubermain(n_runs):
    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6.
    Possible Arguments can be found in main.py
    
    When using GPU for training (i.e. Argument 'use_gpu 1')  it is generally
    not necessary to specify device ids, tasks will be distributed automatically.
    """
    args = []
    
    
    args.append(Argument('experiment', ['Example_lorenz']))

    #select modalities
    args.append(Argument('use_gaussian', [1], add_to_name_as="g"))
    args.append(Argument('use_ordinal', [1], add_to_name_as="o"))
    args.append(Argument('use_poisson', [1], add_to_name_as="p"))
    
    #add noise to observations
    args.append(Argument('gaussian_observation_noise', [0.1], add_to_name_as="noise"))
    #train on GPU
    args.append(Argument('use_gpu', [0]))
    #initial learning rate
    args.append(Argument('learning_rate', [1e-3]))
    args.append(Argument('run', list(range(1, 1 + n_runs))))
    #encoder model
    args.append(Argument('enc_model', ['CNN']))
    #teacher forcing interval tau
    args.append(Argument('n_interleave', [10], add_to_name_as="n"))
    #determine whether to sample TF signal from encoder model
    args.append(Argument('sample_rec', [1]))
    #sequence length during training
    args.append(Argument('seq_len', [300]))
    args.append(Argument('batch_size', [16]))
    args.append(Argument('batches_per_epoch', [25]))
    #layer norm stabilizes training
    args.append(Argument('layer_norm', [1]))
    #latent dimension M
    args.append(Argument('dim_z', [20],add_to_name_as="dz"))
    #encoder/forcing dimension K
    args.append(Argument('dim_forcing', [20],add_to_name_as="df"))
    #bases of dendPLRNN
    args.append(Argument('n_bases', [15]))
    
    args.append(Argument('n_epochs', [1000]))
    args.append(Argument('save_step', [200]))
    args.append(Argument('save_img_step', [200]))
    
    #whether to estimate the noise covariance in latent space or fix at 1
    args.append(Argument('estimate_noise_covariances', [0]))
    
      
    
    return args


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # number of runs for each experiment
    n_runs = 50
    # number if runs to run in parallel
    n_cpu = 25
    # number of processes run parallel on a single GPU
    n_proc_per_gpu = 1

    args = ubermain(n_runs)
    run_settings(*create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu))
