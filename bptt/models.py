import os
import torch as tc
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import utils
from bptt import saving
from bptt import PLRNN_model
import random
from typing import Optional
from sae.encoder import *
from sae.decoder import *


def load_args(model_path):
    args_path = os.path.join(model_path, 'hypers.pkl')
    args = np.load(args_path, allow_pickle=True)
    return args

class Model(nn.Module):
    def __init__(self, args=None, data_set=None):
        super().__init__()
        self.latent_model = None
        self.device = None
        self.args = args
        self.data_set = data_set
        
        if args is not None:
            self.dg=args.dim_g
            self.do=args.dim_o
            self.dp=args.dim_p
            # cast args to dictionary
            self.args = vars(args)
            self.init_from_args()

    def forward(self, x, n=None, s=None):

        '''
        forward pass.
        x: shape (S, T, N)

        '''
        S, T, dx = x.size()

        # Encoder z
        z_enc, entropy = self.E(x.view(-1, dx))
        _, df = z_enc.size()

        # forward mapped latent z
        z_ = self.latent_model(z_enc.view(S, T, -1),s, n)

        z_predicted = z_[:, :, :df]
                
        # return: z_enc: latent states from encoder, z_: forward predicted latent states from encoder, entropy from encoder
        return z_enc.view(S, T, -1), z_predicted, entropy
        
        
    def svae_pass(self,x):
    
        S, T, dx = x.size()

        # Encoder z
        z_enc, entropy = self.E(x.view(-1, dx))
        x_list = []
        
        params = self.latent_model.get_parameters()
        z_forward=self.latent_model.latent_step(z_enc,*params)

        if self.args['use_gaussian']:
            # Decoder
            x_g = self.D_gaussian(z_enc).reshape(S, T, self.dg)
            # map forward stepped latent dynamics to obs. space
            x_list.append(x_g)
        else:
            x_g=tc.zeros(0)

        #decoding is not necessary for computation of likelihood

        return z_enc.view(S, T, -1), z_forward.view(S, T, -1),x_g, entropy
        

    #for multiple shooting    
    def generate_latent_trajectory_and_shooting_nodes(self, length, sequence_indices):
    
        if sequence_indices is None:
            z0 = self.z0[:-1]
            z0_target = self.z0[1:]
        else:
            z0 = self.z0[sequence_indices]
            z0_target = self.z0[sequence_indices + 1]

        hidden_out = self.latent_model.generate_from_z0(length, z0)
        
        return hidden_out, z0_target



    def to(self, device: tc.device):
        self.device = device
        return super().to(device)

    def get_latent_parameters(self):
        '''
        Return a list of all latent model parameters:
        A: (dz, )
        W: (dz, dz)
        h: (dz, )

        For BE-models additionally:
        alpha (db, )
        thetas (dz, db)
        '''
        return self.latent_model.get_parameters()

    def get_num_trainable(self):
        '''
        Return the number of total trainable parameters
        '''
        return sum([p.numel() if p.requires_grad else 0 
                    for p in self.parameters()])

    def init_from_args(self):
        # resume from checkpoint?
        model_path = self.args['load_model_path']
        if model_path is not None:
            epoch = None
            if self.args['resume_epoch'] is None:
                epoch = utils.infer_latest_epoch(model_path)
            self.init_from_model_path(model_path, epoch)
        else:
            self.init_submodules()

    def init_z0_model(self, learn_z0: bool):
        z0model = None
        if learn_z0 and not self.args['use_inv_tf']:
            z0model = Z0Model(self.args['dim_g'], self.args['dim_z'])
        return z0model

    def init_obs_model(self, fix_output_layer):
        dz, dx = self.args['dim_z'], self.args['dim_g']
        output_layer = None
        if fix_output_layer:
            # fix observation model (identity mapping z<->x)
            output_layer = nn.Linear(dz, dx, bias=False)
            B = tc.zeros((dx, dz))
            
            if dz>dx:
              for i in range(dx):
                  B[i, i] = 1
            output_layer.weight = nn.Parameter(B, False)
        else:
            # learnable 
            output_layer = nn.Linear(dz, dx, bias=False)
        return output_layer

    def init_from_model_path(self, model_path, epoch=None):
        # load arguments
        self.args = load_args(model_path)

        # init using arguments
        self.init_submodules()

        # restore model parameters
        self.load_state_dict(self.load_statedict(model_path, 'model', epoch=epoch))
        
        
    def init_z0(self):
          z0 = nn.Parameter(tc.randn(self.args['number_of_sequences'] + 1, self.args['dim_z']), requires_grad=True)
          return z0

    def init_submodules(self):
        '''
        Initialize latent model, output layer and z0 model.
        '''
        # TODO: Add RNN/LSTM as separate models w/o explicit latent model.

        dz, df, dg, do, dc, dp, sample_rec, learn_covariances = self.args['dim_z'], self.args['dim_forcing'], self.args[
            'dim_g'], self.args['dim_o'], self.args['scale_ordinal_data'], self.args['dim_p'], self.args['sample_rec'], 0
        #, self.args['estimate_noise_covariances']
        dx = dg + do + dp
        dh=20
        
        train_svae=self.args['train_svae']

        self.latent_model = PLRNN_model.PLRNN(dx, dz, df,
                                              self.args['n_bases'],
                                              latent_model=self.args['latent_model'],
                                              clip_range=self.args['clip_range'],
                                              layer_norm=self.args['layer_norm'], ds=self.args['dim_s'])

        self.z0_model = self.init_z0_model(self.args['learn_z0'])
        
        if self.args["train_multiple_shooting"]:
          self.z0 = self.init_z0()
        
        
        self.output_layer = self.init_obs_model(self.args['fix_obs_model'])
        # Init encoder architectures

        if self.args['enc_model'] == 'MLP':
            self.E = EncoderMLP(dx, df)
        if self.args['enc_model'] == 'Linear':
            self.E = EncoderLinear(dx, df)
        elif self.args['enc_model'] == 'CNN':
            self.E = StackedConvolutions(dx, df, sample_rec)
        elif self.args['enc_model'] == 'POG':
            if self.data_set:
              dataloader = self.data_set.get_rand_dataloader()
              inp, target,_=next(iter(dataloader))
            else:
              inp=tc.randn(self.args['batch_size'], self.args['seq_len'], dx)
            self.E = ProductOfGaussians(dx, df,dh, inp, batch_size=1)

        elif self.args['enc_model'] == 'Identity':
            self.E = EncoderIdentity()

            # Decoder/observation models
            
        if self.args['use_gaussian'] == 1:
          if self.args['dec_model_gaussian'] == 'Linear':
            self.D_gaussian = Decoder_LinearObservation(df, dg, learn_covariances)
          elif self.args['dec_model_gaussian'] == 'MLP':
            self.D_gaussian = Decoder_MLP(df, dg)
          elif self.args['dec_model_gaussian'] == 'Identity':
            self.D_gaussian = DecoderIdentity(df,dg,learn_covariances)
          elif self.args['dec_model_gaussian'] == 'categorical':
            self.D_gaussian = DecoderCategorical(df, dg)
                
        if self.args['use_ordinal'] == 1:
          if self.args["preprocessing"] == 1:
            self.D_ordinal = Decoder_LinearObservation(df, do, learn_covariances)
          else:
            if self.args['dec_model_ordinal'] == 'cumulative_link':
              self.D_ordinal = Decoder_cumulative_link(df, do, dc)
            elif self.args['dec_model_ordinal'] == 'categorical':
              self.D_ordinal = DecoderCategorical(df, do)
            elif self.args['dec_model_ordinal'] == 'Identity':
              self.D_gaussian = DecoderIdentity(df,do,learn_covariances)
            
        if self.args['use_poisson'] == 1:
          if self.args["preprocessing"]==1:
            self.D_poisson = Decoder_LinearObservation(df, dp, learn_covariances)
          elif self.args['dec_model_poisson']== 'cumulative_link':
            self.D_poisson = Decoder_cumulative_link(df, dp, 2*dc-1)
          elif self.args['dec_model_poisson']== 'categorical':
            self.D_poisson = Decoder_cumulative_link(df, dp)
          elif self.args['dec_model_poisson']== 'poisson':
            self.D_poisson = Decoder_Poisson(df, dp)
          elif self.args['dec_model_poisson']== 'zip':
            self.D_poisson = Decoder_zip(df, dp)
          elif self.args['dec_model_poisson']== 'negative_binomial':
            self.D_poisson = Decoder_NegativeBinomial(df, dp)


       
       
       


    def load_statedict(self, model_path, model_name, epoch=None):
        if epoch is None:
            epoch = self.args['n_epochs']
        path = os.path.join(model_path, '{}_{}.pt'.format(model_name, str(epoch)))
        state_dict = tc.load(path)
        return state_dict

    @tc.no_grad()
        
    
    def generate_free_trajectory(self, data, T):
        dz = self.args['dim_z']
        df = self.args['dim_forcing']
        # z0 = self.E(data[[0]])
        
        data,s=self.split_input_data(data)
        
        states, entropy = self.E(data[:, :])

        if dz == df:
            z0 = states[0, :]
        else:
            z0 = tc.randn((1, dz))
            z0[:, :df] = states[0, :]
        
        if self.args["train_multiple_shooting"]:
          z0=self.z0[0].unsqueeze(0)
            
        noisy_sample=False
        if noisy_sample==True:
          z0=z0+tc.randn(z0.shape)

        # latent traj is T x dz
        latent_traj = self.latent_model.generate(T, z0, s)
        # cast to b x T x dz for output layer and back to T x dx
        
        z_readout = latent_traj[:, :df]
        
        
        decoded_data = []

        if self.args['use_gaussian'] == 1:
            x_gaussian = self.D_gaussian(z_readout)
            decoded_data.append(x_gaussian)

        if self.args['use_ordinal'] == 1:
            # cumulative link decoder with ordinal scale and dimension
            x_ordinal = self.D_ordinal(z_readout)
            decoded_data.append(x_ordinal)

        if self.args['use_poisson'] == 1:
            x_poisson = self.D_poisson(z_readout.unsqueeze(0)).squeeze(0)
            decoded_data.append(x_poisson)

        obs_traj = tc.cat(decoded_data, axis=-1)

        # T x dx, T x dz
        return obs_traj, latent_traj

    def plot_simulated(self, data: tc.Tensor, T: int, modality=int):
    
        X, Z = self.generate_free_trajectory(data, T)
        fig = plt.figure()
        plt.title('simulated')
        plt.axis('off')

        if modality == 1:
            start = 0
            stop = min([self.dg, 5])
        if modality == 2:
            start = self.dg
            stop = self.dg + min([self.do, 5])
        if modality == 3:
            start = self.dg + self.do
            stop = self.dg + self.do + min([self.dp, 5])

        X = X[:, start:stop]

        plot_list = [X, Z]

        names = ['x', 'z']
        for i, x in enumerate(plot_list):
            fig.add_subplot(len(plot_list), 1, i + 1)
            plt.plot(x.cpu())
            plt.title(names[i])
        plt.xlabel('time steps')
        
        
    def plot_simulated_3D(self, data: tc.Tensor, time_steps: int):

        if self.dg==3:
          X, Z = self.generate_free_trajectory(data, time_steps)
          fig = plt.figure()
          plt.title('simulated 3D plot')
          plt.axis('off')
          ground_truth=np.asarray(data)
          generated=np.asarray(X)
          fig = plt.figure()
          ax = fig.gca(projection="3d")
          ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label="ground truth")
          ax.plot(generated[:, 0], generated[:, 1], generated[:, 2], label="generated")
          plt.legend()
          plt.title("3D plot generated vs. ground truth")
      

    def plot_obs_simulated(self, data: tc.Tensor, modality=int):
        time_steps = len(data)
        
        #data,s=self.split_input_data(data)
        
        X, Z = self.generate_free_trajectory(data, time_steps)
        fig = plt.figure()
        plt.title('observations')
        plt.axis('off')
        n_units = data.shape[1]
        max_units = min([n_units, 5])
        max_time_steps = 1000

        if modality == 1:
            start = 0
            stop = min([self.dg, 7])
        if modality == 2:
            start = self.dg
            stop = self.dg + min([self.do, 7])
        if modality == 3:
            start = self.dg + self.do
            stop = self.dg + self.do + min([self.dp, 7])

        X = X[:, :]
        data = data[:, :]

        for i in range(start, stop):
            fig.add_subplot(stop - start, 1, i + 1 - start)
            plt.plot(data[:max_time_steps, i].cpu())
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(X[:max_time_steps, i].cpu())
            ax.set_ylim(lim)
        plt.legend(['data', 'x simulated'])
        plt.xlabel('time steps')

    @tc.no_grad()
    def plot_latent_vs_encoded(self, data: tc.Tensor,
                               rand_seq: Optional[bool] = True):
        '''
        Plot the overlap of latent, teacher forced trajectory
        with the latent trajectory inferred by the encoder.
        '''
        T = self.args['seq_len']
        N = self.args['n_interleave']
        dz = self.args['dim_z']

        df = self.args['dim_forcing']

        T_full, _ = data.size()
        max_units = min([df, 10])

        # input and model prediction
        if rand_seq:
            t = random.randint(0, T_full - T)
        else:
            t = 0
        input_ = data[t: t + T]
        enc_z, z, _ = self(input_.unsqueeze(0), N)
        enc_z.squeeze_(0)
        z.squeeze_(0)[:, :df]

        enc_z = enc_z[:, :df]
        z = z[:, :df]

        z_ = tc.cat([enc_z[[0]], z], 0)

        # x axis
        x = np.arange(T)

        # plot
        fig = plt.figure()
        plt.title('Latent trajectory')
        plt.axis('off')
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            plt.plot(x, enc_z[:, i].cpu(), label='Encoded z')
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(x, z_[:-1, i].cpu(), label='Generated z')
            ax.set_ylim(lim)
            plt.scatter(x[::N], enc_z[::N, i].cpu(), marker='2',
                        label='TF', color='r')
            ax.set_ylim(lim)
            plt.legend(prop={"size": 5})
            plt.ylabel(f'$x_{i}$')
        plt.xlabel('t')

    @tc.no_grad()
    def plot_reconstruction(self, data: tc.Tensor,
                            rand_seq: Optional[bool] = True):
        '''
        Plot reconstruction of the input sequence
        passed through the Autoencoder.
        '''
        data,s=self.split_input_data(data)
        
        T = self.args['seq_len']
        T_full, dx = data.size()
        max_units = min([dx, 10])
        
        

        # input and model prediction
        if rand_seq:
            t = random.randint(0, T_full - T)
        else:
            t = 0
        input_ = data[t: t + T]
        enc_z, _, _ = self(input_.unsqueeze(0))
        
        x_list = []
        
        if self.args['use_gaussian']:
            if self.args['dec_model_gaussian'] == 'categorical':
              x_g = self.D_gaussian(enc_z[0]).reshape(1, T, self.dg)
            else:
              x_g = self.D_gaussian(enc_z).reshape(1, T, self.dg)
            x_list.append(x_g)

        if self.args['use_ordinal']:
            x_o = self.D_ordinal(enc_z[0]).view(1, T, self.do)
            x_list.append(x_o)

        if self.args['use_poisson']:
            # decoding happens directly in the likelihoods, but send through individual observation models for saving
            x_p = self.D_poisson(enc_z.view(1, T, -1)).view(1, T, self.dp)
            x_list.append(x_p)

        x_rec = tc.cat(x_list, axis=-1)        
        rec=x_rec.squeeze(0)
        
        # x axis
        x = np.arange(T)

        # plot
        fig = plt.figure()
        plt.title('Reconstruction')
        plt.axis('off')
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            plt.plot(x, input_[:, i].cpu(), label='GT')
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(x, rec[:, i].cpu(), label='Pred')
            ax.set_ylim(lim)
            plt.legend(prop={"size": 5})
            plt.ylabel(f'$x_{i}$')
        plt.xlabel('t')

    @tc.no_grad()
    
    def split_input_data(self, data):
      dim_s=self.args["dim_s"]
      if dim_s>0:
        s=data[:,-dim_s:]
        data=data[:,:-dim_s]
      else:
        s=None
      return data, s
    
    
    def plot_prediction(self, data: tc.Tensor, modality: int,
                        rand_seq: Optional[bool] = True):

        T = self.args['seq_len']
        N = self.args['n_interleave']
        T_full, dx = data.size()
        max_units = min([dx, 5])

        rand_seq=False
        
        data,s=self.split_input_data(data)

        
        # input and model prediction
        if rand_seq:
            t = random.randint(0, T_full - T)
        else:
            t = 0
        input_ = data[t: t + T]
        
        if s is not None:
          s_ = s[t: t + T].unsqueeze(0)
        else:
          s_=None
        
        _, lat_z,_ = self(input_.unsqueeze(0), N, s_)
    
        x_list=[]
        
        if self.args['use_gaussian']:
            # Decoder
            if self.args['dec_model_gaussian'] == 'categorical':
              x_g = self.D_gaussian(lat_z[0]).reshape(1, T, self.dg)
            else:
              x_g = self.D_gaussian(lat_z).reshape(1, T, self.dg)
            x_list.append(x_g)

        if self.args['use_ordinal']:
            x_o = self.D_ordinal(lat_z[0]).view(1, T, self.do)
            x_list.append(x_o)

        if self.args['use_poisson']:
            # decoding happens directly in the likelihoods, but send through individual observation models for saving
            x_p = self.D_poisson(lat_z.view(1, T, -1)).view(1, T, self.dp)
            x_list.append(x_p)

        x_pred = tc.cat(x_list, axis=-1)        
        pred=x_pred.squeeze(0)

        # x axis
        x = np.arange(T)

        # plot
        fig = plt.figure()
        plt.title('Prediction')
        plt.axis('off')

        if modality == 1:
            start = 0
            stop = min([self.dg, 5])
        if modality == 2:
            start = self.dg
            stop = self.dg + min([self.do, 5])
        if modality == 3:
            start = self.dg + self.do
            stop = self.dg + self.do + min([self.dp, 5])

        for i in range(start, stop):
            fig.add_subplot(stop - start, 1, i + 1 - start)
            plt.plot(x, input_[:, i].cpu(), label='GT')
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(x, pred[:, i].cpu(), label='Pred')
            ax.set_ylim(lim)
            plt.scatter(x[::N], input_[::N, i].cpu(), marker='2',
                        label='TF-obs', color='r')
            ax.set_ylim(lim)
            plt.legend(prop={"size": 5})
            plt.ylabel(f'$x_{i}$')
        plt.xlabel('t')


class Z0Model(nn.Module):
    '''
    MLP that predicts an optimal initial latent state z0 given
    an inital observation x0.

    Takes x0 of dimension dx and returns z0 of dimension dz, by
    predicting dz-dx states and then concatenating x0 and the prediction:
    z0 = [x0, MLP(x0)]
    '''
    def __init__(self, dx: int, dz: int):
        super(Z0Model, self).__init__()
        # TODO: MLP currently only affine transformation
        # maybe try non-linear, deep variants?
        self.MLP = nn.Linear(dx, dz-dx, bias=False)

    def forward(self, x0):
        return tc.cat([x0, self.MLP(x0)], dim=1)
