from copy import deepcopy
import torch as tc
from torch import optim
from torch import nn
from bptt import models
from bptt import regularization
from bptt import saving
from bptt.dataset import GeneralDataset
from tensorboardX import SummaryWriter
from argparse import Namespace
from timeit import default_timer as timer
import datetime
from bptt.tau_estimation import estimate_forcing_interval
import math

class SVAE:
    """
    Train a model as a SVAE.
    """

    def __init__(self, args: Namespace, data_set: GeneralDataset,
                 writer: SummaryWriter, save_path: str, device: tc.device):
        # dataset, model, device, regularizer
        self.device = device
        self.data_set = data_set
        self.model = models.Model(args, data_set)
        self.regularizer = regularization.Regularizer(args)
        self.to_device()
        self.rec_annealing=args.rec_annealing

        # optimizer
        self.optimizer = optim.RAdam(self.model.parameters(), args.learning_rate)
        
        # others
        self.n_epochs = args.n_epochs
        self.gradient_clipping = args.gradient_clipping
        self.writer = writer
        self.use_reg = args.use_reg
        self.saver = saving.Saver(writer, save_path, args, self.data_set, self.regularizer)
        self.save_step = args.save_step
        self.loss_fn = nn.MSELoss()

        self.rec_loss_fn = nn.MSELoss()
        self.decoder_ordinal = args.dec_model_ordinal
        self.decoder_gaussian = args.dec_model_gaussian
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.dim_forcing = args.dim_forcing
        self.use_zip = args.use_zip
        self.use_gaussian = args.use_gaussian
        self.use_ordinal = args.use_ordinal
        self.ordinal_scaling = args.ordinal_scaling
        
        self.estimate_noise_covariances=args.estimate_noise_covariances

        if self.estimate_noise_covariances:
          self.R_z = nn.Parameter(tc.ones(args.dim_forcing)*1, requires_grad=True)
        else:
          self.R_z = nn.Parameter(tc.ones(args.dim_forcing)*1, requires_grad=False)
        

        #multimodal args
        self.dim_g = args.dim_g
        self.dim_o = args.dim_o
        self.dim_p = args.dim_p
        self.zip_scaling = args.zip_scaling

        #loss
        self.alpha_latent = args.alpha_latent
        self.alpha_obs = args.alpha_obs
        self.alpha_rec = args.alpha_rec

        # observation noise
        noise_fraction = args.gaussian_noise_level
     #   data_std = tc.std(self.data_set.data, dim=0, keepdim=True)
        data_std = 1
        self.noise_levels = data_std * noise_fraction

        # scheduler
        e = args.n_epochs
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [int(0.1*e), int(0.8*e), int(0.9*e)], 0.1)

    def to_device(self) -> None:
        '''
        Moves members to computing device.
        '''
        self.model.to(self.device)
        self.data_set.to(self.device)
        self.regularizer.to(self.device)

    def split_input(self, inp):

        # split input and target according to dimensionality of multimodal observations. For non-observed modalities, dimensions are set to zero
        inp_g = inp[:, :, :self.dim_g]
        inp_o = inp[:, :, self.dim_g:self.dim_g + self.dim_o]
        inp_p = inp[:, :, self.dim_g + self.dim_o:]

        return (inp_g, inp_o, inp_p)
        
    def mahalonobis_distance(residual, matrix):
            return - 0.5 * (residual.t() @ residual * tc.inverse(tc.diag(matrix ** 2))).sum()
            
    def latent_likelihood(self,z_lat, z_enc):
      
          def mahalonobis_distance(residual, matrix):
            distance=0
            for i in range(residual.shape[0]):
              distance+=- 0.5 * (residual[i].t() @ residual[i] * tc.inverse(tc.diag(matrix ** 2))).sum()
              return distance
              
              
          def log_det(diagonal_matrix):
              return - tc.log(diagonal_matrix).sum()
              
          LL_z = mahalonobis_distance(z_lat -  z_enc, self.R_z)
          
          T=z_lat.shape[1]
          
          return LL_z+log_det(self.R_z)*(T-1)/2
            
            
    def compute_loss(self, enc_z, forward_z, rec_g: tc.Tensor,inp: tc.Tensor, entropy: tc.Tensor, epoch) -> tc.Tensor:
        '''
        Compute Loss w/ optional MAR loss.
        '''
        
        def log_det(diagonal_matrix):
            return - tc.log(diagonal_matrix).sum()
        
        loss_rec = .0
        loss = .0  
      
        (inp_g, inp_o, inp_p) = self.split_input(inp)
       # (rec_g, rec_o, rec_p) = self.split_input(x_rec)
        
        dim_z=enc_z.shape[-1]
        
        llz=-self.latent_likelihood(enc_z[:, 1:, :], forward_z[:,:-1, :])
        
        if self.use_gaussian == 1:
            if self.decoder_gaussian == 'categorical':
              for i in range(self.batch_size):
                    loss_rec += -self.model.D_ordinal.categorical_log_likelihood(inp_g[i],enc_z[i])*self.ordinal_scaling
                    
            else:
              # reconstruction loss from autoencoder (GT_vs_reconstructed)
              loss_rec += -self.model.D_gaussian.log_likelihood(inp_g,enc_z)

        if self.use_ordinal == 1:
            if self.decoder_ordinal == 'cumulative_link':
                for i in range(self.batch_size):
                    loss_rec += -self.model.D_ordinal.cumulative_link_log_likelihood(inp_o[i, :, :], enc_z[i]) / (
                                self.seq_len * self.batch_size)
        if self.use_zip == 1:
            # switch dimensions to allow broadcasting of likelihood across batches
            inp_p = inp_p.permute(0, 2, 1)
            enc_z_p = enc_z.permute(0, 2, 1)

            loss_rec += -self.model.D_poisson.log_likelihood(inp_p[:, :, :], enc_z_p[:, :, :]) / (
                        self.seq_len * self.batch_size) * self.zip_scaling
          
        loss = llz-entropy+loss_rec
        
        
        if self.use_reg:
            lat_model_parameters = self.model.latent_model.get_latent_parameters()
            loss += self.regularizer.loss(lat_model_parameters)
        
        
        return loss 


    def train(self):

        cum_T = 0.

        if self.gradient_clipping == 1:
            self.estimate_gc_norm()
        else:
            self.gc = self.gradient_clipping

        for epoch in range(1, self.n_epochs + 1):
            # enter training mode
            self.model.train()

            # measure time
            T_start = timer()

            # sample random sequences every epoch
            dataloader = self.data_set.get_rand_dataloader()
            for idx, (inp, target, _) in enumerate(dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                
                
                enc_z, forward_z, rec, entropy = self.model.svae_pass(inp)
                loss = self.compute_loss(enc_z,forward_z, rec,inp, entropy, epoch)
                    
                loss.backward()
                if self.gradient_clipping >= 1:
                    nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                             max_norm=self.gc)
                self.optimizer.step()

            self.scheduler.step()

            # timing
            T_end = timer()
            T_diff = T_end-T_start
            cum_T += T_diff
            cum_T_str = str(datetime.timedelta(seconds=cum_T)).split('.')[0]

            print(f"Epoch {epoch} took {round(T_diff, 2)}s | Cumulative time (h:mm:ss):" 
             f" {cum_T_str} | Loss = {loss.item()}")

            if epoch % self.save_step == 0:
                self.saver.epoch_save(self.model, epoch)


 