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


class multiple_shooting:
    """
    Train a model via multiple shooting.
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

        self.decoder_ordinal = args.dec_model_ordinal
        self.decoder_gaussian = args.dec_model_gaussian
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.dim_forcing = args.dim_forcing
        self.use_poisson = args.use_poisson
        self.use_gaussian = args.use_gaussian
        self.use_ordinal = args.use_ordinal
        
        self.ordinal_scaling = args.ordinal_scaling
        
        self.reg_z0=args.reg_z0
        
        self.R_z = nn.Parameter(tc.ones(args.dim_z), requires_grad=False)

        #multimodal args
        self.dim_g = args.dim_g
        self.dim_o = args.dim_o
        self.dim_p = args.dim_p
        self.poisson_scaling = args.poisson_scaling
        self.ordinal_scaling=args.ordinal_scaling
       
        # observation noise
        self.noise_levels = args.gaussian_noise_level
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
            
        
    def compute_loss(self, target: tc.Tensor, lat_z: tc.Tensor, z0_target: tc.Tensor,
                     seq_indices: tc.Tensor) -> tc.Tensor:
        '''
        Compute Loss w/ optional MAR loss.
        '''
        
        loss_obs = .0
        loss_ms= .0
        
        ##observation loss
                         
        (target_g, target_o, target_p) = self.split_input(target)
            
        if self.use_gaussian == 1:
          if self.decoder_gaussian == 'categorical':
                for i in range(self.batch_size):
                      loss_obs += -self.model.D_ordinal.categorical_log_likelihood(target_g[i],lat_z[i, :, :self.dim_forcing])*self.ordinal_scaling      
          else:
                loss_obs += -self.model.D_gaussian.log_likelihood(target_g,lat_z[:, :, :])

        if self.use_ordinal == 1:
            if self.decoder_ordinal == 'cumulative_link':
                for i in range(self.batch_size):
                    loss_obs += -self.model.D_ordinal.cumulative_link_log_likelihood(target_o[i, :, :],
                                                                                     lat_z[i, :, :])*self.ordinal_scaling / (
                                            self.seq_len * self.batch_size)
        if self.use_poisson == 1:
            # switch dimensions to allow broadcasting of likelihood across batches
            target_p = target_p.permute(0, 2, 1)
            lat_z_p = lat_z.permute(0, 2, 1)
            loss_obs += -self.model.D_poisson.log_likelihood(target_p[:, :, :], lat_z_p[:, :, :]) / (
                        self.seq_len * self.batch_size) * self.poisson_scaling
                        
       #multiple shooting loss
        loss_ms += self.reg_z0 * self.loss_fn(lat_z[:, -1, :], z0_target)
        # train last shooting node
        if len(self.data_set) - 1 in seq_indices:
          pass
       #     loss_ms -= self.model.obs_model.log_likelihood(x=self.data_set.data_point_for_last_shooting_node.unsqueeze(0)
       #                                                 ,z=self.model.z0[-1].unsqueeze(0).unsqueeze(0))
        loss=loss_obs+loss_ms
        loss_ratio=loss_obs/loss_ms
        
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
            for idx, (sequences, targets, sequence_indices) in enumerate(dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                
                z, z0_target = self.model.generate_latent_trajectory_and_shooting_nodes(length=self.seq_len,
                                                                                        sequence_indices=sequence_indices)
                loss = self.compute_loss(target=sequences, lat_z=z, z0_target=z0_target, seq_indices=sequence_indices)

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