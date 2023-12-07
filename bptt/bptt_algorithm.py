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

class BPTT:
    """
    Train a model with (truncated) BPTT.
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
        self.n_epochs=args.n_epochs

        # estimate forcing interval
        if args.estimate_forcing:
            dataloader = self.data_set.get_rand_dataloader()
            inp, target=next(iter(dataloader))
            print(f"Estimating forcing interval...")
            tau_ac = estimate_forcing_interval(inp[0].cpu().numpy(),
                                               False, mode="ACORR")[0]
            tau_mi = estimate_forcing_interval(inp[0].cpu().numpy(),
                                               False, mode="MI")[0]
            mn = min(tau_ac, tau_mi)
            print(f"Estimated forcing interval: min(AC {tau_ac}, MI {tau_mi}) ---> {mn}")
            self.n = mn
        else:
            print(f"Forcing interval set by user: {args.n_interleave}")
            self.n = args.n_interleave
        

        # optimizer
        self.optimizer = optim.RAdam(self.model.parameters(), args.learning_rate)
        
        # others
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.gradient_clipping = args.gradient_clipping
        self.writer = writer
        self.use_reg = args.use_reg
        self.saver = saving.Saver(writer, save_path, args, self.data_set, self.regularizer)
        self.save_step = args.save_step


        # losses
        self.loss_fn = nn.MSELoss()
        self.rec_loss_fn = nn.MSELoss()
        self.decoder_ordinal = args.dec_model_ordinal
        self.decoder_gaussian = args.dec_model_gaussian
        self.decoder_poisson=args.dec_model_poisson


        #dimensions
        self.dim_forcing = args.dim_forcing
        self.use_poisson = args.use_poisson
        self.use_gaussian = args.use_gaussian
        self.use_ordinal = args.use_ordinal
        
        self.preprocessing = args.preprocessing
        self.estimate_noise_covariances=args.estimate_noise_covariances


        #select whether noise covariance in latent space is jointly estimated or fixed
        if self.estimate_noise_covariances:
          self.R_z = nn.Parameter(tc.ones(args.dim_forcing)*1, requires_grad=True)
        else:
          self.R_z = nn.Parameter(tc.ones(args.dim_forcing)*1, requires_grad=False)
          
        #multimodal args
        self.dim_s=args.dim_s
        self.dim_g = args.dim_g
        self.dim_o = args.dim_o
        self.dim_p = args.dim_p
        self.ordinal_scaling = args.ordinal_scaling
        self.poisson_scaling = args.poisson_scaling

        #loss args
        self.alpha_latent = args.alpha_latent
        self.alpha_obs = args.alpha_obs
        self.alpha_rec = args.alpha_rec

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
        
    #employ if covariances in latent space are estimated, otherwise use MSE loss and weight with alpha_latent
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

    def compute_gaussian_loss(self, inp, enc_z, target, lat_z):
        if self.decoder_gaussian == 'categorical':
            return sum(-self.model.D_ordinal.categorical_log_likelihood(inp[i], enc_z[i]) * self.ordinal_scaling
                       for i in range(self.batch_size))
        else:
            return -self.model.D_gaussian.log_likelihood(inp, enc_z)
            
    def compute_ordinal_loss(self, inp, enc_z, target, lat_z):
        #if preprocessing, compute Gaussian likelihood
        if self.preprocessing:
            return -self.model.D_ordinal.log_likelihood(inp, enc_z)
        elif self.decoder_ordinal == 'cumulative_link':
            return sum(-self.model.D_ordinal.cumulative_link_log_likelihood(inp[i], enc_z[i]) * self.ordinal_scaling 
                       for i in range(self.batch_size)) / (self.seq_len * self.batch_size)
        elif self.decoder_ordinal == 'categorical':
            return sum(-self.model.D_ordinal.categorical_log_likelihood(inp[i], enc_z[i]) * self.ordinal_scaling
                       for i in range(self.batch_size))
        else:
            return 0  # Default return value (or raise an error if unsupported decoder)

    def compute_poisson_loss(self, inp, enc_z, target, lat_z):
        #if preprocessing, compute Gaussian likelihood
        if self.preprocessing:
            return -self.model.D_poisson.log_likelihood(inp, enc_z)
        elif self.decoder_poisson == 'categorical':
            return sum(-self.model.D_ordinal.categorical_log_likelihood(inp[i], enc_z[i]) * self.ordinal_scaling
                       for i in range(self.batch_size))
        else:
            return (-self.model.D_poisson.log_likelihood(inp, enc_z) * self.poisson_scaling) / (self.seq_len * self.batch_size)


    def compute_loss(self, enc_z, lat_z,
                     inp: tc.Tensor, target: tc.Tensor, entropy: tc.Tensor, epoch) -> tc.Tensor:
        '''
        Compute Loss w/ optional MAR loss.
        '''
        # return: enc_z:latent states from encoder, lat_z:forward predicted latent states gen.model, rec:x_ from encoder to decoder, pred:predicted x from decoder model, inp:data, target:shifted data

        loss_rec = .0
        loss_obs = .0
        loss = .0

        (inp_g, inp_o, inp_p) = self.split_input(inp)
        (target_g, target_o, target_p)= self.split_input(target)
        
        if self.use_gaussian:
          loss_rec += self.compute_gaussian_loss(inp_g, enc_z, target_g, lat_z)
          loss_obs += self.compute_gaussian_loss(target_g, lat_z, target_g, lat_z)
            
        if self.use_ordinal:
            loss_rec += self.compute_ordinal_loss(inp_o, enc_z, target_o, lat_z)
            loss_obs += self.compute_ordinal_loss(target_o, lat_z, target_o, lat_z)
        
        if self.use_poisson:
            loss_rec += self.compute_poisson_loss(inp_p, enc_z, target_p, lat_z)
            loss_obs += self.compute_poisson_loss(target_p, lat_z, target_p, lat_z)


        # loss in latent space remains Gaussian
        if self.estimate_noise_covariances:
          loss_latent=-self.latent_likelihood(enc_z[:, 1:, :], lat_z[:, :-1, :self.dim_forcing])
        else:
          loss_latent = self.loss_fn(enc_z[:, 1:, :], lat_z[:, :-1, :self.dim_forcing])
          
        if self.rec_annealing:
          alpha_rec = self.alpha_rec * (1 - epoch / self.n_epochs)
          is_early_epoch = epoch / self.n_epochs < 0.1

          alpha_obs = 0 if is_early_epoch else self.alpha_obs
          alpha_latent = 0 if is_early_epoch else self.alpha_latent
        else:
            alpha_rec, alpha_obs, alpha_latent = self.alpha_rec, self.alpha_obs, self.alpha_latent

        loss += alpha_rec * loss_rec + alpha_obs * loss_obs +  alpha_latent*loss_latent + entropy

        #regularize latent magnitudes to avoid divergences
        latent_energy=False
        if latent_energy:
          loss=tc.linalg.vector_norm(enc_z[:, 1:, :])*0.01

        if self.use_reg:
            lat_model_parameters = self.model.latent_model.get_latent_parameters()
            loss += self.regularizer.loss(lat_model_parameters)

        return loss

    def compute_loss_simple(self, pred: tc.Tensor, target: tc.Tensor) -> tc.Tensor:
        '''
        Compute Loss w/ optional MAR loss.
        '''
        loss = .0
        loss += self.loss_fn(pred, target)

        if self.use_reg:
            lat_model_parameters = self.model.latent_model.get_latent_parameters()
            loss += self.regularizer.loss(lat_model_parameters)

        return loss


    def train(self):
        n = self.n

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
                
                
                #input handling, concatenated to the end of the input data
                
                if self.dim_s>0:
                  s=inp[:,:,-self.dim_s:]
                  inp=inp[:,:,:-self.dim_s]
                  target=target[:,:,:-self.dim_s]
                else:
                  s=None
                
                # add noise
                inp += tc.randn_like(inp) * self.noise_levels

               
                enc_z, lat_z, entropy = self.model(inp, n, s)
                loss = self.compute_loss(enc_z, lat_z, inp, target, entropy, epoch)

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


    def estimate_gc_norm(self):
        '''
        Estimate gradient clipping value as suggested by
        Pascanu, 2012: On the difficulty of training Recurrent Neural Networks.
        https://arxiv.org/abs/1211.5063
        Tracks gradient norms across 10 epochs of training and 
        computes the mean. GC clipping value is then set to 5 times
        that value.
        '''
        print("Estimating Gradient Clipping Value ...")
        params = deepcopy(self.model.state_dict())
        N_samples = 0
        running_g_norm = 0.
        for e in range(15):
            dataloader = self.data_set.get_rand_dataloader()
            for _, (inp, target) in enumerate(dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(inp, self.n)
                loss = self.compute_loss(pred, target)

                loss.backward()
                if e > 5:
                    running_g_norm += nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                                            max_norm=1e10)
                    N_samples += 1
                self.optimizer.step()
        
        gc_estimate = 5 * running_g_norm / N_samples
        self.gc = gc_estimate
        self.model.load_state_dict(params)
        print(f"Estimated Gradient Clipping Value: {self.gc}")