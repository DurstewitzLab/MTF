import torch as tc
import torch.nn as nn
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorboardX import utils as tb_utils
import seaborn as sns

import main_eval
import utils


class Saver:
    def __init__(self, writer, save_path, args, data_set, regularizer):
        self.writer = writer
        self.save_path = save_path
        self.args = args
        self.data_set = data_set
        self.model = None
        self.current_epoch = None
        self.current_model = None
        self.regularizer = regularizer
        self.initial_save()

    def initial_save(self):
        if self.args.use_tb:
            self.save_dataset()

 #   def save_dataset(self):
    
    
 #     dataset_gaussian = self.data_set.timeseries['gaussian'].train_data.cpu().numpy()[:1000]
 #     plt.plot(dataset_gaussian)
 #     plt.title('Gaussian Observations')
 #     plt.xlabel('time steps')
 #     figure = plt.gcf()
 #     save_figure_to_tb(figure, self.writer, text='Gaussian data set', global_step=None)
 #     plt.close()

    def save_dataset(self):

        if self.args.use_gaussian == 1:
            dataset_gaussian = self.data_set.timeseries['gaussian'].train_data.cpu().numpy()[:1000]
            plt.plot(dataset_gaussian)
            plt.title('Gaussian Observations')
            plt.xlabel('time steps')
            figure = plt.gcf()
            save_figure_to_tb(figure, self.writer, text='Gaussian data set', global_step=None)
            plt.close()

        if self.args.use_ordinal == 1:
            dataset_ordinal = self.data_set.timeseries['ordinal'].train_data.cpu().numpy()[:1000,:self.args.dim_o]
            plt.plot(dataset_ordinal)
            plt.title('Ordinal Observations')
            plt.xlabel('time steps')
            figure = plt.gcf()
            save_figure_to_tb(figure, self.writer, text='Ordinal data set', global_step=None)
            plt.close()

        if self.args.use_poisson == 1:
            dataset_poisson = self.data_set.timeseries['poisson'].train_data.cpu().numpy()[:1000]
            plt.plot(dataset_poisson)
            plt.title('Poisson Observations')
            plt.xlabel('time steps')
            figure = plt.gcf()
            save_figure_to_tb(figure, self.writer, text='Poisson data set', global_step=None)
            plt.close()

    

    def epoch_save(self, model, epoch):
        # update members
        self.current_epoch = epoch
        self.current_model = model
        
        # switch to evaluation mode
        self.current_model.eval()

        if self.args.use_tb:
            with tc.no_grad():
              #  self.save_loss_terms()
              #  self.save_metrics()

                # save plots and params w/ different step
                s = self.args.save_img_step
                if (self.current_epoch % s == 0):
                    if self.args.train_multiple_shooting:
                      self.save_simulated()
                    else:
                      self.save_prediction()
                      self.save_simulated()
                    # self.save_latent_vs_encoded()
                    # self.save_reconstruction()
                    # self.save_parameters()

                    # save state dict (parameters)
                    path = os.path.join(self.save_path,
                            f'model_{epoch}.pt')
                    while (not os.path.exists(path)):
                        tc.save(self.current_model.state_dict(),
                                path)

    def save_loss_terms(self):
        # get the first input and target sequence of the dataset 
        # to compute the loss
        
        input_, target, _ = self.data_set[0]

        # latent model parameters
        latent_model = self.current_model.latent_model
        latent_model_parameters = latent_model.get_latent_parameters()

        # MSE
        with tc.no_grad():
            enc_z, z, entropy = self.current_model(input_.unsqueeze(0), self.args.n_interleave)
            lat_loss = nn.functional.mse_loss(z[:, :-1, :self.args.dim_forcing], enc_z[:, 1:, :self.args.dim_forcing])


        self.writer.add_scalar(tag='Latent Loss', scalar_value=lat_loss, global_step=self.current_epoch)
        self.writer.add_scalar(tag='Entropy', scalar_value=entropy, global_step=self.current_epoch)

        # manifold attractor regularization (MAR)
        if self.args.use_reg:
            loss_reg = self.regularizer.loss(latent_model_parameters)
            self.writer.add_scalar(tag='MAR Loss', scalar_value=loss_reg, global_step=self.current_epoch)

        # A norm
        A, W, h = latent_model_parameters
        max_eig_A = tc.diag(A).max().detach().item()
        self.writer.add_scalar(tag='max. eig(A)', scalar_value=max_eig_A, global_step=self.current_epoch)

        # Keep in mind: We clip the gradients from the last backward pass of the training loop at
        # current epoch here, which are already clipped during training
        # so this line has the sole purpose of getting the total_norm from the last gradients
        total_norm = nn.utils.clip_grad_norm_(self.current_model.parameters(),
                                              self.args.gradient_clipping)
        self.writer.add_scalar(tag='total_grad_norm', scalar_value=total_norm, global_step=self.current_epoch)

        if not self.args.no_printing:
            print("Epoch {}/{}: Loss {:.6f}, L2-norm A {:.1f}".format(
                str(self.current_epoch).zfill(4), self.args.n_epochs,
                float(loss), float(max_eig_A)))
        

    def save_metrics(self):
        """Evaluate metrics on a subset of the training data, then save them to tensorboard"""
        import main_eval
        main_eval.DATA_GENERATED = None
        #gaussian metrics
        for metric in self.args.gaussian_metrics:
            data_batch= self.data_set.test_tensor()
         #   data_batch = utils.read_data(self.args.gaussian_data_path)
            data_subset = data_batch[:, :self.args.dim_g]           
            print(data_subset.shape)
            metric_value = main_eval.eval_model_on_data_with_metric(model=self.current_model, data=data_subset, metric=metric)
            metric_value = metric_value[0]  # only take first metric value, e.g. mse 1 step ahead, and klz mc
            tag = 'metric_{}'.format(metric)
            self.writer.add_scalar(tag=tag, scalar_value=metric_value, global_step=self.current_epoch)
            if not self.args.no_printing:
                print("{}: {:.3f}".format(metric, metric_value))

    def split_dataset(self, data):

        # split input and target according to dimensionality of multimodal observations. For non-observed modalities, dimensions are set to zero
        data_g = data[:, :self.args.dim_g]
        data_o = data[:, self.args.dim_g:self.args.dim_g + self.args.dim_o]
        data_p = data[:, self.args.dim_g + self.args.dim_o:]

        return (data_g, data_o, data_p)

    def save_parameters(self):
        '''
        Save all parameters to tensorboard.
        '''
        par_dict = {**dict(self.current_model.state_dict())}
        par_to_tb(par_dict, epoch=self.current_epoch, writer=self.writer)

    def save_prediction(self):
        '''
        Save a GT-Prediction plot to tensorboard.
        '''
        data = self.data_set.test_tensor()
        
       # data_g, data_o, data_p=self.split_dataset(data)

        if self.args.use_gaussian == 1:
            modality = 1
            self.current_model.plot_prediction(data, modality)
            save_plot_to_tb(self.writer, text='Gaussian observations, GT vs Prediction',
                            global_step=self.current_epoch)
        if self.args.use_ordinal == 1:
            modality = 2
            self.current_model.plot_prediction(data, modality)
            save_plot_to_tb(self.writer, text='Ordinal observations, GT vs Prediction',
                            global_step=self.current_epoch)
        if self.args.use_ordinal == 1:
            modality = 3
            self.current_model.plot_prediction(data, modality)
            save_plot_to_tb(self.writer, text='Poisson observations, GT vs Prediction',
                            global_step=self.current_epoch)

        plt.close()

    def save_reconstruction(self):
        '''
        Save a reconstruction plot to tensorboard.
        '''
        data = self.data_set.test_tensor()
        self.current_model.plot_reconstruction(data)
        save_plot_to_tb(self.writer, text='Reconstructed Data',
                        global_step=self.current_epoch)
        plt.close()

    def save_latent_vs_encoded(self):
        '''
        Save a reconstruction plot to tensorboard.
        '''
        data =  self.data_set.test_tensor()
        self.current_model.plot_latent_vs_encoded(data)
        save_plot_to_tb(self.writer, text='Latent vs Encoded',
                        global_step=self.current_epoch)
        plt.close()

    def save_simulated(self):
        T = 500
        data = self.data_set.test_tensor()[:T]

        if self.args.use_gaussian == 1:
            modality = 1
            label = "Gaussian data simulated"

            self.current_model.plot_simulated(data, T, modality)
            figure = plt.gcf()
            save_figure_to_tb(figure, self.writer, text=label,
                              global_step=self.current_epoch)
            plt.close()
            
            
            self.current_model.plot_simulated_3D(data, time_steps=3000)
            figure = plt.gcf()
            save_figure_to_tb(figure, self.writer, text=label+ ' 3D',
                              global_step=self.current_epoch)
            plt.close()
            
            

            self.current_model.plot_obs_simulated(data, modality)
            save_plot_to_tb(self.writer, text=label + ' against data'.format(0),
                            global_step=self.current_epoch)
            plt.close()

        if self.args.use_ordinal == 1:
            modality = 2
            label = "Ordinal data simulated"
            self.current_model.plot_simulated(data, T, modality)
            figure = plt.gcf()
            save_figure_to_tb(figure, self.writer, text=label,
                              global_step=self.current_epoch)
            plt.close()

            self.current_model.plot_obs_simulated(data, modality)
            save_plot_to_tb(self.writer, text=label + ' against data'.format(0),
                            global_step=self.current_epoch)
            plt.close()

        if self.args.use_poisson == 1:
            modality = 3
            label = "Poisson data simulated"

            self.current_model.plot_simulated(data, T, modality)
            figure = plt.gcf()
            save_figure_to_tb(figure, self.writer, text=label,
                              global_step=self.current_epoch)
            plt.close()

            self.current_model.plot_obs_simulated(data, modality)
            save_plot_to_tb(self.writer, text=label + ' against data'.format(0),
                            global_step=self.current_epoch)
            plt.close()

    def get_min_max(self, values):
        list_ = list(values)
        indices = [i for i in range(len(list_)) if list_[i] == 1]
        return min(indices), max(indices)


    def plot_as_image(self):
        time_steps = 1000
        data_generated = self.current_model.gen_model.get_observed_time_series(time_steps=time_steps + 1000)
        data_generated = data_generated[1000:1000 + time_steps]
        data_ground_truth = self.data_set.data[0][:time_steps]
        data_generated = data_generated[:(data_ground_truth.shape[0])]  # in case trial data is shorter than time_steps

        plt.subplot(121)
        plt.title('ground truth')
        plt.imshow(data_ground_truth, aspect=0.05, origin='lower', interpolation='none', cmap='Blues_r')
        plt.xlabel('observations')
        plt.ylabel('time steps')
        plt.subplot(122)
        plt.title('simulated')
        plt.imshow(data_generated, aspect=0.05, origin='lower', interpolation='none', cmap='Blues_r')
        plt.ylabel('time steps')
        plt.xlabel('observations')

        save_plot_to_tb(self.writer, text='curve image'.format(), global_step=self.current_epoch)


def initial_condition_trial_to_tb(gen_model, epoch, writer):
    for i in range(len(gen_model.z0)):
        trial_z0 = gen_model.z0[i].unsqueeze(0)
        x = gen_model.get_observed_time_series(800, trial_z0)  # TODO magic length of trial
        plt.figure()
        plt.title('trial {}'.format(i))
        plt.plot(x)
        figure = plt.gcf()
        save_figure_to_tb(figure, writer, text='curve_trial{}'.format(i + 1), global_step=epoch)


def data_plot(x):
    x = x.cpu().detach().numpy()
    plt.ylim(top=4, bottom=-4)
    plt.xlim(right=4, left=-4)
    plt.scatter(x[:, 0], x[:, -1], s=3)
    plt.title('{} time steps'.format(len(x)))
    return plt.gcf()


def save_plot_to_tb(writer, text, global_step=None):
    figure = plt.gcf()
    save_figure_to_tb(figure, writer, text, global_step)


def save_figure_to_tb(figure, writer, text, global_step=None):
    image = tb_utils.figure_to_image(figure)
    writer.add_image(text, image, global_step=global_step)


def save_data_to_tb(data, writer, text, global_step=None):
    if type(data) is list:
        for i in range(len(data)):
            plt.figure()
            plt.title('trial {}'.format(i))
            plt.plot(data[i])
            figure = plt.gcf()
            save_figure_to_tb(figure=figure, writer=writer, text='curve_trial{}_data'.format(i), global_step=None)
    else:
        plt.figure()
        plt.plot(data)
        figure = plt.gcf()
        # figure = data_plot(data)
        save_figure_to_tb(figure=figure, writer=writer, text=text, global_step=global_step)


def par_to_tb(par_dict, epoch, writer):
    for key in par_dict.keys():
        par = par_dict[key].cpu()
        
        if len(par.shape) > 2:
          pass
        else:
          if len(par.shape) == 1:
              par = np.expand_dims(par, 1)
          # tranpose weight matrix of nn.Linear
          # to get true weight (Wx instead of xW)
          elif '.weight' in key:
              par = par.T
          par_to_image(par, par_name=key)
          save_plot_to_tb(writer, text='par_{}'.format(key), global_step=epoch)
          plt.close()


def par_to_image(par, par_name):
    plt.figure()
    # plt.title(par_name)
    sns.set_context('paper', font_scale=1.)
    sns.set_style('white')
    max_dim = max(par.shape)
    use_annot = not (max_dim > 20)
    sns.heatmap(data=par, annot=use_annot, linewidths=float(use_annot), cmap='Blues_r', square=True, fmt='.2f',
                yticklabels=False, xticklabels=False)
