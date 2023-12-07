import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from bptt import helpers as h
import utils as u
import math

class EncoderMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EncoderMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Linear(in_dim // 2, in_dim // 2),
            nn.LayerNorm(in_dim // 2, elementwise_affine=False),
            nn.Mish(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.Linear(in_dim // 4, in_dim // 4),
            nn.LayerNorm(in_dim // 4, elementwise_affine=False),
            nn.Mish(),
            nn.Linear(in_dim // 4, out_dim)
        )
    
    def forward(self, X):
        return self.layers(X)
        

class EncoderLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EncoderLinear, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
    
    
    def forward(self, X):
        entropy=tc.zeros(1)
        return self.layers(X), entropy
        
        
class EncoderIdentity(nn.Module):
    def __init__(self):
        super(EncoderIdentity, self).__init__()
        self.layers = nn.Sequential(
            nn.Identity()
        )
    
    def forward(self, X):
        entropy=tc.zeros(1)
        return self.layers(X), entropy    



        
        
class StackedConvolutions(nn.Module):
    def __init__(self, dim_x, dim_z, sample_rec, kernel_size=[11, 7, 5, 3], stride=[1], padding=[5,3,2,1], num_convs=(3, 1)):
        super(StackedConvolutions, self).__init__()

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.sample_rec=sample_rec

        assert (len(kernel_size) == num_convs[0] + num_convs[1])
        assert (len(kernel_size) == len(stride) or len(stride) == 1)
        assert (len(kernel_size) == len(padding) or len(padding) == 1)

        if len(stride) == 1:
            stride *= len(kernel_size)
        if len(padding) == 1:
            padding *= len(kernel_size)

        mean_convs = []
        for i in range(num_convs[0]):
            mean_convs.append(nn.Conv1d(
                in_channels=dim_x if i == 0 else dim_z,
                out_channels=dim_z,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i]
            ))
        self.mean_conv = nn.Sequential(*mean_convs)

        logvar_convs = []
        for i in range(num_convs[1]):
            logvar_convs.append(nn.Conv1d(
                in_channels=dim_x if i == 0 else dim_z,
                out_channels=dim_z,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i]
            ))
        self.logvar_conv = nn.Sequential(*logvar_convs)

    def to_batch(self, x):
        x = x.T
        x = x.unsqueeze(0)
        return x

    def from_batch(self, x):
        x = x.squeeze(0)
        x = x.T
        return x

    def get_sample(self, mean, log_sqrt_var):
        sample = mean + tc.exp(log_sqrt_var) * tc.randn(mean.shape[0], self.dim_z)
        return sample

    def get_entropy(self, log_sqrt_var):
        entropy = tc.sum(log_sqrt_var) / log_sqrt_var.shape[0]
        return entropy

    def forward(self, x):
        
        x = self.to_batch(x)    

        mean = self.mean_conv(x)
        log_sqrt_var = self.logvar_conv(x)

        mean = self.from_batch(mean)
        log_sqrt_var = self.from_batch(log_sqrt_var)
        
        sample = self.get_sample(mean, log_sqrt_var)
        


        if self.sample_rec==1:
          entropy = self.get_entropy(log_sqrt_var)

          return sample, entropy
        else:
          entropy=tc.zeros(1)
          return mean, entropy


#train a modality specific CNN and combine via MoE or PoE
class POE_StackedConvolutions(nn.Module):
    def __init__(self, dim_g, dim_o, dim_p, dim_z, sample_rec, kernel_size=[11, 7, 5, 3], stride=[1], padding=[5,3,2,1], num_convs=(3, 1)):
        super(POE_StackedConvolutions, self).__init__()
        self.dim_g = dim_g
        self.dim_o = dim_o
        self.dim_p = dim_p
        self.dim_z = dim_z
        self.sample_rec = sample_rec
        
        # Create three separate StackedConvolutions for each modality
        self.encoder_g = StackedConvolutions(dim_g, dim_z, sample_rec, kernel_size, stride, padding, num_convs)
        self.encoder_o = StackedConvolutions(dim_o, dim_z, sample_rec, kernel_size, stride, padding, num_convs)
        self.encoder_p = StackedConvolutions(dim_p, dim_z, sample_rec, kernel_size, stride, padding, num_convs)

    def split_modalities(self, x):
        # Split the concatenated array into modalities modalities
        g = x[:, :self.dim_g]
        o = x[:, self.dim_g:self.dim_g+self.dim_o]
        p = x[:, self.dim_g+self.dim_o:]
        return g, o, p

    def forward(self, x):
    
        x = self.encoder_g.to_batch(x)


        g, o, p = self.split_modalities(x)
        
        epsilon = 1e-7
        # Process each modality separately
        mean_g, log_sqrt_var_g = self.encoder_g.get_mean_and_log_var(g)
        mean_o, log_sqrt_var_o = self.encoder_o.get_mean_and_log_var(o)
        mean_p, log_sqrt_var_p = self.encoder_p.get_mean_and_log_var(p)
        
        #select between mixture of experts and product of experts formulation
        MOE=True
        POE=False
        
        if MOE:
          weights_g = 1/3
          weights_o = 1/3
          weights_p = 1/3
  

          mean_combined = weights_g * mean_g + weights_o * mean_o + weights_p * mean_p
          var_combined = (weights_g * (tc.exp(2 * log_sqrt_var_g) + (mean_g - mean_combined)**2) +
                          weights_o * (tc.exp(2 * log_sqrt_var_o) + (mean_o - mean_combined)**2) +
                          weights_p * (tc.exp(2 * log_sqrt_var_p) + (mean_p - mean_combined)**2))
  
          log_sqrt_var_combined = 0.5 * tc.log(var_combined)
                
        if POE:
          precision_g = -2 * log_sqrt_var_g
          precision_o = -2 * log_sqrt_var_o
          precision_p = -2 * log_sqrt_var_p
          
          # Compute the combined precision and mean
          precision_combined = precision_g.exp() + precision_o.exp() + precision_p.exp()+ 1e-7
          mean_combined = (mean_g * precision_g.exp() + mean_o * precision_o.exp() + mean_p * precision_p.exp()) / (precision_combined)
          
          # Convert combined precision back to log variance
          log_var_combined = -tc.log(precision_combined + 1e-7)
          log_sqrt_var_combined = 0.5 * log_var_combined
          
          
        # Get the combined sample
        sample_combined = self.encoder_g.get_sample(mean_combined, log_sqrt_var_combined)
        entropy_combined = self.encoder_g.get_entropy(log_sqrt_var_combined)
        

        # Return the combined sample and entropy
        if self.sample_rec == 1:
            return sample_combined, entropy_combined
        else:
            return mean_combined, entropy_combined



class Encoder_Amortized_RNN(nn.Module):
    def __init__(self, dim_x, dim_z, time_length=20):
        super(Encoder_Amortized_RNN, self).__init__()

        # Dimensionalities
        self.d_x = dim_x
        self.d_z = dim_z
        self.d_hidden = dim_z
        self.time_length = time_length

        # Linear layers for mean and log-variance
        self.mean = nn.Linear(self.d_hidden, self.d_z, bias=False)
        self.logvar = nn.Linear(self.d_hidden, self.d_z, bias=True)

        # RNN layer
        self.rnn = nn.RNN(self.d_x, self.d_hidden, batch_first=True)

    #split sequence into smaller subsequences to speed up runtime and avoid gradient problems
    def preprocess_input(self, x):
        T = x.shape[0]
        x = x.unsqueeze(0)
        
        if T % self.time_length == 0:
            minibatches = T // self.time_length
            data = x.view(minibatches, self.time_length, self.d_x)
        else:
            data = x.view(1, -1, self.d_x)
            minibatches = 1

        return data, minibatches

    def get_rnn_parameters(self):
        b1, b2, hidden, xh = self.rnn.parameters()
        return xh, hidden, b2, b1

    def forward(self, x):
      data, minibatches = self.preprocess_input(x)
  
      # Initialize hidden state
      hidden = tc.zeros(1, minibatches, self.d_hidden)
  
      # Pass the entire sequence through the RNN
      out, _ = self.rnn(data, hidden)
  
      # Reshape RNN output for the linear layers
      out_reshaped = out.contiguous().view(-1, self.d_hidden)
  
      # Compute mu and log_stddev for the entire sequence
      mu = self.mean(out_reshaped)
      log_stddev = self.logvar(out_reshaped)
  
      # Sampling for the entire sequence
      sample_seq = mu + tc.exp(log_stddev) * tc.randn(mu.size(), device=x.device)
  
      # Compute entropy
      entropy = tc.sum(log_stddev) / (data.shape[0] * self.time_length)
  
      return sample_seq.view(minibatches, self.time_length, self.d_z), entropy
        
        
class TransformerEncoder(nn.Module):
    def __init__(self, dim_x, dim_z, sample_rec, sequence_length, num_layers=3):
        super(TransformerEncoder, self).__init__()

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.sample_rec = sample_rec
        self.sequence_length=sequence_length

        self.embedding = nn.Linear(dim_x, dim_z)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_z, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mean_projection = nn.Linear(dim_z, dim_z)
        self.logvar_projection = nn.Linear(dim_z, dim_z)

    def get_sample(self, mean, log_sqrt_var):
        sample = mean + tc.exp(log_sqrt_var) * tc.randn(mean.shape[0], self.dim_z)
        return sample

    def get_entropy(self, log_sqrt_var):
        entropy = tc.sum(log_sqrt_var) / log_sqrt_var.shape[0]
        return entropy
        
    def positional_encoding(self, seq_len, d_model):
        """
        Compute positional encodings for input sequence.
        """
        pe = tc.zeros(seq_len, d_model)
        position = tc.arange(0, seq_len, dtype=tc.float).unsqueeze(1)
        div_term = tc.exp(tc.arange(0, d_model, 2).float() * -(tc.log(tc.tensor(10000.0)) / d_model))
        pe[:, 0::2] = tc.sin(position * div_term)
        pe[:, 1::2] = tc.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe
        

    def forward(self, x):
        T = x.shape[0]
        if T % self.sequence_length == 0:  # Check if T is divisible by self.sequence_length
          
          minibatches = T // self.sequence_length
          x = x.view(minibatches, self.sequence_length, self.dim_x)
        else:
          x = x.view(1, -1, self.dim_x)

            
        x = self.embedding(x)

        # Add positional encoding
        pe = self.positional_encoding(x.size(0), self.dim_z)
        x = x + pe
        
        x = x.permute(1, 0, 2)  # Transformer expects input as (length, batch, features)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2) 

        x=x.reshape(1,-1,self.dim_z).squeeze(0)

        mean = self.mean_projection(x)
        log_sqrt_var = self.logvar_projection(x)

        sample = self.get_sample(mean, log_sqrt_var)

        if self.sample_rec == 1:
            entropy = self.get_entropy(log_sqrt_var)
            return sample, entropy
        else:
            entropy = tc.zeros(1)
            return mean, entropy 





#encoder model from Kramer, Bomer et al (ICML 2022)
class ProductOfGaussians(nn.Module):
    """Product of gaussians used as recognition model.
    Arguments:
        dim_x (int): dimensionality of observation space
        dim_x (int): dimensionality of latent space
        dim_hidden (int): number of neurons of hidden layer
        rec_dict (dict):
            * A (tc.tensor): shape (dim_z, dim_z)
            * QinvChol (tc.tensor): shape (dim_z, dim_z)
            * Q0invChol (tc.tensor): shape (dim_z, dim_z)
    """

    def __init__(self, dim_x, dim_z, dim_hidden, X_true, batch_size, rec_dict=None):
        super(ProductOfGaussians, self).__init__()

        # TODO: Initialize the weights of the NN layers to have 0 mean wrt training data

        """ 
        - the weight matrices are stored as their own transpose, i.e. w_in has shape (dim_hidden, dim_x)
        - w_in has shape (dim_hidden, dim_x)
        - w_in_out has shape (batch_size, dim_hidden)
        NOTE: Even though the initialization of the hidden layers doesnt seem to make sense, 
        the results obtained with this set-up are very good.
        (results in folder 'lorentz_relu_meanCenteredInit_deeperRecModel')
        However, maybe the initialization of the hidden layers like this doesnt even have that big
        of an impact and only the initialization of the input layer is of importance."""

        self.initInstanceVariables(batch_size, dim_hidden, dim_x, dim_z, rec_dict)
        self.init_encoder(X_true[0])

        if rec_dict is not None:
            self.load_state_dict(rec_dict, strict=False)

    def initInstanceVariables(self, batch_size, dim_hidden, dim_x, dim_z, rec_dict):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_hidden = dim_hidden
        self.rec_dict = rec_dict
        self.batch_size = 50
        self.A = nn.Parameter(0.9 * tc.rand(dim_z, dim_z), requires_grad=True)
        self.QinvChol = nn.Parameter(tc.rand(dim_z, dim_z), requires_grad=True)
        self.Q0invChol = nn.Parameter(tc.rand(dim_z, dim_z), requires_grad=True)

        self.useRecognitionModelClipping = False

        #for testing only
        self.last_AA = None
        self.last_BB = None
        self.last_mean = None
        self.last_cov = None
        
    def init_weights_and_bias(self, regularization_weights, layer, firstLayer=True):
    
      nn.init.orthogonal_(layer.weight)
      
      if firstLayer:
          layer.weight = nn.Parameter(
              (layer.weight.t() / tc.matmul(layer.weight, regularization_weights.t()).std(dim=1)).t())
      else:
          layer.weight = nn.Parameter(
              (layer.weight.t() / tc.matmul(layer.weight, regularization_weights).std(dim=1)).t())
  
      if firstLayer:
          layer.bias = nn.Parameter(-(tc.matmul(layer.weight, regularization_weights.t())).mean(dim=1))
      else:
          layer.bias = nn.Parameter(-(tc.matmul(layer.weight, regularization_weights)).mean(dim=1))

    def init_encoder(self, x_true):

        # TODO: does it make difference if we initiate the weights after first layer?

        """Encoder for mean & covariance of numerical data"""
        self.fc_mean_in = nn.Linear(self.dim_x, self.dim_hidden)
        self.init_weights_and_bias(x_true, self.fc_mean_in, firstLayer=True)
        self.fc_mean_h1 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_mean_h3 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_mean_out = nn.Linear(self.dim_hidden, self.dim_z)

        self.fc_cov_in = nn.Linear(self.dim_x, self.dim_hidden)
        self.init_weights_and_bias(x_true, self.fc_cov_in, firstLayer=True)
        self.fc_cov_h1 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_cov_h3 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_cov_out = nn.Linear(self.dim_hidden, self.dim_z * self.dim_z)

    def activateRecognitionModelClipping(self):
        if self.useRecognitionModelClipping is True:
            print("Userwarning: activating RecognitionModelClipping allthough it is True allready")
        self.useRecognitionModelClipping = True

    def deactivateRecognitionModelClipping(self):
        if self.useRecognitionModelClipping is False:
            print("Userwarning: deactivating RecognitionModelClipping allthough it is False allready")
        self.useRecognitionModelClipping = False

    def encode_mean(self, x):
        x = x.view(-1, self.dim_x)
        x_mean = F.relu(self.fc_mean_in(x))
        x_mean = F.relu(self.fc_mean_h1(x_mean))
        x_mean = F.relu(self.fc_mean_h3(x_mean))
        x_mean = self.fc_mean_out(x_mean)
        return x_mean

    def encode_cov(self, x):
        x = x.view(-1, self.dim_x)
        x_cov = F.relu(self.fc_cov_in(x))
        x_cov = F.relu(self.fc_cov_h1(x_cov))
        x_cov = F.relu(self.fc_cov_h3(x_cov))
        x_cov = self.fc_cov_out(x_cov)
        return x_cov

    def forward(self, x):
    
        """x = numerical data"""
        # cov is actually not the covariance matrix but instead a part of the Matrix used in the Kalman filter to
        # calculate the cholesky decomposition and hence the correct covariance matrix


        batch_size=x.shape[0]
        # shape (BATCH_SIZE, dim_z * dim_z)
        mean = self.encode_mean(x)
        cov = self.encode_cov(x)
        

        self.AA, BB, lambdaMu = h.calculate_cholesky_factors(mean, cov, self.dim_z, self.QinvChol,
                                                             self.Q0invChol, self.A, batch_size)

        # compute cholesky decomposition
        # the_chol[0] has shape (BATCH_SIZE, dim_z, dim_z)
        # the_chol[1] has shape ((BATCH_SIZE-1), dim_z, dim_z)

        self.the_chol = h.blk_tridag_chol(self.AA, BB)
        ib = h.blk_chol_inv(self.the_chol[0], self.the_chol[1], lambdaMu, lower=True,
                                transpose=False)
        self.mu_z = h.blk_chol_inv(self.the_chol[0], self.the_chol[1], ib, lower=False,
                                   transpose=True)
        self.ln_determinant = -2 * tc.log(tc.diagonal(self.the_chol[0], dim1=-2, dim2=-1)).sum()
        
        
        
        sample=self.getSample(batch_size=batch_size, noise=True)
        entropy=self.evalEntropy()/batch_size
        
        return sample, entropy
        

    def getSample(self, batch_size, noise=True):
        """
        Reparameterization to get samples of z.
        """
        
        normSamps = tc.randn(batch_size, self.dim_z)
        
        R = h.blk_chol_inv(self.the_chol[0], self.the_chol[1], normSamps, lower=False, transpose=True)

        if noise:
            return self.mu_z + R
        else:
            return self.mu_z

    def evalEntropy(self, alpha=0.5):
        """
        Differential entropy of a gaussian distributed random variable can be calculated via determinant of the
        covariance matrix.
        """
        a = 1

        entropy = -self.calcEntropy(self.batch_size, a, self.ln_determinant)/self.batch_size

        return entropy

    def calcEntropy(self, T, a, ln_determinant):
        return a * (ln_determinant / 2 + self.dim_z * T / 2.0 * (1 + tc.log(tc.tensor(2 * math.pi))))

    def getHessian(self):
        choleskyFactor = h.construct_bidiagonal(self.the_chol[0], self.the_chol[1])
        hessian = choleskyFactor #@ choleskyFactor.t()
        return hessian

    def getFullCholeskyFactor(self, x):
        oldBatchSize = self.batch_size
        self.batch_size = x.shape[0]
        self.forward(x)
        self.batch_size = oldBatchSize
        return self.the_chol[0], self.the_chol[1]

    def setBatchSize(self, newBatchSize):
        self.batch_size = newBatchSize