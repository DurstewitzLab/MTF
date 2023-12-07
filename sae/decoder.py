import torch as tc
import torch.nn as nn
import numpy as np

class DecoderMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DecoderMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim // 2),
          #  nn.Linear(out_dim // 4, out_dim // 4),
            nn.LayerNorm(out_dim // 2, elementwise_affine=False),
            nn.Mish(),
         #   nn.Linear(out_dim // 4, out_dim // 2),
            nn.Linear(out_dim // 2, out_dim // 2),
            nn.LayerNorm(out_dim // 2, elementwise_affine=False),
            nn.Mish(),
            nn.Linear(out_dim // 2, out_dim)
        )
    
    def forward(self, X):
        return self.layers(X)
        
        
class Decoder_LinearObservation(nn.Module):
    def __init__(self, in_dim, out_dim, estimate_noise_covariances):
        super(Decoder_LinearObservation, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
        
        
        if estimate_noise_covariances:
          self.R_x = nn.Parameter(tc.ones(out_dim)*1, requires_grad=True)
        else:
         self.R_x = nn.Parameter(tc.ones(out_dim)*1, requires_grad=False)

    def forward(self, X):
        return self.layers(X)
        
        
    def log_likelihood(self,x, z):
    
        def mahalonobis_distance(residual, matrix):
          distance=0
          for i in range(residual.shape[0]):
            distance+=- 0.5 * (residual[i].t() @ residual[i] * tc.inverse(tc.diag(matrix ** 2))).sum()
            return distance
              
        def log_det(diagonal_matrix):
            return - tc.log(diagonal_matrix).sum()
            
        LL_x = mahalonobis_distance(x -  self.layers(z), self.R_x)
        
        T=x.shape[1]
        
        return LL_x+log_det(self.R_x)*(T-1)/2
        
        

class DecoderIdentity(nn.Module):
    def __init__(self, in_dim, out_dim, estimate_noise_covariances):
        super(DecoderIdentity, self).__init__()
        self.layers = nn.Sequential(
            nn.Identity()
        )
        if estimate_noise_covariances:
          self.R_x = nn.Parameter(tc.ones(out_dim)*1, requires_grad=True)
        else:
         self.R_x = nn.Parameter(tc.ones(out_dim)*1, requires_grad=False)
    
        self.out_dim=out_dim
    
    def forward(self, X):
    
        if len(X.shape)==3:
          return self.layers(X[:,:,:self.out_dim])
          
        else:
          return self.layers(X[:,:self.out_dim])
          
        
    def log_likelihood(self,x, z):
    
        def mahalonobis_distance(residual, matrix):
          distance=0
          for i in range(residual.shape[0]):
            distance+=- 0.5 * (residual[i].t() @ residual[i] * tc.inverse(tc.diag(matrix ** 2))).sum()
            return distance
            
            
        def log_det(diagonal_matrix):
            return - tc.log(diagonal_matrix).sum()
        dim_x=x.shape[-1]
        LL_x = mahalonobis_distance(x -  self.layers(z[:,:,:dim_x]), self.R_x)
        
        T=x.shape[1]
        
        return LL_x+log_det(self.R_x)*(T-1)/2




class DecoderCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DecoderCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(1, 8, 2, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.Mish(),
            nn.Conv1d(8, 8, 3, 1, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.Mish(),
            nn.ConvTranspose1d(8, 4, 2, 2, bias=False),
            nn.BatchNorm1d(4),
            nn.Mish(),
            nn.Conv1d(4, 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(4),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(4 * in_dim * 4, out_dim)
        )
    
    def forward(self, X):
        return self.layers(X.unsqueeze(1))


class Decoder_cumulative_link(nn.Module):
    def __init__(self, dim_z, dim_x, dim_c):
        super().__init__()

        self.dx = dim_x
        self.dz = dim_z
        self.dc = dim_c

        self.beta_0 = nn.Parameter(tc.randn(self.dx, self.dc - 1), requires_grad=True)
        self.beta = nn.Parameter(tc.randn(self.dx, self.dz), requires_grad=True)

    def forward(self, z):
        probabilities = self.get_category_probabilities(z)
        x = tc.argmax(probabilities, dim=2) + 1
        x = x.float()
        x[tc.any(z.isnan(), dim=1), :] = np.nan

        return x

    def get_category_probabilities(self, z):
        linear_predictor = self.calculate_linear_predictor(z)
        cumulative_probabilities = self.inverse_logit_link_function(linear_predictor)
        probabilities = self.calculate_probabilities_from_cumulative_probabilities(cumulative_probabilities)

        probabilities[(probabilities < 10e-5) & (probabilities > -10e-5)] = 0.

        return probabilities

    def calculate_linear_predictor(self, z):
        lin_pred = (self.beta @ z.T).T
        beta_0 = self.reparameterize_beta_0()

        lin_pred = beta_0.unsqueeze(0) - lin_pred.unsqueeze(2)

        return lin_pred

    def inverse_logit_link_function(self, linear_predictor):
        cumul_prob = tc.sigmoid(linear_predictor)

        # Add cumulative probability for the last category (always one)
        pad = tc.nn.ConstantPad1d((0, 1, 0, 0), 1.0)
        cumul_prob = pad(cumul_prob)

        return cumul_prob

    def calculate_probabilities_from_cumulative_probabilities(self, cumul_prob):
        prob = tc.diff(cumul_prob, prepend=tc.zeros((cumul_prob.shape[0], cumul_prob.shape[1], 1)), dim=2)
        return prob

    def reparameterize_beta_0(self):
        # Re-parameterize beta_0 such that consecutive parameters are strongly ordered
        # This is required to ensure we have always positive non zero probabilities

        beta_0_tilde = tc.cumsum(tc.exp(self.beta_0), dim=1)
        beta_0_tilde = beta_0_tilde - tc.exp(self.beta_0[:, 0]).unsqueeze(1) + self.beta_0[:, 0].unsqueeze(1)
        # beta_0_tilde[:, 0] = 0.

        return beta_0_tilde

    def cumulative_link_log_likelihood(self, x, z):
        # Calculate cumulative probabilities

        probabilities = self.get_category_probabilities(z)

        # Ignore 'nan' data points
        probabilities = probabilities[~tc.isnan(x)]

        # For each i and t find the correct categorical indices
        categorical_indices = x[~tc.isnan(x)].long() - 1
        if -1 in categorical_indices:
          categorical_indices[categorical_indices==-1] = 1
        
        categorical_indices = categorical_indices.unsqueeze(-1)

        probabilities = tc.gather(probabilities, dim=1, index=categorical_indices)

        # If probabilities are too small set them to 10e-10 to avoid exploding log
        probabilities[probabilities < 10e-10] = 10e-10

        ll_cumulative_link = tc.sum(tc.log(probabilities))

        assert tc.isfinite(ll_cumulative_link)

        return ll_cumulative_link
        
        
        
class Decoder_Poisson(nn.Module):
    def __init__(self, dz, dq):
        super(Decoder_Poisson, self).__init__()

        self.dz = dz
        #dimension of Poisson data
        self.dq = dq

        # Only gamma is needed for a simple Poisson regression
        self.gamma = nn.Parameter(tc.randn((self.dq, self.dz)) * 0.001, requires_grad=True)
        
    def forward(self, z):
        gamma_z = tc.exp(z @ self.gamma.T)
        gamma_z = tc.nan_to_num(gamma_z, nan=0.0001, posinf=None, neginf=None)
        data = tc.poisson(tc.abs(gamma_z) + 0.01)
        return data
                    
    def log_likelihood(self, q_p, z_p):
        q = q_p.permute(0, 2, 1)
        z = z_p.permute(0, 2, 1)
        gamma_z = self.gamma @ z.T

        log_q_factorial = tc.sum(-tc.lgamma(q[~tc.isnan(q)] + 1))
        q_term = q.T * gamma_z - tc.exp(gamma_z)

        q_term_sum = tc.sum(q_term)

        log_lik = log_q_factorial + q_term_sum

        return log_lik
        
        
class Decoder_zip(nn.Module):
    def __init__(self, dz, dq):
        super(Decoder_zip, self).__init__()

        self.dz = dz
        self.dq = dq

        self.beta = nn.Parameter(tc.randn((self.dq, self.dz))*0.01, requires_grad=True)
        self.gamma = nn.Parameter(tc.randn((self.dq, self.dz))*0.05, requires_grad=True)
        
    def forward(self, z):
        pi = tc.exp(z @ self.beta.T) / (1 + tc.exp(z @ self.beta.T))
        gamma_z = tc.exp(z @ self.gamma.T)
        gamma_z = tc.nan_to_num(gamma_z, nan=0.0001, posinf=None, neginf=None)
        poisson = tc.poisson(tc.abs(gamma_z) + 0.01)        
        masking = (tc.bernoulli(1-pi) > 0).float()  # Ensure masking is a float for multiplication

        data = masking * poisson  # Include zero inflation using masking

        return data
                    

    def log_likelihood(self, q_p, z_p):
        q = q_p.permute(0, 2, 1)
        z = z_p.permute(0, 2, 1)

        beta_z = self.beta @ z.T
        gamma_z = tc.nn.functional.softplus(self.gamma @ z.T)  # Ensure positive rate

        # Log probability of q being 0
        log_pi_z = beta_z - tc.log(1 + tc.exp(beta_z))
        q_equal_0_term = tc.sum(log_pi_z[q.T == 0])
        
        # Log probability of q under Poisson when q > 0
        q_larger_0_term = q.T * gamma_z - tc.exp(gamma_z) - tc.lgamma(q.T + 1)
        q_larger_0_term = tc.sum(q_larger_0_term[q.T > 0])
        
        # Log probability of q not being 0
        log_1_minus_pi_z = -tc.log(1 + tc.exp(beta_z))
        q_larger_0_term += tc.sum(log_1_minus_pi_z[q.T > 0])

        log_lik = q_equal_0_term + q_larger_0_term

        return log_lik
    

class Decoder_NegativeBinomial(nn.Module):
    def __init__(self, dz, dq):
        super(Decoder_NegativeBinomial, self).__init__()

        self.dz = dz  # Dimension of latent space
        self.dq = dq  # Dimension of Poisson data

        # Parameters for Negative Binomial regression: mean (gamma) and dispersion (phi)
        self.gamma = nn.Parameter(tc.randn((self.dq, self.dz)) * 0.001, requires_grad=True)
        # Dispersion parameter phi needs to be positive, initialize with small positive values
        self.phi = nn.Parameter(tc.rand(self.dq) * 0.001 + 0.1, requires_grad=True)

    def forward(self, z):
        
        # Compute the mean rate (mu) of the Negative Binomial distribution
        mu = tc.exp(z @ self.gamma.T)
        mu = tc.nan_to_num(mu, nan=0.0001, posinf=None, neginf=None)

        # The dispersion parameter (phi) is used to model the variance
        # phi is broadcasted to match the batch and sequence dimensions of mu
        phi_expanded = self.phi.unsqueeze(0).expand_as(mu)

        # Generate data from the Negative Binomial distribution
        data = tc.distributions.NegativeBinomial(mu, phi_expanded).sample()
        return data

    def log_likelihood(self, q_p, z_p):
        q = q_p.permute(0, 2, 1)  # Permute to match the [batch_size, sequence_length, dq]
        z = z_p.permute(0, 2, 1)
        

        z = z.transpose(1, 2)
        
        gamma_z = self.gamma @ z.transpose(1, 2)
        mu = tc.exp(tc.matmul(z, self.gamma.T))

        # The log likelihood of the Negative Binomial distribution
        phi_expanded = self.phi.unsqueeze(0).expand_as(mu)
        q_transposed = q.transpose(1, 2).int() 
        
        test_dist = tc.distributions.NegativeBinomial(mu, phi_expanded)
        log_lik = test_dist.log_prob(q_transposed).sum()
       

        return log_lik


class DecoderCategorical(nn.Module):
    def __init__(self, dz, dc):
        super(DecoderCategorical, self).__init__()
        
        self.dim_c=dc
        self.dim_z=dz
        self.beta = nn.Parameter(tc.rand(self.dim_c, self.dim_z)*0.01, requires_grad=True)
    
    def forward(self, Z):
        T,dz=Z.shape
        cats=tc.zeros(Z.shape[0], self.dim_c)
        probs=tc.zeros(Z.shape[0], self.dim_c)
        
        for ind in range(T):
          norm=1+tc.sum(tc.exp(self.beta[:-1] @ Z[ind]))
          probs[ind,:-1]=tc.exp(self.beta[:-1]@Z[ind])/norm
          probs[ind,-1]=1/norm
          
          most_likely_cat=tc.argmax(probs[ind], axis=0)
          cats[ind,most_likely_cat]=1
        return cats
        
        
    def categorical_log_likelihood(self, C, Z):

        #renormalize to ensure numerical stability
        beta_z=self.beta[:] @ Z.t()
        beta_z_norm=self.beta[:-1] @ Z.t()
        constant=tc.max(beta_z, axis=0)[0]
        b_z_shifted = beta_z - constant
        b_z_norm_shifted=beta_z_norm-constant
      
        normalizationTerm=1+tc.sum(tc.exp(b_z_norm_shifted), axis=0)
      
        #parallelized computation
        result=(C[:,-1]==1)*-tc.log(normalizationTerm)+(C[:,-1]==0)*(tc.sum(b_z_shifted*(C.t()==1), axis=0)- tc.log(normalizationTerm))    
        result=tc.sum(result)
      

        return result


