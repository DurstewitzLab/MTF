from typing import Optional, Tuple
from bptt.dataset import GeneralDataset
import torch.nn as nn
import torch as tc
import math
from torch.linalg import pinv


class PLRNN(nn.Module):
    """
    Piece-wise Linear Recurrent Neural Network (Durstewitz 2017)

    Args:
        dim_x: Dimension of the observations
        dim_z: Dimension of the latent states (number of hidden neurons)
        n_bases: Number of bases to use in the BE-PLRNN
        clip_range: latent state clipping value
        latent_model: Name of the latent model to use. Has to be in LATENT_MODELS
        layer_norm: Use LayerNormalization (no learnable parameters currently)
    """

    LATENT_MODELS = ['PLRNN', 'clipped-PLRNN', 'dendr-PLRNN']

    def __init__(self, dim_x: int, dim_z: int, dim_force: int, n_bases: int, clip_range: float,
                 latent_model: str, layer_norm: bool, ds=None):
        super(PLRNN, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.ds=ds
        self.n_bases = n_bases
        self.use_bases = False
        self.d_f = dim_force
        
        if latent_model == 'PLRNN':
            if n_bases > 0:
                print("Chosen model is vanilla PLRNN, the bases Parameter has no effect here!")
            self.latent_step = PLRNN_Step(dz=self.d_z, clip_range=clip_range, layer_norm=layer_norm, ds=self.ds)
        else:
            if latent_model == 'clipped-PLRNN':
                assert n_bases > 0, "n_bases has to be a positive integer for clipped-PLRNN!"
                self.latent_step = PLRNN_Clipping_Step(self.n_bases, dz=self.d_z, clip_range=clip_range,
                                                       layer_norm=layer_norm, dataset=None, ds=self.ds)
                self.use_bases = True
            elif latent_model == 'dendr-PLRNN':
                assert n_bases > 0, "n_bases has to be a positive integer for dendr-PLRNN!"
                self.latent_step = PLRNN_Basis_Step(self.n_bases, dz=self.d_z, clip_range=clip_range, 
                                                    layer_norm=layer_norm, dataset=None, ds=self.ds)
                self.use_bases = True
            else:
                raise NotImplementedError(f"{latent_model} is not yet implemented. Use one of: {self.LATENT_MODELS}.")

    def get_latent_parameters(self):
        '''
        Split the AW matrix and return A, W, h.
        A is returned as a 1d vector!
        '''
        AW = self.latent_step.AW
        A = tc.diag(AW)
        W = AW - tc.diag(A)
        h = self.latent_step.h
        return A, W, h

    def get_basis_expansion_parameters(self):
        alphas = self.latent_step.alphas
        thetas = self.latent_step.thetas
        return alphas, thetas

    def get_parameters(self):
        params = self.get_latent_parameters()
        if self.use_bases:
            params += self.get_basis_expansion_parameters()
        return params

    def forward(self, x, s, n=None, z0=None, B=None):
        '''
        Forward pass with observations interleaved every n-th step.
        Credit @Florian Hess
        '''

        # switch dimensions for performance reasons
        x_ = x.permute(1, 0, 2)
        T, b, dx = x_.size()

        # no interleaving obs. if n is not specified
        if n is None:
            n = T + 1
            
            
        if s is not None:
          s_ = s.permute(1, 0, 2)
        else:
          s_ = [None] * T 


        # pre-compute pseudo inverse
        B_PI = None
        if B is not None:
            B_PI = pinv(B)

        # initial state
        if z0 is None:
            z = tc.randn(size=(b, self.d_z), device=x.device)
            z = self.teacher_force(z, x_[0], B_PI)
        else:
            z = z0

        # stores whole latent state trajectory
        Z = tc.empty(size=(T, b, self.d_z), device=x.device)
        # gather parameters
        params = self.get_parameters()
        for t in range(T):
            # interleave observation every n time steps
            if (t % n == 0) and (t > 0):
                z = self.teacher_force(z, x_[t], B_PI)
            z = self.latent_step(z, s_[t],*params)
            Z[t] = z

        return Z.permute(1, 0, 2)
        
        
        
        
    def generate_step(self, z0, inputs=None):
        

        params = self.get_parameters()
        z0 = z0.unsqueeze(0)
        z = self.latent_step(z0, inputs, *params)

        return z.squeeze(0)
    

    @tc.no_grad()

    def generate(self, T, z0, inputs):
        '''
        Generate a trajectory of T time steps given
        an initial condition z0. If no initial condition
        is specified, z0 is teacher forced.
        '''
        # holds the whole generated trajectory
        Z = tc.empty((T, 1, self.d_z), device=z0.device)
        
        
        if inputs is None:
            inputs = [None] * T
        else:
            if inputs.shape[0] < T:
                missing = T - inputs.shape[0]
                inputs = tc.cat([inputs, tc.zeros(missing, inputs.shape[1])], dim=0)
            inputs = inputs.unsqueeze(1)
        
        

        Z[0] = z0
        params = self.get_parameters()
        for t in range(1, T):
            Z[t] = self.latent_step(Z[t-1], inputs[t-1], *params)

        return Z.squeeze_(1)

    def teacher_force(self, z: tc.Tensor, x: tc.Tensor,
                      B_PI: Optional[tc.Tensor] = None) -> tc.Tensor:
        '''
        Apply teacher forcing to the latent state vector z.
        If B_PI is None, identity mapping is assumed the first
        dx entries of z are teacher forced. If B_PI is not None,
        z is estimated using the least-squares solution.
        '''
        if B_PI is not None:
            z = x @ B_PI.t()
        else:
            z[:, :self.d_f] = x
        return z
        
    def generate_from_z0(self, T, z0):

        number_of_sequences = z0.shape[0]

        # stores whole latent state trajectory
        Z = tc.empty(size=(T, number_of_sequences, self.d_z))

        # gather parameters
        params = self.get_parameters()

        # z0 is freely trained in multiple shooting
        z = z0
        Z[0] = z0

        for t in range(1, T):
          z = self.latent_step(z, *params)
          Z[t] = z

        return Z.permute(1, 0, 2)

        
    def mahalonobis_distance(residual, matrix):
            return - 0.5 * (residual.t() @ residual * tc.inverse(tc.diag(matrix ** 2))).sum()
            
    def log_det(diagonal_matrix):
            return - tc.log(diagonal_matrix).sum()
        
    def log_likelihood_z(self, z):
    
        params = self.get_parameters()

        distance_z = mahalonobis_distance(z[1:, :] - self.latent_step(z[:-1, :], *params), self.R_z)
        time_steps = x.shape[0]
        
        constant_z = - 0.5 * self.d_z * tc.log(tc.tensor(2 * math.pi)) * time_steps
        ll_z = distance_z0 + distance_z + log_det(self.R_z) * (time_steps - 1) + constant_z
        

        return ll_z / time_steps
        
        
    def log_likelihood_x(self, x):


        distance_x = mahalonobis_distance(x - self.observation(z), self.R_x)
        constant_x = - 0.5 * self.d_x * tc.log(tc.tensor(2 * math.pi)) * time_steps
        ll_x = distance_x + log_det(self.R_x) * time_steps + constant_x
        
        
        time_steps = x.shape[0]

        ll_x = distance_x + log_det(self.R_x) * time_steps + constant_x
        

        return ll_x / time_steps


class Latent_Step(nn.Module):
    def __init__(self, dz, ds, clip_range=None, layer_norm=False):
        super(Latent_Step, self).__init__()
        self.clip_range = clip_range
        #self.nonlinearity = nn.ReLU()
        self.dz = dz
        self.ds = ds

        if self.ds is not None:
            self.C = nn.Parameter(tc.randn(self.dz, self.ds), requires_grad=True)
        if layer_norm:
            self.norm = lambda z: z - z.mean(dim=1, keepdim=True)
        else:
            self.norm = nn.Identity()

    def init_AW_random_max_ev(self):
        AW = tc.eye(self.dz) + 0.1 * tc.randn(self.dz, self.dz)
        max_ev = tc.max(tc.abs(tc.linalg.eigvals(AW)))
        return nn.Parameter(AW / max_ev, requires_grad=True)

    def init_uniform(self, shape: Tuple[int]) -> nn.Parameter:
        # empty tensor
        tensor = tc.empty(*shape)
        # value range
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)

    def init_thetas_uniform(self, dataset: GeneralDataset) -> nn.Parameter:
        '''
        Initialize theta matrix of the basis expansion models such that 
        basis thresholds are uniformly covering the range of the given dataset
        '''
        mn, mx = dataset.data.min().item(), dataset.data.max().item()
        tensor = tc.empty((self.dz, self.db))
        # -mx to +mn due to +theta formulation in the basis step formulation
        nn.init.uniform_(tensor, -mx, -mn)
        return nn.Parameter(tensor, requires_grad=True)

    def init_AW(self):
        '''
        Talathi & Vartak 2016: Improving Performance of Recurrent Neural Network
        with ReLU Nonlinearity https://arxiv.org/abs/1511.03771.
        '''
        matrix_random = tc.randn(self.dz, self.dz)
        matrix_positive_normal = (1 / self.dz) * matrix_random.T @ matrix_random
        matrix = tc.eye(self.dz) + matrix_positive_normal
        max_ev = tc.max(tc.abs(tc.linalg.eigvals(matrix)))
        matrix_spectral_norm_one = matrix / max_ev
        return nn.Parameter(matrix_spectral_norm_one, requires_grad=True)

    def clip_z_to_range(self, z):
        if self.clip_range is not None:
            tc.clip_(z, -self.clip_range, self.clip_range)
        return z

    def add_input(self, s):
        if s is not None:
            s = tc.nan_to_num(s, nan=0)
            input = tc.einsum('ij,bj->bi', (self.C, s))
        else:
            input = 0
        return input

class PLRNN_Step(Latent_Step):
    def __init__(self, *args, **kwargs):
        super(PLRNN_Step, self).__init__(*args, **kwargs)
        self.AW = self.init_AW()
        self.h = self.init_uniform((self.dz, ))

    def forward(self, z, A, W, h):
        z_activated = tc.relu(self.norm(z))
        z = A * z + z_activated @ W.t() + h + self.add_input(s)
        return self.clip_z_to_range(z)

class PLRNN_Basis_Step(Latent_Step):
    def __init__(self, db, dataset=None, *args, **kwargs):
        super(PLRNN_Basis_Step, self).__init__(*args, **kwargs)
        self.AW = self.init_AW()
        self.h = self.init_uniform((self.dz, ))
        self.db = db

        if dataset is None:
            self.thetas = nn.Parameter(tc.randn(self.dz, self.db))
        else:
            self.thetas = self.init_thetas_uniform(dataset)

        self.alphas = self.init_uniform((self.db, ))

    def forward(self, z, A, W, h, alphas, thetas):
        z_norm = self.norm(z).unsqueeze(-1)
        # thresholds are broadcasted into the added dimension of z
        be = tc.sum(alphas * tc.relu(z_norm + thetas), dim=-1)
        z = A * z + be @ W.t() + h + self.add_input(s)
        return self.clip_z_to_range(z)

class PLRNN_Clipping_Step(Latent_Step):
    def __init__(self, db, dataset=None, *args, **kwargs):
        super(PLRNN_Clipping_Step, self).__init__(*args, **kwargs)
        self.AW = self.init_AW()
        self.h = self.init_uniform((self.dz, ))
        self.db = db
        self.use_bases = True

        if dataset is None:
            self.thetas = nn.Parameter(tc.randn(self.dz, self.db))
        else:
            self.thetas = self.init_thetas_uniform(dataset)

        self.alphas = self.init_uniform((self.db, ))

    def forward(self, z, s, A, W, h, alphas, thetas):
        z_norm = self.norm(z).unsqueeze(-1)
        be_clip = tc.sum(alphas * (tc.relu(z_norm + thetas) - tc.relu(z_norm)), dim=-1)
        z = A * z + be_clip @ W.t() + h + self.add_input(s)
        return z
        

        
class shallow_PLRNN_Step(Latent_Step):
    def __init__(self, db, dataset=None, *args, **kwargs):
        super(shallow_PLRNN_Step, self).__init__(*args, **kwargs)
        
        self.d_hidden = db        
        self.use_bases = False
        self.init_parameters()


    def init_parameters(self):
        r1 = 1.0 / (self.d_hidden ** 0.5)
        r2 = 1.0 / (self.dz ** 0.5)
        self.W1 = nn.Parameter(uniform_(tc.empty(self.dz, self.d_hidden), -r1, r1))
        self.W2 = nn.Parameter(uniform_(tc.empty(self.d_hidden, self.dz), -r2, r2))
        self.A = nn.Parameter(uniform_(tc.empty(self.dz), a=0.5, b=0.9))
        self.h2 = nn.Parameter(uniform_(tc.empty(self.d_hidden), -r1, r1))
        self.h1 = nn.Parameter(tc.zeros(self.dz))
        
        self.AW=self.A
        
    def forward(self, z, s):
        return self.A * z + tc.relu(z @ self.W2.T + self.h2) @ self.W1.T + self.h1+self.add_input(s)


    def jacobian(self, z):
        """Compute the Jacobian of the model at state z. Expects z to be a 1D tensor."""
        #assert z.ndim() == 1
        return tc.diag(self.A) + self.W1 @ tc.diag(self.W2 @ z > -self.h2).float() @ self.W2
