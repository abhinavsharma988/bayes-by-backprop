import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BNNLayer(nn.Module):

    NegHalfLog2PI = -.5 * math.log(2.0 * math.pi)
    softplus = lambda x: math.log(1 + math.exp(x))


    def __init__(self, n_input, n_output, activation, prior_mean, prior_rho):
        # --- Check activation ---
        assert activation in {'relu', 'softmax', 'tanh', 'none'}, 'Activation Type Not Found'
        # --- Initialize parent class ---
        super(BNNLayer, self).__init__()
        # --- Random Gaussian block (for sampling) ---
        self._gaussian_block = np.random.randn(10000)
        # --- Store layer dimensions ---
        self.n_input = n_input
        self.n_output = n_output
        # --- Weight parameters (posterior mean and rho) ---
        self.W_mean = nn.Parameter(torch.ones((n_input, n_output)) * prior_mean)
        self.W_rho  = nn.Parameter(torch.ones(n_input, n_output) * prior_rho)
        # --- Bias parameters ---
        self.b_mean = nn.Parameter(torch.ones(1, n_output) * prior_mean)
        self.b_rho  = nn.Parameter(torch.ones(1, n_output) * prior_rho)
        # --- Prior variance (scalar tensor) ---
        self.prior_var = torch.full(
            (1, 1),
            (math.log(1 + math.exp(prior_rho)))**2
        )
        # --- Activation function selection ---
        self.act = None
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'softmax':
            self.act = F.softmax
        elif activation == 'tanh':
            self.act = F.tanh
        # --- Helper: numpy → torch ---
        self._Var = lambda x: torch.from_numpy(x).float()

    def forward(self, X, mode):
        # --- Check mode ---
        assert mode in {'forward', 'MAP', 'MC'}, 'BNNLayer Mode Not Found'
        # --- Output shape ---
        _shape = (X.size()[0], self.n_output)
        # =========================
        # STEP 1: Compute mean
        # =========================
        Z_Mean = torch.mm(X, self.W_mean) + self.b_mean.expand(*_shape)
        # =========================
        # CASE 1: MAP (deterministic)
        # =========================
        if mode == 'MAP':
            if self.act is not None:
                return self.act(Z_Mean)
            else:
                return Z_Mean
        # =========================
        # STEP 2: Compute std
        # =========================
        Z_Std = torch.sqrt(
            torch.mm(
                torch.pow(X, 2),
                torch.pow(F.softplus(self.W_rho), 2)
            )
            +
            torch.pow(
                F.softplus(self.b_rho.expand(*_shape)),
                2
            )
        )
        # =========================
        # STEP 3: Sample noise
        # =========================
        Z_noise = self._random(_shape)
        Z = Z_Mean + Z_Std * Z_noise
        # =========================
        # CASE 2: MC (sampling only)
        # =========================
        if mode == 'MC':
            if self.act is not None:
                return self.act(Z)
            else:
                return Z
        # =========================
        # STEP 4: Prior std
        # =========================
        Prior_Z_Std = torch.sqrt(

            torch.mm(
                torch.pow(X, 2),
                self.prior_var.expand(self.n_input, self.n_output)
            )
            +
            self.prior_var.expand(*_shape)
        ).detach()

        # =========================
        # STEP 5: KL divergence
        # =========================
        layer_KL = self.sample_KL(
            Z,
            Z_Mean, Z_Std,
            Z_Mean.detach(), Prior_Z_Std
        )

        # =========================
        # STEP 6: Activation
        # =========================
        if self.act is not None:
            out = self.act(Z)
        else:
            out = Z

        return out, layer_KL

    def _random(self, shape):

        Z_noise = np.random.choice(
            self._gaussian_block,
            size = shape[0] * shape[1])
        Z_noise = np.expand_dims(Z_noise, axis=1).reshape(*shape)
        return self._Var(Z_noise)
    @staticmethod
    def log_gaussian(x, mean, std):

        return (
            BNNLayer.NegHalfLog2PI
            - torch.log(std)
            - 0.5 * torch.pow(x - mean, 2) / torch.pow(std, 2)
        )
    @staticmethod
    def sample_KL(x, mean1, std1, mean2, std2):

        log_prob1 = BNNLayer.log_gaussian(x, mean1, std1)
        log_prob2 = BNNLayer.log_gaussian(x, mean2, std2)

        return (log_prob1 - log_prob2).sum()






#############OLD CODE##############

# import numpy as np
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable


# class BNNLayer(nn.Module):
#     NegHalfLog2PI = -.5 * math.log(2.0 * math.pi)
#     softplus = lambda x: math.log(1 + math.exp(x))

#     def __init__(self, n_input, n_output, activation, prior_mean, prior_rho):
#         assert activation in {'relu', 'softmax', 'tanh','none'}, 'Activation Type Not Found'

#         super(BNNLayer, self).__init__()

#         # Instantiate a large Gaussian block to sample from, much faster than generating random sample every time
#         self._gaussian_block = np.random.randn(10000)

#         self.n_input = n_input
#         self.n_output = n_output

#         self.W_mean = nn.Parameter(torch.ones((n_input, n_output)) * prior_mean)
#         self.W_rho = nn.Parameter(torch.ones(n_input, n_output) * prior_rho)

#         self.b_mean = nn.Parameter(torch.ones(1, n_output) * prior_mean)
#         self.b_rho = nn.Parameter(torch.ones(1, n_output) * prior_rho)

#         #self.prior_var = Variable(torch.ones(1, 1) * BNNLayer.softplus(prior_rho) ** 2)
#         self.prior_var = torch.full((1, 1), (math.log(1 + math.exp(prior_rho)))**2) # converting calcualted value to tensor
#         # Set activation function
#         self.act = None
#         if activation == 'relu':
#             self.act = F.relu
#         elif activation == 'softmax':
#             self.act = F.softmax
#         elif activation == 'tanh':
#             self.act = F.tanh

#         self._Var = lambda x: torch.from_numpy(x).float()
        
#     def forward(self, X, mode):
#         assert mode in {'forward', 'MAP', 'MC'}, 'BNNLayer Mode Not Found'

#         _shape = (X.size()[0], self.n_output)

#         # Z: pre-activation. Local reparam. trick is used.
#         Z_Mean = torch.mm(X, self.W_mean) + self.b_mean.expand(*_shape)

#         if mode == 'MAP': return self.act(Z_Mean) if self.act is not None else Z_Mean

#         Z_Std = torch.sqrt(
#             torch.mm(torch.pow(X, 2),
#                      torch.pow(F.softplus(self.W_rho), 2)) +
#             torch.pow(F.softplus(self.b_rho.expand(*_shape)), 2)
#         )

#         Z_noise = self._random(_shape)
#         Z = Z_Mean + Z_Std * Z_noise

#         if mode == 'MC': return self.act(Z) if self.act is not None else Z

#         # Stddev for the prior
#         Prior_Z_Std = torch.sqrt(
#             torch.mm(torch.pow(X, 2),
#                      self.prior_var.expand(self.n_input, self.n_output)) +
#             self.prior_var.expand(*_shape)
#         ).detach()

#         # KL[posterior(w|D)||prior(w)]
#         layer_KL = self.sample_KL(Z,
#                                   Z_Mean, Z_Std,
#                                   Z_Mean.detach(), Prior_Z_Std)

#         out = self.act(Z) if self.act is not None else Z
#         return out, layer_KL

#     def _random(self, shape):
#         Z_noise = np.random.choice(self._gaussian_block, size=shape[0] * shape[1])
#         Z_noise = np.expand_dims(Z_noise, axis=1).reshape(*shape)
#         return self._Var(Z_noise)

#     @staticmethod
#     def log_gaussian(x, mean, std):
#         return BNNLayer.NegHalfLog2PI - torch.log(std) - .5 * torch.pow(x - mean, 2) / torch.pow(std, 2)

#     @staticmethod
#     def sample_KL(x, mean1, std1, mean2, std2):
#         log_prob1 = BNNLayer.log_gaussian(x, mean1, std1)
#         log_prob2 = BNNLayer.log_gaussian(x, mean2, std2)
#         return (log_prob1 - log_prob2).sum()
