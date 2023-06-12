import numpy as np
import torch
import torch.nn as nn
from Climate_discovery.BaryNet_supervised import BaryNetSupervised


# The objective function is max_latent min_transport max_test_functions
# BaryNetSupervised contains the inner minimax problem: min_transport max_test_functions
class BaryNetUnsupervised(BaryNetSupervised):
    def __init__(self,sample_dim,latent_dim):
        super().__init__(sample_dim=sample_dim,label_dim=latent_dim)
        self.sample_dim = sample_dim
        self.latent_dim = latent_dim
        # The map to the latent variable space
        # 56 -> 5 -> 5 -> 1
        self.latent = nn.Sequential(nn.Linear(self.sample_dim, (self.latent_dim + 4)),
                                    nn.LeakyReLU(0.1),
                                    nn.BatchNorm1d((self.latent_dim + 4), affine=False, track_running_stats=False),
                                    nn.Linear((self.latent_dim + 4), self.latent_dim + 4),
                                    nn.LeakyReLU(0.1),
                                    nn.BatchNorm1d(self.latent_dim + 4, affine=False, track_running_stats=False),
                                    nn.Linear(self.latent_dim + 4, self.latent_dim, bias=False), )

        # Enforce Lipschitz constraint
        # In this case, use scale-invariant nonlinearity like ReLU for Z(x)
        # self.latent.parameters().clamp_(-0.01,0.01)

    # If the latent variable is provided: input = (X,Z)
    # Else input = X, and we compute Z
    # Saves time if the inner BaryNetSupervised is trained separately,
    # because the computation of the grad of self.latent is avoided
    def forward(self, input):
        if isinstance(input, torch.Tensor):
            X = input
            Z = self.latent(X)
        else:
            X,Z = input
        L = super(BaryNetUnsupervised, self).forward(input=(X,Z))
        return L

    # sample of barycenter, produced after the training of the transport net
    def barycenter(self, inputs):
        if isinstance(inputs, torch.Tensor):
            X = inputs
            with torch.no_grad():
                Z = self.latent(X)
        else:
            X,Z = inputs
        Y = self.transport((X,Z), is_train=False)
        return Y