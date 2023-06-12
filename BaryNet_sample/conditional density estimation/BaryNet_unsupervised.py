import numpy as np
import torch
import torch.nn as nn
from BaryNet_supervised import BaryNetSupervised


# The objective function is max_latent min_transport max_test_functions
# BaryNetSupervised contains the inner minimax problem: min_transport max_test_functions
class BaryNetUnsupervised(BaryNetSupervised):
    def __init__(self,sample_dim,label_dim):
        super().__init__(sample_dim=sample_dim,label_dim=label_dim)
        self.d = sample_dim
        self.k = label_dim
        # The map to the latent variable space
        self.latent = nn.Linear(self.d, self.k, bias=False)

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
        return Y.numpy()