import numpy as np
import torch
import torch.nn as nn

# sample_dim refers to the dimension of the sample space (the X space)
# label_dim refers to the dimension of the label space (the Z space)

# Each pass takes time O(N)
# Depending on formulation, it could be optimized as either a minimax problem (inf transport + sup test function)
# or a maxmin problem (sup test function + inf transport)

class BaryNetSupervised(nn.Module):

    def __init__(self,sample_dim,label_dim):
        super().__init__()
        # The cost function c from C(XxY) of the transport problem
        self.cost = nn.MSELoss(reduction='sum')
        self.d = sample_dim
        self.k = label_dim

        # By Theorem 2.44 [Villani, Topics in OT], the transport map has the form y = x + grad term,
        # so if the residual term is sublinear, it would be easier to optimize using ResNet.
        # See He et al., "Deep residual learning for image recognition"
        # We are using ResNet implicitly, by defining the residual part of the transport map
        self.transport_res = nn.Sequential( nn.Linear(self.d + self.k, self.d + self.k + 4),
                                            nn.ReLU(),
                                            nn.Linear(self.d + self.k + 4, self.d + self.k + 4),
                                            nn.ReLU(),
                                            nn.Linear(self.d + self.k + 4, self.d), )
        # Initialize the transport map without translation
        #last_lin_bias = list(self.transport_res.state_dict().values())[-1]
        #with torch.no_grad():
        #    last_lin_bias.copy_(torch.zeros(last_lin_bias.size()))

        # The inverse transport map as a residual network
        self.inverse_res = nn.Sequential( nn.Linear(self.d + self.k, self.d + self.k + 4),
                                            nn.ReLU(),
                                            nn.Linear(self.d + self.k + 4, self.d + self.k + 4),
                                            nn.ReLU(),
                                            nn.Linear(self.d + self.k + 4, self.d), )

        # Constraint: psi_Z integrate to zero with respect to the z-margin. See the duality formula for barycenter
        # This is enforced by subtracting its expectation, estimated from the sample {z_i}
        # Any constant term can be discarded, so the last layer has bias=False
        self.test_Z = nn.Sequential(nn.Linear(self.k, self.k + 4),
                                    nn.ReLU(),
                                    nn.Linear(self.k + 4, 1, bias=False),)
        # Universality can be achieved by width >= d + 4 with arbitrary depth
        self.test_Y = nn.Sequential(nn.Linear(self.d, self.d + 4),
                                    nn.ReLU(),
                                    nn.Linear(self.d + 4, self.d + 4),
                                    nn.ReLU(),
                                    nn.Linear(self.d + 4, 1), )

    def transport(self,input,is_train):
        # Default: input = (X,Z) where X,Z are tensors
        X, _ = input
        XZ = torch.cat(input, dim=1)
        with torch.set_grad_enabled(is_train):
            Y = X + self.transport_res(XZ)
        return Y

    def forward(self, input):
        X, Z = input
        Y = self.transport(input,is_train=True)

        # The output is a 1xN tensor, so we flatten it by squeeze
        Psi_Z_unnormalized = self.test_Z(Z).squeeze()
        # PyTorch does not support operation on original data with grad, so we make a clone
        Psi_Z = Psi_Z_unnormalized.clone()
        # Constraint: Integral over z-marginal vanishes
        Psi_Z -= torch.mean(Psi_Z_unnormalized)
        Psi_Y = self.test_Y(Y).squeeze()

        # Use the average transport cost instead of total transport cost
        # so that we don't need to adjust the learning rate if batch size changes
        L = (self.cost(X,Y) - torch.dot(Psi_Y,Psi_Z)) / X.size()[0]
        return L

    # sample of barycenter, produced after the training of the transport net
    def barycenter(self, input):
        Y = self.transport(input,is_train=False)
        return Y

    # Training the inverse transport net
    def reconstruction_error(self, input):
        # Default: input = (X,Z) or (X,Y,Z) where X,Y,Z are tensors
        # X is the original sample set, Y is the barycenter, and Z is the latent variable
        if len(input)==2:
            X,Z = input
            # Compute the barycenter without the need to backprop on the transport net
            Y = self.transport(input,is_train=False)
        else:
            X, Y, Z = input
        # Reconstruct the sample
        YZ = torch.cat((Y,Z), dim=1)
        X_recover = Y + self.inverse_res(YZ)
        return self.cost(X, X_recover) / X.size()[0]

    # Recover the conditional density given z by transporting the barycenter using the inverse transport map S(x,z)
    def reconstruct_conditional(self, input, z):
        # Default: input = (X,Z) where X,Z are tensors
        Y = self.transport(input, is_train=False)
        Z_conditional = torch.zeros(Y.size()[0]).fill_(z).unsqueeze(dim=1)
        YZ = torch.cat((Y, Z_conditional), dim=1)
        with torch.no_grad():
            recovered_conditional = Y + self.inverse_res(YZ)
        return recovered_conditional