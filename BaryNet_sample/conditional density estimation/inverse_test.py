from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from BaryNet_supervised import BaryNetSupervised

# Training the inverse transport map

X,Z = torch.load('.\labeled_data.pt')
N = X.size()[0]
d,k = X.size()[1], Z.size()[1]


model = torch.load('BaryNet_unsupervised.pth')
Y = model.transport((X,Z), is_train=False)

optimizer_inverse = torch.optim.SGD(model.inverse_res.parameters(), lr=1e-5)
for t in range(2000):
    model.zero_grad()
    L = model.reconstruction_error((X,Y,Z))
    # Reconstruct the sample
    print(np.sqrt(float(L)))
    L.backward()
    optimizer_inverse.step()

torch.save(model, './BaryNet_unsupervised_inverse.pth')
#torch.save(inverse_res,'./inverse_res.pth')