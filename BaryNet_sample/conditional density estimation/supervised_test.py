import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from BaryNet_supervised import BaryNetSupervised
from MiniMaxSolver import MiniMaxSolver


# ------This is the testing ground for supervised BaryNet------


# For 2D data, plot the original sample and its barycenter
def plot_sample(sample, xrange=None, yrange=None, title=""):
    plt.scatter(sample[:, 0], sample[:, 1])
    # set the range of the axis
    if xrange: plt.xlim(-xrange, xrange)
    if yrange: plt.ylim(-yrange, yrange)
    plt.title(title)
    plt.savefig("{}.png".format(title))  # save the figure
    plt.clf()  # clear the figure


def plot_3D(X, title=""):
    fig = plt.figure()
    plt.clf()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 2], X[:, 1])
    ax.set_xlim3d(-4, 4)
    ax.set_zlim3d(-6, 6)
    ax.set_ylim3d(-1, 11)
    plt.title(title)
    plt.savefig("{}.png".format(title))


# Solve minimax by quasi implicit gradient descent
def quasi_implicit_descent(model, lr, epoch):
    # Minimax problem: min over transport, max over test_function
    optimizer = MiniMaxSolver(model, model.transport_res, (model.test_Y, model.test_Z), input=(X, Z), lr=lr)
    for t in range(epoch):
        optimizer.step(Constrained=False, Normalization=None, TestMode=True)
        if (t + 1) % 100 == 0:
            barycenter = model.barycenter((X, Z))
            plot_3D(barycenter, title="Barycenter, round={}".format(t + 1))


X, Z = torch.load('.\sample_diverging_gaussians.pt')
N = X.size()[0]
d, k = X.size()[1], Z.size()[1]

# Initialize the network
torch.manual_seed(25)
model = BaryNetSupervised(sample_dim=d, label_dim=k)

# Test quasi implicit descent
# Learning rate
lr = 5e-4
epoch = 4000

quasi_implicit_descent(model, lr, epoch)

torch.save(model, './BaryNet_trained.pth')
