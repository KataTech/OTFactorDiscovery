import numpy as np
import torch
from BaryNet_unsupervised import BaryNetUnsupervised
from MiniMaxSolver import MiniMaxSolver
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3D(X_list, title=""):
    fig = plt.figure()
    plt.clf()
    ax = Axes3D(fig)
    ax.scatter(X_list[:, 0], X_list[:, 1], X_list[:, 2])
    plt.title(title)
    plt.savefig(title + '.png')
    plt.close('all')


def plot_histogram(X, title=''):
    plt.hist(X, 50, density=True, alpha=0.75)
    plt.xlim(-40, 40)
    plt.title(title)
    plt.savefig(title + '.png')
    plt.close('all')


def plot_correlation(Z, n=15, title=''):
    # Assume that the coldest date is the n-th day
    time_of_year = torch.load('climate_data\\time_of_year_label_temperature.pt').numpy()
    sin_time_of_year = time_of_year[:, 0]
    cos_time_of_year = time_of_year[:, 1]
    adjust = 2 * np.pi / 365 * n
    seasonal_effect = np.cos(adjust) * cos_time_of_year + np.sin(adjust) * sin_time_of_year
    plt.scatter(seasonal_effect, Z)
    plt.xlabel('time of year')
    plt.ylabel('latent variable')
    plt.title('Correlation between time of year and latent variable')
    plt.savefig(title + '.png')
    plt.close('all')
    print(np.corrcoef([seasonal_effect, Z]))


def plot_latent():
    abs_time = torch.load('climate_data\\absolute_time_temperature.pt').numpy()
    Z = torch.load('Continental_latent_variable round=50000.pt').numpy()
    abs_time *= 10
    abs_time += 2009
    plt.scatter(abs_time, Z)
    plt.xlabel('year')
    plt.ylabel('latent variable')
    plt.xlim(2009, 2019)
    plt.title('Latent variable z')
    plt.show()


# Solve minimax by quasi implicit gradient descent
def quasi_implicit_descent(model, min_list, max_list, input, lr, epoch):
    # Minimax problem: min over transport, max over test_function
    optimizer = MiniMaxSolver(model, min_list, max_list, input=input, lr=lr)
    optimizer.batch_size = 1000
    for t in range(epoch):
        optimizer.step_OMD(Constrained=False, TestMode=True)
        # Enforce Lipschitz constraint
        with torch.no_grad():
            for para in model.latent.parameters():
                para.clamp_(-0.1, 0.1)
        if (t + 1) % 500 == 0:
            with torch.no_grad():
                Z = model.latent(input)
            Z = Z.numpy()[:, 0]
            plot_correlation(Z, title='full round={}'.format(t + 1))
            torch.save(model.state_dict(), 'Continental_temperature_discovery.pth')


def factor_discovery():
    X = torch.load('climate_data\\temperature_corrected.pt')

    model = BaryNetUnsupervised(sample_dim=X.size()[1], latent_dim=1)
    model.load_state_dict(torch.load('Continental_temperature_discovery.pth'))

    lr = 1e-5
    epoch = 12500
    min_list = (model.transport_res,)
    max_list = (model.test_Y, model.test_Z, model.latent)
    quasi_implicit_descent(model, min_list, max_list, input=X, lr=lr, epoch=epoch)

    torch.save(model.state_dict(), 'Continental_temperature_discovery.pth')
    with torch.no_grad():
        torch.save(model.latent(X), 'Continental_latent_variable.pt')


factor_discovery()
