# This script is dedicated to the flow-based optimal transport 
# barycenter solvers. 

import numpy as np

def gaussian_kernel(X, sigma=1.0): 
    """
    Computes the Gaussian kernel matrix for the given data points. 

    Inputs: 
        - x: the data points. N-by-d numpy array where N is the number of 
            samples and d is the dimension of each x_i vector. 
        - sigma: the standard deviation of the Gaussian kernel. 
    """
    # Compute the squared Euclidean distance matrix
    distances = np.sum(X**2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X.T**2, axis=0, keepdims=True)
    # Compute the kernel matrix using the Gaussian kernel
    K_x = np.exp(-distances / (2 * sigma**2))
    return K_x



def gaussian_kernel_grad(y, l, gauss_kernel, sigma=1.0): 
    """
    Returns the gradient of the kernel matrix at entry (y, l) with respect the index of y. 
    """
    return -gauss_kernel[y, l] * (y - l) / sigma**2

def gaussian_kernel_kl_grad(x, y, lam, k_y, k_z): 
    """
    Computes the gradient of the loss function with respect to y. 
    """
    # compute the gradient of the kernel matrix
    grad = np.zeros_like(y)
    for i in range(y.shape[0]): 
        grad[i] = np.sum(gaussian_kernel_grad(i, np.arange(y.shape[0]), k_y) * k_z[i, :], axis=0) / np.sum(k_z[i, :] * k_y[i, :])
        grad[i] -= np.sum(gaussian_kernel_grad(i, np.arange(y.shape[0]), k_y), axis=0) / np.sum(k_y[i, :])
        # print("Dimension of Gradient: {}".format(grad[i].shape))
        # print("Dimension of y {}".format(y.shape))
        # print("Dimension of x {}".format(x.shape))
        # print("Dimension of lam {}".format(lam.shape))
        grad[i] = y[i] - x[i] + lam * grad[i]
    return grad

def compute_barycenter(x, z, y_init, lam, barycenter_cost_grad, kern_y=gaussian_kernel, kern_z=gaussian_kernel, epsilon=0.001, lr=0.01, max_iter=1000, verbose=0): 
    """
    Computes the barycenter with a flow-based approach. In other words, 
    we run gradient descent on y_i with respect to the barycenter objective
    until it reaches convergence. 

    Inputs: 
        - x: the observed data points. N-by-d numpy array where N is the number 
            of samples and d is the dimension of each x_i vector. 
        - z: the hidden factors. N-by-k numpy array where k is the dimension of 
            each z_i vector. 
        - y_init: the initial starting points of the barycenter points. N-by-d 
            numpy array.
        - lam: the regularization parameter for controlling the independence 
            between y and z.
        - barycenter_cost_grad: a function that computes the gradient of the 
            barycenter objective with respect to y. 
        - kern_y: the kernel to use for estimating the distribution of y
        - kern_z: the kernel to use for estimating the distribution of z
        - epsilon: the convergence threshold.
        - lr: the learning rate for gradient descent.
        - max_iter: the maximum number of iterations to run.
    """
    y = y_init
    iter = 0
    # pre-compute the kernel matrix for Z since that remains constant
    k_z = kern_z(z)
    # iterate until maximum iteration or convergence
    while iter < max_iter: 
        # update the iteration count 
        iter += 1
        k_y = kern_y(y)
        # compute the gradient vector of this iteration
        grad = barycenter_cost_grad(x, y, lam, k_y, k_z)
        # run a gradient descent step
        y = y - lr * grad
        # check for convergence
        if np.linalg.norm(grad) < epsilon:
            break
        if verbose >= 1 and iter % 100 == 0:
            print("Iteration {}: gradient norm = {}".format(iter, np.linalg.norm(grad)))
    if verbose >= 2 :
        print("Final gradient norm = {}".format(np.linalg.norm(grad)))
        print("Number of iterations = {}".format(iter))
    return y