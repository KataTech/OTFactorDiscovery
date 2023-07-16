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
    # print("distances: {}".format(distances))
    # Compute the kernel matrix using the Gaussian kernel
    K_x = np.exp(-distances / (2 * sigma**2))
    return K_x


def gaussian_kernel_grad(y, i, l, gauss_kernel, sigma=1.0, verbose=0, second_kernel = None): 
    """
    Returns the gradient of the kernel matrix at entry (i, l) with respect the index of y. 
    """
    if second_kernel is None: 
        result = np.multiply(-gauss_kernel[i, l].reshape((y.shape[0], 1)), y[i, :] - y[l, :]) / sigma**2
        if verbose > 10: print(f"Gaussian Kernel Grad (i = {i}, l = {l}); Use Case 1: \n{result}")
    else: 
        result = np.multiply(-(gauss_kernel[i, l] * second_kernel[i, l]).reshape((y.shape[0], 1)), y[i, :] - y[l, :]) / sigma**2
        if verbose > 10: print(f"Gaussian Kernel Grad (i = {i}, l = {l}); Use Case 2: \n{result}")
    if verbose > 10: 
        print("Shape of gradient = {}".format(result.shape))
    return result

def gaussian_kernel_kl_grad(y, x, lam, k_y, k_z, verbose = 0): 
    """
    Computes the gradient of the loss function with respect to y. 
    """
    # compute the gradient of the kernel matrix
    grad = np.zeros_like(y)
    for i in range(y.shape[0]): 
        if verbose > 2:
            print("Iteration {}".format(i))
        grad[i] = np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose = verbose, second_kernel = k_z), axis=0) / np.sum(k_z[i, :] * k_y[i, :])
        if i == 0 and verbose > 10:     
            print(f"Numerator Sum: \n{np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose=verbose, second_kernel = k_z), axis=0)} \nwith dimensions {np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose=verbose, second_kernel = k_z), axis=0).shape}")
            print("Gradient after update one: \n{}\n".format(grad[i]))
        grad[i] -= np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose = verbose), axis=0) / np.sum(k_y[i, :])
        if i == 0 and verbose > 10: 
            print(f"Numerator Sum 2: \n{np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose=verbose), axis=0)} \nwith shape {np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose=verbose), axis=0).shape}")
            print("Gradient after update two: \n{}".format(grad[i]))
        # print("Dimension of Gradient: {}".format(grad[i].shape))
        # print("Dimension of y {}".format(y.shape))
        # print("Dimension of x {}".format(x.shape))
        # print("Dimension of lam {}".format(lam.shape))
        grad[i] = y[i] - x[i] + lam * grad[i]
    return grad

def compute_barycenter(x, z, y_init, lam, barycenter_cost_grad=gaussian_kernel_kl_grad, kern_y=gaussian_kernel, kern_z=gaussian_kernel, 
                       epsilon=0.001, lr=0.01, max_iter=1000, verbose=0, adaptive_lr=False, growing_lambda=True, 
                       warm_stop = 200, max_lambda = 300, monitor=None): 
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
        - verbose: the verbosity level for debugging 
        - adaptive_lr: whether to use adaptive learning rate or not.
    """
    y = y_init
    iter = 0
    # pre-compute the kernel matrix for Z since that remains constant
    k_z = kern_z(z)
    old_grad_norm = float('inf')
    # pre-compute the growth rate of the lambda and the stopping iteration
    if growing_lambda: 
        lambda_growth = max_lambda / warm_stop
        lam = 0.0
    if monitor is not None: 
        monitor_iters = 0
        monitor.eval({"y": y, "Lambda": lam, "Iteration": iter, "Gradient Norm": None})
    # iterate until maximum iteration or convergence
    while iter < max_iter: 
        # update the iteration count 
        iter += 1
        k_y = kern_y(y)
        # compute the gradient vector of this iteration
        grad = barycenter_cost_grad(y, x, lam, k_y, k_z, verbose=verbose)
        # run a gradient descent step
        y = y - lr * grad
        # check for convergence
        if iter > warm_stop and np.linalg.norm(grad) < epsilon:
            break
        # adaptive update of the learning rate
        if adaptive_lr and lr < 1 and lr > 0.0001: 
            lr = lr * 1.01 if (np.linalg.norm(grad) < old_grad_norm) else lr * 0.5
        old_grad_norm = np.linalg.norm(grad)
        # print the gradient norm every 100 iterations
        if verbose >= 1 and iter % 100 == 0:
            print("Iteration {}: gradient norm = {}".format(iter, np.linalg.norm(grad)))
        # update the lambda value if necessary
        if growing_lambda and iter < warm_stop: 
            lam += lambda_growth
        # perform monitor functionality if necessary 
        if monitor is not None and monitor_iters == 0: 
            monitor.eval({"y": y, "Lambda": lam, "Iteration": iter, "Gradient Norm": old_grad_norm})
            monitor_iters = monitor.get_monitoring_skip()
        if monitor is not None and monitor_iters != 0: 
            monitor_iters -= 1
    # print the final gradient norm and number of iterations
    if verbose >= 2 :
        print("Final gradient norm = {}".format(np.linalg.norm(grad)))
        print("Number of iterations = {}".format(iter))
    return y

def gaussian_kernel_single(x_i, x_l, sigma): 
    """
    Computes a single entry of the gaussian kernel
    """
    return np.exp(np.linalg.norm(x_i - x_l) / (-2 * sigma ** 2))

def gaussian_kernel_duo(Y, Y_center, sigma = 1.0): 
    """
    Compute the kernel matrix using Y and Y_center
    """
    n = Y.shape[0]
    kern = np.zeros((n, n))
    for i in range(n): 
        for l in range(n): 
            kern[i, l] = np.exp(-(np.linalg.norm(Y[i, :] - Y_center[l, :])) / (2 * sigma**2))
    return kern

def kl_barycenter_loss(Y, Y_center, X, Z, lam, kern_y = gaussian_kernel_duo, kern_z = gaussian_kernel, verbose = 0): 
    """
    Computes the loss function of the barycenter problem.

    Inputs: 
        - X: the observed data points. N-by-d numpy array where N is the number
            of samples and d is the dimension of each x_i vector.
        - Y: the barycenter points. N-by-d numpy array.
        - Z: the hidden factors. N-by-k numpy array where k is the dimension of
            each z_i vector.
        - lam: the regularization parameter for controlling the independence
            between y and z.
        - kern_y: the kernel to use for estimating the distribution of y
        - kern_z: the kernel to use for estimating the distribution of z 
    """
    # compute the kernel matrices
    k_y = kern_y(Y, Y_center)
    k_z = kern_z(Z)
    # compute the loss function
    loss = 0
    for i in range(X.shape[0]): 
        temp = 0
        temp += np.linalg.norm(Y[i, :] - X[i, :])**2 / 2
        temp += lam * np.log(np.sum(k_y[i, :] * k_z[i, :]) / (np.sum(k_y[i, :]) * np.sum(k_z[i, :])))
        if verbose > 0: 
            print(f"The loss value evaluation for entry i = {i}: {temp}")
        loss += temp 
    return loss

class SemiSupervisedOT(): 
    """
    A class dedicated to solving the semi-supervised OT problem
    """
    def __init__(self, kernel_y = "gaussian", kernel_z = "gaussian", kernel_y_bandwidth = [1.0], 
                 kernel_z_bandwidth = [1.0], regularizer = "kl_divergence"): 
        """
        Initialize the setting of the solver. The default solver uses gaussian kernels with sigma = 1.0 
        for both the transported observations, y, and the hidden factors, z. 

        Inputs: 
            kernel_y: the kernel to use for the transported observations, y. 
            kernel_z: the kernel to use for the latent factors, z. 
            kernel_y_bandwidth: the parameters for the y kernel. 
            kernel_z_bandwidth: the parameters for the z kernel. 
            regularizer: the type of regularizer to use for enforcing the independence condition. 
        """
        self.kern_y_params = kernel_y_bandwidth
        self.kern_z_params = kernel_z_bandwidth
        self.regularizer = regularizer
        if kernel_y == "gaussian":
            self.kernel_y = kernel_y
        else: 
            raise NotImplementedError
        if kernel_z == "gaussian": 
            self.kernel_z = gaussian_kernel_single
        else: 
            raise NotImplementedError
        
    
    def initialize(self, X: np.ndarray, Z: np.ndarray, K: int, Z_map: np.ndarray): 
        """
        Pre-compute attributes based on the supplied training data. 

        Inputs: 
            X: a N-by-M numpy array of observations where each row represents an observation
                for a total of N observations with M attributes each. 
            Z: a N-by-1 numpy array of labels. If an entry is -1, then that means the label
                is unknown. Otherwise, it should be an integer between 0 and K - 1, inclusive. 
            K: the total number of classes. 
            Z_map: a K-by-1 vector indicating the value associated with the K-th class. 
        """
        # set global variables
        self.X = X
        self.Z = Z
        self.N, self.M = X.shape
        self.K = K
        # conduct sanity checks for the dimensions of the input variables
        assert self.Z.shape[0] == self.N, "ERROR: The number of entries in Z does not match N."
        # initialize the probability matrix P, where the entry (i, k) represents the probability 
        # that the i-th observation belongs to class k
        self.P = np.zeros((self.N, self.K))
        for i in range(self.N): 
            if self.Z[i] == -1: 
                # assign uniform probability to each class if the observation class is unknown
                self.P[i, :] = 1 / float(self.K)
            else: 
                self.P[i, self.Z[i]] = 1.0
        # pre-compute the kernel with respect to each class
        self.kern_z = np.zeros((self.K, self.K))
        for k in range(self.K): 
            for kk in range(self.K): 
                self.kern_z[k, kk] = self.kernel_z(Z_map[k], Z_map[kk], *self.kern_z_params)

    def compute_kernel(self, mat: np.ndarray, kern_mode: str, sigma: float, second_kern:np.ndarray): 
        """
        Compute kernel for the matrix 'mat' according to the specified kernel 
        type 'kern_mode'. 

        Requires that 'mat' has a dimension of N-K-M. 
        Inputs: 
            mat: the matrix to compute the kernel for.
            kern_mode: the type of kernel to compute. 
            sigma: the parameter for the kernel. 
            compute_grad: whether the gradient of the kernel is needed
            second_kernel: the z kernel, supplied if we are computing y kernel

        Returns the computed kernel and its associated derivative if requested. 
        """
        # conduct sanity check on required condition
        assert mat.shape == (self.N, self.K, self.M), f"ERROR: Incorrect dimension {mat.shape} of the input matrix for kernel processing."
        if kern_mode != "gaussian": 
            raise NotImplementedError
        # initialize the gaussian kernels
        kern = np.zeros((self.N, self.K, self.N, self.K, 1))
        kern_joint = np.zeros((self.N, self.K, self.N, self.K, 1))
        kern_grad = np.zeros((self.N, self.K, self.N, self.K, self.M))
        kern_joint_grad = np.zeros((self.N, self.K, self.N, self.K, self.M))
        # populate entries of the gaussian kernels
        for i in range(self.N): 
            for k in range(self.K): 
                for j in range(self.N): 
                    for k_prime in range(self.K): 
                        kern[i, k, j, k_prime] = np.exp(np.linalg.norm(mat[i, k] - mat[j, k_prime]) / (-2 * sigma ** 2))
                        kern_joint[i, k, j, k_prime] = kern[i, k, j, k_prime] * second_kern[k, k_prime]
                        kern_grad[i, k, j, k_prime] = (kern[i, k, j, k_prime] / (-1 * sigma ** 2)) * (mat[i, k] - mat[j, k_prime])
                        kern_joint_grad[i, k, j, k_prime] = kern_grad[i, k, j, k_prime] * second_kern[k, k_prime]
        return kern, kern_joint, kern_grad, kern_joint_grad

    def gradient(self, Y, lam): 
        """
        Compute the gradient with respect to Y. 
        """
        grad = np.zeros_like(Y)
        kern_y, kern_yz, kern_y_grad, kern_yz_grad = self.compute_kernel(Y, self.kernel_y, 
                                                                         *self.kern_y_params,
                                                                         second_kern = self.kern_z)
        for i in range(self.N): 
            for k in range(self.K): 
                grad[i, k] = (np.sum(kern_yz_grad[i, k, :, :], axis = (0, 1)) / np.sum(kern_yz[i, k, :, :]))
                grad[i, k] -= (np.sum(kern_y_grad[i, k, :, :], axis = (0, 1)) / np.sum(kern_y[i, k, :, :]))
                grad[i, k] *= lam
                grad[i, k] += Y[i, k] - self.X[i]
                grad[i, k] *= self.P[i, k]
        return grad 
    
    def estimate(self, Y, i, k, sigma): 
        """
        Perform a kernel estimation of the sample represented by Y[i, k] sample belonging in the 
        barycenter estimation. 
        """
        return np.sum(np.exp(np.linalg.norm(Y[i, k] - Y.flatten()) / (2 * sigma ** 2)))


    def probability_update(self, Y, sigma, verbose = 0): 
        """
        Update the probability matrix P using the latest values of Y.
        """
        old_P = np.copy(self.P)
        for i in range(self.N): 
            # skip over entries with known Z values 
            if self.Z[i] != -1: 
                continue
            for k in range(self.K): 
                if verbose > 1: 
                    print(f"Weighted estimate of observation {i} in class {k}: {old_P[i, k] * self.estimate(Y, i, k, sigma)}")
                self.P[i, k] = old_P[i, k] * self.estimate(Y, i, k, sigma) / np.sum(old_P[i, :] * self.estimate(Y, i, np.arange(self.K), sigma))

    def augment_y(self, Y): 
        """
        Augment the Y matrix from the input format to the desired format. 
        """
        # TODO: write the code for transforming the input Y matrix in the train function to have input indices (i, k) 
        # this should be written in a "smart" way that takes into account the masking,. 
        Y_new = np.zeros((Y.shape[0], self.K, Y.shape[1]))
        for i in range(self.K): 
            Y_new[:, i, :] = Y
        return Y_new

    def train(self, Y_init, lr = 0.001, epsilon = 0.001, max_iter = 1000, growing_lambda=True, fixed_lam=0.0, 
              warm_stop = 50, max_lambda = 150, monitor=None, verbose = 0): 
        """
        Perform training on the semi-supervised optimal transport learning model. 

        Inputs: 
            Y_init: a N-K-M numpy array indicating the initial condition of the transported observations, 
                    such that entry Y_init[i, k, :] represents the i-th observation under the assumption
                    that it belongs to the class k.
            lr: the learning rate of the gradient descent algorithm. 
            epsilon: the acceptable error.
            max_iter: the maximum number of iteration for the optimizer.
            growing_lambda: indicates whether the lambda term grows with the optimizer.
            fixed_lam: the lambda value if growing_lambda is false.
            warm_stop: the iteration to stop incrementing lambda if growing_lambda is true. 
            max_lambda: the maximum value of lambda if growing_lambda is true. 
            monitor: the monitor object for reporting the optimizer's state during iterations.
            verbose: argument for getting updates 
        """
        Y = self.augment_y(Y_init)
        # TODO: implement functionalities for using monitors
        if monitor is not None: 
            raise NotImplementedError
        iter = 0
        # initialize the lambda values 
        if growing_lambda is True: 
            lam = 0.0
            lambda_per_iter = max_lambda / warm_stop
        else: 
            lam = fixed_lam
        while iter < max_iter: 
            # update the iteration counter
            iter += 1
            # compute the gradient with respect to Y currently
            grad = self.gradient(Y, lam)
            if verbose > 1: 
                print(grad)
            grad_norm = np.linalg.norm(grad.flatten())
            # perform gradient descent step
            Y = Y - grad * lr
            # perform a probability update 
            self.probability_update(Y, *self.kern_y_params, verbose)
            # check for early convergence to local minimum
            if grad_norm < epsilon: 
                if growing_lambda and iter > warm_stop: 
                    break
            # update lambda if necessary
            if growing_lambda and iter < warm_stop: 
                lam += lambda_per_iter
            # display conditions of the optimization procedure
            if verbose > 0 and iter % 100 == 0: 
                print("Iteration {}: gradient norm = {}".format(iter, grad_norm))
        # reached convergence... 
        if verbose > 0: 
            print("FINAL: Gradient norm = {} at iteration {}".format(grad_norm, iter))
        return Y
        
    def select_best(self, Y, verbose = 0): 
        predictions = np.zeros((Y.shape[0], Y.shape[2]))
        assignments = np.zeros(Y.shape[0])
        for i in range(self.N): 
            assignments[i] = np.argmax(self.P[i])
            predictions[i, :] = Y[i, assignments[i], :]            
            if verbose > 0: 
                print(f"Selecting class {assignments[i]} for observation {i}")
        return predictions, assignments
            


            


                
