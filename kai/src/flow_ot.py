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
        lambda_growth = (max_lambda - lam) / warm_stop
    if monitor is not None: 
        monitor_iters = 0
        monitor.eval({"y": y, "Lambda": lam, "Iteration": iter, "Gradient Norm": None})
    # iterate until maximum iteration or convergence
    while iter < max_iter: 
        # update the iteration count 
        iter += 1
        k_y = kern_y(y)
        if iter == 1 and verbose == True: 
            print("Kernel Y from First Iteration:\n", k_y)
        # compute the gradient vector of this iteration
        grad = barycenter_cost_grad(y, x, lam, k_y, k_z, verbose=verbose)
        if iter == 1: 
            print("Gradient from First Iteration:\n", grad)
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
            self.kernel_z = kernel_z
        else: 
            raise NotImplementedError
        
    def get_index(self, i, k): 
        """
        Compute the index in the aggregated representations of Y and its kernel given
        the (i, k) indices. 
        """
        return i * self.K + k

    def gaussian_kernel(self, X, sigma): 
        """
        Compute the gaussian kernel of a matrix. 
        """
        # Compute the squared Euclidean distance matrix
        distances = np.sum(X**2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X.T**2, axis=0, keepdims=True)
        # Compute the kernel matrix using the Gaussian kernel
        K_x = np.exp(-distances / (2 * sigma ** 2))
        return K_x
    
    def construct_z(self): 
        Z = np.zeros((self.N, self.K, self.D))
        for i in range(self.N): 
            Z[i] = self.Z_map
        return Z.reshape((self.N * self.K, self.D))
    
    def initialize(self, X: np.ndarray, labels: np.ndarray, K: int, Z_map: np.ndarray): 
        """
        Pre-compute attributes based on the supplied training data. 

        Inputs: 
            X: a N-by-M numpy array of observations where each row represents an observation
                for a total of N observations with M attributes each. 
            labels: a N-by-1 numpy array of labels. If an entry is -1, then that means the label
                is unknown. Otherwise, it should be an integer between 0 and K - 1, inclusive. 
            K: the total number of classes. 
            Z_map: a K-by-1 vector indicating the value associated with the K-th class. 
        """
        # set global variables
        self.X = X
        self.labels = labels
        self.N, self.M = X.shape
        self.K = K
        self.Z_map = Z_map
        self.D = self.Z_map.shape[1]
        # conduct sanity checks for the dimensions of the input variables
        assert self.labels.shape[0] == self.N, "ERROR: The number of entries in labels does not match N."
        assert self.Z_map.shape[0] == self.K, "ERROR: The number of entries in Z-Map does not match K."
        # initialize the probability matrix P, where the entry (i, k) represents the probability 
        # that the i-th observation belongs to class k
        self.P = np.zeros((self.N, self.K))
        for i in range(self.N): 
            if self.labels[i] == -1: 
                # assign uniform probability to each class if the observation class is unknown
                self.P[i, :] = 1 / float(self.K)
            else: 
                self.P[i, self.labels[i]] = 1.0
        # construct full Z matrix 
        self.Z = self.construct_z()
        # pre-compute the Z kernel
        if self.kernel_z == "gaussian": 
            self.kern_z = self.gaussian_kernel(self.Z, *self.kern_z_params)
        else: 
            raise NotImplementedError

    def augment_y(self, Y): 
        """
        Preprocess the Y matrix. Specifically, we want to create K copies of every single 
        observations Y_i within Y. Next, we need to flatten this properly such that every 
        K-consecutive entries represent the different versions of an observation. 

        Returns the processed Y matrix. 
        """
        Y_processed = np.zeros((self.N, self.K, self.M))
        for k in range(self.K): 
            Y_processed[:, k, :] = Y
        return Y_processed.reshape((self.N * self.K, self.M))
    
    def gradient(self, Y, lam, iter, verbose): 
        """
        Compute the gradient matrix storing the individual gradient vectors with respect to each 
        y_i observation. 
        """
        # initialize a gradient matrix 
        grad = np.zeros((self.N * self.K, self.M))
        # compute the current kernel y 
        if self.kernel_y == "gaussian": 
            sigma = self.kern_y_params[0]
            kern_y = self.gaussian_kernel(Y, sigma)
        else: 
            raise NotImplementedError
        # add the probability weights to the current kernel 
        w_kern_y = np.multiply(self.P.reshape((self.N * self.K, 1)), kern_y).T
        if iter == 1 and verbose > 2: 
            print("Weighted Kernel from First Iteration:\n", w_kern_y)
        # iterate over all possible indices of i and k
        for i in range(self.N): 
            for k in range(self.K): 
                if self.P[i, k] == 0: 
                    grad[self.get_index(i, k)] = np.zeros(self.M)
                    continue
                # if verbose > 1 and i == 8: 
                #     print(f"Processing entry ({i}, {k}) ---------------------------------")
                # process the joint kernel term of the gradient 
                joint_prod = w_kern_y[self.get_index(i, k), :] * self.kern_z[self.get_index(i, k), :]
                y_diff = Y[self.get_index(i, k), :] - Y
                weighted_diffs = ((joint_prod * y_diff.T).T) / (-1 * sigma ** 2)
                # if verbose > 1 and i == 8: 
                #     print(f"Joint Prod:\n", joint_prod)
                #     print(f"Weighted Diffs:\n", weighted_diffs)
                grad[self.get_index(i, k)] = np.sum(weighted_diffs, axis = 0) / np.sum(joint_prod)
                # process the y kernel term of the gradient 
                weighted_diffs = ((w_kern_y[self.get_index(i, k)] * y_diff.T).T) / (-1 * sigma ** 2)
                grad[self.get_index(i, k)] -= np.sum(weighted_diffs, axis = 0) / np.sum(w_kern_y[self.get_index(i, k)])
                # process the full gradient 
                grad[self.get_index(i, k)] = Y[self.get_index(i, k)] - self.X[i] + lam * grad[self.get_index(i, k)]
                # weight by the probability of the current index 
                grad[self.get_index(i, k)] *= self.P[i, k]
        return grad
    
    def estimate_p_ik(self, i, k, w_kern_y, P): 
        """
        Compute the weighted likelihood of finding observation Y_i in class K based on 
        the kernel density estimation of the barycenter specified by "w_kern_y"
        """
        return P[i, k] * np.sum(w_kern_y[self.get_index(i, k)])
    
    def probability_update(self, Y, verbose): 
        """
        Perform probability updates using the current positions of the Y matrix. 
        """
        if self.kernel_y != "gaussian":
            raise NotImplementedError
        old_P = np.copy(self.P)
        # compute the weighted estimation of the barycenter distribution
        kern_y = self.gaussian_kernel(Y, *self.kern_y_params)
        w_kern_y = np.multiply(self.P.reshape((self.N * self.K, 1)), kern_y)
        # update the probability by (i, k)-th indexing
        for i in range(self.N): 
            # skip over entries with known Z values 
            if self.labels[i] != -1: 
                continue
            # if verbose > 1: 
            #     print(f"Probability Update for i={i}: {self.P[i, :]}")
            total_weight = sum([self.estimate_p_ik(i, k_prime, w_kern_y, old_P) for k_prime in np.arange(self.K)])
            for k in range(self.K): 
                # update the probability weight by comparing to other classes
                self.P[i, k] = self.estimate_p_ik(i, k, w_kern_y, old_P) / total_weight

    def train(self, Y_init, lr = 0.001, epsilon = 0.001, max_iter = 1000, growing_lambda=True, init_lam=0.0, 
              warm_stop = 50, max_lam = 150, monitor=None, verbose = 0): 
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
        lam = init_lam
        if growing_lambda is True: 
            lambda_per_iter = (max_lam - init_lam) / warm_stop
        while iter < max_iter: 
            # update the iteration counter
            iter += 1
            # compute the gradient with respect to Y currently
            grad = self.gradient(Y, lam, iter, verbose)
            if iter == 1 and verbose > 2: 
                print("Gradient from First Iteration:\n", grad)
            # TODO: remove these print statements after debug session
            if verbose > 4: 
                print(f"Iteration {iter} Gradient Report:")
                print(grad[self.get_index(8, 0)])
                print(grad[self.get_index(8, 1)])
                print(grad[self.get_index(9, 0)])
                print(grad[self.get_index(9, 1)])
                print()
            grad_norm = np.linalg.norm(grad)
            # perform gradient descent step
            Y = Y - grad * lr
            # perform a probability update 
            self.probability_update(Y, verbose)
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
                if verbose > 2: 
                    print(f"Gradient: {grad}")
        # reached convergence... 
        if verbose > 0: 
            print("FINAL: Gradient norm = {} at iteration {}".format(grad_norm, iter))
        return self.select_best(Y)
        
    def select_best(self, Y, verbose = 0): 
        """
        Select the best version of Y across the different classes for every y_i. 
        In the end, you return the predictions, which is a filtered Y matrix containing 
        only one vector for every i-th observation (as opposed to K vectors) and the 
        assignments that indicate the class that is best associated with the i-th 
        observation. 
        """
        predictions = np.zeros((self.N, self.M))
        assignments = np.zeros(self.N, dtype=np.int64)
        for i in range(self.N): 
            assignments[i] = int(np.argmax(self.P[i]))
            if verbose > 0: 
                print(f"Selecting class {assignments[i]} for observation {i}")
            predictions[i, :] = Y[self.get_index(i, assignments[i]), :]            
        return predictions, assignments

    @staticmethod            
    def mask(labels: np.ndarray, percentage: float, seed = None): 
        """
        Mask a certain percentage of the supplied labels by replacing them with negative one. 

        Inputs: 
            - labels: a N-by-1 vector of values ranging from 0 to K - 1 that represent classes 
                    of an observation. 
            - percentage: the percentage of the labels to mask, this is automatically rounded 
                    to the nearest integer in our operation. 
            - seed: the random seed to use for reproducibility. 

        Returns both the masked version and ground truth. 
        """
        element_masked = int(labels.shape[0] * percentage)
        if seed is not None: 
            np.random.seed(seed)
        # randomly select elements to be masked
        mask = np.random.choice(labels.shape[0], element_masked, replace = False)
        masked_labels = np.copy(labels)
        masked_labels[mask] = -1
        return masked_labels, labels
        
        

                
