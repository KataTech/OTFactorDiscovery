import numpy as np
import time

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
        self.P_truth = np.zeros((self.N, self.K))
        for i in range(self.N): 
            if self.labels[i] == -1: 
                # assign uniform probability to each class if the observation class is unknown
                self.P_truth[i, :] = 1 / float(self.K)
            else: 
                self.P_truth[i, self.labels[i]] = 1.0
        self.P_mock = np.copy(self.P_truth)
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
    
    def gradient(self, Y: np.ndarray, lam: float, mock_prob: bool, iter: int, verbose: int): 
        """
        Compute the gradient matrix storing the individual gradient vectors with respect to each 
        y_i observation. 

        Inputs: 
            Y - the post-processed Y matrix for the current iteration. the dimension should be (NK, M). 
            lam - the regularization parameter for the independence condition.
            mock_prob - whether to use the mock probability matrix or the ground truth probability matrix.
            iter - the current iteration of the optimizer.
            verbose - the verbosity level for debugging.

        Returns the gradient matrix. Specifically, the i-th entry of the gradient matrix is the
        M-dimensional gradient vector of the i-th observation.  
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
        P = self.P_mock if mock_prob else self.P_truth
        w_kern_y = np.multiply(P.reshape((self.N * self.K, 1)), kern_y).T
        if iter == 1 and verbose > 2: 
            print("Weighted Kernel from First Iteration:\n", w_kern_y)
        # iterate over all possible indices of i and k
        for i in range(self.N): 
            for k in range(self.K): 
                if P[i, k] == 0: 
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
                grad[self.get_index(i, k)] *= P[i, k]
        return grad
    
    def estimate_p_ik(self, i, k, w_kern_y, P): 
        """
        Compute the weighted likelihood of finding observation Y_i in class K based on 
        the kernel density estimation of the barycenter specified by "w_kern_y"
        """
        return P[i, k] * np.sum(w_kern_y[self.get_index(i, k)])
    
    def probability_update(self, Y, mock_prob, eta, verbose): 
        """
        Perform probability updates using the current positions of the Y matrix. 
        """
        if self.kernel_y != "gaussian":
            raise NotImplementedError
        old_P = np.copy(self.P_truth)
        P = self.P_mock if mock_prob else self.P_truth
        # compute the weighted estimation of the barycenter distribution
        kern_y = self.gaussian_kernel(Y, *self.kern_y_params)
        w_kern_y = np.multiply(P.reshape((self.N * self.K, 1)), kern_y)
        # update the probability by (i, k)-th indexing
        for i in range(self.N): 
            # skip over entries with known Z values 
            if self.labels[i] != -1: 
                continue
            total_weight = sum([self.estimate_p_ik(i, k_prime, w_kern_y, old_P) for k_prime in np.arange(self.K)])
            for k in range(self.K): 
                # update the probability weight by comparing to other classes
                self.P_truth[i, k] = self.estimate_p_ik(i, k, w_kern_y, old_P) / total_weight
                self.P_mock[i, k] += eta * (self.P_truth[i, k] - self.P_mock[i, k])
           

    def train(self, Y_init, lr = 0.001, epsilon = 0.001, max_iter = 1000, growing_lambda=True, init_lam=0.0, 
              warm_stop = 50, max_lam = 150, mock_prob=False, eta=0.01, monitors=None, delayed_prob_update=True, 
              verbose = 0, timeit = False): 
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
            max_lam: the maximum value of lambda if growing_lambda is true. 
            mock_prob: whether to employ the slow-start probability update or not.
            eta: the learning rate of the probability update if slow_update is enabled. 
            monitor: the monitor object for reporting the optimizer's state during iterations.
            delayed_prob_update: whether to start performing the probability updates after warm_stop.
            eta: the learning rate on the probability update
            verbose: argument for getting updates 
            time: whether to time the training process or not.
        
        Returns the best version of Y and the assignments of the observations.
        """
        # create a internal representation for every variable that represents the 
        # the current state of the model
        if timeit: 
            start_time = time.time()
        self._mock_prob = mock_prob
        self._lam = init_lam
        self._epsilon = epsilon
        self._lr = lr
        self._iter = 0
        self._Y = self.augment_y(Y_init)
        if growing_lambda is True: 
            lambda_per_iter = (max_lam - init_lam) / warm_stop
        while self._iter <= max_iter: 
            # compute the gradient with respect to Y currently
            self._grad = self.gradient(self._Y, self._lam, mock_prob, self._iter, verbose)
            if self._iter == 1 and verbose > 2: 
                print("Gradient from First Iteration:\n", self._grad)
            # TODO: remove these print statements after debug session
            if verbose > 4: 
                print(f"Iteration {iter} Gradient Report:")
                print(self._grad[self.get_index(8, 0)])
                print(self._grad[self.get_index(8, 1)])
                print(self._grad[self.get_index(9, 0)])
                print(self._grad[self.get_index(9, 1)])
                print()
            grad_norm = np.linalg.norm(self._grad)
            # perform gradient descent step
            self._Y = self._Y - self._grad * lr
            # perform monitoring if monitors are supplied 
            if monitors is not None: 
                monitors.eval(self, self.get_params())
            # update the state in preparation for the next stage of gradient descent 
             # update the iteration counter
            self._iter += 1
            # perform a probability update if conditions are sufficient
            if delayed_prob_update and self._iter >= warm_stop: 
                self.probability_update(self._Y, mock_prob, eta, verbose)
            elif not delayed_prob_update:
                self.probability_update(self._Y, mock_prob, eta, verbose)
            # check for early convergence to local minimum
            if grad_norm < self._epsilon: 
                # if we have growing lambda, then we need to wait until 
                # the lambda finish growing before we can stop
                if growing_lambda and self._iter > warm_stop:
                    # if furthermore we have probability updates, 
                    # we need to allocate sometime for the probability update 
                    # to run before stopping 
                    if delayed_prob_update and self._iter > 1.5 * warm_stop: 
                        break
                    elif not delayed_prob_update: 
                        break
                # if everything is performed with a fix lambda, we 
                # can stop early without worrying about any delayed effects 
                elif not growing_lambda: 
                    break
            # update lambda if necessary
            if growing_lambda and self._iter < warm_stop: 
                self._lam += lambda_per_iter
            # display conditions of the optimization procedure
            if verbose > 0 and self._iter % 100 == 0: 
                print("Iteration {}: gradient norm = {}".format(self._iter, grad_norm))
                if verbose > 2: 
                    print(f"Gradient: {self._grad}")
        # reached convergence... 
        if verbose > 0: 
            print("FINAL: Gradient norm = {} at iteration {}".format(grad_norm, self._iter))
        if timeit:
            end_time = time.time()
            predictions, assignments = self.select_best(self._Y, mock_prob)
            return predictions, assignments, end_time - start_time
        return self.select_best(self._Y, mock_prob, verbose)
        
    def select_best(self, Y, mock_prob, verbose = 0): 
        """
        Select the best version of Y across the different classes for every y_i. 
        In the end, you return the predictions, which is a filtered Y matrix containing 
        only one vector for every i-th observation (as opposed to K vectors) and the 
        assignments that indicate the class that is best associated with the i-th 
        observation. 

        Inputs:
            Y: the augmented Y matrix.
            mock_prob: whether to use the mock probability matrix or the ground truth probability matrix.
            verbose: the verbosity level for debugging.
        
        Returns the Y matrix containing only the observations corresponding to the predicted class and 
        the predicted class assignments.
        """
        predictions = np.zeros((self.N, self.M))
        assignments = np.zeros(self.N, dtype=np.int64)
        P = self.P_mock if mock_prob else self.P_truth
        for i in range(self.N): 
            assignments[i] = int(np.argmax(P[i]))
            if verbose > 1 and self.labels[i] == -1: 
                print(f"Selecting class {assignments[i]} for observation {i} with probability {P[i, assignments[i]]}")
            predictions[i, :] = Y[self.get_index(i, assignments[i]), :]            
        return predictions, assignments
    
    def get_params(self): 
        """
        Construct and return a dictionary storing the parameters of the model.
        """
        params = {}
        params["iteration"] = self._iter
        params["lam"] = self._lam
        params["Y"] = np.copy(self._Y)
        params["mock_prob"] = self._mock_prob
        params["label"] = np.copy(self.labels)
        params["epsilon"] = self._epsilon
        params["gradient"] = self._grad
        params["lr"] = self._lr
        params["P"] = self.P_mock if self._mock_prob else self.P_truth
        params["sigma_y"] = self.kern_y_params[0]
        params["sigma_z"] = self.kern_z_params[0]
        return params

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
        