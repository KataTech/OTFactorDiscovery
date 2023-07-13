"""
Author: Daniel Wang
Date: 2023-06-23
"""

import numpy as np
import utils

class KQuasicenters:
    """
    This class must be initialized with user-defined values of n_clusters and max_iter.
    Otherwise, the default values (3 and 300, respectively) will be used.
    """
    def __init__(self, n_clusters=3, init='forgy', max_iter=300, random_state=None, verbose=True):
        """
        :param [n_clusters]: Integer. The number of clusters to form as well as the number of quasicenters to generate.
        :param [init]: String. The method used to initialize the quasicenters. One of 'forgy' and '++'.
        :param [max_iter]: Integer. The maximum number of iterations of the k-quasicenters algorithm for a single run.
        :param [verbose]: Boolean. Whether to print warnings and suggestions.

        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.optimoids = None
        self.labels = None
    
    def fit(self, X, cost_metric='squared_euclidean', tolerance=1e-6, max_steps=100, descent_rate=0.1, max_descents=3):
        """
        This function implements the k-quasicenters algorithm.

        The function iteratively updates the quasicenters and the labels of the data points until convergence.

        :param X: NumPy array of shape (n_samples, n_features)
        :param [cost_metric]: String. One of 'squared_euclidean', 'euclidean', 'manhattan', 'Lp' (where p is a positive integer), and 'euclidean^n' (where n is a positive integer).
        :param [tolerance]: Float. This is a stopping criterion. It is used only if cost_metric is 'euclidean', 'Lp', or 'euclidean^n'.
        :param [max_steps]: Integer. This is a stopping criterion for the Weiszfeld algorithm. It is used only if cost_metric is 'euclidean'.
        :param [descent_rate]: Float. This is a learning rate for gradient descent. It is used only if cost_metric is 'Lp' or 'euclidean^n'.
        :param [max_descents]: Integer. This is a stopping criterion for gradient descent. It is used only if cost_metric is 'Lp' or 'euclidean^n'.
        """

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Check if the cost metric is valid and print helpful tips for the user if verbose is True
        if self.verbose:
            print('\n' + 'Using cost metric: ' + cost_metric)
            if cost_metric not in ['squared_euclidean', 'manhattan']:
                print('This iterative algorithm may take a while.')
                if cost_metric == 'euclidean' and (tolerance == 1e-6 or max_steps) == 100:
                    print('Notice: For the cost metric "euclidean", quasicenters are updated using the Weiszfeld algorithm, which is iterative. \n'
                            'You have the option of defining the keyword arguments "tolerance" and "max_steps" when calling KQuasicenters.fit(). \n'
                            'If you omit these, the default values (1e-6 and 100, respectively) from utils.py will be used for the iterative algorithm.')
                elif (
                    utils.is_Lp(cost_metric) or utils.is_euclidean_power(cost_metric)
                     and (tolerance==1e-6 or descent_rate==0.1 or max_descents==3)
                     ):
                    print('Notice: For cost metrics of the form "Lp" and "euclidean^n", quasicenters are updated using gradient descent. \n'
                            'You have the option of defining the keyword arguments "tolerance", "descent_rate", and "max_descents" when calling KQuasicenters.fit(). \n'
                            'If you omit these, the default values (1e-6, 0.1, and 3, respectively) from utils.py will be used.')
        if cost_metric not in ['squared_euclidean', 'manhattan', 'euclidean'] and not utils.is_Lp(cost_metric) and not utils.is_euclidean_power(cost_metric):
            raise ValueError('Invalid cost metric. Please see the documentation for valid choices.')

        # Initialize the quasicenters
        if self.init == 'forgy':
            self.optimoids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        elif self.init == '++':
            self.optimoids = utils.plusplus(X, self.n_clusters, cost_metric=cost_metric, random_state=self.random_state)
        else:
            raise ValueError('Invalid init argument. Please choose from "forgy" and "++".')
        
        # Main loop of the KQuasicenters algorithm
        for _ in range(self.max_iter):
            # Assign labels to each data point
            self.labels = self.predict(X, cost_metric=cost_metric)
            # Update quasicenters to be the cost-minimizer of each cluster
            new_quasicenters = utils.update_quasicenters(X, self.labels, self.n_clusters, self.optimoids, cost_metric=cost_metric, 
                                                   tolerance=tolerance, max_steps=max_steps, descent_rate=descent_rate, max_descents=max_descents)
            # Check for convergence
            if np.allclose(self.optimoids, new_quasicenters):
                break
            self.optimoids = new_quasicenters
    
    def predict(self, X, cost_metric=None):
        """
        Predict the closest cluster each sample in X belongs to.

        For each sample in X, compute the distances to all quasicenters and assign the sample to the closest cluster.

        :param X: NumPy array of shape (n_samples, n_features), the input samples.
        :param [cost_metric]: String. One of 'squared_euclidean', 'euclidean', 'manhattan', 'Lp' (where p is a positive integer), and 'euclidean^n' (where n is a positive integer).
        :return: Labels of each point
        """
        # Calculate the distance from each point to each quasicenters
        distances = utils.calculate_distances(X, cost_metric=cost_metric, quasicenters=self.optimoids)

        # Return the index of the closest quasicenters for each point
        return np.argmin(distances, axis=1)
