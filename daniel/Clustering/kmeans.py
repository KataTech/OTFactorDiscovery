"""
Author: Daniel Wang
Date: 2023-06-23
"""
import numpy as np
import utils

class KMeans:
    """
    This class must be initialized with user-defined values of n_clusters and max_iter.
    Otherwise, the default values (3 and 300, respectively) will be used.
    """
    def __init__(self, n_clusters=3, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
    
    def fit(self, X, cost_metric='squared_euclidean', init='Forgy', tolerance=1e-6, max_steps=100, descent_rate=0.1, random_state=None, verbose=True):
        """
        This function implements the k-means algorithm.

        The function iteratively updates the centroids and the labels of the data points until convergence.

        :param X: NumPy array of shape (n_samples, n_features)
        :param [cost_metric]: String. One of 'squared_euclidean', 'euclidean', 'manhattan', 'Lp' (where p is a positive integer), and 'euclidean^n' (where n is a positive integer).
        :param [init]: String. One of 'Forgy' and '++'.
        :param [tolerance]: Float. This is a stopping criterion, used only if cost_metric is 'euclidean'. Defaults to 1e-6.
        :param [max_steps]: Integer. This is a stopping criterion, used only if cost_metric is 'euclidean'. Defaults to 100.
        :param [descent_rate]: Float. This is a learning rate for gradient descent, used only if cost_metric is 'Lp' or 'euclidean^n'. Defaults to 0.1.
        :param [random_state]: Integer. This will be used as the seed for randomly initializing the centroids. Defaults to None.
        :param [verbose]: Boolean. This determines whether to print suggestions and warnings about optional arguments. Defaults to True.
        """

        if random_state is not None:
            np.random.seed(random_state)

        print('\n' + 'Using cost metric: ' + cost_metric)
        if cost_metric not in ['squared_euclidean', 'manhattan']:
            if verbose:
                print(#'For cost metrics other than "squared_euclidean and manhattan", the centroids are computed using an iterative algorithm. \n'
                    'This iterative algorithm may take a while.')
                if cost_metric == 'euclidean' and tolerance == 1e-6 and max_steps == 100:
                    print('Notice: For the cost metric "euclidean", centroids are updated using the Weiszfeld algorithm, which is iterative. \n'
                            'You have the option of defining the keyword arguments "tolerance" and "max_steps" when calling KMeans.fit(). \n'
                            'If you omit these, the default values (1e-6 and 100, respectively) from utils.py will be used for the iterative algorithm.')
                    
            if utils.is_Lp(cost_metric) or utils.is_euclidean_power(cost_metric):
                if verbose and descent_rate == 0.1:
                    print('Notice: For cost metrics of the form "Lp" and "euclidean^n", centroids are updated using gradient descent. \n'
                            'You have the option of defining the keyword argument "descent_rate". \n'
                            'If you omit this, the default value (0.1) from utils.py will be used.')
            elif cost_metric != 'euclidean':
                raise ValueError('Invalid cost metric. Please see the documentation for valid choices.')

        if init == 'Forgy':
            self.centroids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        elif init == '++':
            self.centroids = utils.plusplus(X, self.n_clusters, cost_metric=cost_metric, random_state=random_state)
        else:
            raise ValueError('Invalid init argument. Please choose from "Forgy" and "++".')

        for _ in range(self.max_iter):
            # Assign labels to each data point
            self.labels = self.predict(X, cost_metric=cost_metric)
            # Update centroids based on the mean of each cluster
            new_centroids = utils.update_centroids(X, self.labels, self.n_clusters, self.centroids, cost_metric=cost_metric, 
                                                   tolerance=tolerance, max_steps=max_steps, descent_rate=descent_rate)
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
    
    def predict(self, X, cost_metric=None):
        distances = utils.calculate_distances(X, cost_metric=cost_metric, centroids=self.centroids)
        return np.argmin(distances, axis=1)
