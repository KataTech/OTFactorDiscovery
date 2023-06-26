"""
Author: Daniel Wang
Date: 2023-06-23
"""
import numpy as np
import utils

class KMeans:
    """
    This class must be initialized with user-defined values of n_clusters and max_iter.
    Otherwise, the default values (3 and 500, respectively) will be used.
    """
    def __init__(self, n_clusters=3, max_iter=500):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.tolerance = None
        self.max_steps = None
        self.learning_rate = None
    
    def fit(self, X, cost_metric='squared_euclidean', tolerance=1e-6, max_steps=100, learning_rate = 0.01, **kwargs):
        """
        This function implements the k-means algorithm.

        The function iteratively updates the centroids and the labels of the data points until convergence.

        :param X: NumPy array of shape (n_samples, n_features)
        :param cost_metric: String. One of 'squared_euclidean', 'euclidean', 'manhattan', 'Lp' (where p is a positive integer), or 'euclidean^n' (where n is a positive integer).
        :param tolerance: Float. The algorithm update_centroid() will stop if the change in the centroid is less than this value.
        :param max_steps: Integer. The algorithm update_centroid() will stop if the number of steps exceeds this value.
        :param kwargs: Keyword arguments. See utils.py for details.
        """
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        if cost_metric in ['squared_euclidean', 'manhattan']:
            pass
        elif cost_metric == 'euclidean' or utils.is_Lp(cost_metric) or utils.is_euclidean_power(cost_metric):
            if tolerance == 1e-6 or max_steps == 100 or learning_rate == 0.01:
                print('Warning: For cost metrics other than "squared_euclidean" and "manhattan", the centroids are computed using an iterative algorithm.'
                      ' You have the option of defining the keyword arguments "tolerance" and "max_steps" when calling KMeans.fit(). If you omit these, '
                      'the default values (1e-6 and 100, respectively) from utils.py will be used. \n'
                      'Specifically, for the cost metrics "Lp" and "euclidean^n", the centroids are computed using gradient descent.'
                      ' You have the option of defining the keyword argument "learning_rate". If you omit this, the default value (0.01) from utils.py will be used.')
            else:
                print('For cost metrics other than "squared_euclidean and manhattan", the centroids are computed using an iterative algorithm. This may take a while.')
        else:
            raise ValueError('Invalid cost metric. Please see the documentation for valid choices.')

        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assign labels to each data point
            self.labels = self._assign_labels(X, cost_metric=cost_metric, **kwargs)
            # Update centroids based on the mean of each cluster
            new_centroids = utils.update_centroids(X, self.labels, self.n_clusters, cost_metric, 
                                                   tolerance=self.tolerance, max_steps=self.max_steps, learning_rate=self.learning_rate, **kwargs)
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids

        # Convert centroids to NumPy array
        self.centroids = np.array(self.centroids)
    
    def predict(self, X):
        return self._assign_labels(X)
    
    def _assign_labels(self, X, cost_metric=None, **kwargs):
        distances = utils.calculate_distances(X, cost_metric=cost_metric, centroids=self.centroids, **kwargs)
        return np.argmin(distances, axis=1)
