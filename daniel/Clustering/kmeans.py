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
    
    def fit(self, X, cost_metric='squared_euclidean', tolerance=1e-6, max_steps=100, **kwargs):
        """
        This function implements the k-means algorithm.

        The function iteratively updates the centroids and the labels of the data points until convergence.

        :param X: NumPy array of shape (n_samples, n_features)
        :param cost_metric: String. One of 'squared_euclidean', 'euclidean', 'manhattan', or 'Lp', where p is a positive integer.
        :param tolerance: Float. The algorithm update_centroid() will stop if the change in the centroid is less than this value.
        :param max_steps: Integer. The algorithm update_centroid() will stop if the number of steps exceeds this value.
        :param kwargs: Keyword arguments. See utils.py for details.
        """
        self.tolerance = tolerance
        self.max_steps = max_steps
        if cost_metric == 'squared_euclidean':
            pass
        elif cost_metric in ['euclidean', 'manhattan'] or utils.is_Lp(cost_metric):
            if tolerance == 1e-6 and max_steps == 100:
                print('Warning: For cost metrics other than "squared_euclidean", the centroids are computed using an iterative algorithm.'
                      ' You have the option of defining the keyword arguments "tolerance" and "max_steps" when calling KMeans.fit(). If you omit these, '
                      'the default values from utils.py (1e-6 and 100, respectively) will be used.')
            else:
                print('For cost metrics other than "squared_euclidean", the centroids are computed using an iterative algorithm. This may take a while.')
        else:
            raise ValueError('Invalid cost metric. Please see the documentation for valid choices.')

        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assign labels to each data point
            self.labels = self._assign_labels(X)
            # Update centroids based on the mean of each cluster
            new_centroids = utils.update_centroids(X, self.labels, self.n_clusters, cost_metric, tolerance=self.tolerance, max_steps=self.max_steps, **kwargs)
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids

        # Convert centroids to NumPy array
        self.centroids = np.array(self.centroids)
    
    def predict(self, X):
        return self._assign_labels(X)
    
    def _assign_labels(self, X):
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)
    
    def _calculate_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=-1)
