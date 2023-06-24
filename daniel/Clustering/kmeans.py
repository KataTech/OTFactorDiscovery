"""
Author: Daniel Wang
Date: 2023-06-23
"""
import numpy as np
from utils import *

class KMeans:
    """
    This class must be initialized with user-defined values of n_clusters and max_iterations.
    Otherwise, the default values (8 and 300, respectively) will be used.
    """
    def __init__(self, n_clusters=8, max_iterations=300):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None
    
    def fit(self, X, cost_metric='squared_euclidean', **kwargs):
        """
        This function implements the k-means algorithm.

        The function iteratively updates the centroids and the labels of the data points until convergence.

        :param X: NumPy array of shape (n_samples, n_features)
        :param cost_metric: String. One of 'squared_euclidean', 'manhattan', 'euclidean', or 'Lp', where p is a positive integer.
        :param kwargs: Keyword arguments for the function update_centroids(). See utils.py for details.
        """
        # Check if the user has specified the keyword arguments "tolerance" and "max_steps"
        user_chosen = False
        if cost_metric == 'squared_euclidean':
            pass
        elif cost_metric in ['manhattan', 'euclidean'] or is_Lp(cost_metric):
            if 'tolerance' not in kwargs or 'max_steps' not in kwargs:
                print('Warning: For cost metrics other than "squared_euclidean", the centroids are computed using an iterative algorithm.'
                      ' Please specify the keyword arguments "tolerance" and "max_steps" when calling KMeans.fit(). Otherwise, the default values from utils.py'
                      ' (1e-6 and 100, respectively) will be used.')
            else:
                print('For cost metrics other than "squared_euclidean", the centroids are computed using an iterative algorithm. This may take a while.')
                user_chosen = True
        else:
            print('Warning: Invalid cost metric. Defaulting to "squared_euclidean".')
            cost_metric = 'squared_euclidean'

        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        
        for _ in range(self.max_iterations):
            # Assign labels to each data point
            self.labels = self._assign_labels(X)
            
            # Update centroids based on the mean of each cluster
            new_centroids = update_centroids(X, self.labels, self.n_clusters, cost_metric, **kwargs)
            
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
