import numpy as np
from utils import *

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
    
    def fit(self, X, cost_metric='squared_euclidean'):
        if cost_metric not in ['squared_euclidean', 'manhattan', 'euclidean', 'Lp'] and not is_Lp(cost_metric):
            print('Invalid cost metric. Defaulting to "squared euclidean".')
            cost_metric = 'squared_euclidean'
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            # Assign labels to each data point
            self.labels = self._assign_labels(X)
            
            # Update centroids based on the mean of each cluster
            new_centroids = update_centroids(X, self.labels, self.n_clusters, cost_metric)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids

        # Convert centroids to NumPy array
        self.centroids = np.array(self.centroids)
        print(self.centroids)
    
    def predict(self, X):
        return self._assign_labels(X)
    
    def _assign_labels(self, X):
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)
    
    def _calculate_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=-1)
