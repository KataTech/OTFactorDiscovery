"""
Author: Daniel Wang
Date: 2023-06-23
"""

import numpy as np
import scipy.optimize as spo
import sklearn.metrics as skm
import re

class KGenCenters:
    """
    This class must be initialized with user-defined values of n_clusters and max_iter.
    Otherwise, the default values (3 and 300, respectively) will be used.
    """
    def __init__(self, n_clusters=3, init='forgy', plusplus_dist=None, max_iter=300, random_state=None, verbose=True):
        """
        :param [n_clusters]: Integer. The number of clusters to form as well as the number of centers to generate.
        :param [init]: String. The method used to initialize the centers. One of 'forgy', 'random partition', and '++'.
        :param [plusplus_dist]: String. The distance metric that will be used for probabiltiies in the k-means++ initialization. Provide ONLY if you want to hard-code a distance that overrides cost_metric.
        :param [link_labels]: NumPy array of shape (n_samples,). Provide ONLY for clustering with must-link constraints. Each element of the array is None or an integer. The choice of integer is arbitrary as long as link_labels[i] = link_labels[j] for all i and j that are must-link.
        :param [max_iter]: Integer. The maximum number of iterations of the k-GenCenters algorithm for a single run.
        :param [random_state]: Integer. The random seed for initializing the centers.
        :param [verbose]: Boolean. Whether to print warnings and suggestions.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.plusplus_dist = plusplus_dist
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.centers = None
        self.assigns = None

    # Private methods (originally from utils.py)
    def _is_Lp(self, cost_metric):
        """
        This function checks whether the argument "cost_metric" has the format "Lp", where p is a number at least unity.
        """
        # Support for non-integer values of p that are at least one.
        pattern = r'^L([1-9]\d*(\.\d+)?)$'
        return re.match(pattern, cost_metric) is not None

    def _is_euclidean_power(self, cost_metric):
        """
        This function checks whether the argument "cost_metric" has the format "euclidean^n", where n is a positive integer.
        """
        pattern = r'^euclidean\^\d+$'
        return re.match(pattern, cost_metric) is not None

    def _weiszfeld(self, X, tolerance, max_steps):
        """
        This function implements the Weiszfeld algorithm for computing the geometric median. 
        The geometric median is a generalization of the median to multi-dimensional space.
        
        :param X: The input data.
        :param tolerance: The tolerance for convergence.
        :param max_steps: The maximum number of steps.
        :return: The geometric median.
        """
        num_points, _ = X.shape

        # Initialize the geometric median as the mean of the points
        geometric_median = np.mean(X, axis=0)

        for iteration in range(max_steps):
            # Calculate the distances from the current geometric median to all points
            distances = np.linalg.norm(X - geometric_median, axis=1)

            # Handle division-by-zero issues
            nonzero_indices = distances != 0
            if np.any(nonzero_indices):
                weights = 1 / np.where(distances != 0, distances, np.finfo(float).eps)  # Replace zero distances with a small epsilon
                weights /= np.sum(weights)  # Normalize the weights
                new_geometric_median = np.dot(weights, X)
            else:
                new_geometric_median = geometric_median

            # Check if the algorithm has converged
            if np.allclose(geometric_median, new_geometric_median, atol=tolerance):
                break
            else:
                geometric_median = new_geometric_median

        return geometric_median

    def _calculate_distances(self, X, cost_metric=None, centers=None):
        """
        This function calculates the distances from each data point to each center.
        
        :param X: The input data.
        :param cost_metric: The cost metric used for clustering.
        :param centers: The centers of the clusters.
        :return: The distances from each data point to each center.
        """
        if cost_metric is None:
            raise ValueError('No value for the argument "cost_metric" was received.')
        if centers is None:
            raise ValueError('No value for the argument "centers" was received.')
        
        if cost_metric in ['squared_euclidean', 'euclidean^2']:
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=-1)**2
        elif cost_metric in ['manhattan', 'L1']:
            distances = np.linalg.norm(X[:, np.newaxis] - centers, ord=1, axis=-1)
        elif cost_metric in ['euclidean', 'euclidean^1', 'L2']:
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=-1)
        elif self._is_Lp(cost_metric):
            p = float(cost_metric[1:])
            distances = np.linalg.norm(X[:, np.newaxis] - centers, ord=p, axis=-1)
        # FIXME: Cost metrics of the form "euclidean^n" with n greater than 3 currently cause overflow. A community fix is welcomed.
        elif self._is_euclidean_power(cost_metric):
            p = 2 # this exponent can be customized by end-users for non-euclidean distances
            n = int(cost_metric[10:])
            distances = np.linalg.norm(X[:, np.newaxis] - centers, ord=p, axis=-1)**n
            
        return distances

    def _update_centers(self, X, labels, n_clusters, centers, cost_metric=None, tolerance=None, max_steps=None, descent_rate=None, max_descents=None):
        """
        This function updates the centers based on the given cost metric.
        
        :param X: The input data.
        :param labels: The cluster labels for the data.
        :param n_clusters: The number of clusters.
        :param centers: The current centers.
        :param cost_metric: The cost metric used for clustering. 
        :param tolerance: The tolerance for convergence.
        :param max_steps: The maximum number of steps for the Weiszfeld algorithm.
        :param descent_rate: The learning rate for gradient descent.
        :param max_descents: The maximum number of descents for gradient descent.
        :return: The updated centers.
        """
        if cost_metric is None:
            raise ValueError('No value for the argument "cost_metric" was received.')
        if tolerance is None:
            raise ValueError('No value for the argument "tolerance" was received.')
        if max_steps is None:
            raise ValueError('No value for the argument "max_steps" was received.')
        if descent_rate is None:
            raise ValueError('No value for the argument "descent_rate" was received.')
        if max_descents is None:
            raise ValueError('No value for the argument "max_descents" was received.')
        
        centers = np.copy(centers)
        if cost_metric in ['squared_euclidean', 'euclidean^2']:
            for k in range(n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    centers[k] = np.mean(cluster_points, axis=0)
        elif cost_metric in ['manhattan', 'L1']:
            for k in range(n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    centers[k] = np.median(cluster_points, axis=0)
        elif cost_metric in ['euclidean', 'euclidean^1', 'L2']:
            for k in range(n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    centers[k] = self._weiszfeld(cluster_points, tolerance, max_steps)
        elif self._is_Lp(cost_metric) or self._is_euclidean_power(cost_metric):
            if self._is_Lp(cost_metric):
                p = float(cost_metric[1:])
                n = 1
            else:
                p = 2
                n = int(cost_metric[10:])
            for k in range(n_clusters):   
                for gd_step in range(max_descents):
                    # existing gradient calculation and update code
                    grad = np.zeros(X.shape[1])
                    num_points = np.sum(labels == k)  # Number of points in cluster k
                    for x in X[labels == k]:
                        distance = np.abs(centers[k] - x)
                        non_zero_indices = distance != 0
                        if np.any(non_zero_indices):  # Check if there are non-zero indices
                            grad += n * np.sum(distance[non_zero_indices]**p, axis=-1)**(n/p - 1) * distance[non_zero_indices]**(p - 1) * np.sign(centers[k] - x)[non_zero_indices]
                    grad[np.isnan(grad)] = 0  # Replace NaN values with zero
                    if np.allclose(grad, 0, atol=tolerance):
                        break
                    centers[k] -= descent_rate * grad / num_points

        return centers

    def _plusplus(self, X, n_clusters, cost_metric=None, random_state=None):
        """
        This function implements the k-means++ initialization method for the centers.
        This method aims to provide a better initialization than random initialization, which can lead to bad local optima.
        
        :param X: The input data.
        :param n_clusters: The number of clusters.
        :param cost_metric: The cost metric used for clustering.
        :param random_state: The seed for the random number generator.
        :return: The initialized centers.
        """
        if cost_metric is None:
            raise ValueError('No value for the argument "cost_metric" was received.')
        if random_state is None:
            np.random.seed(0)
            if self.verbose:
                print('Warning: No value for the argument "random_state" was received by the utils.plusplus() initialization. It is recommended that you set this for reproducibility. \n'
                    'Defaulting to random_state=0.')
        else:
            np.random.seed(random_state)
        
        centers = np.empty((n_clusters, X.shape[1]))
        centers[0] = X[np.random.randint(X.shape[0])]  # Choose the first center randomly
        for k in range(1, n_clusters):
            distances = self._calculate_distances(X, cost_metric, centers[:k])
            probabilities = np.min(distances, axis=1) / np.sum(np.min(distances, axis=1))  # Compute the probability of each point being chosen as the next center
            centers[k] = X[np.random.choice(X.shape[0], p=probabilities)]  # Choose the next center randomly, with probabilities proportional to the distances from the previous centers
        
        return centers

    # Public methods
    def fit(self, X, link_labels=None, cost_metric=None, tolerance=1e-5, max_steps=100, descent_rate=0.1, max_descents=3):
        """
        This function implements the k-GenCenters algorithm.
        The function iteratively updates the centers and the labels of the data points until convergence.

        :param X: NumPy array of shape (n_samples, n_features)
        :param cost_metric: String. One of 'squared_euclidean', 'euclidean', 'manhattan', 'Lp' (where p is a number at least unity), and 'euclidean^n' (where n is a positive integer).
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
                if cost_metric == 'euclidean' and (tolerance == 1e-5 or max_steps) == 100:
                    print('Notice: For the cost metric "euclidean", centers are updated using the Weiszfeld algorithm, which is iterative. \n'
                            'You have the option of defining the keyword arguments "tolerance" and "max_steps" when calling KGenCenters.fit(). \n'
                            'If you omit these, the default values (1e-5 and 100, respectively) from utils.py will be used for the iterative algorithm.')
                elif (
                    self._is_Lp(cost_metric) or self._is_euclidean_power(cost_metric)
                     and (tolerance==1e-5 or descent_rate==0.1 or max_descents==3)
                     ):
                    print('Notice: For cost metrics of the form "Lp" and "euclidean^n", centers are updated using gradient descent. \n'
                            'You have the option of defining the keyword arguments "tolerance", "descent_rate", and "max_descents" when calling KGenCenters.fit(). \n'
                            'If you omit these, the default values (1e-5, 0.1, and 3, respectively) from utils.py will be used.')
        if cost_metric not in ['squared_euclidean', 'manhattan', 'euclidean'] and not self._is_Lp(cost_metric) and not self._is_euclidean_power(cost_metric):
            raise ValueError('Invalid cost metric. Please see the documentation for valid choices.')

        # Initialize the centers
        if self.init == 'forgy':
            self.centers = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        elif self.init == 'random_partition':
            # Create random assignment for each data point
            random_asst = np.random.randint(0, self.n_clusters, size=X.shape[0])
            self.centers = np.array([X[random_asst == i].mean(axis=0) for i in range(self.n_clusters)])
        elif self.init == '++':
            if self.plusplus_dist is not None:
                self.centers = self._plusplus(X, self.n_clusters, cost_metric=self.plusplus_dist, random_state=self.random_state)
            else:
                self.centers = self._plusplus(X, self.n_clusters, cost_metric=cost_metric, random_state=self.random_state)
        else:
            raise ValueError('Invalid init argument. Please choose from "forgy", "random_partition", and "++".')
        
        # Create an array devoid of constraints if link_labels was not provided
        if link_labels is None:
            link_labels = np.full(X.shape[0], np.nan) # Label the unconstrained points with NaN
        # Initialize the cluster assignments to -1
        self.assigns = np.full(X.shape[0], -1)
        # Main loop of the KGenCenters algorithm
        for _ in range(self.max_iter):
            # Compute distances between all points and all centers
            distances = self._calculate_distances(X, cost_metric=cost_metric, centers=self.centers)
            constraint_labels = np.unique(link_labels)
            # Remove NaN from unique_labels
            constraint_labels = constraint_labels[~np.isnan(constraint_labels)]
            
            for label in constraint_labels:
                indices = np.where(link_labels == label)[0]
                # Get the sum of distances between each center and the linked points
                sum_distances = np.sum(distances[indices], axis=0)
                # Unanimously assign linked points to the center that minimizes the sum of distances
                self.assigns[indices] = np.argmin(sum_distances)

            self.assigns[np.isnan(link_labels)] = np.argmin(distances[np.isnan(link_labels)], axis=1)  # Assign unconstrained points to the closest center

            # Update centers to be the cost-minimizer of each cluster
            new_centers = self._update_centers(X, self.assigns, self.n_clusters, self.centers, cost_metric=cost_metric, 
                                                   tolerance=tolerance, max_steps=max_steps, descent_rate=descent_rate, max_descents=max_descents)
            # Check for convergence
            if np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers
    
    def predict(self, X, cost_metric=None):
        """
        Predict the cluster to which each point in X belongs

        For each sample in X, compute the distances to all centers and assign the sample to the closest cluster.

        :param X: NumPy array of shape (n_samples, n_features), the input samples.
        :param cost_metric: String. One of 'squared_euclidean', 'euclidean', 'manhattan', 'Lp' (where p is a number at least unity), and 'euclidean^n' (where n is a positive integer).
        :return: Labels of each point
        """
        if cost_metric is None:
            raise ValueError('You must specify a cost metric when calling KGenCenters.predict().')

        # Calculate the distance from each point to each center
        distances = self._calculate_distances(X, cost_metric=cost_metric, centers=self.centers)

        # Return the index of the closest center for each point
        return np.argmin(distances, axis=1)
    
    def evaluate(self, true_labels):
        """
        Evaluate the clustering against the true labels (if they are available) for accuracy

        :param true_labels: NumPy array of shape (n_samples,), the true labels for each sample
        :return: Float. The accuracy of the clustering
        """
        if self.assigns is None:
            raise ValueError('You must call KGenCenters.fit() before evaluating the clustering.')
        
        # Calculate the confusion matrix between true and predicted labels
        confusion = skm.confusion_matrix(true_labels, self.assigns)
        # Perform a linear sum assignment (also known as the Hungarian algorithm) to maximize the accuracy
        row_ind, col_ind = spo.linear_sum_assignment(confusion, maximize=True)
        # Calculate accuracy
        accuracy = confusion[row_ind, col_ind].sum() / np.sum(confusion)

        return accuracy
    
    def inertia(self, X, cost_metric=None):
        """
        Calculate the inertia of the clustering

        :param X: NumPy array of shape (n_samples, n_features), the input samples.
        :param cost_metric: String. One of 'squared_euclidean', 'euclidean', 'manhattan', 'Lp' (where p is a number at least unity), and 'euclidean^n' (where n is a positive integer).
        :return: Float. The inertia of the clustering
        """
        if cost_metric is None:
            raise ValueError('You must specify a cost metric when calling KGenCenters.inertia().')
        if self.assigns is None:
            raise ValueError('You must call KGenCenters.fit() before calculating inertia.')
        
        # Calculate the distance from each point to each center
        distances = self._calculate_distances(X, cost_metric=cost_metric, centers=self.centers)
        # Calculate inertia
        inertia = np.sum(distances[np.arange(X.shape[0]), self.assigns])

        return inertia

    def voronoi(self, cost_metric=None, x_range=(-10, 10), y_range=(-10, 10), resolution=100):
        """
        Calculate the boundaries of the Voronoi diagram (two-dimensional only)

        :param cost_metric: String. One of 'squared_euclidean', 'euclidean', 'manhattan', 'Lp' (where p is a number at least unity), and 'euclidean^n' (where n is a positive integer).
        :param x_range: Tuple. The range of x values to consider for the Voronoi diagram.
        :param y_range: Tuple. The range of y values to consider for the Voronoi diagram.
        :param resolution: Integer. The number of points along each axis to consider for the Voronoi diagram.
        :return: NumPy array. Contains True on the boundaries of the Voronoi diagram (i.e., points equidistant from multiple centers) and False elsewhere.
        """
        if cost_metric is None:
            raise ValueError('You must specify a cost metric when calling KGenCenters.voronoi().')
        if self.centers is None:
            raise ValueError('You must call KGenCenters.fit() before calculating the Voronoi diagram.')
        
        # Create a grid of points
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        xx, yy = np.meshgrid(x, y)
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Calculate the distance from each point to each center
        distances = self._calculate_distances(grid, cost_metric=cost_metric, centers=self.centers)
        # Calculate the cluster assignment for each point
        assigns = np.argmin(distances, axis=1).reshape(resolution, resolution)

        # Calculate the Voronoi boundaries
        boundaries_x = np.diff(assigns, axis=0) != 0  # Boundaries in the x direction
        boundaries_y = np.diff(assigns, axis=1) != 0  # Boundaries in the y direction

        # Pad the arrays to have the same shape
        boundaries_x = np.pad(boundaries_x, ((0, 1), (0, 0)), 'constant', constant_values=0)
        boundaries_y = np.pad(boundaries_y, ((0, 0), (0, 1)), 'constant', constant_values=0)

        # Combine the boundaries
        boundaries = np.logical_or(boundaries_x, boundaries_y)

        # Add a border of False values to ensure the array shapes match
        boundaries = np.pad(boundaries, pad_width=1, mode='constant', constant_values=False)

        return boundaries
