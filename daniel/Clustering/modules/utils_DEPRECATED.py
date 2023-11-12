"""
Author: Daniel Wang
Date: 2023-06-23

NOTE: THIS MODULE IS DEPRECATED. The functions in the script are private functions that were previously called from kgencenters.py.
The functions have been moved to the kgencenters.py script and are now called from there. This script is no longer used.
"""

import numpy as np
import re

def is_Lp(cost_metric):
    """
    This function checks whether the argument "cost_metric" has the format "Lp", where p is a positive integer.
    """
    pattern = r'^L([1-9]\d*)$'
    return re.match(pattern, cost_metric) is not None


def is_euclidean_power(cost_metric):
    """
    This function checks whether the argument "cost_metric" has the format "euclidean^n", where n is a positive integer.
    """
    pattern = r'^euclidean\^\d+$'
    return re.match(pattern, cost_metric) is not None


def update_centers(X, labels, n_clusters, centers, cost_metric=None, tolerance=None, max_steps=None, descent_rate=None, max_descents=None):
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
                centers[k] = weiszfeld(cluster_points, tolerance, max_steps)
    elif is_Lp(cost_metric) or is_euclidean_power(cost_metric):
    # FIXME: For high values of n, the cost metric euclidean^n produces overflow errors. Users are welcome to contribute a fix (e.g. scaling max(abs()) of the data to 1)
    # However, care should be taken to perform the inverse transform on all output values to avoid confusing the end user
        if is_Lp(cost_metric):
            p = int(cost_metric[1:])
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


def weiszfeld(X, tolerance, max_steps):
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


def plusplus(X, n_clusters, cost_metric=None, random_state=None, verbose=True):
    """
    This function implements the k-means++ initialization method for the centers.
    This method aims to provide a better initialization than random initialization, which can lead to bad local optima.
    
    :param X: The input data.
    :param n_clusters: The number of clusters.
    :param cost_metric: The cost metric used for clustering.
    :param random_state: The seed for the random number generator.
    :param verbose: Whether to print warnings and suggestions.
    :return: The initialized centers.
    """
    if cost_metric is None:
        raise ValueError('No value for the argument "cost_metric" was received.')
    if random_state is None:
        np.random.seed(0)
        if verbose:
            print('Warning: No value for the argument "random_state" was received by the utils.plusplus() initialization. It is recommended that you set this for reproducibility. \n'
                'Defaulting to random_state=0.')
    else:
        np.random.seed(random_state)
    
    centers = np.empty((n_clusters, X.shape[1]))
    centers[0] = X[np.random.randint(X.shape[0])]  # Choose the first center randomly
    for k in range(1, n_clusters):
        distances = compute_distances(X, cost_metric, centers[:k])
        probabilities = np.min(distances, axis=1) / np.sum(np.min(distances, axis=1))  # Compute the probability of each point being chosen as the next center
        centers[k] = X[np.random.choice(X.shape[0], p=probabilities)]  # Choose the next center randomly, with probabilities proportional to the distances from the previous centers
    
    return centers


def compute_distances(X, cost_metric=None, centers=None):
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
    elif is_Lp(cost_metric):
        p = int(cost_metric[1:])
        distances = np.linalg.norm(X[:, np.newaxis] - centers, ord=p, axis=-1)
    elif is_euclidean_power(cost_metric):
        p = 2 # this exponent can be customized by end-users for non-euclidean distances
        n = int(cost_metric[10:])
        distances = np.linalg.norm(X[:, np.newaxis] - centers, ord=p, axis=-1)**n
        
    return distances
