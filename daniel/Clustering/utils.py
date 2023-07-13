"""
Author: Daniel Wang
Date: 2023-06-23
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


def update_quasicenters(X, labels, n_clusters, quasicenters, cost_metric=None, tolerance=None, max_steps=None, descent_rate=None, max_descents=None):
    """
    This function updates the quasicenters based on the given cost metric.
    
    :param X: The input data.
    :param labels: The cluster labels for the data.
    :param n_clusters: The number of clusters.
    :param quasicenters: The current quasicenters.
    :param cost_metric: The cost metric used for clustering. 
    :param tolerance: The tolerance for convergence.
    :param max_steps: The maximum number of steps for the Weiszfeld algorithm.
    :param descent_rate: The learning rate for gradient descent.
    :param max_descents: The maximum number of descents for gradient descent.
    :return: The updated quasicenters.
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
    
    quasicenters = np.copy(quasicenters)
    if cost_metric == 'squared_euclidean':
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                quasicenters[k] = np.mean(cluster_points, axis=0)
    elif cost_metric == 'manhattan':
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                quasicenters[k] = np.median(cluster_points, axis=0)
    elif cost_metric == 'euclidean':
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                quasicenters[k] = weiszfeld(cluster_points, tolerance, max_steps)
    elif is_Lp(cost_metric) or is_euclidean_power(cost_metric):
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
                    distance = np.abs(quasicenters[k] - x)
                    non_zero_indices = distance != 0
                    if np.any(non_zero_indices):  # Check if there are non-zero indices
                        grad += n * np.sum(distance[non_zero_indices]**p, axis=-1)**(n/p - 1) * distance[non_zero_indices]**(p - 1) * np.sign(quasicenters[k] - x)[non_zero_indices]
                grad[np.isnan(grad)] = 0  # Replace NaN values with zero
                if np.allclose(grad, 0, atol=tolerance):
                    break
                quasicenters[k] -= descent_rate * grad / num_points

    return quasicenters


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
    This function implements the k-means++ initialization method for the quasicenters.
    This method aims to provide a better initialization than random initialization, which can lead to bad local optima.
    
    :param X: The input data.
    :param n_clusters: The number of clusters.
    :param cost_metric: The cost metric used for clustering.
    :param random_state: The seed for the random number generator.
    :param verbose: Whether to print warnings and suggestions.
    :return: The initialized quasicenters.
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
    
    cost_metric = 'euclidean^9'
    quasicenters = np.empty((n_clusters, X.shape[1]))
    quasicenters[0] = X[np.random.randint(X.shape[0])]  # Choose the first quasicenter randomly
    for k in range(1, n_clusters):
        distances = calculate_distances(X, cost_metric, quasicenters[:k])
        probabilities = np.min(distances, axis=1) / np.sum(np.min(distances, axis=1))  # Compute the probability of each point being chosen as the next quasicenter
        quasicenters[k] = X[np.random.choice(X.shape[0], p=probabilities)]  # Choose the next quasicenter randomly, with probabilities proportional to the distances from the previous quasicenters
    
    return quasicenters


def calculate_distances(X, cost_metric=None, quasicenters=None):
    """
    This function calculates the distances from each data point to each quasicenter.
    
    :param X: The input data.
    :param cost_metric: The cost metric used for clustering.
    :param quasicenters: The quasicenters of the clusters.
    :return: The distances from each data point to each quasicenter.
    """
    if cost_metric is None:
        raise ValueError('No value for the argument "cost_metric" was received.')
    if quasicenters is None:
        raise ValueError('No value for the argument "quasicenters" was received.')
    
    if cost_metric == 'squared_euclidean':
        distances = np.linalg.norm(X[:, np.newaxis] - quasicenters, axis=-1)**2
    elif cost_metric == 'manhattan':
        distances = np.linalg.norm(X[:, np.newaxis] - quasicenters, ord=1, axis=-1)
    elif cost_metric == 'euclidean':
        distances = np.linalg.norm(X[:, np.newaxis] - quasicenters, axis=-1)
    elif is_Lp(cost_metric):
        p = int(cost_metric[1:])
        distances = np.linalg.norm(X[:, np.newaxis] - quasicenters, ord=p, axis=-1)
    elif is_euclidean_power(cost_metric):
        p = 2 # this exponent can be customized by end-users for non-euclidean distances
        n = int(cost_metric[10:])
        distances = np.linalg.norm(X[:, np.newaxis] - quasicenters, ord=p, axis=-1)**n
        
    return distances
