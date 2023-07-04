"""
Author: Daniel Wang
Date: 2023-06-23
"""
import numpy as np
import re

def is_Lp(cost_metric):
    """
    This function checks if the argument "cost_metric" has the format "Lp", where p is a positive integer.
    """
    pattern = r'^L([1-9]\d*)$'
    return re.match(pattern, cost_metric) is not None

def is_euclidean_power(cost_metric):
    """
    This function checks if the argument "cost_metric" has the format "euclidean^n", where n is a positive integer.
    """
    pattern = r'^euclidean\^\d+$'
    return re.match(pattern, cost_metric) is not None


def update_centroids(X, labels, n_clusters, centroids, cost_metric=None, tolerance=None, max_steps=None, descent_rate=None, **kwargs):
    if cost_metric is None:
        raise ValueError('No value for the argument "cost_metric" was received.')
    if tolerance is None:
        raise ValueError('No value for the argument "tolerance" was received.')
    if max_steps is None:
        raise ValueError('No value for the argument "max_steps" was received.')
    if descent_rate is None:
        raise ValueError('No value for the argument "descent_rate" was received.')
    
    centroids = np.copy(centroids)

    if cost_metric == 'squared_euclidean':
        for k in range(n_clusters):
            centroids[k] = np.mean(X[labels == k], axis=0)
    elif cost_metric == 'manhattan':
        for k in range(n_clusters):
            centroids[k] = np.median(X[labels == k], axis=0)
    elif cost_metric == 'euclidean':
        for k in range(n_clusters):
            centroids[k] = weiszfeld(X[labels == k], tolerance, max_steps)
    elif is_Lp(cost_metric):
        p = int(cost_metric[1:])
        for k in range(n_clusters):   
            grad = np.zeros(X.shape[1])
            num_points = np.sum(labels == k)  # Number of points in cluster k
            for x in X[labels == k]:
                distance = np.abs(centroids[k] - x)
                non_zero_indices = distance != 0
                if np.any(non_zero_indices):  # Check if there are non-zero indices
                    grad += np.sum(distance[non_zero_indices]**p, axis=-1)**(1/p - 1) * distance[non_zero_indices]**(p - 1) * np.sign(centroids[k] - x)[non_zero_indices]
            grad[np.isnan(grad)] = 0  # Replace NaN values with zero
            centroids[k] -= descent_rate * grad / num_points
    elif is_euclidean_power(cost_metric):
        p = 2 # this exponent can be customized by end-users for non-euclidean distances
        n = int(cost_metric[10:])
        for k in range(n_clusters):
            grad = np.zeros(X.shape[1])
            num_points = np.sum(labels == k)  # Number of points in cluster k
            for x in X[labels == k]:
                distance = np.abs(centroids[k] - x)
                non_zero_indices = distance != 0
                if np.any(non_zero_indices):  # Check if there are non-zero indices
                    grad += n * np.sum(distance[non_zero_indices]**p, axis=-1)**(2/p - 1) * distance[non_zero_indices]**(p - 1) * np.sign(centroids[k] - x)[non_zero_indices]
            grad[np.isnan(grad)] = 0  # Replace NaN values with zero
            centroids[k] -= descent_rate * grad / num_points

    return centroids

def calculate_distances(X, cost_metric=None, centroids=None, **kwargs):
    if cost_metric is None:
        raise ValueError('No value for the argument "cost_metric" was received.')
    if centroids is None:
        raise ValueError('No value for the argument "centroids" was received.')
    
    if cost_metric in ['euclidean', 'squared_euclidean']:
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1) # missing the squared case, but not necessary since x^2 is monotonic
    elif cost_metric == 'manhattan':
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, ord=1, axis=-1)
    elif is_Lp(cost_metric):
        p = int(cost_metric[1:])
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, ord=p, axis=-1)
    elif is_euclidean_power(cost_metric):
        p = 2 # this exponent can be customized by end-users for non-euclidean distances
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, ord=p, axis=-1) # missing the n-th power, but not necessary since x^n is monotonic
    return distances


def weiszfeld(X, tolerance, max_steps):
    """
    This function implements the Weiszfeld algorithm for computing the geometric median, as described here:
    https://en.wikipedia.org/wiki/Geometric_median#Computation (link is correct as of 2023-06-23)
    """
    num_points, _ = X.shape

    # Initialize the geometric median as the mean of the points
    geometric_median = np.mean(X, axis=0)

    for iteration in range(max_steps):
        # Calculate the distances from the current geometric median to all points
        distances = np.linalg.norm(X - geometric_median, axis=1)

        # Update the geometric median
        weights = 1 / distances
        weights /= np.sum(weights)  # Normalize the weights
        new_geometric_median = np.dot(weights, X)

        # Check if the algorithm has converged
        if np.linalg.norm(new_geometric_median - geometric_median) < tolerance:
            break
        else:
            geometric_median = new_geometric_median

    return geometric_median