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
    pattern = r'^L\d+$'
    return re.match(pattern, cost_metric) is not None

def update_centroids(X, labels, n_clusters, cost_metric, tolerance=1e-6, max_iterations=100, **kwargs):
    centroids = np.zeros((n_clusters, X.shape[1]))

    if cost_metric == 'squared_euclidean':
        for k in range(n_clusters):
            centroids[k] = np.mean(X[labels == k], axis=0)
    elif cost_metric == 'manhattan':
        for k in range(n_clusters):
            centroids[k] = np.median(X[labels == k], axis=0)
    elif cost_metric == 'euclidean':
        for k in range(n_clusters):
            centroids[k] = weiszfeld(X[labels == k], tolerance, max_iterations)

    elif is_Lp(cost_metric) and int(cost_metric[1:])!= 2:
        p = int(cost_metric[1:])
        for k in range(n_clusters):
            pass
    else:
        for k in range(n_clusters):
            centroids[k] = np.mean(X[labels == k], axis=0)

    return centroids

def weiszfeld(X, tolerance, max_iterations):
    """
    This function implements the Weiszfeld algorithm for computing the geometric median, as described here:
    https://en.wikipedia.org/wiki/Geometric_median#Computation (link is correct as of 2023-06-23)
    """
    num_points, _ = X.shape

    # Initialize the geometric median as the mean of the points
    geometric_median = np.mean(X, axis=0)

    for iteration in range(max_iterations):
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