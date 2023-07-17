"""
Author: Daniel Wang
Date: 2023-06-23
"""

import numpy as np
import scipy.optimize as spo
import sklearn.metrics as skm
import utils

class KQuasicenters:
    """
    This class must be initialized with user-defined values of n_clusters and max_iter.
    Otherwise, the default values (3 and 300, respectively) will be used.
    """
    def __init__(self, n_clusters=3, init='forgy', max_iter=300, random_state=None, verbose=True):
        """
        :param [n_clusters]: Integer. The number of clusters to form as well as the number of optimoids to generate.
        :param [init]: String. The method used to initialize the optimoids. One of 'forgy', 'random partition', and '++'.
        :param [max_iter]: Integer. The maximum number of iterations of the k-quasicenters algorithm for a single run.
        :param [verbose]: Boolean. Whether to print warnings and suggestions.

        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.optimoids = None
        self.assigns = None
    
    def fit(self, X, cost_metric='squared_euclidean', tolerance=1e-5, max_steps=100, descent_rate=0.1, max_descents=3):
        """
        This function implements the k-quasicenters algorithm.

        The function iteratively updates the optimoids and the labels of the data points until convergence.

        :param X: NumPy array of shape (n_samples, n_features)
        :param [cost_metric]: String. One of 'squared_euclidean', 'euclidean', 'manhattan', 'Lp' (where p is a positive integer), and 'euclidean^n' (where n is a positive integer).
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
                    print('Notice: For the cost metric "euclidean", optimoids are updated using the Weiszfeld algorithm, which is iterative. \n'
                            'You have the option of defining the keyword arguments "tolerance" and "max_steps" when calling KQuasicenters.fit(). \n'
                            'If you omit these, the default values (1e-5 and 100, respectively) from utils.py will be used for the iterative algorithm.')
                elif (
                    utils.is_Lp(cost_metric) or utils.is_euclidean_power(cost_metric)
                     and (tolerance==1e-5 or descent_rate==0.1 or max_descents==3)
                     ):
                    print('Notice: For cost metrics of the form "Lp" and "euclidean^n", optimoids are updated using gradient descent. \n'
                            'You have the option of defining the keyword arguments "tolerance", "descent_rate", and "max_descents" when calling KQuasicenters.fit(). \n'
                            'If you omit these, the default values (1e-5, 0.1, and 3, respectively) from utils.py will be used.')
        if cost_metric not in ['squared_euclidean', 'manhattan', 'euclidean'] and not utils.is_Lp(cost_metric) and not utils.is_euclidean_power(cost_metric):
            raise ValueError('Invalid cost metric. Please see the documentation for valid choices.')

        # Initialize the optimoids
        if self.init == 'forgy':
            self.optimoids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        elif self.init == 'random_partition':
            # Create random assignment for each data point
            random_asst = np.random.randint(0, self.n_clusters, size=X.shape[0])
            self.optimoids = np.array([X[random_asst == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # Create a color scheme for the labels
            colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
            color_dict = {i: color for i, color in enumerate(colors[:self.n_clusters])}
            import matplotlib.pyplot as plt
            # Create a plot of the initial assignments and optimoids
            plt.figure(figsize=(8, 6))
            plt.scatter(X[:, 0], X[:, 1], c=[color_dict[label] for label in random_asst], s=20)
            plt.scatter(self.optimoids[:, 0], self.optimoids[:, 1], marker='X', c='red', s=200, zorder=9)
            plt.title('Initial Assignments and Optimoids (random_partition)', size=16)
            plt.show()
        elif self.init == '++':
            self.optimoids = utils.plusplus(X, self.n_clusters, cost_metric=cost_metric, random_state=self.random_state)
        else:
            raise ValueError('Invalid init argument. Please choose from "forgy" and "++".')
        
        # Main loop of the KQuasicenters algorithm
        for _ in range(self.max_iter):
            # Assign labels to each data point
            self.assigns = self.predict(X, cost_metric=cost_metric)
            
            # Update optimoids to be the cost-minimizer of each cluster
            new_optimoids = utils.update_optimoids(X, self.assigns, self.n_clusters, self.optimoids, cost_metric=cost_metric, 
                                                   tolerance=tolerance, max_steps=max_steps, descent_rate=descent_rate, max_descents=max_descents)
            # Check for convergence
            if np.allclose(self.optimoids, new_optimoids):
                break
            self.optimoids = new_optimoids
    
    def predict(self, X, cost_metric=None):
        """
        Predict the closest cluster each sample in X belongs to.

        For each sample in X, compute the distances to all optimoids and assign the sample to the closest cluster.

        :param X: NumPy array of shape (n_samples, n_features), the input samples.
        :param [cost_metric]: String. One of 'squared_euclidean', 'euclidean', 'manhattan', 'Lp' (where p is a positive integer), and 'euclidean^n' (where n is a positive integer).
        :return: Labels of each point
        """
        # Calculate the distance from each point to each optimoid
        distances = utils.calculate_distances(X, cost_metric=cost_metric, optimoids=self.optimoids)

        # Return the index of the closest optimoid for each point
        return np.argmin(distances, axis=1)
    
    def evaluate(self, true_labels):
        """
        Evaluate the clustering performance for accuracy

        :param true_labels: NumPy array of shape (n_samples,), the true labels for each sample
        :return: Float. The accuracy of the clustering
        """
        if self.assigns is None:
            raise ValueError('You must call KQuasicenters.fit() before evaluating the clustering.')
        
        # Calculate the confusion matrix between true and predicted labels
        confusion = skm.confusion_matrix(true_labels, self.assigns)
        # Perform a linear sum assignment (also known as the Hungarian algorithm) to maximize the accuracy
        row_ind, col_ind = spo.linear_sum_assignment(confusion, maximize=True)
        # Calculate accuracy
        accuracy = confusion[row_ind, col_ind].sum() / np.sum(confusion)

        return accuracy

        
