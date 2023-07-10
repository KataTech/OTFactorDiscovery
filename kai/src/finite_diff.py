import numpy as np
import matplotlib.pyplot as plt 

def perturb_X(X, epsilon, direction, data_id): 
    """
    Perturbed each vector in X to move in the direction of the input vector 
    by a magnitude of epsilon. Supplemental to the finite difference checking method. 
    
    Inputs: 
    - X: a matrix to perturb where each row represents a separate data point
    - epsilon: the magnitude of the perturbation
    - direction: the direction of perturbation, typically one of the standard basis vectors
    - data_id: the data entry in X to experience the perturbation
    """
    # Create a copy of the original X 
    X_perturbed = X.copy()
    X_perturbed[data_id, :] += epsilon * direction
    return X_perturbed

def finite_difference(eval_f, epsilon, data_matrix, data_id, parameters, test_vectors = None): 
    """
    Perform a gradient approximation of the objective function, eval_f, using the finite difference method. 

    Inputs: 
    - eval_f: a function that evaluates the objective given the data_matrix and other parameters
    - epsilon: the magnitude of perturbation used in the finite difference computation
    - data_matrix: the data matrix associated with eval_f, should be a n-by-d numpy array 
    - data_id: the observation that we are approximating the gradient with respect to
    - parameters: additional parameters to eval_f
    - test_vectors: the collection of directional vectors for computing finite difference, 
                    should be a d-by-d numpy array
    
    Returns a d-dimensional vector representing the derivating of eval_f with respect to data_id
    element of the data_matrix. 
    """
    return [(eval_f(perturb_X(data_matrix, epsilon, direction, data_id), *parameters) - eval_f(perturb_X(data_matrix, -1 * epsilon, direction, data_id), *parameters)) / (2 * epsilon) for direction in test_vectors]

def comp(approx, actual, mode = "euclidean"):
    """
    Compare the approximation and the actual. If mode is "euclidean", then we compute the euclidean distance. 
    """
    if mode == "euclidean": 
        return np.linalg.norm(approx - actual)
    raise NotImplementedError

def plot_finite_difference(data_matrix, params_f, params_grad, data_id, eval_f, grad_f, test_vectors, epsilons, log_scale=True): 
    """
    Plot the change in finite difference approximation and direct computation of the gradient as a
    function of the epsilon value for a specific evaluation of an objective function. 

    Inputs: 
    - data_matrix: a n-by-d data matrix that serves as the optimization variable where n is the 
                   number of samples and d is the dimension of each sample. 
    - params_f: the rest of the parameter expected of the eval_f function
    - params_grad: the rest of the parameter expected of the grad_f function
    - data_id: the index of the data point for which we are interested in
    - eval_f: the objective function to optimize over
    - grad_f: the gradient function
    - test_vectors: the set of directional vectors, typically this should be the standard basis in R^d. 
    - epsilons: the set of epsilon values to plot. 

    Outputs a visualization of the change in approximation error across epsilon values. 
    """
    errors = []
    for epsilon in epsilons: 
        fin_diff = finite_difference(eval_f, epsilon, data_matrix, data_id, params_f, test_vectors)
        actual_grad = grad_f(data_matrix, *params_grad)[data_id]
        errors.append(comp(fin_diff, actual_grad))
    errors = np.array(errors)
    if log_scale: 
        plt.plot(np.log(epsilons), np.log(errors))
    else:     
        plt.plot(epsilon, errors)
    plt.xlabel("Epsilon")
    plt.ylabel("L2 Norm of Gradient Difference")
    plt.title("Change in Gradient Difference over Epsilons")
    plt.show()
    