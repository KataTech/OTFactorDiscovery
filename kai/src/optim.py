import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from .loss import *

def etgw(L: Loss, x_init, y_init, lr, epsilon=0.001, max_iter = 1000): 
    """
    Explicit Twisted Gradient Descent (ETGD)

    This function implements the ETGD algorithm from
    Essid, Tabak, and Trigila (2019). 
    Notably, the optimization here is: 

            min_x max_y L(x, y)

    Inputs: 
        - L: the loss function, should take x and y as arguments
        - x_init: the initial value of x
        - y_init: the initial value of y
        - lr: the learning rate
        - epsilon: the stopping criterion
        - max_iter: the maximum number of iterations
    
    Outputs: 
        - x: the final value of x
        - y: the final value of y
        - x_list: a list of all x values
        - y_list: a list of all y values
    """
    # set initial values 
    x = x_init
    y = y_init
    # initialize lists to store values
    x_list = [x]
    y_list = [y]
    gradient_norm = [np.linalg.norm(L.gradient(x, y, as_numpy=True))]
    # initialize iteration counter
    i = 0
    # iterate until convergence
    while i < max_iter: 
        grad_L = L.gradient(x, y)
        x = x - lr * grad_L[0]
        y = y + lr * grad_L[1]
        x_list.append(x)
        y_list.append(y)
        gradient_norm.append(np.linalg.norm(L.gradient(x, y, as_numpy=True)))
        if gradient_norm[-1] < epsilon:
            break
        i += 1
    return x, y, x_list, y_list, gradient_norm

def itgw(L: Loss, x_init, y_init, lr, epsilon=0.001, max_iter = 1000):
    """ 
    Implicit Twisted Gradient Descent (ITGD)

    This function implements the ITGD algorithm from
    Essid, Tabak, and Trigila (2019).
    Notably, the optimization here is: 

            min_x max_y L(x, y)
    
    Different from the explicit version, the implicit version 
    attempts to leverage the gradient at the next iteration. 
    Since computing the gradient of the next point is difficult,
    we instead use gradient and hessian of the current point to 
    approximate it. 

    Inputs: 
        - L: the loss function, should take x and y as arguments
        - x_init: the initial value of x
        - y_init: the initial value of y
        - lr: the learning rate
        - epsilon: the stopping criterion
        - max_iter: the maximum number of iterations
    """
    # set initial values 
    x = x_init
    y = y_init
    # initialize lists to store values
    x_list = [x]
    y_list = [y]
    gradient_norm = [np.linalg.norm(L.gradient(x, y, as_numpy=True))]
    # initialize iteration counter
    i = 0
    # iterate until convergence
    while i < max_iter: 
        grad_L = L.gradient(x, y, as_numpy=True)
        hess_L = L.hessian(x, y, as_numpy=True)
        z = np.array([x, y]) - lr * np.linalg.inv(np.array([[1, 0], [0, -1]]) + lr * hess_L) @ grad_L
        x = z[0]
        y = z[1]
        x_list.append(x)
        y_list.append(y)
        gradient_norm.append(np.linalg.norm(L.gradient(x, y, as_numpy=True)))
        if gradient_norm[-1] < epsilon:
            break
        i += 1
    return x, y, x_list, y_list, gradient_norm


def plot_3D(f, x_list = None, y_list = None, grid = [[-50, 50], [-50, 50]], fineness = 0.2): 
    """
    Code adapted from Chapter 12 of "Python Programming and Numerical Methods" by Kong, Siauw, and Bayen. 
    Link: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter12.02-3D-Plotting.html    

    Inputs: 
        - f: the function to plot
        - x_list: a list of x values to plot on the function surface
        - y_list: a list of y values to plot on the function surface
        - grid: the domain of the function
        - fineness: the fineness of the mesh grid
    """
    # set-up 3D plot
    fig = plt.figure(figsize = (12,10))
    ax = plt.axes(projection='3d')
    # define the domain of the function
    x = np.arange(grid[0][0], grid[0][1], fineness)
    y = np.arange(grid[1][0], grid[1][1], fineness)
    # generate the mesh grid for the function
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)
    # visualize the function
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    fig.colorbar(surf, shrink=0.5, aspect=8)
    # visualize the path of the optimization algorithm if provided
    if x_list is not None and y_list is not None:
        x_arr = np.array(x_list)
        y_arr = np.array(y_list)
        ax.plot(x_arr, y_arr, zs = f(x_arr, y_arr), zdir = 'z', marker = 'o', color = 'r', linewidth = 2)
    plt.show()

def vis_gradient(y_list, y_axis_title = "Gradient Norm", x_axis_title = "Iteration", x_list = None): 
    """
    Visualize the gradient norm over iterations of an optimization procedure, in order to determine convergence. 

    Inputs: 
        - y_list: a list of gradient norms per iteration. 
        - y_axis_title: the title of the y-axis
        - x_axis_title: the title of the x-axis
        - x_list: a list of iteration numbers.
    """
    if x_list is None: 
        x_list = np.arange(len(y_list))
    y_list = np.array(y_list)
    plt.plot(x_list, y_list)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.show()

def run_trials(TRIALS, f, grad_f, hess_f): 
    """
    Run multiple trials of ITGD to find the optimal point of a function.

    Inputs:
        - TRIALS: the number of trials to run
        - f: the function to optimize
        - grad_f: the gradient of the function
        - hess_f: the hessian of the function
    """
    np.random.seed(0)
    TRIALS = 100
    x_init = np.random.randint(0, 1, TRIALS)
    y_init = np.random.randint(0, 1, TRIALS)
    x_opts = []
    y_opts = []

    for trial in range(TRIALS):
        x_optim, y_optim, _, _, _ = itgw(f, grad_f, hess_f, x_init[trial], y_init[trial], 0.01, epsilon=0.0001, max_iter = 1000) 
        x_opts.append(x_optim)
        y_opts.append(y_optim)
    return x_opts, y_opts

def visualize_optimal_points(x_values, y_values, lr = 0.01):
    """
    Visualize the distribution of optimal points found by ITGD over multiple trials.
    Inputs: 
        x_values: list of x values of optimal points
        y_values: list of y values of optimal points
    """
    grid_size = 2  # Size of the grid
    resolution = 100  # Resolution of the heatmap (higher values give smoother results)
    
    # Create a grid
    grid = np.zeros((resolution, resolution))
    x_grid = np.linspace(0, grid_size, resolution)
    y_grid = np.linspace(0, grid_size, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute the density of optimal points on the grid
    for x, y in zip(x_values, y_values):
        # Find the nearest grid cell
        x_index = np.argmin(np.abs(x_grid - x))
        y_index = np.argmin(np.abs(y_grid - y))
        
        # Increment the density at that grid cell
        grid[y_index, x_index] += 1
    
    # Plot the heatmap
    plt.imshow(grid, cmap='hot', origin='lower', extent=[0, grid_size, 0, grid_size])
    plt.colorbar(label='Optimal Points Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Optimal Points Heatmap ($\epsilon = 0.0001, \eta = {lr}$)')
    plt.show()