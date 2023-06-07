"""
Module Description: This module contains helpful definitions for the saddle point optimizer.
Author: Daniel Wang
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SaddlePointOptimizer:
    """
    func MUST be a Sympy expression of x and y.
    """
    def __init__(self, func):
        self.L = func
        # Define x and y as symbols
        x, y = sp.symbols('x y', real=True)
        # Calculate the gradients
        self.grad_x = sp.diff(self.L, x)
        self.grad_y = sp.diff(self.L, y)
        # Store the gradients as a column vector
        self.grad = sp.Matrix([[self.grad_x], [self.grad_y]])

    def optimize(self, x_init, y_init, eta=0.1, max_steps=1e4, tolerance=1e-5, make_plot=True):
        # Initialize the starting point as a NumPy array
        z = np.array([[x_init], [y_init]], dtype=float)
        # Define the matrix J
        J = np.array([[1, 0], [0, -1]])
        # Create an array to store the values of z during optimization
        z_values = np.empty((2, int(max_steps + 1)))
        now_step = 0

        while now_step < max_steps:
            x, y = sp.symbols('x y', real=True)
            # Substitute the current z values into the gradient
            G = np.array(self.grad.subs({x: z[0,0], y: z[1,0]}).evalf(), dtype=float)
            # Check if the norm of the gradient is below the tolerance
            if np.linalg.norm(G) < tolerance:
                break
            # Update z using the optimization step formula
            z -= eta * np.matmul(J, G)
            # Append the updated z value to the list
            z_values[:, now_step] = z.flatten()
            now_step += 1

        # Convert the list of z values to a NumPy array
        self.history = z_values[:, :now_step]

        if make_plot:
            self.plot()

    def plot(self):
        x_values = self.history[0, :]
        y_values = self.history[1, :]
        x, y = sp.symbols('x y', real=True)
        # Define the range of x and y values for plotting
        x_range = np.linspace(min)
        y_range = np.linspace(-10, 10, 100)
        # Create a meshgrid from the x and y ranges
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        # Create a callable function for the expression self.L
        func_callable = sp.lambdify((x, y), self.L, modules='numpy')
        # Evaluate the function at each point of the meshgrid
        z_mesh = func_callable(x_mesh, y_mesh)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the function surface
        ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis')

        # Extract the x, y, and z values from the optimization history
        z_values_x = self.history[:, 0]
        z_values_y = self.history[:, 1]
        z_values_z = np.array([self.L.subs({x: x_val, y: y_val}).evalf() for x_val, y_val in zip(z_values_x, z_values_y)])

        # Plot the path traced by z during optimization
        ax.plot(z_values_x, z_values_y, z_values_z, 'r.-')

        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('L')
        ax.set_title('Optimization Process')

        # Enable mouse rotation
        ax.mouse_init()

        # Show the plot
        plt.show()

