"""
Module Description: This module contains helpful definitions for the saddle point optimizer.
Author: Daniel Wang
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SaddlePoint:
    """
    func MUST be a Sympy expression of x and y.
    """
    def __init__(self, func):
        self.L = func
        # Define x and y as symbols
        x, y = sp.symbols('x y', real=True)
        # Calculate the gradient
        self.grad = sp.Matrix([sp.diff(self.L, x), sp.diff(self.L, y)])
        # Calculate the Hessian
        self.hess = sp.hessian(self.L, (x, y))

    def optimize(self, xy_init, algo="naive_gradient", eta=0.1, max_steps=1e4, tolerance=1e-5):
        # Initialize the starting point as a NumPy array
        z = np.array([[xy_init[0]], [xy_init[1]]], dtype=float)
        # Define the matrix J
        J = np.array([[1, 0], [0, -1]])
        # Create an array to store the values of z during optimization
        z_values = np.empty((2, int(max_steps)))
        # Store the initial z value in the array
        z_values[:, 0] = z.flatten()
        # Initialize the step counter
        now_step = 1
        # Define x and y as symbols
        x, y = sp.symbols('x y', real=True)

        if algo == "naive_gradient":
            while now_step < max_steps:
                # Substitute the current z values into the gradient
                G = np.array(self.grad.subs({x: z[0,0], y: z[1,0]}).evalf(), dtype=float)
                # Check if the norm of the gradient is below the tolerance
                if np.linalg.norm(G) < tolerance:
                    break
                # Update z
                z -= eta * np.matmul(J, G)
                # Store the current z value in the array
                z_values[:, now_step] = z.flatten()
                now_step += 1
        elif algo == "implicit_gradient":
            while now_step < max_steps:
                # Substitute the current z values into the gradient
                G = np.array(self.grad.subs({x: z[0,0], y: z[1,0]}).evalf(), dtype=float)
                # Substitute the current z values into the hessian
                H = np.array(self.hess.subs({x: z[0,0], y: z[1,0]}).evalf(), dtype=float)
                # Check if the norm of the gradient is below the tolerance
                if np.linalg.norm(G) < tolerance:
                    break
                # Update z
                z -= eta * np.matmul(np.linalg.inv(J+eta*H), G)
                # Store the current z value in the array
                z_values[:, now_step] = z.flatten()
                now_step += 1
        else:
            raise ValueError("Invalid algorithm choice.")
        

        # Convert the list of z values to a NumPy array
        self.history = z_values[:, :now_step]


    def plot(self, pitch, yaw):
        reach = np.max(np.abs(self.history))
        # Extract the x and y values from the optimization history
        x_values = self.history[0, :]
        y_values = self.history[1, :]
        x, y = sp.symbols('x y', real=True)

        # Create a callable function for the expression self.L
        func_callable = sp.lambdify((x, y), self.L, modules='numpy')

        # Define the range of x and y values for plotting based on data
        x_range = np.linspace(-1.1 * reach, 1.1 * reach, 100)
        y_range = np.linspace(-1.1 * reach, 1.1 * reach, 100)

        # Create a meshgrid from the x and y ranges
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)

        # Evaluate the function at each point of the meshgrid
        f_mesh = func_callable(x_mesh, y_mesh)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the function surface
        ax.plot_surface(x_mesh, y_mesh, f_mesh, cmap='viridis')

        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('L')
        ax.set_title('Optimization Process')

        # Set the perspective of the plot
        ax.view_init(elev=pitch, azim=yaw)

        # Extract the x, y, and z values from the optimization history
        f_values = func_callable(x_values, y_values)

        # Plot the path traced by z during optimization as shaded spheres
        ax.plot(x_values, y_values, f_values, c='r', linewidth=0.5, zorder=10)

        # Show the plot
        plt.tight_layout()  # Prevent overlap of labels
        plt.show()

