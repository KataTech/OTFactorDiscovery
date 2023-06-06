"""
Module Description: This module contains helpful definitions for the saddle point optimizer.
Author: Daniel Wang
Date: June 5, 2023
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class Function:
    def __init__(self, func):
        self.L = sp.sympify(func)  # convert the lambda function to a SymPy expression
        self.grad_x, self.grad_y = self.calculate_gradients()  # Calculate the gradients with respect to x and y
        self.grad = sp.Matrix([self.grad_x, self.grad_y])  # Store the gradient as a matrix

    def calculate_gradients(self):
        x, y = sp.symbols('x y', real=True)  # Define symbols for variables x and y
        lag = self.L(x, y)  # Create a lagrangian expression
        grad_x = sp.diff(lag, x)  # Calculate the partial derivative with respect to x
        grad_y = sp.diff(lag, y)  # Calculate the partial derivative with respect to y
        return grad_x, grad_y
    
    def calculate_hessian(self):
        x, y = sp.symbols('x y', real=True)  # Define symbols for variables x and y
        lag = self.L(x, y)  # Evaluate the function being optimized
        hessian = sp.Matrix([[sp.diff(lag, var1, var2) for var1 in (x, y)] for var2 in (x, y)])  # Calculate the Hessian matrix
        return hessian
    
    def optimize(self, x_init, y_init, eta=0.1, max_steps=1e4, tolerance=1e-5, make_plot=True):
        now_step = 0
        z = np.array([x_init, y_init])  # Initialize z as a NumPy array
        J = sp.Matrix([[1, 0], [0, -1]])  # Define the matrix J
        z_values = [z.copy()]  # Create an empty list to store z values

        while G.norm() > tolerance and now_step < max_steps:
            G = self.grad.subs({x: z[0], y: z[1]})  # Substitute the current z values into the gradient
            z -= eta * J * G  # Update z using the optimization step formula
            z_values.append(z.copy())  # Append the updated z value to z_values
            now_step += 1

        self.history = np.array(z_values)  # Convert z_values to a NumPy array for easier manipulation

    def plot(self):
        # Define the range of x and y values for plotting
        x_range = np.linspace(-10, 10, 100)
        y_range = np.linspace(-10, 10, 100)
        
        # Create a meshgrid from the x and y ranges
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        
        # Evaluate the function at each point of the meshgrid
        z_mesh = self.L(x_mesh, y_mesh)
        
        # Plot the function surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis')
        
        # Plot the path traced by z during optimization
        z_values_x = self.history[:, 0]
        z_values_y = self.history[:, 1]
        z_values_z = self.L(z_values_x, z_values_y)
        ax.plot(z_values_x, z_values_y, z_values_z, 'r.-')
        
        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('L')
        ax.set_title('Optimization Process')
        
        # Show the plot
        plt.show()

