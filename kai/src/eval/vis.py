"""
A file containing all the visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def plot_two_curves(y1, y2, x, title, left_axis_name, right_axis_name, x_axis_name): 
    """
    Plot two curves on the same plot. The scales should be different so that 
    both curves are visible on the screen with different colors,
    """
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, 'b-')
    ax1.set_xlabel(x_axis_name)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(left_axis_name, color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'r-')
    ax2.set_ylabel(right_axis_name, color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.title(title)
    plt.show()