"""
A file containing all the visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def plot_two_curves(y1, y2, x, title, left_axis_name, right_axis_name, x_axis_name, 
                    save = False, save_path = None, y3 = None, y3_axis_name = None): 
    """
    Plot two curves on the same plot. The scales should be different so that 
    both curves are visible on the screen with different colors,
    """
    if y3 is None: 
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

    # if the third curve is provided, plot it in a figure 
    # below the figure of the first two curves, but share the same x-axis
    else: 
        fig, axs = plt.subplots(2, 1, sharex = True, height_ratios = [3, 1])
        axs[0].plot(x, y1, 'b-')
        axs[0].set_ylabel(left_axis_name, color='b')
        axs[0].tick_params('y', colors='b')
        ax2 = axs[0].twinx()
        ax2.plot(x, y2, 'r-')
        ax2.set_ylabel(right_axis_name, color='r')
        ax2.tick_params('y', colors='r')

        axs[1].plot(x, y3, 'g-')
        axs[1].set_ylabel(y3_axis_name, color='g')
        axs[1].tick_params('y', colors='g')

    plt.title(title)
    if save: 
        plt.savefig(save_path)
    else: 
        plt.show()
    
    