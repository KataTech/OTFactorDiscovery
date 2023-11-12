import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import seaborn as sns 

from src.eval.vis import plot_two_curves
from constants import OUTPUT_PATH

### TEST for vis2plot
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x)
y3 = np.tan(x)
plot_two_curves(y1, y2, x, "Test", "sin", "exp", "x", True, f"{OUTPUT_PATH}/test.png", y3, "tan")


### TEST for animation 
# np.random.seed(125)
# means = np.linspace(0, 50, 50)
# fig, ax = plt.subplots()

# def update(frame): 
#     ax.clear()
#     sns.kdeplot(np.random.normal(means[frame], 1, 1000), ax = ax)
#     ax.set_xlim([0, 50])
#     ax.set_ylim([0, 1])

# animation = FuncAnimation(fig, update, frames = len(means), interval = 50, repeat=False)
# plt.show()

