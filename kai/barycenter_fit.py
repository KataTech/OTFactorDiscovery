import src.eval.vis as vis
import numpy as np 

x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.exp(x)
vis.plot_two_curves(y1, y2, x, "Test", "sin(x)", "exp(x)", "x")