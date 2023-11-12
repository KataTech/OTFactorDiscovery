"""
A collection of monitor generators.
"""
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from ..model.monitor import Monitor

class Monitor_Gen():
    """
    A monitor generator template. This should be inherited by
    all monitor generators and should not be used directly itself. 
    Akin to an abstract class. 
    """
    def __init__(self, monitor_skipping, state_tracker, reporter_args, 
                 tracker_args): 
        self.monitor_skipping = monitor_skipping
        self.state_tracker = state_tracker
        self.reporter_args = reporter_args
        self.tracker_args = tracker_args

    def tracker(self, state_tracker, model, params, *args): 
        return

    def reporter(self, model, params, *args): 
        return
    
    def generate(self): 
        return Monitor(self.tracker, self.reporter, self.monitor_skipping, 
                       self.state_tracker, self.reporter_args, self.tracker_args)

class Barycenter_Fit_Gen(Monitor_Gen):
    """
    Generates a monitor that tracks the barycenter fit of the 
    model groups. Specifically, it measures the KL divergence 
    between the distribution of each class as they move towards the 
    barycenter. 

    LIMITATION: Currently, you can only compare between 
    two groups of observations. 
    """

    def tracker(self, state_tracker, model, params, *args): 
        """
        Tracks the internal state of the model 
        relevant to the barycenter fit. 

        Inputs: 
            self: the monitor object itself.
            state_tracker: a dictionary tracking relevant states 
                            of the model. In this case, it should 
                            be a dictionary of lists of KL divergences
                            between the distribution of each class. 
        """
        # track the current regularizer value 
        state_tracker["Iteration"].append(params["iteration"])
        state_tracker["Lambda"].append(params["lam"])
        # tracl the current y points per group 
        Y_compact = model.select_best(params["Y"], params["mock_prob"])[0]
        y_a = Y_compact[params["label"] == 0]
        y_b = Y_compact[params["label"] == 1]
        # compute the kl divergence 
        p_a = np.histogram(y_a)[0] / len(y_a)
        p_b = np.histogram(y_b)[0] / len(y_b)
        state_tracker["KL"].append(stats.entropy(p_a, p_b))
        # track the current hypothesis test result 
        truth_sample = state_tracker["GROUND_TRUTH"]
        state_tracker["P_VALUE"].append(stats.ks_2samp(truth_sample, Y_compact.flatten())[1])
        return

    def reporter(self, model, params, *args): 
        return super().reporter(model, params, *args)

class Gaussian_Vis_Gen(Monitor_Gen): 
    """
    Generates a monitor that tracks the kernel density estimation plot 
    of the barycenter distributions as the training procedure progresses. 
    """
    def tracker(self, state_tracker, model, params, *args): 
        """
        Tracks the internal state of the model 
        relevant to the gaussian visualizations. 

        Inputs: 
            self: the monitor object itself.
            state_tracker: a dictionary tracking relevant states 
                            of the model. In this case, it should 
                            be a dictionary containing kdeplot 
                            axes of each iteration. 
        """
        Y_compact = model.select_best(params["Y"], params["mock_prob"])[0]
        y_a = Y_compact[params["label"] == 0]
        y_b = Y_compact[params["label"] == 1]
        state_tracker["Iteration"].append(params["iteration"])
        state_tracker["SAMPLE_A"].append(y_a)
        state_tracker["SAMPLE_B"].append(y_b)
        return 

    def reporter(self, model, params, *args):
        return super().reporter(model, params, *args)