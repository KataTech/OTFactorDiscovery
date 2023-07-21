import numpy as np 

import src.model.ssl_ot as sot 
from src.model.monitor import Monitors
from src.eval.monitor_gen import Barycenter_Fit_Gen
from src.eval import vis
from constants import hyperparams as hp
from constants import gauss_params as gp

# set the random seed for reproducibility
np.random.seed(hp["seed"])

# generate gaussian experiment samples 
num_classes = 2
num_samples = gp["num_samples"]
gauss_A = np.random.normal(gp["mu_A"], gp["sigma_A"], num_samples)
gauss_B = np.random.normal(gp["mu_B"], gp["sigma_B"], num_samples)
x = np.concatenate((gauss_A, gauss_B)).reshape((2 * num_samples, 1))
labels = np.concatenate((np.zeros(num_samples), np.ones(num_samples))).reshape((2 * num_samples, 1)).astype(int)
z_map = np.array([[0, 1], [1, 0]])

# initialize the monitor(s)
state_tracker = {
    "Iteration": [],
    "Lambda": [],
    "KL": [],
    "P_VALUE": [],
    "GROUND_TRUTH": np.random.normal((gp["mu_A"] + gp["mu_B"]) / 2, gp["sigma_A"], 2 * num_samples)
}
bary_gen = Barycenter_Fit_Gen(hp["monitoring_skip"], state_tracker, [], [])
mons = Monitors([bary_gen.generate()])

# run the training iteration
ssl = sot.SemiSupervisedOT(kernel_y_bandwidth = [hp["sigma_y"]], kernel_z_bandwidth = [hp["sigma_z"]])
ssl.initialize(x, labels, num_classes, z_map)
y = ssl.train(x, hp["lr"], hp["epsilon"], hp["max_iter"], hp["growing_lambda"], hp["init_lam"], 
          hp["warm_stop"], hp["max_lam"], hp["mock_prob"], hp["eta"], mons, hp["verbose"])
state_tracker = mons.get_monitors()[0].get_states()

# plot the barycenter fit
vis.plot_two_curves(state_tracker["KL"], state_tracker["P_VALUE"], state_tracker["Iteration"], 
                    "Barycenter Fit", "KL Divergence", "P-Value", "Iteration")