import numpy as np 
import scipy.stats as stats
import sys
from datetime import datetime

import src.model.ssl_ot as sot
from src.eval.vis import plot_two_curves 
from constants import hyperparams as hp 
from constants import gauss_params as gp 
from constants import OUTPUT_PATH

# set the random seed for reproducibility 
np.random.seed(hp["seed"])
if len(sys.argv) > 1: 
    file_id = sys.argv[1]
file_id = datetime.now()

# generate gaussian experiment samples
num_classes = 2
num_samples = gp["num_samples"]
gauss_A = np.random.normal(gp["mu_A"], gp["sigma_A"], num_samples)
gauss_B = np.random.normal(gp["mu_B"], gp["sigma_B"], num_samples)
x = np.concatenate((gauss_A, gauss_B)).reshape((2 * num_samples, 1))
labels = np.concatenate((np.zeros(num_samples), np.ones(num_samples))).reshape((2 * num_samples, 1)).astype(int)
z_map = np.array([[0, 1], [1, 0]])

# initialize the set of max-lambda values to try 
max_lam_list = [0, 10, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
init_lambda = 0
result_y = []
kl_divergence = []
p_value = []

# run the training iteration
for iter in range(len(max_lam_list)): 
    print(f"\nTraining with max_lambda = {max_lam_list[iter]}")
    ssl = sot.SemiSupervisedOT(kernel_y_bandwidth = [hp["sigma_y"]], kernel_z_bandwidth = [hp["sigma_z"]])
    ssl.initialize(x, labels, num_classes, z_map)
    y, prediction, elapsed_time = ssl.train(x, hp["lr"], hp["epsilon"], hp["max_iter"], True, init_lambda, 
                                            hp["warm_stop"], max_lam_list[iter], hp["mock_prob"], hp["eta"], None, hp["verbose"], 
                                            timeit=True)
    result_y.append(y)
    print(f"\nTraining time: {elapsed_time} seconds\n\n")
    # compute the kl divergence between the barycenter points of the two groups 
    y_a = y[labels == 0]
    y_b = y[labels == 1]
    p_a = np.histogram(y_a)[0] / len(y_a)
    p_b = np.histogram(y_b)[0] / len(y_b)
    kl_divergence.append(stats.entropy(p_a, p_b))
    # compute the p-value between the barycenter points of the two groups 
    p_value.append(stats.ks_2samp(y_a, y_b)[1])

plot_two_curves(kl_divergence, p_value, max_lam_list, 
                "KL Divergence and P-Value vs. Max Lambda", 
                "KL Divergence", "P-Value", "Max Lambda", True,
                f"{OUTPUT_PATH}/max_lam_exp_{file_id}.png")

