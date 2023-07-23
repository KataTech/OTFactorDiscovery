import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime

import src.model.ssl_ot as sot 
from src.model.monitor import Monitors
from src.eval.monitor_gen import Barycenter_Fit_Gen, Gaussian_Vis_Gen
from src.eval import vis
from constants import hyperparams as hp
from constants import gauss_params as gp
from constants import OUTPUT_PATH


# set the random seed for reproducibility
np.random.seed(hp["seed"])
file_id = datetime.now()

# generate gaussian experiment samples 
num_classes = 2
num_samples = gp["num_samples"]
gauss_A = np.random.normal(gp["mu_A"], gp["sigma_A"], num_samples)
gauss_B = np.random.normal(gp["mu_B"], gp["sigma_B"], num_samples)
x = np.concatenate((gauss_A, gauss_B)).reshape((2 * num_samples, 1))
labels = np.concatenate((np.zeros(num_samples), np.ones(num_samples))).reshape((2 * num_samples, 1)).astype(int)
z_map = np.array([[0, 1], [1, 0]])

# initialize the monitor(s)
state_trackers = [{
    "Iteration": [],
    "Lambda": [],
    "KL": [],
    "P_VALUE": [],
    "GROUND_TRUTH": np.random.normal((gp["mu_A"] + gp["mu_B"]) / 2, gp["sigma_A"], 2 * num_samples)
}, {
    "Iteration": [],
    "Lambda": [],
    "SAMPLE_A": [], 
    "SAMPLE_B": [],
}]


bary_mon = Barycenter_Fit_Gen(hp["monitoring_skip"], state_trackers[0], [], [])
gauss_mon = Gaussian_Vis_Gen(5 * hp["monitoring_skip_2"], state_trackers[1], [], [])
mons = Monitors([bary_mon.generate(), gauss_mon.generate()])

# run the training iteration
ssl = sot.SemiSupervisedOT(kernel_y_bandwidth = [hp["sigma_y"]], kernel_z_bandwidth = [hp["sigma_z"]])
ssl.initialize(x, labels, num_classes, z_map)
y, prediction, elapsed_time = ssl.train(x, hp["lr"], hp["epsilon"], hp["max_iter"], hp["growing_lambda"], hp["init_lam"], 
                                        hp["warm_stop"], hp["max_lam"], hp["mock_prob"], hp["eta"], mons, hp["verbose"], 
                                        timeit=True)
print(f"\nTraining time: {elapsed_time} seconds")
state_tracker = mons.get_monitors()[0].get_states()

# plot the barycenter fit
state_tracker = state_trackers[0]
vis.plot_two_curves(state_tracker["KL"], state_tracker["P_VALUE"], state_tracker["Iteration"], 
                    "Barycenter Fit", "KL Divergence", "P-Value", "Iteration", save = True,
                    save_path = f"{OUTPUT_PATH}/barycenter_fit_kl_p_val{file_id}.png", y3 = state_tracker["Lambda"], y3_axis_name="Lambda")

# make the animation plot of the gaussian distributions
state_tracker = state_trackers[1]
fig, ax = plt.subplots()

def update(frame): 
    ax.clear()
    sns.kdeplot(data=state_tracker["SAMPLE_A"][frame],
                color='crimson', label='A', fill=True, ax=ax)
    sns.kdeplot(data=state_tracker["SAMPLE_B"][frame],
                color='limegreen', label='B', fill=True, ax=ax)
    ax.set_title(f"Iteration {state_tracker['Iteration'][frame]}")
    ax.set_xlim([-5, 5])
    ax.set_ylim([0, 0.5])

anim = FuncAnimation(fig, update, frames = len(state_tracker["SAMPLE_A"]), interval = 500, repeat=False)
anim.save(f"{OUTPUT_PATH}/barycenter_fit_gaussian_vis_{file_id}.gif", writer=PillowWriter(fps=5))
plt.show()