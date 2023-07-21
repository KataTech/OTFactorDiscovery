DATA_PATH = "kai/data"
OUTPUT_PATH = "kai/output"
AMR_UTI_PATH = "kai/data/amr-uti-antimicrobial-resistance-in-urinary-tract-infections-1.0.0"

hyperparams = {
    "seed": 125, 
    "sigma_y": 1,
    "sigma_z": 1,
    "lr": 0.001,
    "epsilon": 0.001,
    "max_iter": 2000,
    "growing_lambda": True,
    "init_lam": 0,
    "warm_stop": 500,
    "max_lam": 1000,
    "mock_prob": False,
    "eta": 0.01,
    "verbose": True,
    "monitoring_skip": 5
}

gauss_params = {
    "mu_A": -1,
    "sigma_A": 1,
    "mu_B": 1,
    "sigma_B": 1,
    "num_samples": 100
}