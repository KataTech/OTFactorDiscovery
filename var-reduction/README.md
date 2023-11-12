# Variability Reduction via Optimal Transport
An optimal transport based approach to reduce unwanted variability from data distributions. 

### Brief Summary
This directory contains the numerical simulations associated with the variability reduction via optimal transport project. Implemented by Kai M. Hung as a project for AM-SURE 2023 at New York University, Courant Institute of Mathematical Sciences. 

### Start-up Guide 
To begin, we recommend the user to create a conda virtual environment via `conda create --name <env> --file requirements.txt` where `<env>` should be replaced with your desired environment name. For users unfamiliar, we refer you to the official conda [documentation](https://docs.conda.io/en/latest/). 

For an illustration of the method's efficacy, we recommend running `python3 barycenter_fit.py` where you will see (1) a plot of the KL divergence between the proposed barycenter and the true barycenter over time and (2) an animation for applying the variation reduction method to a dataset with two gaussianly distributed groups. This experiment demonstrates that our method will convert the data distribution to converge to the barycenter with respect to the factors that we want to "filter out". 

You may also want to edit `constants.py` first to tune the `OUTPUT_URL` and experiment hyperparameters to your liking. 

### Table of Content

Here we summarize all files present in this directory and their purpose.
```
+-- src/
|   +-- eval/ : contains evaluation scripts 
    |   +-- finite_diff.py/ : finite difference-based gradient checking
    |   +-- monitor_gen.py/ : generator for monitors that are used in this project
    |   +-- monitor.py/ : a class of monitor objects used to capture information throughout the algorithm iterations
    |   +-- vis.py/ : visualization tools 
|   +-- model/ : contains model scripts
    |   +-- flow_ot.py/ : variability reduction via OT
    |   +-- ssl_ot.py/ : semi-supervised variability reduction via OT
|   +-- optim/ : contain scripts for optimization sandbox environemnt - NOT used for finalized product
    |   +-- loss.py/ : automatic gradient derivation class for simple functions 
    |   +-- optim.py/ : optimization algorithms 
+-- barycenter_fit.py : an experiment script for running a specific iteration of the variability reduction algorithm
+-- constants.py : contains constants such as hyperparameter and output paths 
+-- flow_ot_exp.ipynb : a notebook demonstrating algorithms over iris dataset
+-- gaussian.ipynb : a notebook demonstrating algorithms over synthetic gaussians
+-- lambda_tuning.py : an experiment script for tuning the lambda (trade-off between data deformation and independence guarantees)
+-- run_experiment.py : a wrapper to run experiments and track their parameter settings, e.g. `python3 run_experiments.py barycenter_fit.py`
+-- ssl_test.ipynb : a notebook demonstrating semi-supervised algorithm
+-- .gitignore
+-- README.md 
+-- requirements.txt
```