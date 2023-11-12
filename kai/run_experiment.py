import subprocess
import sys 
import time
from datetime import datetime 

from constants import hyperparams as hp
from constants import gauss_params as gp 
from constants import OUTPUT_PATH

experiment_executable = sys.argv[1]
file_name = experiment_executable.split("/")[-1].split(".")[0]
file_ID = datetime.now()

# save the hyperparameter values to the output file
with open(f"{OUTPUT_PATH}/{file_name}_{file_ID}.txt", "w") as f:
    f.write(f"Model hyperparameter values:\n{hp}\n")
    f.write(f"Gaussian parameters:\n{gp}\n")

# run the experiment and track its total elapsed time. 
print(f"Starting experiment....")
exp_start_time = time.time()

output = subprocess.run(["python", experiment_executable, str(file_ID)], capture_output=True)

exp_end_time = time.time()
exp_elapsed_time = exp_end_time - exp_start_time

# save the output to a file 
with open(f"{OUTPUT_PATH}/{file_name}_{file_ID}.txt", "w") as f:
    f.write(output.stdout)
    f.write("\n\nTotal executable elapsed time: " + str(exp_elapsed_time) + " seconds\n")

print(f"\n\nTotal executable elapsed time: {exp_elapsed_time} seconds\n")