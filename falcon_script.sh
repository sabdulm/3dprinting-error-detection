#!/bin/bash

#SBATCH --job-name="MG-3DResnet"	 # job name
#SBATCH --partition=peregrine-gpu 		 # partition to which job should be submitted
#SBATCH --qos=gpu_long			  		 # qos type
#SBATCH --nodes=1                 		 # node count
#SBATCH --ntasks=1                		 # total number of tasks across all nodes
#SBATCH --cpus-per-task=6         		 # cpu-cores per task
#SBATCH --mem=120G                  		 # total memory per node
#SBATCH --gres=gpu:a100-sxm4-80gb:1  # A100 80GB
#SBATCH --time=120:10:00 				 #  wall time

source activate 3dprint


# kaggle competitions download -c early-detection-of-3d-printing-issues -p /tmp/
# unzip -q -d /tmp/ /tmp/early-detection-of-3d-printing-issues.zip

echo "python3 train_convnet.py"
python3 train_convnet.py
