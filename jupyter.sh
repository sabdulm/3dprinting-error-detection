#!/bin/bash
#SBATCH --job-name="Jupyter-GPU" 	  # a name for your job
#SBATCH --partition=peregrine-gpu		  # partition to which job should be submitted
#SBATCH --qos=gpu_medium					  # qos type
#SBATCH --nodes=1                		  # node count
#SBATCH --ntasks=1               		  # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        		  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G         				  # total memory per node
#SBATCH --gres=gpu:nvidia_a100_3g.39gb:1  # Request 1 GPU
#SBATCH --time=03:00:00          		  # total run time limit (HH:MM:SS)

module purge
module load python/anaconda
source activate 3dprint

port=8888
ssh -N  -f -R $port:localhost:$port falcon
jupyter-notebook --no-browser  --port=$port
