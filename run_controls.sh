#!/bin/bash

#SBATCH --partition=private-teodoro-gpu
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=0-00:01  # (DD-HH:MM)
#SBATCH --output=logs/job_output_%j.txt
#SBATCH --error=logs/job_error_%j.err

REGISTRY=/home/users/b/borneta/sif
SIF=dl.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=run_controls_test.py

srun apptainer run --nv ${IMAGE} python ${SCRIPT}
