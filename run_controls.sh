#!/bin/bash

#SBATCH --job-name=run_controls
#SBATCH --partition=private-teodoro-gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/run_controls/job_%j.txt
#SBATCH --error=logs/run_controls/job_%j.err

REGISTRY=/home/users/b/borneta/sif
SIF=dl.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=run_controls.py

srun apptainer run --nv ${IMAGE} python ${SCRIPT}
