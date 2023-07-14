#!/bin/bash

#SBATCH --job-name=run_controls
#SBATCH --partition=private-teodoro-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32gb
#SBATCH --time=0-24:00:00
#SBATCH --output=/home/users/b/borneta/hai_project/logs/run_controls/job_%j.txt
#SBATCH --error=/home/users/b/borneta/hai_project/logs/run_controls/job_%j.err

REGISTRY=/home/users/b/borneta/sif
SIF=torch-image.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=run_controls.py

srun apptainer run --nv ${IMAGE} python ${SCRIPT}
