#!/bin/bash

#SBATCH --job-name=run_gnn
#SBATCH --partition=private-teodoro-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --time=6-00:00:00
#SBATCH --output=/home/users/b/borneta/hai_project/logs/run_gnn/job_%j.txt
#SBATCH --error=/home/users/b/borneta/hai_project/logs/run_gnn/job_%j.err

REGISTRY=/home/users/b/borneta/sif
SIF=torch-image.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=run_gnn.py

srun apptainer run --nv ${IMAGE} python ${SCRIPT}
