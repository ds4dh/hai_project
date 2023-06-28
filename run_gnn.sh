#!/bin/bash

#SBATCH --job-name=run_gnn
#SBATCH --partition=private-teodoro-gpu
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32gb
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/run_gnn_%j.txt
#SBATCH --error=logs/run_gnn_%j.err

REGISTRY=/home/users/b/borneta/sif
SIF=dl.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=run_gnn.py

srun apptainer run --nv ${IMAGE} python ${SCRIPT}
