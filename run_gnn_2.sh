#!/bin/bash

### Initialize stuff
#SBATCH --job-name=run_gnn
#SBATCH --partition=private-teodoro-gpu
#SBATCH --output=/home/users/b/borneta/hai_project/logs/run_gnn/job_%j.txt
#SBATCH --error=/home/users/b/borneta/hai_project/logs/run_gnn/job_%j.err
#SBATCH --time=1-00:00:00

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=1

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=12

# Load modules or your own conda environment here
REGISTRY=/home/users/b/borneta/sif
SIF=dl.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=run_gnn.py

# Ray tune network things
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]} 
ip=$node_1
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 start-head.sh $ip $redis_password &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i start-worker.sh $ip_head $redis_password &
  sleep 5
done
##############################################################################################

#### call your code below
srun apptainer run --nv ${IMAGE} python ${SCRIPT}
exit