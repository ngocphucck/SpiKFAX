#!/bin/bash
#SBATCH --job-name=run_1
#SBATCH --output=run_1.out
#SBATCH --error=run_1.err

#SBATCH --partition=P2
#SBATCH --gres=gpu:1

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

# Run the experiment

python exp_all.py --dataset CIFAR10-DVS --model lenet5 --optimizer spikefac_stab --lr 1e-5 --epochs 60 --seed 0
