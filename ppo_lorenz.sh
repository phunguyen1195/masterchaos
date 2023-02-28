#!/bin/bash



#SBATCH --job-name=Lorenz   # job name
#SBATCH --output=log/lorenz/Lorenz-%j.out # output log file
#SBATCH --error=log/lorenz/Lorenz-%j.err  # error file
#SBATCH --time=24:00:00  # 5 hour of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH -c 2        # cpus-per-task
#SBATCH --partition=gpu # GPU partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu     # Request 1 GPU
#SBATCH --mail-user=phu.c.nguyen@sjsu.edu
#SBATCH --mail-type=END

python test_stablebaselines3_SAC_1d_learningratedecay.py