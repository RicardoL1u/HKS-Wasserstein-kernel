#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=v5_192
#SBATCH -N 2
#SBATCH --mail-type=all
#SBATCH --mail-user=2018212874@bupt.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate myenv

python3 exp_script.py