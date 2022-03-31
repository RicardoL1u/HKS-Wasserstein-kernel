#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=v5_192
#SBATCH -N 1
#SBATCH --mail-type=all
#SBATCH --mail-user=2018212874@bupt.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate myenv

# python3 main.py -d MUTAG -m 1 -s 0 -c 100 -g 1.0 -w 0.6 -path ./
python3 exp_script.py