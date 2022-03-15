#! /bin/bash

#SBATCH --job-name=test
#SBATCH --partition=v5_192
#SBATCH -N 1
#SBATCH --mail-type=all
#SBATCH --mail-user=2018212874@bupt.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate myenv
python3 baselines.py -d MUTAG
python3 baselines.py -d PTC_MR 
python3 baselines.py -d MUTAG -gs
python3 baselines.py -d PTC_MR -gs
