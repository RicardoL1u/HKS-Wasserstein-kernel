#! /bin/bash

#SBATCH --job-name=test
#SBATCH --partition=v5_192
#SBATCH -N 1
#SBATCH --mail-type=all
#SBATCH --mail-user=2018212874@bupt.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate myenv
python3 baselines.py -d MUTAG -gs
python3 baselines.py -d PTC_MR -gs
python3 baselines.py -d NCI1 -gs
python3 baselines.py -d PROTEINS -gs
python3 baselines.py -d DD -gs
python3 baselines.py -d ENZYMES -gs


