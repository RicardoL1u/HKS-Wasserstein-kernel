#! /bin/bash

#SBATCH --job-name=test
#SBATCH --partition=v5_192
#SBATCH -N 1
#SBATCH --mail-type=all
#SBATCH --mail-user=2018212874@bupt.edu.cn
#SBATCH --output=baseline.out
#SBATCH --error=baseline.err

source activate myenv
python3 baselines.py -d MUTAG -gs  --seed 7
python3 baselines.py -d PTC_MR -gs --seed 7
python3 baselines.py -d PROTEINS -gs --seed 7
python3 baselines.py -d DD -gs --seed 7
python3 baselines.py -d ENZYMES -gs --seed 7

python3 baselines.py -d MUTAG -gs  --seed 77
python3 baselines.py -d PTC_MR -gs --seed 77
python3 baselines.py -d PROTEINS -gs --seed 77
python3 baselines.py -d DD -gs --seed 77
python3 baselines.py -d ENZYMES -gs --seed 77

python3 baselines.py -d MUTAG -gs  --seed 777
python3 baselines.py -d PTC_MR -gs --seed 777
python3 baselines.py -d PROTEINS -gs --seed 777
python3 baselines.py -d DD -gs --seed 777
python3 baselines.py -d ENZYMES -gs --seed 777

python3 baselines.py -d MUTAG -gs  --seed 4396
python3 baselines.py -d PTC_MR -gs --seed 4396
python3 baselines.py -d PROTEINS -gs --seed 4396
python3 baselines.py -d DD -gs --seed 4396
python3 baselines.py -d ENZYMES -gs --seed 4396

python3 baselines.py -d MUTAG -gs  --seed 1205
python3 baselines.py -d PTC_MR -gs --seed 1205
python3 baselines.py -d PROTEINS -gs --seed 1205
python3 baselines.py -d DD -gs --seed 1205
python3 baselines.py -d ENZYMES -gs --seed 1205


