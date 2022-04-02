#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=v5_192
#SBATCH -N 1
#SBATCH --mail-type=all
#SBATCH --mail-user=2018212874@bupt.edu.cn
#SBATCH --output=PROTEINS.out
#SBATCH --error=%j.err

source activate myenv

# python3 main.py -d MUTAG -m 0 -s 2 -hl 800  -cv -gs 
# python3 main.py -d MUTAG -m 1 -s 0 -hl 800  -cv -gs

python3 main.py -d BZR -m 0 -s 2 -hl 800  -cv -gs 
python3 main.py -d BZR -m 1 -s 0 -hl 800  -cv -gs

# python3 main.py -d DD -m 0 -s 2 -hl 800  -cv -gs --sinkhorn
# python3 main.py -d DD -m 1 -s 0 -hl 800  -cv -gs --sinkhorn

# python3 main.py -d DD -m 0 -s 2 -hl 800  -cv -gs --sinkhorn
# python3 main.py -d ENZYMES -m 0 -s 2 -hl 800  -cv -gs 
# python3 main.py -d ENZYMES -m 1 -s 0 -hl 800  -cv -gs 

# python3 main.py -d NCI1 -m 0 -s 2 -hl 800  -cv -gs --sinkhorn
# python3 main.py -d NCI1 -m 1 -s 0 -hl 800  -cv -gs --sinkhorn