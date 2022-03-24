#! /bin/bash

#SBATCH --job-name=test
#SBATCH --partition=v5_192
#SBATCH -N 1
#SBATCH --mail-type=all
#SBATCH --mail-user=2018212874@bupt.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate myenv

# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.45 -hl 500
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.45 -hl 600
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.45 -hl 700
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.45 -hl 800
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.45 -hl 900
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.45 -hl 1000

python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.00 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.05 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.10 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.15 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.20 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.25 -hl 800
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.30 -hl 800
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.35 -hl 800
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.40 -hl 800
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.45 -hl 800
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.50 -hl 800
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.55 -hl 800
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.60 -hl 800
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.65 -hl 800
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.70 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.75 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.80 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.85 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.90 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 0.95 -hl 800
python3 main.py -d PTC_MR -m 1 -s 0 -cv -w 1.00 -hl 800




# li's hks
# python3 main.py -d MUTAG -m 0 -s 1 -cv -gs --sinkhorn
# python3 main.py -d PTC_MR -m 0 -s 1 -cv -gs --sinkhorn
# python3 main.py -d ENZYMES -m 0 -s 1 -cv -gs

# python3 main.py -d MUTAG -m 0 -s 1 -cv -gs -w 0.5
# python3 main.py -d PTC_MR -m 0 -s 1 -cv -gs -w 0.5
# python3 main.py -d ENZYMES -m 0 -s 1 -cv -gs -w 0.5

# python3 main.py -d MUTAG -m 0 -s 1 -cv -gs -w 0.6
# python3 main.py -d PTC_MR -m 0 -s 1 -cv -gs -w 0.6
# python3 main.py -d ENZYMES -m 0 -s 1 -cv -gs -w 0.6

# dual exp
# python3 main.py -d MUTAG -m 0 -s 3 -cv -gs -w 0.4
# python3 main.py -d PTC_MR -m 0 -s 3 -cv -gs -w 0.4
# python3 main.py -d ENZYMES -m 0 -s 2 -cv -gs -w 0.6

# wks 
# python3 main.py -d MUTAG -m 1 -s 1 -cv -gs 
# python3 main.py -d PTC_MR -m 1 -s 1 -cv -gs 
# python3 main.py -d ENZYMES -m 1 -s 0 -cv -gs -w 0.5 

# python3 main.py -d MUTAG -m 1 -s 0 -cv -gs -w 0.6
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -gs -w 0.6
# python3 main.py -d ENZYMES -m 1 -s 0 -cv -gs -w 0.6 


# python3 main.py -d MUTAG -m 1 -s 0 -cv -gs -w 0.45
# python3 main.py -d PTC_MR -m 1 -s 0 -cv -gs -w 0.45