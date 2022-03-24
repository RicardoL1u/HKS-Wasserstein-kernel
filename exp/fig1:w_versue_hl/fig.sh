#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=v5_192
#SBATCH -N 1
#SBATCH --mail-type=all
#SBATCH --mail-user=2018212874@bupt.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate myenv

# bash的登号是严格不能有空格的
method="WKS"
method_num=1
dataset="PTC_MR"
rm -r $dataset
path="exp/fig1:w_versue_hl"
start=100
end=1300
step=100


# -------------
pwd
cd ../..
pwd

for ((i=$start; i<=$end; i=i+$step))
do
    python3 main.py -d $dataset -m $method_num -s 0 -cv -w 0.45 -hl $i -p $path 
done

cd $path
python3 fig.py -d $dataset -m $method -s $start -e $end -step $step