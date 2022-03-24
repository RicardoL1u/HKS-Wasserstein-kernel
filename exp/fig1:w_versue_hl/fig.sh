#!/bin/bash
# bash的登号是严格不能有空格的
method="WKS"
dataset="MUTAG"
path="exp/fig1:w_versue_hl"
method_num=0
if method=="WKS"; then
    method_num=1
fi
echo $method_num
pwd
cd ../..
pwd

for ((i=1; i<=2; i++))
do
    declare -i hl=$i*100
    # echo $hl
    python3 main.py -d $dataset -m $method_num -s 0 -cv -w 0.45 -hl $hl -p $path 
done