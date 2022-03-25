import os
import numpy as np
import subprocess
import time
method="WKS"
dataset="PTC_MR"
start=0.0
end=0.15
step=0.05

# ===============================
now_path = os.getcwd()
output_path = os.path.join(os.getcwd(),dataset)
output_path = os.path.join(output_path,method)
method_num = 0
if method == "WKS":
    method_num = 1

os.system("pwd")
os.chdir("../..")
os.system("pwd")
# os.system("python3 main.py -d MUTAG")

sub_procs = []
for w in np.arange(start,end+step,step):
    sub_procs.append(subprocess.Popen(["python3","main.py","-d",dataset,"-m",f'{method_num}',"-w",f'{w}',"-hl",'800','-p',output_path]))
    # print(f"python3 main.py -d {dataset} -m {method_num} -w {w} -hl 800 -p {output_path}")

for p in sub_procs:
    while p.poll() == None:
        time.sleep(1)
        print("sleep over")

os.chdir(now_path)
os.system(f'python3 fig.py -d {dataset} -m {method} -s {start} -e {end} -step {step}')