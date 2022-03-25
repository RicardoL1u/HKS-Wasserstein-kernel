import os
import numpy as np
import subprocess
import time
import shutil
method="WKS"
dataset="PTC_MR"
start=0.0
end=1.00
step=0.05

# ===============================
shutil.rmtree(os.path.join(dataset,method),ignore_errors=True)
now_path = os.getcwd()
output_path = os.getcwd()
method_num = 0
if method == "WKS":
    method_num = 1

os.system("pwd")
os.chdir("../..")
os.system("pwd")

sub_procs = []
for w in np.arange(start,end+step,step):
    w = np.round(w,2)
    sub_procs.append(subprocess.Popen(["python3","main.py","-d",dataset,"-m",f'{method_num}',"-w",f'{w}',"-hl",'1000','-p',output_path,'-n',f'{w}','-cv']))

for p in sub_procs:
    while p.poll() == None:
        time.sleep(1)
    print("sleep over")

os.chdir(now_path)
# os.system(f'python3 fig.py -d {dataset} -m {method} -s {start} -e {end} -step {step}')
os.system(f'zip -r {method}_{dataset}_w.zip {os.path.join(dataset,method)}/')