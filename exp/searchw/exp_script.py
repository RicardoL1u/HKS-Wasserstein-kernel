import os
import numpy as np
import subprocess
import time
import shutil
dataset="PTC_MR"
resultzip = f'{dataset}_w_hl=800.zip'


# ===============================
shutil.rmtree(dataset,ignore_errors=True)
os.makedirs(os.path.join(dataset,"HKS"))
os.makedirs(os.path.join(dataset,"WKS"))
now_path = os.getcwd()
output_path = os.getcwd()

os.system("pwd")
os.chdir("../..")
os.system("pwd")

sub_procs = []
sub_procs.append(subprocess.Popen(["python3","main.py","-d",dataset,"-m","0",'-s','0',"-hl",'800','-p',output_path,'-n','searchw0','-cv','-gs']))
sub_procs.append(subprocess.Popen(["python3","main.py","-d",dataset,"-m","0",'-s','1',"-hl",'800','-p',output_path,'-n','searchw1','-cv','-gs']))
sub_procs.append(subprocess.Popen(["python3","main.py","-d",dataset,"-m","0",'-s','2',"-hl",'800','-p',output_path,'-n','searchw2','-cv','-gs']))

sub_procs.append(subprocess.Popen(["python3","main.py","-d",dataset,"-m","1",'-s','0',"-hl",'800','-p',output_path,'-n','searchw0','-cv','-gs']))
sub_procs.append(subprocess.Popen(["python3","main.py","-d",dataset,"-m","1",'-s','1',"-hl",'800','-p',output_path,'-n','searchw1','-cv','-gs']))


for p in sub_procs:
    while p.poll() == None:
        time.sleep(1)
    print("sleep over")

os.chdir(now_path)
# os.system(f'python3 fig.py -d {dataset} -m {method} -s {start} -e {end} -step {step}')
os.system(f'zip -r {resultzip} {dataset}/')