import os
import numpy as np
import shutil
method="WKS"
sample=0
dataset="MUTAG"
C = 1.0
g = 10
hl = 800
sinkhorn = ''
resultzip = f'{method}{sample}_{dataset}_w_c={C}_g={g}_hl={hl}{sinkhorn}.zip'
start=0.4
end=0.6
step=0.1
ws = [0.00,0.05,0.10,0.15,0.20,0.25,
              0.30,0.35,0.40,0.45,0.50,0.55,
              0.60,0.65,0.70,0.75,0.80,0.85,
              0.90,0.95,1.00]
# ===============================
shutil.rmtree(os.path.join(dataset,method),ignore_errors=True)
os.makedirs(os.path.join(dataset,method))
now_path = os.getcwd()
output_path = os.getcwd()
method_num = 0
if method == "WKS":
    method_num = 1

os.system("pwd")
os.chdir("../..")
os.system("pwd")

sub_procs = []
for w in ws:
    w = np.round(w,2)
    name = '{:1.2f}'.format(w)
    os.system(f'python3 main.py -d {dataset} -m {method_num} -s {sample} -c {C} -g {g} -w {w} -hl {hl} -p {output_path} -n {name} -cv {sinkhorn}')


os.chdir(now_path)
# os.system(f'python3 fig.py -d {dataset} -m {method} -s {start} -e {end} -step {step}')
os.system(f'zip -r {resultzip} {os.path.join(dataset,method)}/')
os.system('git add .')
os.system(f'git commit -a -m \"exp:record for {resultzip}\"')