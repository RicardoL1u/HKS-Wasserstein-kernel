import os
import pandas as pd
# dataset="PROTEINS"
# sinkhorn = ""
resultzip = 'figw.zip'


# ===============================
now_path = os.getcwd()
output_path = os.path.join(os.getcwd(),'results')
if not os.path.exists(output_path):
    os.mkdir(output_path)

os.system("pwd")
os.chdir("../w")
os.system("pwd")
best_pd = pd.read_csv("best.csv")
os.chdir('../..')

ws = [0.00,0.05,0.10,0.15,0.20,0.25,
              0.30,0.35,0.40,0.45,0.50,0.55,
              0.60,0.65,0.70,0.75,0.80,0.85,
              0.90,0.95,1.00]

for index,row in best_pd.iterrows():
    if row['dataset'] != 'PROTEINS':
        continue
    if row["sinkhorn"] == 'True':
        sinkhorn = '--sinkhorn'
    else:
        sinkhorn = ''
    dataset = row['dataset']
    if row['method'] == 'HKS':
        method_num = 0
        sample = 2
    else:
        method_num = 1
        sample = 0
    C = row['C']
    g = row['gamma']
    hl = 800
    for w in ws:
        name = '{:1.2f}'.format(w)
        os.system(f'python3 main.py -d {dataset} -m {method_num} -s {sample} -c {C} -g {g} -w {w} -hl {hl} -p {output_path} -n {name} -cv {sinkhorn}')

os.chdir(now_path)
os.system(f'zip -r {resultzip} {output_path}')
os.system('git add .')
os.system(f'git commit -a -m \"exp:record for {resultzip}\"')