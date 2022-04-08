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
hls = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]

for index,row in best_pd.iterrows():
    if row['dataset'] != 'PTC_MR':
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
    w = row['w']
    for hl in hls:
        name = '{:1.2f}'.format(w)
        os.system(f'python3 main.py -d {dataset} -m {method_num} -s {sample} -c {C} -g {g} -w {w} -hl {hl} -p {output_path} -n {name} -cv {sinkhorn}')

os.chdir(now_path)
os.system(f'zip -r {resultzip} {output_path}')
os.system('git add .')
os.system(f'git commit -a -m \"exp:record for {resultzip}\"')