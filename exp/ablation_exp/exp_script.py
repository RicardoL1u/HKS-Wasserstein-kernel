import os
import pandas as pd
# dataset="PROTEINS"
# sinkhorn = ""
resultzip = 'ablation.zip'


# ===============================
# shutil.rmtree(dataset,ignore_errors=True)
# os.makedirs(os.path.join(dataset,"HKS"))
# os.makedirs(os.path.join(dataset,"WKS"))
now_path = os.getcwd()
output_path = os.path.join(os.getcwd(),'results')
if not os.path.exists(output_path):
    os.mkdir(output_path)

os.system("pwd")
best_pd = pd.read_csv("best.csv")
os.chdir("../..")
os.system("pwd")

for index,row in best_pd.iterrows():
    if row["sinkhorn"] == True:
        sinkhorn = '--sinkhorn'
    else:
        sinkhorn = ''
    dataset = row['dataset']
    method_num = row['method']
    sample = row['sample']
    C = row['c']
    g = row['gamma']
    hl = row['hl']
    for w in [0.00,1.00]:
        name = '{:1.2f}'.format(w)
        os.system(f'python3 main.py -d {dataset} -m {method_num} -s {sample} -c {C} -g {g} -w {w} -hl {hl} -p {output_path} -n {name} -cv {sinkhorn}')

os.chdir(now_path)
os.system(f'zip -r {resultzip} {output_path}')
os.system('git add .')
os.system(f'git commit -a -m \"exp:record for {resultzip}\"')