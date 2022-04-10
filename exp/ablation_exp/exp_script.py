import os
import pandas as pd
resultzip = 'ablation.zip'


# ===============================
# shutil.rmtree(dataset,ignore_errors=True)
# os.makedirs(os.path.join(dataset,"HKS"))
# os.makedirs(os.path.join(dataset,"WKS"))

dataset_list = ['MUTAG','PTC_MR','ENZYMES','PROTEINS','DD']
method_list = ['HKS','WKS']
now_path = os.getcwd()
output_path = os.path.join(os.getcwd(),'results')
if not os.path.exists(output_path):
    os.mkdir(output_path)

os.chdir('../..')

for dataset in dataset_list:
    for method in method_list:
        path_to_file = os.path.join(output_path,dataset+'_'+method+'.csv')
        best_pd = pd.read_csv(path_to_file)
        for index,row in best_pd.iterrows():
            if dataset == 'DD':
                sinkhorn = '--sinkhorn'
            else:
                sinkhorn = ''
            # method_num = row['method']
            if method == 'HKS':
                method_num = 0
                sample = 2
            else:
                method_num = 1
                sample = 0
            C = row['C']
            g = row['gamma']
            hl = 800
            for w in [1.00]: 
                name = '{:1.2f}'.format(w)
                os.system(f'python3 main.py -d {dataset} -m {method_num} -s {sample} -c {C} -g {g} -w {w} -hl {hl} -p {output_path} -n {name} -cv {sinkhorn}')

os.chdir(now_path)
# os.system(f'zip -r {resultzip} {output_path}')
# os.system('git add .')
# os.system(f'git commit -a -m \"exp:record for {resultzip}\"')