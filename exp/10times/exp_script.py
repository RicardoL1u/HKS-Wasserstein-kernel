import os

resultzip = '5times.zip'

dataset_list = ['MUTAG','PTC_MR','ENZYMES','PROTEINS','DD']
method_list = ['HKS','WKS']
now_path = os.getcwd()
output_path = os.path.join(os.getcwd(),'results')
if not os.path.exists(output_path):
    os.mkdir(output_path)

os.chdir('../..')

for dataset in dataset_list:
    for method in method_list:
        if dataset == 'DD':
            sinkhorn = '--sinkhorn'
        else:
            sinkhorn = ''
        if method == 'HKS':
            method_num = 0
            sample = 2
        else:
            method_num = 1
            sample = 0
        hl = 800
        for seed in [7,77,777,4396,1205]:
            name ='seed_{:04d}'.format(seed)
            os.system(f'python3 main.py -d {dataset} -m {method_num} -s {sample} --seed {seed} -gs -hl {hl} -p {output_path} -n {name} -cv {sinkhorn}')
os.chdir(now_path)
os.system(f'zip -r {resultzip} {output_path}')
os.system('git add .')
os.system(f'git commit -a -m \"exp:record for {resultzip}\"')