import os

from ot import sinkhorn
dataset="PROTEINS"
sinkhorn = ""
resultzip = f'{dataset}_w_hl=800_no_cv{sinkhorn}.zip'


# ===============================
# shutil.rmtree(dataset,ignore_errors=True)
# os.makedirs(os.path.join(dataset,"HKS"))
# os.makedirs(os.path.join(dataset,"WKS"))
now_path = os.getcwd()
output_path = os.getcwd()

os.system("pwd")
os.chdir("../..")
os.system("pwd")

# os.system(f'python3 main.py -d {dataset} -m 0 -s 0 -hl 800 -p {output_path} -cv -gs')
# os.system(f'python3 main.py -d {dataset} -m 0 -s 1 -hl 800 -p {output_path} -cv -gs')
os.system(f'python3 main.py -d {dataset} -m 0 -s 2 -hl 800 -p {output_path} -cv -gs {sinkhorn}')
os.system(f'python3 main.py -d {dataset} -m 1 -s 0 -hl 800 -p {output_path} -cv -gs {sinkhorn}')
# os.system(f'python3 main.py -d {dataset} -m 1 -s 1 -hl 800 -p {output_path} -cv -gs')



os.chdir(now_path)
# os.system(f'python3 fig.py -d {dataset} -m {method} -s {start} -e {end} -step {step}')
os.system(f'zip -r {resultzip} {os.path.join(dataset)}/')
os.system('git add .')
os.system(f'git commit -a -m \"exp:record for {resultzip}\"')