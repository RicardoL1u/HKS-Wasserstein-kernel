import os

dataset = 'MUTAG'

seeds = [
    # 7,
    # 77,
    777,
    4396,
    1205
]
for seed in seeds:
    os.system(f'python3 baselines.py -d {dataset} -gs  --seed {seed}')