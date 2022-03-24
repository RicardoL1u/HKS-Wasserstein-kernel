import csv
import os
import numpy as np
Path = "./results/PTC_MR/WKS"
result_files = [open(Path+i,'r') for i in os.listdir(Path)]
result_readers = [csv.reader(f) for f in result_files]
for r in result_readers:
    next(r)

x_axis = np.arange(0.30,0.75,0.05).tolist()

