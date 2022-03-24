# import csv
import os
import numpy as np
import pandas as pd
x_axis = np.arange(0.30,0.75,0.05).tolist()

Path = "./results/PTC_MR/WKS/"
# Get list of all files in a given directory sorted by name
list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(Path, x)),
                        os.listdir(Path) ) )
result_files = [pd.read_csv(Path+i) for i in list_of_files]