import pandas as pd
import os
dataset = "PTC_MR"
HKS_data_path = os.path.join(dataset,"HKS")
WKS_data_path = os.path.join(dataset,"WKS")

hks_files = sorted( filter( lambda x: os.path.isfile(os.path.join(HKS_data_path, x)),
                            os.listdir(HKS_data_path) ) )
wks_files = sorted( filter( lambda x: os.path.isfile(os.path.join(WKS_data_path, x)),
                            os.listdir(WKS_data_path) ) )        

hks_results = [pd.read_csv(HKS_data_path+i) for i in hks_files]   
wks_results = [pd.read_csv(WKS_data_path+i) for i in wks_files]

result = hks_results
result = result.extend(wks_results)

import numpy as np  
means = np.array([np.mean(file["accuracy"]) for file in result])
stds = np.array([np.std(file["accuracy"]) for file in result])
print(means,stds)
