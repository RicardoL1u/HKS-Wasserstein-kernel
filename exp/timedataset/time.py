import re
from datetime import datetime
import numpy as np

datasets = ['MUTAG','PTC_MR','PROTEINS','ENZYMES','DD']
methods = ['HKS','WKS']

ws_regex = r"(.*): ready to compute the wass diss with "
we_regex = r"(.*): have computed the wass diss with sinkhorn ="
ns_regex = r"(.*): ready to generate node embeddings"
ne_regex = r"(.*): the node embedding have been generated"
def stat_file(file:str)->np.ndarray:
    with open(file, 'r') as file:
        data = file.readlines()
    wass_time = []
    node_time = []
    for i in range(len(data)):
        sm = re.match(ws_regex, data[i])
        if bool(sm):
            em = re.match(we_regex, data[i+1])
            i = i + 1
            start_time = datetime.strptime(sm.group(1),'%Y/%m/%d %I:%M:%S')
            end_time = datetime.strptime(em.group(1),'%Y/%m/%d %I:%M:%S')
            wass_time.append((end_time - start_time).total_seconds())
            continue
        sm = re.match(ns_regex,data[i])
        if bool(sm):
            while not bool(re.match(ne_regex, data[i])):
                i = i + 1
            em = re.match(ne_regex, data[i])
            start_time = datetime.strptime(sm.group(1),'%Y/%m/%d %I:%M:%S')
            end_time = datetime.strptime(em.group(1),'%Y/%m/%d %I:%M:%S')
            node_time.append((end_time - start_time).total_seconds())
            continue


    return np.array(wass_time),np.array(node_time)

def seconds_to_format(times):
    seconds = np.mean(times)
    float_time = f'{np.round(seconds-int(seconds),decimals=2)}'[1:]
        # print(float_time)
    mean = datetime.fromtimestamp(int(seconds)) - datetime.strptime("1970-01-01 08:00:00",'%Y-%m-%d %I:%M:%S')
    
    print('Mean Time: {}{} +- {:2.2f} sec'.format(
                            mean,
                            float_time,  
                            np.std(times)))
    return ['{}{}'.format(mean,float_time),np.std(times)]

column_list = ['dataset','method','node emb mean','node emb std','data num','wass distance mean','wass dsitance std','data num']
data = []
for dataset in datasets:
    for method in methods:
        file = dataset+'_'+method+'.log'
        print(file)
        wass_time,node_time = stat_file(file)
        data_unit = [dataset,method]
        data_unit.extend(seconds_to_format(node_time))
        data_unit.append(len(node_time))
        data_unit.extend(seconds_to_format(wass_time))
        data_unit.append(len(wass_time))
        data.append(data_unit)
import pandas as pd
df = pd.DataFrame(data,columns = column_list)
df.to_csv('time.csv')
        
