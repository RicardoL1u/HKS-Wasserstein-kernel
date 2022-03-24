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
means = np.array([np.mean(file["accuracy"]) for file in result_files])
stds = np.array([np.std(file["accuracy"]) for file in result_files])
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')
# plot
fig, ax = plt.subplots()
fig.set_size_inches(4, 4)
ax.fill_between(x_axis, means-stds, means+stds, alpha=.5, linewidth=0)
ax.plot(x_axis, means, linewidth=2)
ax.set(xlim=(0.25, 0.75), xticks=np.arange(0.25, 0.80, 0.05),
       ylim=(0.6, 0.7), yticks=np.arange(0.5, 0.75, 0.025))
# plt.show()
plt.savefig("fig1:varied_W.png", bbox_inches='tight', pad_inches=0)
