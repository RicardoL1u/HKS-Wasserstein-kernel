# import csv
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Provide the dataset name',
                            choices=['MUTAG','PTC_MR',"NCI1","PROTEINS","DD",'ENZYMES'])
    parser.add_argument('-m', '--method', type=str, help='Provide the signature name',
                            choices=['HKS','WKS'])
    parser.add_argument('--sample', type=int,default=0,required=True , help='Provide the sample number')
    parser.add_argument('-s','--start', type = float, required=True, help = "start value of x")
    parser.add_argument('-e','--end', type = float, required=True, help = "end value of x")
    parser.add_argument('-step','--step', type = float, required=True, help = "end value of x")
    args = parser.parse_args()

    x_axis = np.arange(args.start,args.end+args.step,args.step).tolist()
    Path = "./"+args.dataset+"/"+args.method+"/"
    # Get list of all files in a given directory sorted by name
    list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(Path, x)),
                            os.listdir(Path) ) )
    result_files = [pd.read_csv(Path+i) for i in list_of_files]
    means = np.array([np.mean(file["accuracy"]) for file in result_files])
    stds = np.array([np.std(file["accuracy"]) for file in result_files])
    print(means,stds)
    plt.style.use('_mpl-gallery')
    # plot
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)
    ax.fill_between(x_axis, means-stds, means+stds, alpha=.5, linewidth=0)
    ax.plot(x_axis, means, linewidth=2)
    ax.set(xlim=(args.start-args.step, args.end+args.step), xticks=x_axis,
        ylim=(0.50, 0.80), yticks=np.arange(0.50, 0.80, 0.025))
    plt.show()
    plt.savefig(f"fig1:varied_w_hl=800_{args.method}{args.sample}_{args.dataset}.png", bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()