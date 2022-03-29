# """
# =============================================================
# Graph classification on MUTAG using the shortest path kernel.
# =============================================================

# Script makes use of :class:`grakel.ShortestPath`
# """
# from __future__ import print_function
# print(__doc__)
import argparse
from asyncio.log import logger

import numpy as np
import sys
sys.path.append("..")
import utilities
import sklearn.model_selection

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath
from grakel.kernels import RandomWalkLabeled
from grakel.kernels import WeisfeilerLehman
from grakel.kernels import GraphletSampling
from grakel.kernels import GraphHopper

import logging
logging.basicConfig(format='%(asctime)s: %(message)s',datefmt='%Y/%m/%d %I:%M:%S',level=logging.DEBUG)

kernel_list = [
    # ShortestPath,
    # WeisfeilerLehman,
    GraphletSampling,
    GraphletSampling,
    RandomWalkLabeled,
    # GraphHopper
]

kernel_param_dict_list = [
    # {'normalize':True},
    # {'normalize':True},
    {'normalize':True,'k':3},
    {'normalize':True,'k':4},
    {'normalize':True},
    # {'normalize':True,'method_type':'fast'}
]

def main():
    np.random.seed(1205) 


    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Provide the dataset name',
                            choices=['MUTAG','PTC_MR',"PROTEINS","DD",'ENZYMES'])
    parser.add_argument('-gs','--gridsearch', default=False, action='store_true', help='Enable grid search')
    args = parser.parse_args()
      
    print()
    print("=============================================================")
    print(f"Graph classification on {args.dataset} using the baseline kernels.")
    print("=============================================================")
    print()

    # Loads the given dataset
    logging.info(f'loading the {args.dataset} dataset')
    DATASET = fetch_dataset(args.dataset, verbose=False,as_graphs=False)
    G_ori, y_ori = DATASET.data, DATASET.target
    
    logging.info(f'before drop huge graphs, there are {len(y_ori)} graphs in {args.dataset}')
    index_list = []
    for i in range(len(G_ori)):
        if len(G_ori[i][1]) > 3 and  len(G_ori[i][1]) < 620:
            # [Graphs[i], node_labels[i], edge_labels[i]]
            index_list.append(i)
    G = [G_ori[i] for i in index_list]
    y = y_ori[index_list]
    logging.info(f'after drop huge graphs, there are {len(y)} graphs in {args.dataset}')

    
    # Transform to Kernel
    # Here the flags come into play
    if args.gridsearch:
        # iterate over the iterations too
        param_grid = [
            # C is the hype parameter of SVM
            # The strength of the regularization is inversely proportional to C. 
            # Must be strictly positive. The penalty is a squared l2 penalty.
            {'C': np.logspace(-3,3,num=7)}
        ]

    for i,kernel in enumerate(kernel_list):
        logging.info(f'read to conduct {str(kernel)} on {args.dataset}')
        logging.info(f'with gridserach = {args.gridsearch} and crossvalidation = True' )
        gk = kernel(**kernel_param_dict_list[i])
        cv = sklearn.model_selection.StratifiedKFold(n_splits=10,shuffle=True)
        M = gk.fit_transform(G)
        kernel_matrices = [M]
        accuracy_scores = []
        
        for train_index, test_index in cv.split(kernel_matrices[0], y):
            K_train = [K[train_index][:, train_index] for K in kernel_matrices]
            K_test  = [K[test_index][:, train_index] for K in kernel_matrices]
            y_train, y_test = y[train_index], y[test_index]
            
            
            # Gridsearch
            if args.gridsearch:
                gs, best_params,unit_result,param = utilities.custom_grid_search_cv(SVC(kernel='precomputed'), 
                        param_grid, K_train, y_train, cv=5)
                # Store best params
                # C_ = best_params['params']['C']
                # gamma_ = kernel_params[best_params['K_idx']]
                y_pred = gs.predict(K_test[best_params['K_idx']])
            else:
                # Uses the SVM classifier to perform classification
                clf = SVC(kernel="precomputed")
                clf.fit(K_train[0], y_train)
                y_pred = clf.predict(K_test[0])
            accuracy_scores.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        logging.info(f'have conducted {str(kernel)} on {args.dataset}')
        # Computes and prints the classification accuracy
        print('Mean 10-fold accuracy of '+str(kernel)+' in '+args.dataset+': {:2.2f} +- {:2.2f} %'.format(
                            np.mean(accuracy_scores) * 100,  
                            np.std(accuracy_scores)/np.sqrt(10) * 100))

if __name__ == "__main__":
    main()
