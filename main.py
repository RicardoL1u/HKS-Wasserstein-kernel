import argparse
from math import fabs
import os
from random import shuffle
import wass_dis
import utilities
import numpy as np
np.random.seed(1205) 
import sklearn.model_selection
import dgl.data
# import igraph as ig
import sklearn.metrics
import pandas as pd
from sklearn.svm import SVC
import datetime
import signature

# global parameter
method_dict = {
    0:'HKS',
    1:'WKS',
}

signature_dict = {
    0:signature.HKS,
    1:signature.WKS,
}
sampleways_dict = {
    0:{
        0:signature.get_random_samples,
        1:signature.get_random_samples_li,
        2:signature.get_random_samples_based_exp_dual,
        # 3:signature.get_sample4WKS
    },
    1:{
        0:signature.get_sample4WKS,
        1:signature.get_random_samples_li,
        2:signature.get_random_samples_based_exp_dual
    }
}

def main():
    print()
    print("=============================================================")
    print("Graph classification using the HKS/WKS and Wassertain kernel.")
    print("=============================================================")
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Provide the dataset name',
                            choices=['MUTAG','PTC_MR',"NCI1","PROTEINS","DD",'ENZYMES'])
    parser.add_argument('-m','--method',type = int ,default=0,help='0 for hks,1 for wks')
    parser.add_argument('-w','--weight',type = float ,default=0.4,help='the relative important metric between generated node signature and node feature')
    parser.add_argument('-s','--samplemethods',type = int,default=0,help='choose different sample methods')
    parser.add_argument('-cv','--crossvalidation', default=False, action='store_true', help='Enable a 10-fold crossvalidation')
    parser.add_argument('-gs','--gridsearch', default=False, action='store_true', help='Enable grid search')
    parser.add_argument('--sinkhorn', default=False, action='store_true', help='Use sinkhorn approximation')
    parser.add_argument('-hl','--hlen', type = int, required=False, default=800, help = "number of sample points in signature, would be 100*h")
    parser.add_argument('-c','--C', type = float, required=False, default=1, help = "the strength of the regularization of SVM is inversely proportionaly to C")
    parser.add_argument('-g','--gamma', type = float, required=False, default=10, help = "Gammas in eps(-gamma*M):")
    parser.add_argument('-p','--path', type = str, required=False, default=None, help = "Path for experiment records")
    parser.add_argument('-n','--name', type = str, required=False, default=None, help = "name for experiment records")
    # parser.add_argument('--h_min', type = int, required=False, default=5, help = "(Min) number of sample points in HKS, would be 2^n")
    # parser.add_argument('--h_max', type = int, required=False, default=10, help = "(Max) number of sample points in HKS, would be 2^n")

    args = parser.parse_args()
    dataset = args.dataset
    
    if args.path == None:
        results_path = os.path.join('results', dataset)
    else:
        results_path = os.path.join(args.path, dataset)
    results_path = os.path.join(results_path,method_dict[args.method])

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Transform to Kernel
    # Here the flags come into play
    if args.gridsearch:
        # Gammas in eps(-gamma*M):
        gammas = np.logspace(-4,1,num=6)  
        # iterate over the iterations too
        param_grid = [
            # C is the hype parameter of SVM
            # The strength of the regularization is inversely proportional to C. 
            # Must be strictly positive. The penalty is a squared l2 penalty.
            {'C': np.logspace(-3,3,num=7)}
        ]
        # hs = np.arange(5,10)*100
        ws = np.arange(3,8)*0.1
        
    else:
        ws = [args.weight]
        # hs = [args.hlen]
        C = [args.C]
        gammas = [args.gamma]

    #---------------------------------
    # Embeddings
    #---------------------------------
    # Load the data and generate the embeddings 
    print(f'Generating {method_dict[args.method]} embeddings by {sampleways_dict[args.method][args.samplemethods]} with hl={args.hlen} for {dataset}.')
    if not args.gridsearch:
        print(f'with hlen = {args.hlen}, C = {args.C} and gammas = {args.gamma}')
    data = dgl.data.LegacyTUDataset(name=dataset)
    graphs, y = zip(*[graph for graph in data])
    graphs = list(graphs)
    y = list(y)
    y = np.array([unit.item() for unit in y])
    
    # Load the data and generate the embeddings 
    # Calculate the wass dis with the given number of samples points in HKS
    wasserstein_distances = wass_dis.pairwise_wasserstein_distance(graphs,args.hlen,signature_dict[args.method],sampleways_dict[args.method][args.samplemethods],ws,args.sinkhorn)
    
    # Save Wasserstein distance matrices
    # for i, D_w in enumerate(wasserstein_distances):
    #     filext = 'wasserstein_distance_matrix'
    #     if args.sinkhorn:
    #         filext += '_sinkhorn'
    #     filext += f'_it{i}.npy'
    #     np.save(os.path.join(output_path,filext), D_w)
    # print('Wasserstein distances computation done. Saved to file.')
    # print()

    kernel_matrices = []
    kernel_params = []
    for i, current_w in enumerate(ws):
        # Generate the full list of kernel matrices from which to select
        M = wasserstein_distances[i]
        for g in gammas:
            K = np.exp(-g*M)
            kernel_matrices.append(K)
            kernel_params.append((current_w, g))

    # Check for no hyperparameter:
    if not args.gridsearch:
        assert len(kernel_matrices) == 1
    print('Kernel matrices computed.')
    print()

    #---------------------------------
    # Classification
    #---------------------------------
    # Run hyperparameter search if needed
    print(f'Running SVMs, crossvalidation: {args.crossvalidation}, gridsearch: {args.gridsearch}.')

    # Contains accuracy scores for each cross validation step; the
    # means of this list will be used later on.
    accuracy_scores = []
    # Hyperparam logging
    best_C = []
    best_w = []
    best_gamma = []

    cv = sklearn.model_selection.StratifiedKFold(n_splits=10,shuffle=True)
    
    fold_result = []
    param = []
    for train_index, test_index in cv.split(kernel_matrices[0], y):
        K_train = [K[train_index][:, train_index] for K in kernel_matrices]
        K_test  = [K[test_index][:, train_index] for K in kernel_matrices]
        y_train, y_test = y[train_index], y[test_index]
       
        # Gridsearch
        if args.gridsearch:
            gs, best_params,unit_result,param = utilities.custom_grid_search_cv(SVC(kernel='precomputed'), 
                    param_grid, K_train, y_train, cv=5)
            fold_result.append(unit_result)
            # Store best params
            C_ = best_params['params']['C']
            w_,gamma_ = kernel_params[best_params['K_idx']]
            y_pred = gs.predict(K_test[best_params['K_idx']])
        else:
            gs = SVC(C=C[0], kernel='precomputed').fit(K_train[0], y_train)
            y_pred = gs.predict(K_test[0])
            w_,gamma_, C_ =ws[0], gammas[0], C[0] 
        best_C.append(C_)
        best_w.append(w_)
        best_gamma.append(gamma_)

        accuracy_scores.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        if not args.crossvalidation:
            break

    #---------------------------------
    # Printing and logging
    #---------------------------------
    if args.crossvalidation:
        print('Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %'.format(
                    np.mean(accuracy_scores) * 100,  
                    np.std(accuracy_scores) * 100))
    else:
        print('Final accuracy: {:2.3f} %'.format(np.mean(accuracy_scores)*100))
    
    if args.crossvalidation or args.gridsearch:
        if args.name != None:
            extension = "_"+args.name+"_"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        else:
            extension = "_"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if args.crossvalidation:
            extension += '_crossvalidation'
        if args.gridsearch:
            extension += '_gridsearch'
        results_filename = os.path.join(results_path, f'results_{dataset}'+f'_{method_dict[args.method]}'+extension+'.csv')
        n_splits = 10 if args.crossvalidation else 1
        pd.DataFrame(np.array([best_w,best_C, best_gamma, accuracy_scores]).T, 
                columns=[['w','C', 'gamma', 'accuracy']], 
                index=['fold_id{}'.format(i) for i in range(n_splits)]).to_csv(results_filename)
        print(f'Results saved in {results_filename}.')
    else:
        print('No results saved to file as --crossvalidation or --gridsearch were not selected.')

if __name__ == "__main__":
    main()
