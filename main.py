import argparse
from cmath import log
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
import logging
logging.basicConfig(format='%(asctime)s: %(message)s',datefmt='%Y/%m/%d %I:%M:%S',level=logging.DEBUG)

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

def get_wass_distances(args,graphs,output_path,ws)->list:
    pre_wass_dis = []
    com_wass_dis = []
    pre_wass_index = []
    com_ws = []
    logging.info('ready to load the pre-computed wass distances')
    for w in ws:
        filext = 'wasserstein_distance_matrix'
        if args.sinkhorn:
            filext += '_sinkhorn'
        nw = '{:1.2f}'.format(w)
        filext += f'_it{args.samplemethods}_{nw}_{args.sinkhorn}.npy'
        if os.path.exists(os.path.join(output_path,filext)):
            pre_wass_index.append(True)
            pre_wass_dis.append(np.load(os.path.join(output_path,filext)))
        else:
            pre_wass_index.append(False)
            com_ws.append(w)
    logging.info(f'after loading pre-computed wass distances, there are distance with w={com_ws} remained' )
    if len(com_ws) > 0:
        com_wass_dis = wass_dis.pairwise_wasserstein_distance(graphs,args.hlen,signature_dict[args.method],
                            sampleways_dict[args.method][args.samplemethods],com_ws,args.sinkhorn)
    
    assert len(com_wass_dis) + len(pre_wass_dis) == len(ws), f'there are {len(com_wass_dis)} computed matrix and {len(pre_wass_dis)} pre-com matrix howevr ws number = {len(ws)}'
    
    wasserstein_distances = []
    cnt_pre = 0
    cnt_com = 0
    for i,v in enumerate(pre_wass_index):
        if v:
            wasserstein_distances.append(pre_wass_dis[cnt_pre])
            cnt_pre+=1
        else:
            wasserstein_distances.append(com_wass_dis[cnt_com])
            cnt_com+=1

    # Save Wasserstein distance matrices
    for i, D_w in enumerate(com_wass_dis):
        filext = 'wasserstein_distance_matrix'
        if args.sinkhorn:
            filext += '_sinkhorn'
        nw = '{:1.2f}'.format(com_ws[i])
        filext += f'_it{args.samplemethods}_{nw}_{args.sinkhorn}.npy'
        np.save(os.path.join(output_path,filext), D_w)
        logging.info(f'have saved {filext}')
    logging.info('Wasserstein distances computation done. Saved to file.')
    return wasserstein_distances

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

    output_path = os.path.join('output', dataset)
    if args.path == None:
        results_path = os.path.join('results', dataset)
    else:
        results_path = os.path.join(args.path, dataset)
    results_path = os.path.join(results_path,method_dict[args.method])
    output_path = os.path.join(output_path,method_dict[args.method])

    for path in [output_path, results_path]:	
        if not os.path.exists(path):
            os.makedirs(path)

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
        ws = [0.00,0.05,0.10,0.15,0.20,0.25,
              0.30,0.35,0.40,0.45,0.50,0.55,
              0.60,0.65,0.70,0.75,0.80,0.85,
              0.90,0.95,1.00]
        # ws = [0.4,0.5,0.6]
        
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
    graphs_ori, y_ori = zip(*[graph for graph in data])
    graphs_ori = list(graphs_ori)
    y_ori = list(y_ori)

    logging.info(f'before drop huge graphs, there are {len(y_ori)} graphs in {args.dataset}')
    index_list = []
    for i,graph in enumerate(graphs_ori):
        if graph.number_of_nodes() < 620:
            index_list.append(i)
    graphs = [graphs_ori[i] for i in index_list]
    y = [y_ori[i] for i in index_list]
    y = np.array([unit.item() for unit in y])
    logging.info(f'after drop huge graphs, there are {len(y)} graphs in {args.dataset}')



    # Load the data and generate the embeddings 
    # Calculate the wass dis with the given w
    wasserstein_distances = get_wass_distances(args,graphs,output_path,ws)
    print()

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
    logging.info(f'Running SVMs, crossvalidation: {args.crossvalidation}, gridsearch: {args.gridsearch}.')

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
    logging.info('gridsearch and crossvalidation done')
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
