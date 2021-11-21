import argparse
import os
from random import shuffle
import wass_dis
import utilities
import numpy as np
import sklearn.model_selection
import igraph as ig
import sklearn.metrics
import pandas as pd
from sklearn.svm import SVC

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Provide the dataset name (MUTAG or Enzymes)',
                            choices=['MUTAG', 'ENZYMES'])
    parser.add_argument('--crossvalidation', default=False, action='store_true', help='Enable a 10-fold crossvalidation')
    parser.add_argument('--gridsearch', default=False, action='store_true', help='Enable grid search')
    parser.add_argument('--sinkhorn', default=False, action='store_true', help='Use sinkhorn approximation')
    parser.add_argument('--h_min', type = int, required=False, default=5, help = "(Min) number of sample points in HKS, would be 2^n")
    parser.add_argument('--h_max', type = int, required=False, default=10, help = "(Max) number of sample points in HKS, would be 2^n")

    args = parser.parse_args()
    dataset = args.dataset
    
    data_path = os.path.join('./data',dataset)
    output_path = os.path.join('output', dataset)
    results_path = os.path.join('results', dataset)

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
        hs = np.arange(1,10)*100
    else:
        gammas = [0.001]
        hs = [8]

    #---------------------------------
    # Embeddings
    #---------------------------------
    # Load the data and generate the embeddings 
    embedding_type = 'continuous' if dataset == 'ENZYMES' else 'discrete'
    print(f'Generating {embedding_type} embeddings for {dataset}.')
    # if dataset == 'ENZYMES':
    #     label_sequences = compute_wl_embeddings_continuous(data_path, h)
    # else:
    #     label_sequences = compute_wl_embeddings_discrete(data_path, h)
    graph_filenames = utilities.retrieve_graph_filenames(data_path)
    graphs = [ig.read(filename) for filename in graph_filenames]
    wasserstein_distances = [wass_dis.pairwise_wasserstein_distance(graphs,t) for t in hs]


    sinkhorn = False
    # Save Wasserstein distance matrices
    for i, D_w in enumerate(wasserstein_distances):
        filext = 'wasserstein_distance_matrix'
        if sinkhorn:
            filext += '_sinkhorn'
        filext += f'_it{i}.npy'
        np.save(os.path.join(output_path,filext), D_w)
    print('Wasserstein distances computation done. Saved to file.')
    print()




    kernel_matrices = []
    kernel_params = []
    for i, current_h in enumerate(hs):
        # Generate the full list of kernel matrices from which to select
        M = wasserstein_distances[i]
        for g in gammas:
            K = np.exp(-g*M)
            kernel_matrices.append(K)
            kernel_params.append((current_h, g))

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
    # Load labels
    label_file = os.path.join(data_path, 'Labels.txt')
    
    y = np.array(utilities.read_labels(label_file))

    # Contains accuracy scores for each cross validation step; the
    # means of this list will be used later on.
    accuracy_scores = []
    # np.random.seed(42)
    np.random.seed(1205) #Mean 10-fold accuracy: 72.43 +- 8.22 %
    # np.random.seed(1205542) #Mean 10-fold accuracy: 70.73 +- 8.27 %
    # np.random.seed(2018212874) #Mean 10-fold accuracy: 68.48 +- 10.15 %
    # Hyperparam logging
    best_C = []
    best_h = []
    best_gamma = []

    cv = sklearn.model_selection.StratifiedKFold(n_splits=10,shuffle=True)

    for train_index, test_index in cv.split(kernel_matrices[0], y):
        K_train = [K[train_index][:, train_index] for K in kernel_matrices]
        K_test  = [K[test_index][:, train_index] for K in kernel_matrices]
        y_train, y_test = y[train_index], y[test_index]
       
        # Gridsearch
        if args.gridsearch:
            gs, best_params = utilities.custom_grid_search_cv(SVC(kernel='precomputed'), 
                    param_grid, K_train, y_train, cv=5)
            # Store best params
            C_ = best_params['params']['C']
            h_,gamma_ = kernel_params[best_params['K_idx']]
            y_pred = gs.predict(K_test[best_params['K_idx']])
        else:
            gs = SVC(C=100, kernel='precomputed').fit(K_train[0], y_train)
            y_pred = gs.predict(K_test[0])
            h_,gamma_, C_ =8, gammas[0], 100 
        best_C.append(C_)
        best_h.append(h_)
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
        extension = ''
        if args.crossvalidation:
            extension += '_crossvalidation'
        if args.gridsearch:
            extension += '_gridsearch'
        results_filename = os.path.join(results_path, f'results_{dataset}'+extension+'.csv')
        n_splits = 10 if args.crossvalidation else 1
        pd.DataFrame(np.array([best_h,best_C, best_gamma, accuracy_scores]).T, 
                columns=[['h','C', 'gamma', 'accuracy']], 
                index=['fold_id{}'.format(i) for i in range(n_splits)]).to_csv(results_filename)
        print(f'Results saved in {results_filename}.')
    else:
        print('No results saved to file as --crossvalidation or --gridsearch were not selected.')

if __name__ == "__main__":
    main()