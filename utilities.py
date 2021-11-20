import os
import sklearn.model_selection
import sklearn.model_selection._validation
import sklearn.metrics
import sklearn.base
import numpy as np

#################
# File loaders 
#################
def read_labels(filename):
    '''
    Reads labels from a file. Labels are supposed to be stored in each
    line of the file. No further pre-processing will be performed.
    '''
    labels = []
    with open(filename) as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]
    return labels

def retrieve_graph_filenames(data_directory):
    """
    retrieve the graph filenames based on the given dataset

    Parameters
    ----
    data_directory: string

    Returns
    ----
    graph_filenames: list[string]
    """

    # Load graphs
    files = os.listdir(data_directory)
    graphs = [g for g in files if g.endswith('gml')]
    graphs.sort()
    return [os.path.join(data_directory, g) for g in graphs]

#######################
# Hyperparameter search
#######################

def custom_grid_search_cv(model, param_grid, precomputed_kernels, y, cv=5):
    '''
    Custom grid search based on the sklearn grid search for an array of precomputed kernels
    '''
    # 1. Stratified K-fold
    cv = sklearn.model_selection.StratifiedKFold(n_splits=cv, shuffle=False)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = [] # list of dict, its the same for every split
        # run over the kernels first
        for K_idx, K in enumerate(precomputed_kernels):
            # Run over parameters
            for p in list(sklearn.model_selection.ParameterGrid(param_grid)):
                sc = sklearn.model_selection._validation._fit_and_score(sklearn.base.clone(model), K, y, 
                        scorer=sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score), 
                        train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)

                # TODO: seems the svm is failed to fit the MUTAG
                # print(sc)
                # print("what you know")
                split_results.append(sc["test_scores"])
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = sklearn.base.clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]