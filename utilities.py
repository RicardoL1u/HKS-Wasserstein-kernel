import os
import sklearn.model_selection
import sklearn.model_selection._validation
import sklearn.metrics
import sklearn.base
import numpy as np

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
    print(best_idx)
    # Return the fitted model and the best_parameters
    ret_model = sklearn.base.clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]

def custom_grid_search_cv1(model, param_grid, precomputed_kernels, y, cv=5):
    '''
    Custom grid search based on the sklearn grid search for an array of precomputed kernels
    '''
    # 1. Stratified K-fold
    cv = sklearn.model_selection.StratifiedKFold(n_splits=cv, shuffle=True)
    results = []
    params = [] # list of dict, its the same for every split
    for K_idx, K in enumerate(precomputed_kernels):
        for p in list(sklearn.model_selection.ParameterGrid(param_grid)):
            split_results = []
            params.append({'K_idx': K_idx, 'params': p})
            for train_index, test_index in cv.split(precomputed_kernels[0], y):
                # run over the kernels first
                # Run over parameters
                sc = sklearn.model_selection._validation._fit_and_score(sklearn.base.clone(model), K, y, 
                        scorer=sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score), 
                        train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)

                # TODO: seems the svm is failed to fit the MUTAG
                # print(sc)
                # print("what you know")
                split_results.append(sc["test_scores"])
            results.append(split_results)
    # Collect results and average
    results = np.array(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = sklearn.base.clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx],results[best_idx]