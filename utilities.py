import os
import sklearn.model_selection
import sklearn.model_selection._validation
import sklearn.metrics
import sklearn.base
import numpy as np
import igraph as ig

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

def read_gml(filename):
	node_features = []
	g = ig.read(filename)
		
	if not 'label' in g.vs.attribute_names():
		g.vs['label'] = list(map(str, [l for l in g.vs.degree()]))    
	
	node_features = g.vs['label']

	adj_mat = np.asarray(g.get_adjacency().data)
	
	return node_features, adj_mat


def load_continuous_graphs(data_directory):
    graph_filenames = retrieve_graph_filenames(data_directory)

    # initialize
    node_features = []
    adj_mat = []
    n_nodes = []

    # Iterate across graphs and load initial node features
    for graph_fname in graph_filenames:
        node_features_cur, adj_mat_cur = read_gml(graph_fname)
        # Load features
        node_features.append(np.asarray(node_features_cur).astype(float).reshape(-1,1))
        adj_mat.append(adj_mat_cur.astype(int))
        n_nodes.append(adj_mat_cur.shape[0])

    # Check if there is a node_features.npy file 
    # containing continuous attributes
    # PS: these were obtained by processing the TU Dortmund website
    # If none is present, keep degree or label as features.
    attribtues_filenames = os.path.join(data_directory, 'node_features.npy')
    if os.path.isfile(attribtues_filenames):
        node_features = np.load(attribtues_filenames,allow_pickle=True)

    n_nodes = np.asarray(n_nodes)
    node_features = np.asarray(node_features)

    return node_features, adj_mat, n_nodes

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