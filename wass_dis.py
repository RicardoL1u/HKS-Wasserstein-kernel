import HKS
import numpy as np
import ot

def pairwise_wasserstein_distance(X,T,  sinkhorn=False):
    """
    Pairwise computation of the Wasserstein distance between embeddings of the 
    graphs in X.
    args:
        X (List[ig.graphs]): List of graphs
        T (int): the num of sample points for HKS
        sinkhorn (bool): Indicates whether sinkhorn approximation should be used
    """
    
    # Embed the nodes
    node_embeddings_matrice = HKS.CalculateHKS4Graphs(X,T)

    # Compute the Wasserstein distance
    pairwise_distances = _compute_wasserstein_distance(node_embeddings_matrice, sinkhorn=sinkhorn, 
                                     sinkhorn_lambda=1e-2)
    return pairwise_distances


def _compute_wasserstein_distance(node_embeddings_matrice, sinkhorn=False, 
                                   sinkhorn_lambda=1e-2):
    '''
    Generate the Wasserstein distance matrix for the graphs embedded 
    in label_sequences
    '''
    # Get the iteration number from the embedding file
    n = len(node_embeddings_matrice)
    
    M = np.zeros((n,n))
    # Iterate over pairs of graphs
    for graph_index_1, graph_1 in enumerate(node_embeddings_matrice):
        for graph_index_2, graph_2 in enumerate(node_embeddings_matrice[graph_index_1:]):
            # Get cost matrix
            costs = ot.dist(graph_1, graph_2, metric='euclidean')

            if sinkhorn:
                mat = ot.sinkhorn(np.ones(len(graph_1))/len(graph_1), 
                                    np.ones(len(graph_2))/len(graph_2), costs, sinkhorn_lambda, 
                                    numItermax=50)
                M[graph_index_1, graph_index_2 + graph_index_1] = np.sum(np.multiply(mat, costs))
            else:
                M[graph_index_1, graph_index_2 + graph_index_1] = \
                    ot.emd2([], [], costs)
                    
    M = (M + M.T)
    return M