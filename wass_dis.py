import signature
import numpy as np
import ot
import logging

def pairwise_wasserstein_distance(X,T,signature_method,sample_method,weight:list,sinkhorn=False):
    """
    Pairwise computation of the Wasserstein distance between embeddings of the 
    graphs in X.
    args:
        X (List[DGL.DGLGraph]): List of graphs
        T (int): the num of sample points for HKS
        sinkhorn (bool): Indicates whether sinkhorn approximation should be used
    """
    
    # Embed the nodes
    # logging.basicConfig(format='%(asctime)s: %(message)s',datefmt='%Y/%m/%d %I:%M:%S',level=logging.DEBUG)
    # logging.info("ready to generate node embeddings")
    node_signature_matrice = signature.CalculateSignature4Graphs(X,signature_method,sample_method,T)
    node_attr_matrice = signature.GraphAttrMatrice(X)
    # logging.info("the node embedding have been generated")
    wasserstein_distances = []
    # Compute the Wasserstein distance
    for w in weight:
        graph_node_embeddings = []
        for i in range(len(X)):
            graph_node_embeddings.append(np.concatenate((w*node_signature_matrice[i],(1-w)*node_attr_matrice[i]),axis=1))
        # msg = f"the wass diss with sinkhorn = {sinkhorn} and w = {w}"
        # logging.debug("ready to compute "+msg )
        wasserstein_distances.append(_compute_wasserstein_distance(graph_node_embeddings, 
            sinkhorn=sinkhorn, sinkhorn_lambda=10))
        # logging.debug("have computed "+msg)
    return wasserstein_distances


def _compute_wasserstein_distance(node_embeddings_matrice, sinkhorn=False, 
                                   sinkhorn_lambda=0.01,isImport=False):
    '''
    Generate the Wasserstein distance matrix for the graphs embedded 
    in label_sequences
    '''
    # Get the iteration number from the embedding file
    n = len(node_embeddings_matrice)
    
    M = np.zeros((n,n))

    for i in range(len(node_embeddings_matrice)):
        node_embeddings_matrice[i] = np.nan_to_num(node_embeddings_matrice[i],posinf=1.0,neginf=1e-7,nan=0)
    # Iterate over pairs of graphs
    for graph_index_1, graph_1 in enumerate(node_embeddings_matrice):
        for graph_index_2, graph_2 in enumerate(node_embeddings_matrice[graph_index_1:]):
            
            # print(graph_1.tolist())
            # print(np.dot(graph_1,graph_2.T))
            # Get cost matrix
            costs = ot.dist(graph_1, graph_2, metric='sqeuclidean')
            # print(costs)
            # print(graph_1-graph_2)

            graph1_dis = np.ones(graph_1.shape[0])/graph_1.shape[0]
            graph2_dis = np.ones(graph_2.shape[0])/graph_2.shape[0]

            if sinkhorn:
                mat = ot.sinkhorn(graph1_dis, graph2_dis, costs, sinkhorn_lambda, numItermax=50)
                M[graph_index_1, graph_index_2 + graph_index_1] = np.sum(np.multiply(mat, costs))
            else:
                M[graph_index_1, graph_index_2 + graph_index_1] = ot.emd2(graph1_dis,graph2_dis, costs)
                    
    M = (M + M.T)
    return M

