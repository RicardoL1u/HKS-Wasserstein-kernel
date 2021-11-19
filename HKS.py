import numpy as np
import igraph as ig

def HKS(graph, T):
    """
    Compute the Heat Kernel Signature for each node in the given graph

    Parameters
    ----------
    graph: igraph.Graph
    T: list
    
    Returns
    ----------
    embeddings: numpy.array
        shape (N * len(T))

    """
    
    adj_matrix = np.array(graph.get_adjacency().data)
    deg_vector = np.array(graph.degree())
    deg_matrix = np.diagflat(deg_vector)
    graph_laplacian = deg_matrix - adj_matrix
    eigenvalues,eigenvectors = np.linalg.eig(graph_laplacian)
    embeddings = np.zeros((len(deg_vector),len(T)))
    for i in range(len(deg_vector)):
        embedding = np.array([np.sum(np.exp(-eigenvalues*t)*eigenvectors[i]*eigenvectors[i]) for t in T])
        embeddings[i] = embedding
    return np.transpose(embeddings)