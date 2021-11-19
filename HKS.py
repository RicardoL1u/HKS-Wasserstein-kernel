import numpy as np
import igraph as ig

def get_random_samples(T=8):
    return np.random.random((T))

def HKS(graph):
    """
    Compute the Heat Kernel Signature for each node in the given graph

    Parameters
    ----------
    graph: igraph.Graph

    Returns
    ----------
    embeddings: numpy.array
        shape (len(T) * N)
        this shape is used to consistent with ot.dist

    """
    
    adj_matrix = np.array(graph.get_adjacency().data)
    deg_vector = np.array(graph.degree())
    deg_matrix = np.diagflat(deg_vector)
    graph_laplacian = deg_matrix - adj_matrix
    sample_points = get_random_samples()
    eigenvalues,eigenvectors = np.linalg.eig(graph_laplacian)
    embeddings = np.zeros((len(deg_vector),len(sample_points)))
    for i in range(len(deg_vector)):
        embedding = np.array([np.sum(np.exp(-eigenvalues*t)*eigenvectors[i]*eigenvectors[i]) for t in sample_points])
        embeddings[i] = embedding
    return embeddings

def CalculateHKS4Graphs(graphs):
    """
    Calculate generate the matrix the node embeddings for each given graph

    Parameters
    ----------
    graphs: list of igraph.Graph

    Returns
    ----------
    feature_matrices: list of matrix of node embeddings

    """
    matrices = [HKS(graph) for graph in graphs]
    return matrices