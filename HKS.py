import numpy as np
import igraph as ig

def HKS(graph, T,intervel):
    adj_matrix = np.array(graph.get_adjacency().data)
    deg_vector = np.array(graph.degree())
    deg_matrix = np.diagflat(deg_vector)
    graph_laplacian = deg_matrix - adj_matrix
    eigenvalues,eigenvectors = np.linalg.eig(graph_laplacian)
    embeddings = np.array((len(deg_vector),T))
    embeddings