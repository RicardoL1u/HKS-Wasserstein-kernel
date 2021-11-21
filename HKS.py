import numpy as np
import igraph as ig

def get_random_samples(T=8):
    np.random.seed(1205)
    # return np.random.random((T))*1500
    # return np.linspace(0,5000,T)
    return np.random.standard_exponential((T))

def get_random_samples_based_exp(T=8,lambda_ = 1):
    np.random.seed(542)
    beta = 1/lambda_
    return np.random.exponential(scale=lambda_,size=(T))

def get_random_samples_based_exp_dual(T=8,lambda_ = 1):
    np.random.seed(542)
    beta = 1/lambda_
    zero_list = np.zeros((int(T/2)))
    samples_left = np.random.exponential(scale=beta,size=(int(T/2)))
    samples_right = np.maximum(50-np.random.exponential(scale=beta,size=(int(T/2))),zero_list)
    # np.maximum()
    return np.concatenate((samples_left,samples_right))

# def get_random_samples_based_hypoexp(lambdas,T=8):
#     np.random.seed(42)
#     samples_list = [np.random.exponential(scale=1/lambda_,size=(T)) for lambda_ in lambdas]
#     sample_list = np.zeros((T))
#     for sample in samples_list:
#         samples_list += sample
#     return sample_list/len(lambdas)


def HKS(graph,T):
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
    eigenvalues,eigenvectors = np.linalg.eig(graph_laplacian)
    sample_points = get_random_samples_based_exp_dual(T,np.mean(eigenvalues))
    embeddings = np.zeros((len(deg_vector),len(sample_points)))
    for i in range(len(deg_vector)):
        embedding = np.array([np.sum(np.exp(-eigenvalues*t)*eigenvectors[i]*eigenvectors[i]) for t in sample_points])
        embeddings[i] = embedding
    return embeddings

def CalculateHKS4Graphs(graphs,T):
    """
    Calculate generate the matrix the node embeddings for each given graph

    Parameters
    ----------
    graphs: list of igraph.Graph

    Returns
    ----------
    feature_matrices: list of matrix of node embeddings

    """
    matrices = [HKS(graph,T) for graph in graphs]
    return matrices