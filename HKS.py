import numpy as np
import igraph as ig

def get_random_samples(lambda2,lambdaLast,T=8):
    """
    sample HKS uniformly over the logarithm scaled temporal domain
    
    Parameters
    ---
    lambda2: the 2nd eigenvalue of eigendecomposition for the given graph,
             represent the base frequency of the graph spectral
    lambdaLast: the biggest eigenvalue represents the maximum frequency

    Returns:
    ---
    sample points: numpy.array
    """
    t_min = 4*np.log(10/lambdaLast)
    t_max = 4*np.log(10/lambda2)
    points = np.log(np.linspace(start=t_min,stop=t_max,num=T))
    return points

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


def HKS(graph,T,categorical=True,isHeuristics=False):
    """
    Compute the Heat Kernel Signature for each node in the given graph

    Parameters
    ----------
    graph: igraph.Graph

    Returns
    ----------
    tuple
    embeddings: numpy.array
        shape (len(T) * N)
        this shape is used to consistent with ot.dist
    

    """
    
    adj_matrix = np.array(graph.get_adjacency().data)
    deg_vector = np.array(graph.degree())
    deg_matrix = np.diagflat(deg_vector)
    graph_laplacian = deg_matrix - adj_matrix
    eigenvalues,eigenvectors = np.linalg.eig(graph_laplacian)
    sorted_eigenvalues = np.sort(eigenvalues)
    sample_points = get_random_samples(sorted_eigenvalues[1],sorted_eigenvalues[-1],T)
    embeddings = np.zeros((len(deg_vector),len(sample_points)))

    for i in range(len(deg_vector)):
        if isHeuristics:
            embedding = np.array([np.sum(np.exp(-eigenvalues*t)*eigenvectors[i]*eigenvectors[i])/np.sum(np.exp(-eigenvalues*t)) for t in sample_points])
        else:
            embedding = np.array([np.sum(np.exp(-eigenvalues*t)*eigenvectors[i]*eigenvectors[i]) for t in sample_points])
        
        embeddings[i] = embedding
    if categorical:
        embeddings = np.concatenate((embeddings,GetNodeAttrMat(graph,eigenvalues)),axis=1)
    return embeddings,eigenvalues

def GetNodeAttrMat(graph,eigenvalues,categorical = True):
    if categorical:
        labels = np.array(graph.vs['label'],dtype=int)
        num_labels = 7
        return np.eye(num_labels)[labels] * eigenvalues[:,None]
        # return np.eye(num_labels)[labels]
    else:
        return np.zeros((4,4))
    

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