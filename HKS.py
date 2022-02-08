import numpy as np
# import igraph as ig
import os
import dgl
import torch

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
    # print(lambda2,lambdaLast)
    t_min = 4*np.log(10)/lambdaLast
    t_max = 4*np.log(10)/lambda2
    points = np.log(np.linspace(start=np.log(t_min),stop=np.log(t_max),num=T))
    return points

def get_random_samples_li():
    t0 =  0.01
    alpha1 = 2 
    tauScale = 15
    tau = np.arange(0,tauScale,1/16)
    return t0*alpha1**tau

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


def WKS(graph,N=200):
    w = 0.5
    wks_variance = 6
    adj_matrix = graph.adj()
    deg_vector = graph.out_degrees()
    deg_matrix = torch.diag(deg_vector)
    graphical_laplacian = deg_matrix - adj_matrix
    eigenvalues,eigenvectors = torch.linalg.eig(graphical_laplacian)
    eigenvalues = np.abs(eigenvalues.numpy())
    eigenvectors = eigenvectors.numpy()

    sorted_eigen = np.sort(eigenvalues)
    sorted_eigen[sorted_eigen<1e-6]=1e-6
    log_eigenvalue = np.log(sorted_eigen)
    # print(np.abs(sorted_eigen))
    # print(np.max(np.abs(sorted_eigen),1e-6))
    # log_eigenvalue = np.log(np.max(np.abs(sorted_eigen),1e-6))
    e_set = np.linspace(log_eigenvalue[1],log_eigenvalue[-1]/1.02,N)
    sigma =(e_set[1]-e_set[0])*wks_variance
    wks = np.zeros((len(deg_vector),N))
    for i in range(len(deg_vector)):
        wks[i] = np.array([np.sum(np.exp(-(e-log_eigenvalue)*(e-log_eigenvalue)/(2*sigma*sigma))*eigenvectors[i]*eigenvectors[i])/np.sum(np.exp(-(e-log_eigenvalue)*(e-log_eigenvalue)/(2*sigma*sigma))) for e in e_set])    
        # embeddings[i] = embedding
    # print(wks.shape)
    wks = np.concatenate(((1-w)*wks,w*GetNodeAttrMat(graph)),axis=1)
    # print(wks.shape)
    return wks

def HKS(graph,T,categorical,isHeuristics=False):
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
    w = 0.4
    adj_matrix = graph.adj()
    deg_vector = graph.out_degrees()
    deg_matrix = torch.diag(deg_vector)
    graphical_laplacian = deg_matrix - adj_matrix
    eigenvalues,eigenvectors = torch.linalg.eig(graphical_laplacian)
    eigenvalues = eigenvalues.numpy()
    eigenvectors = eigenvectors.numpy()

    # sorted_eigen = np.sort(eigenvalues)
    # print(len(sorted_eigen),lambda2,lambdaLast)
    # sample_points = get_random_samples(lambda2,lambdaLast,T)
    sample_points = get_random_samples_li()
    embeddings = np.zeros((len(deg_vector),len(sample_points)))

    # HKS part
    for i in range(len(deg_vector)):
        if isHeuristics:
            embedding = np.array([np.sum(np.exp(-eigenvalues*t)*eigenvectors[i]*eigenvectors[i])/\
                np.sum(np.exp(-eigenvalues*t)) for t in sample_points])
        else:
            embedding = np.array([np.sum(np.exp(-eigenvalues*t)*eigenvectors[i]*eigenvectors[i]) for t in sample_points])    
        embeddings[i] = embedding

    # for i in range(len(deg_vector)):
    #     embeddings[i] = np.array([np.sum(np.exp(-eigenvalues*t)*eigenvectors[i]*eigenvectors[i]) for t in sample_points])    
    #     # embeddings[i] = embedding

    embeddings = np.concatenate(((1-w)*embeddings,w*GetNodeAttrMat(graph)),axis=1)
    return embeddings

def GetNodeAttrMat(graph):
    return (graph.ndata['feat']).numpy()
    # if categorical:
    #     labels = np.array(graph.vs['attr'],dtype=int)
    #     num_labels = 7
    #     return 2.5*np.eye(num_labels)[labels]
    # else:
    #     attribtues_filenames = os.path.join(data_directory, 'node_features.npy')
    #     if os.path.isfile(attribtues_filenames):
    #         node_features = np.load(attribtues_filenames,allow_pickle=True)
    #     return node_features
    

def CalculateHKS4Graphs(graphs,T,categorical):
    """
    Calculate generate the matrix the node embeddings for each given graph

    Parameters
    ----------
    graphs: list of igraph.Graph

    Returns
    ----------
    feature_matrices: list of matrix of node embeddings

    """
    # matrices = [HKS(graph,T,categorical) for graph in graphs]
    matrices = [WKS(graph) for graph in graphs]
    return matrices