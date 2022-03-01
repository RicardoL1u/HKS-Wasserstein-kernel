from distutils.log import debug
import numpy as np
# import igraph as ig
import os
import dgl
import torch
import dgl.data

problem_graphs = []
problem_labels = []
cnt = 0

def get_random_samples(lambda2,lambdaLast,T=8) -> np.ndarray :
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

def get_random_samples_li() -> np.ndarray :
    t0 =  0.01
    alpha1 = 2 
    tauScale = 15
    tau = np.arange(0,tauScale,1/16)
    return t0*alpha1**tau

def get_random_samples_based_exp(T=8,lambda_ = 1) -> np.ndarray :
    np.random.seed(542)
    beta = 1/lambda_
    return np.random.exponential(scale=lambda_,size=(T))

def get_random_samples_based_exp_dual(T=8,lambda_ = 1) -> np.ndarray :
    np.random.seed(542)
    beta = 1/lambda_
    zero_list = np.zeros((int(T/2)))
    samples_left = np.random.exponential(scale=beta,size=(int(T/2)))
    samples_right = np.maximum(50-np.random.exponential(scale=beta,size=(int(T/2))),zero_list)

    return np.concatenate((samples_left,samples_right))


def WKS(graph,N=200) -> np.ndarray :
    # w = 0.
    global cnt
    cnt = cnt + 1
    adj_matrix = graph.adj()
    deg_vector = graph.out_degrees()
    deg_matrix = torch.diag(deg_vector)
    graphical_laplacian = deg_matrix - adj_matrix
    eigenvalues,eigenvectors = torch.linalg.eig(graphical_laplacian)
    eigenvalues = np.abs(eigenvalues.numpy())
    eigenvalues[eigenvalues<1e-6] = 1
    eigenvectors = eigenvectors.numpy()
    sorted_eigen = np.sort(eigenvalues)
    log_eigenvalue = np.log(sorted_eigen)
    e_set = np.linspace(log_eigenvalue[1],log_eigenvalue[-1]/1.02,N)
    wks_variance = 6 * 60 / len(eigenvalues)
    sigma =(e_set[1]-e_set[0])*wks_variance
    sigma = 1
    wks = np.zeros((len(deg_vector),N))
    # debugmark = False
    # for e in e_set:
    #     if np.sum(np.exp(-(e-log_eigenvalue)*(e-log_eigenvalue)/(2*sigma*sigma))) == 0:
    #         debugmark = True
    #         print(e)
    #         # break
    # if debugmark:
    #     problem_labels.append(cnt) 
    #     print(e_set)
    #     print(sigma)
    #     print(graph)
    #     print(sorted_eigen)
    #     print(log_eigenvalue)
    #     problem_graphs.append(graph)
    for i in range(len(deg_vector)):
        wks[i] = np.array([np.sum(np.exp(-(e-log_eigenvalue)*(e-log_eigenvalue)/(2*sigma*sigma))*eigenvectors[i]*eigenvectors[i])/np.sum(np.exp(-(e-log_eigenvalue)*(e-log_eigenvalue)/(2*sigma*sigma))) for e in e_set])    

    return wks

def HKS(graph,T,isHeuristics=False) -> np.ndarray :
    """
    Compute the Heat Kernel Signature for each node in the given graph

    Parameters
    ----------
    graph: DGL.DGLgraph

    Returns
    ----------
    tuple
    embeddings: numpy.array
        shape (len(T) * N)
        this shape is used to consistent with ot.dist
    

    """
    # w = 0.4
    adj_matrix = graph.adj()
    deg_vector = graph.out_degrees()
    deg_matrix = torch.diag(deg_vector)
    graphical_laplacian = deg_matrix - adj_matrix
    eigenvalues,eigenvectors = torch.linalg.eig(graphical_laplacian)
    eigenvalues = eigenvalues.numpy()
    eigenvectors = eigenvectors.numpy()

    # sorted_eigen = np.sort(eigenvalues)
    # print(len(sorted_eigen),lambda2,lambdaLast)
    # sample_points = get_random_samples(sorted_eigen[1],sorted_eigen[-1],T)
    # sample_points = get_random_samples_li()
    sample_points = get_random_samples_based_exp_dual(T,np.mean(eigenvalues))
    hks = np.zeros((len(deg_vector),len(sample_points)))

    # HKS part
    for i in range(len(deg_vector)):
        if isHeuristics:
            hks[i] = np.array([np.sum(np.exp(-eigenvalues*t)*eigenvectors[i]*eigenvectors[i])/\
                np.sum(np.exp(-eigenvalues*t)) for t in sample_points])
        else:
            hks[i] = np.array([np.sum(np.exp(-eigenvalues*t)*eigenvectors[i]*eigenvectors[i]) for t in sample_points])    

    return hks

def GetNodeAttrMat(graph):
    return (graph.ndata['feat']).numpy()
    

def CalculateSignature4Graphs(graphs,method,T):
    """
    Calculate generate the matrix the node embeddings for each given graph

    Parameters
    ----------
    graphs: list of igraph.Graph

    Returns
    ----------
    feature_matrices: list of matrix of node embeddings

    """
    w = 0.4
    if method==0:
        matrices = [np.concatenate(((1-w)*HKS(graph,T),w*GetNodeAttrMat(graph)),axis=1) for graph in graphs]
    elif method==1:
        matrices = [np.concatenate(((1-w)*WKS(graph,T),w*GetNodeAttrMat(graph)),axis=1) for graph in graphs]
    if len(problem_graphs) > 0:
        dgl.data.utils.save_graphs('./graph.bin',problem_graphs)
    return matrices