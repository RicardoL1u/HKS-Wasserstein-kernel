import grakel
import grakel.kernels
import random
import numpy as np
import dgl
import wass_dis

from signature import WKS
from signature import get_sample4WKS

def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]

    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0

    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]

    return matrix

def get_dglGraph(adj_mat):
    src, dst = np.nonzero(adj_mat)
    g = dgl.DGLGraph((src,dst))
    return g


def test0(g0,g1,kernel):
    kernel.fit_transform([g0])
    kernel.transform([g1])

def testWKS(g0,g1):
    wass_dis.pairwise_wasserstein_distance([g0,g1],800,signature_method=WKS,sample_method=get_sample4WKS,weight=[0.4])


if __name__ == '__main__':
    import timeit
    adj0 = random_adjacency_matrix(5)
    adj1 = random_adjacency_matrix(5)

    g0 = grakel.Graph(adj0)
    g1 = grakel.Graph(adj1)
    
    SP = grakel.kernels.ShortestPath(normalize=True,with_labels=False)
    RW = grakel.kernels.RandomWalk(normalize=True)
    GL3 = grakel.kernels.GraphletSampling(normalize=True,k=3)
    GL4 = grakel.kernels.GraphletSampling(normalize=True,k=4)
    # For Python>=3.5 one can also write:
    print(timeit.timeit("test0(g0,g1,SP)", globals=locals(),number=10))
    print(timeit.timeit("test0(g0,g1,RW)", globals=locals(),number=10))
    print(timeit.timeit("test0(g0,g1,GL4)", globals=locals(),number=10))
    print(timeit.timeit("test0(g0,g1,GL3)", globals=locals(),number=10))

    g0 = get_dglGraph(adj0)
    g1 = get_dglGraph(adj1)

    print(timeit.timeit("testWKS(g0,g1)", globals=locals(),number=10))


