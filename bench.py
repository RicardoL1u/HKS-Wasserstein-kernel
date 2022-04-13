import grakel
import grakel.kernels
import random
import numpy as np
import dgl
import wass_dis
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s: %(message)s',datefmt='%Y/%m/%d %I:%M:%S',level=logging.DEBUG)


from signature import WKS
from signature import get_sample4WKS

from signature import HKS
from signature import get_random_samples_based_exp_dual

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

def testHKS(g0,g1):
    wass_dis.pairwise_wasserstein_distance([g0,g1],800,signature_method=HKS,sample_method=get_random_samples_based_exp_dual,weight=[0.4])

if __name__ == '__main__':
    import timeit

    column_list = ['n','SP','RW','GL3','GL4','HKS','WKS']
    

    SP = grakel.kernels.ShortestPath(normalize=True,with_labels=False)
    RW = grakel.kernels.RandomWalk(normalize=True)
    GL3 = grakel.kernels.GraphletSampling(normalize=True,k=3)
    GL4 = grakel.kernels.GraphletSampling(normalize=True,k=4)

    data = []
    ns = [10,int(10**1.5),100,int(100**2.5)]
    # ns = [10]
    for n in ns:
        for i in range(5):
            data_unit = [n]
            cost = []
            logging.info(f'This is {i}-th graph with node number = {n}')
            adj0 = random_adjacency_matrix(n)
            adj1 = random_adjacency_matrix(n)

            g0 = grakel.Graph(adj0)
            g1 = grakel.Graph(adj1)
            
            # For Python>=3.5 one can also write:
            logging.info('SP test')
            cost.append(timeit.timeit("test0(g0,g1,SP)", globals=locals(),number=10))
            # logging.info('RW test')
            # cost.append(timeit.timeit("test0(g0,g1,RW)", globals=locals(),number=10))
            # logging.info('GL3 test')
            # cost.append(timeit.timeit("test0(g0,g1,GL3)", globals=locals(),number=10))
            # logging.info('GL4 test')
            # cost.append(timeit.timeit("test0(g0,g1,GL4)", globals=locals(),number=10))

            g0 = get_dglGraph(adj0)
            g1 = get_dglGraph(adj1)

            logging.info('HKS test')
            cost.append(timeit.timeit("testHKS(g0,g1)", globals=locals(),number=10))
            logging.info('WKS test')
            cost.append(timeit.timeit("testWKS(g0,g1)", globals=locals(),number=10))
            cost = np.array(cost)
            cost = np.round(cost,4)
            data_unit.extend(cost)
            data.append(data_unit)
            logging.info(f'This is data-point is {data_unit}')
    
    df = pd.DataFrame(data,columns=column_list)
    df.to_csv('bench_mean.csv')

