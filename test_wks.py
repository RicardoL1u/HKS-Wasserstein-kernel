import dgl.data
import unittest
from matplotlib.pyplot import ginput
# from sklearn.covariance import graphical_lasso
import torch
import numpy
from dgl.data.utils import load_graphs
from zmq import proxy_steerable
import signature
# 继承 unittest.TestCase 就创建了一个测试样例。
class TestPipeline(unittest.TestCase):
    
    def test_wks(self):
        # prb offset
        # [1, 15, 22, 24, 28, 33, 40, 41, 47, 54, 57, 61, 72, 74, 81, 82, 89, 101, 111]
        print('hi')
        glist,label = load_graphs('./graph.bin')
        print(len(glist))
        g = glist[0]
        
        signature.WKS(g)
        # for g in glist:        
        #     HKS.WKS(g)
        # print(len(glist))
        # print(HKS.problem_labels)

if __name__ == "__main__":
    unittest.main()