import dgl.data
import unittest
# from sklearn.covariance import graphical_lasso
import torch
import numpy
from dgl.data.utils import load_graphs
import HKS
# 继承 unittest.TestCase 就创建了一个测试样例。
class TestPipeline(unittest.TestCase):
    
    def test_wks(self):
        print('hi')
        glist = load_graphs('./graph.bin')
        print(glist)
        g = glist[0]
        print(g)
        HKS.WKS(g)