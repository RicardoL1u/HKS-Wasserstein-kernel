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
        glist,label = load_graphs('./graph.bin')
        print(len(glist))
        offset_list =[]
        cnt = 0
        for g in glist:
            print(g)
            HKS.WKS(g)

if __name__ == "__main__":
    unittest.main()