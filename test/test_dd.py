import dgl.data
import unittest
import sys
sys.path.append("..")
# from sklearn.covariance import graphical_lasso
import torch
import numpy as np
import signature
# 继承 unittest.TestCase 就创建了一个测试样例。
class TestDGL(unittest.TestCase):
    def test_dd(self):
        data = dgl.data.LegacyTUDataset('PROTEINS')
        graphs, y = zip(*[graph for graph in data])
        graphs = list(graphs)
        y = np.array([unit.item() for unit in y])
        # print(signature.GetNodeAttrMat(graphs[0]))
        print(np.unique(signature.GetNodeAttrMat(graphs[0]),return_counts=True))
        print(graphs[0])
        # max_type = 0
        # min_type = 123456789
        # for g in graphs:
        #     max_type = np.max((max_type,np.max(signature.GetNodeAttrMat(g))))
        #     min_type = np.min((min_type,np.abs(np.min(signature.GetNodeAttrMat(g)))))
        #     if np.min(signature.GetNodeAttrMat(g)) < 0:
        #         print(g)
        #         # print(signature.GetNodeAttrMat(g))
        # print(max_type,min_type)
        # print()
if __name__ == "__main__":
    unittest.main()