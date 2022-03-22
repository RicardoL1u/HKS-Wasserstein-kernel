import dgl.data
import unittest
import sys
sys.path.append("..")
# from sklearn.covariance import graphical_lasso
import torch
import numpy as np
import signature

# 继承 unittest.TestCase 就创建了一个测试样例。
class TestPipeline(unittest.TestCase):
    # def test_eig(self):
    #     A = torch.randn(2, 2, dtype=torch.complex128)
    #     print()
    #     print(torch.linalg.eig(A))
    #     A = A.numpy()
    #     print(numpy.linalg.eig(A))
    # def test_bug(self):
    #     sort_eigen = 
    def test_dataset(self):
        data = dgl.data.LegacyTUDataset('ENZYMES')
        print("original dataset length : ", len(data.graph_lists))
        g,label = data[123]
        gf = signature.GetNodeAttrMat(g).shape
        gwks = signature.WKS(g,signature.get_sample4WKS,40)
        ghks = signature.HKS(g,signature.get_sample4WKS,800)
        # print(gwks)
        print()
        print(ghks[:,ghks.shape[1]-40:])
        print(g.number_of_nodes())
        # print(signature.GetNodeAttrMat(g))

        print()


        


      

if __name__ == "__main__":
    unittest.main()