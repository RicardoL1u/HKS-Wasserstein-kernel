import dgl.data
import unittest
# from sklearn.covariance import graphical_lasso
import torch
import numpy as np

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
        data = dgl.data.LegacyTUDataset('PTC_FM')
        distr = np.zeros((1000))
        # data = dgl.data.MUTAGDataset() 
        for i in range(len(data)):
            g,label = data[i]
            node_num = len(g.out_degrees())
            if node_num >999:
                node_num = 999
            distr[node_num]+=1
        print(distr)


        


      

if __name__ == "__main__":
    unittest.main()