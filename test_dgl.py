import dgl.data
import unittest
# from sklearn.covariance import graphical_lasso
import torch
import numpy

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
        # data = dgl.data.MUTAGDataset() 
        print(data)
        g,label = data[132]
        print(g)
        print(label)
        print(g.out_degrees())
        print(g.in_degrees())
        # print((g.ndata['attr']).numpy())

        # print(g)
        # print(g.adj())

        # print(g.ndata["label"])
        
        # print(g.degree())

        # graphs, labels = zip(*[data[i] for i in range(4)])
        # graphs = list(graphs)
        # labels = list(labels)
        # print(graphs)
        # print(labels)
        # print(type(graphs))
        # print(type(labels))
        # for i in range(128,129):
        #     print(data[i])
        #     g, label = data[i]
            
        #     print(g.ndata['attr'])
        #     print(label)
        #     g, label = data[i+150]
        #     print(g)
        #     print(label)


      

if __name__ == "__main__":
    unittest.main()