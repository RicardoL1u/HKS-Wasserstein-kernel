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
    # def test_eig(self):
    #     A = torch.randn(2, 2, dtype=torch.complex128)
    #     print()
    #     print(torch.linalg.eig(A))
    #     A = A.numpy()
    #     print(numpy.linalg.eig(A))
    # def test_bug(self):
    #     sort_eigen = 
    def test_dataset(self):
        self.dataset_analysis("MUTAG")
        self.dataset_analysis("PTC_MR")
        self.dataset_analysis("NCI1")
        self.dataset_analysis("PROTEINS")
        self.dataset_analysis("DD")
        self.dataset_analysis("ENZYMES")
    def dataset_analysis(self,dataset:str):
        print('\n=====================================================\n')
        data = dgl.data.LegacyTUDataset(dataset)
        print(f"{dataset} dataset length : ", len(data.graph_lists))
        graphs, y = zip(*[graph for graph in data])
        graphs = list(graphs)
        y = np.array([unit.item() for unit in y])
        print(np.unique(y,return_counts=True))

        graph_size_list = []
        for g in graphs:
            graph_size_list.append(g.number_of_nodes())
        graph_size_list = np.array(graph_size_list)
        print(np.unique(graph_size_list,return_counts=True))
        print(np.mean(graph_size_list))
        print(np.std(graph_size_list))
        print(np.sum(graph_size_list))
        print(graphs[0])
        print(graphs[0].ndata['feat'].shape)
        print(np.unique(graphs[0].ndata['feat'].numpy(),return_counts=True))


      

if __name__ == "__main__":
    unittest.main()