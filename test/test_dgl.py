import dgl.data
import unittest
import sys
sys.path.append("..")
# from sklearn.covariance import graphical_lasso
import torch
import numpy as np
import signature
import logging
# 继承 unittest.TestCase 就创建了一个测试样例。
class TestDGL(unittest.TestCase):
    def test_dataset(self):
        self.dataset_analysis("MUTAG")
        self.dataset_analysis("PTC_MR")
        # self.dataset_analysis("NCI1")
        self.dataset_analysis("PROTEINS")
        self.dataset_analysis("DD")
        self.dataset_analysis("ENZYMES")
    def dataset_analysis(self,dataset:str):
        logging.basicConfig(format='%(asctime)s: %(message)s',datefmt='%Y/%m/%d %I:%M:%S',level=logging.DEBUG)
        logging.debug('\n=====================================================\n')
        data = dgl.data.LegacyTUDataset(dataset)
        msg = f"{dataset} dataset length : ", len(data.graph_lists)
        logging.info(msg)
        graphs, y = zip(*[graph for graph in data])
        graphs = list(graphs)
        y = np.array([unit.item() for unit in y])
        logging.info(np.unique(y,return_counts=True))

        graph_size_list = []
        for g in graphs:
            graph_size_list.append(g.number_of_nodes())
        graph_size_list = np.array(graph_size_list)
        logging.info(np.unique(graph_size_list,return_counts=True))
        logging.info(np.mean(graph_size_list))
        logging.info(np.std(graph_size_list))
        logging.info(np.sum(graph_size_list))
        logging.info(graphs[0])
        logging.info(graphs[0].ndata['feat'].shape)
        logging.info(np.unique(graphs[0].ndata['feat'].numpy(),return_counts=True))


      

if __name__ == "__main__":
    unittest.main()