import unittest

from numpy.testing._private.utils import assert_equal
import utilities
import numpy as np
import igraph as ig
import HKS
import wass_dis
from sklearn.model_selection import StratifiedKFold

import wass_dis
# 继承 unittest.TestCase 就创建了一个测试样例。
class TestPipeline(unittest.TestCase):

    #这些方法的命名都以 test 开头。 这个命名约定告诉测试运行者类的哪些方法表示测试。
    # def test_pipeline(self):
    #     graph_filenames = utilities.retrieve_graph_filenames("./data/MUTAG")
    #     graphs = [ig.read(filename) for filename in graph_filenames]
    #     wasserstein_distances = wass_dis.pairwise_wasserstein_distance(graphs)

    #     gammas = 0.001
    #     M = wasserstein_distances
    #     g = gammas
    #     K = np.exp(-g*M)
    #     print(type(K))
    #     print(K)
    #     kernel_matrices = []
    #     kernel_params = []
    #     kernel_matrices.append(K)
    #     kernel_params.append((g))

    #     # label_file = os.path.join(data_path, 'Labels.txt')
    
    #     y = np.array(utilities.read_labels("./data/MUTAG/Labels.txt"))
    #     print(y)
    #     cv = StratifiedKFold(n_splits=10,shuffle=True)

    #     for train_index, test_index in cv.split(kernel_matrices[0], y):
    #         K_train = [K[train_index][:, train_index] for K in kernel_matrices]
    #         K_test  = [K[test_index][:, train_index] for K in kernel_matrices]
    #         y_train, y_test = y[train_index], y[test_index]


    def test_pred(self):
        y = np.array(utilities.read_labels("./data/MUTAG/Labels.txt"))
        print(y)

    def test_if(self):
        if True:
            s = "string"
        print(s+"1")

if __name__ == "__main__":
    unittest.main()