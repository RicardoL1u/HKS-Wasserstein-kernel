import unittest

from numpy.testing._private.utils import assert_equal
import utilities
import numpy as np
import igraph as ig
import HKS
import wass_dis
# 继承 unittest.TestCase 就创建了一个测试样例。
class TestStringMethods(unittest.TestCase):

    #这些方法的命名都以 test 开头。 这个命名约定告诉测试运行者类的哪些方法表示测试。
    # def test_direct_file_list(self):
    #     graph_filenames = utilities.retrieve_graph_filenames("./data/MUTAG")
    #     graphs = [ig.read(filename) for filename in graph_filenames]
    #     test_graph = graphs[0]
    #     # print(test_graph.get_edgelist()[0:10])
    #     test_graph.vs[3]["foo"] =  "bar"
    #     # print(test_graph.es['weight'])
    #     # print(test_graph.vs['label'])
    #     # print(test_graph.vs["id"])
    #     # print(ig.summary(test_graph))
        
    #     graph_A = test_graph.get_adjacency()
    #     adj_matrix = np.array(graph_A.data)
    #     degree_vector = np.array(test_graph.degree())
    #     degree_matrix  = np.diagflat(degree_vector)
    #     graph_laplacian = degree_matrix - adj_matrix
    #     eigenvalues,eigenvectors = np.linalg.eig(graph_laplacian)
    #     eigenvalues = np.sort(eigenvalues)

    #     print(eigenvalues)
    #     print(eigenvalues[1],eigenvalues[-1])
    #     T = HKS.get_random_samples(eigenvalues[1],eigenvalues[-1])
    #     # embedding = [np.sum(np.exp(-eigenvalues*t)*eigenvectors[0]*eigenvectors[0]) for t in T]
    #     print(T)

    def test_what(self):
        embedding = np.array([t for t in [1,2,3]/
        np.sum(np.array([1,2,3]))])
        print(embedding)

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()