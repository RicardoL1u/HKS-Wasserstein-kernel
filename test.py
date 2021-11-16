import unittest
import utilities
import igraph as ig

# 继承 unittest.TestCase 就创建了一个测试样例。
class TestStringMethods(unittest.TestCase):

    #这些方法的命名都以 test 开头。 这个命名约定告诉测试运行者类的哪些方法表示测试。
    def test_direct_file_list(self):
        graph_filenames = utilities.retrieve_graph_filenames("./data/MUTAG")
        graphs = [ig.read(filename) for filename in graph_filenames]
        test_graph = graphs[0]
        print(test_graph.get_edgelist()[0:10])
        test_graph.vs[3]["foo"] =  "bar"
        print(test_graph.es['weight'])
        print(test_graph.vs['label'])
        print(test_graph.vs["id"])
        print(ig.summary(test_graph))
        
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()