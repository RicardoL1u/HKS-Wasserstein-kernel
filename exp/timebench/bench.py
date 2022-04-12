import grakel
import random

def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]

    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0

    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]

    return matrix

def test0(adj_mat0,adj_mat1,kernel):
    kernel.fit_transform([grakel.Graph(adj_mat0)])
    kernel.transform([grakel.Graph(adj_mat1)])


if __name__ == '__main__':
    import timeit
    temp0 = random_adjacency_matrix(5)
    temp1 = random_adjacency_matrix(5)
    SP = grakel.kernels.ShortestPath(normalize=True,with_labels=False)
    # For Python>=3.5 one can also write:
    print(timeit.timeit("test(temp0,temp1,SP)", globals=locals(),number=10))