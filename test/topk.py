import numpy as np
import time
from scipy.linalg import eigh as largest_eigh
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

np.set_printoptions(suppress=True)
np.random.seed(0)
N=5000
k=100
X = np.random.random((N,N)) - 0.5
X = np.dot(X, X.T) #create a symmetric matrix

# Benchmark the dense routine
start = time.time()
evals_large, evecs_large = largest_eigh(X, eigvals=(N-k,N-1))
elapsed = (time.time() - start)
print ("eigh elapsed time: ", elapsed)

# Benchmark the sparse routine
start = time.time()
evals_large_sparse, evecs_large_sparse = largest_eigsh(X, k, which='LM')
elapsed = (time.time() - start)
print ("eigsh elapsed time: ", elapsed)

# Benchmark the full eigenvalues
start = time.time()
eigenvalues, eigenvectors = np.linalg.eig(X)
elapsed = (time.time() - start)
print ("full elapsed time: ", elapsed)