"""
=============================================================
Graph classification on MUTAG using the shortest path kernel.
=============================================================

Script makes use of :class:`grakel.ShortestPath`
"""
from __future__ import print_function
print(__doc__)
import argparse

import numpy as np
import sklearn.model_selection

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Provide the dataset name (MUTAG or PTC_FM)',
                            choices=['MUTAG', 'PTC_FM'])
    args = parser.parse_args()
      
    # Loads the MUTAG dataset
    MUTAG = fetch_dataset(args.dataset, verbose=False)
    G, y = MUTAG.data, MUTAG.target

    gk = ShortestPath(normalize=True)
    cv = sklearn.model_selection.StratifiedKFold(n_splits=10,shuffle=True)

    accuracy_scores = []
    for train_index, test_index in cv.split(G, y):
        G_train = [G[i] for i in train_index]
        G_test  = [G[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]
        K_train = gk.fit_transform(G_train)
        K_test = gk.transform(G_test)
        # Uses the SVM classifier to perform classification
        clf = SVC(kernel="precomputed")
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        accuracy_scores.append(sklearn.metrics.accuracy_score(y_test, y_pred))


    # Computes and prints the classification accuracy
    print('Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %'.format(
                        np.mean(accuracy_scores) * 100,  
                        np.std(accuracy_scores) * 100))

if __name__ == "__main__":
    main()
