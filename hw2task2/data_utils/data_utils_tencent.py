# coding : utf-8
# create by ztypl on 2018/5/15

import numpy as np
from scipy.sparse import load_npz

def load_train_data(path="../data/tencent/"):
    edges_train = np.load(path + "train_edges.npy")
    return edges_train


def load_test_data(path="../data/tencent/"):
    edges_test = np.load(path + "test_edges.npy")
    edges_test_false = np.load(path + "test_edges_false.npy")
    return edges_test, edges_test_false


def load_matrix(path="../data/tencent/"):
    matrix = load_npz(path + "adj_train.npz")
    matrix = matrix + matrix.T.multiply(matrix.T > matrix) - matrix.multiply(matrix.T > matrix)
    return matrix
