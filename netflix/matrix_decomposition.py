# coding : utf-8
# create by ztypl on 2018/5/30

import sys
import numpy as np
from scipy import sparse
from sklearn.metrics import mean_squared_error


def load_data(path='data/'):
    train = sparse.load_npz(path + "train.npz")
    test = sparse.load_npz(path + "test.npz")
    test_X = np.c_[test.tocoo().row, test.tocoo().col]
    test_y = test.tocoo().data
    return train, test_X, test_y


def J(X, A, U, V, lbd):
    return 0.5 * np.linalg.norm(A * (X - U @ V.T), ord='fro') ** 2 \
           + lbd * np.linalg.norm(U) ** 2 \
           + lbd * np.linalg.norm(V) ** 2


def rmse(U, V, test_X, test_y):
    X_pred = U @ V.T
    pred_y = X_pred[test_X[:,0].reshape(-1), test_X[:,1].reshape(-1)]
    return np.sqrt(mean_squared_error(test_y, pred_y))


def decomposition(X, test_X, test_y, k=50, lbd=1e-2, learning_rate=1e-2):
    X = X.toarray()
    A = X.astype('bool').astype('int')
    U = np.random.uniform(-1e-2, 1e-2, (X.shape[0], k))
    V = np.random.uniform(-1e-2, 1e-2, (X.shape[1], k))
    delta_loss = np.inf
    old_loss = np.inf
    i = 0
    while delta_loss >= 1e-2:
        dU = (A * (U @ V.T - X)) @ V + 2 * lbd * U
        dV = (A * (U @ V.T - X)) @ U + 2 * lbd * V
        new_U = U - learning_rate * dU
        new_V = V - learning_rate * dV
        loss = J(X, A, new_U, new_V, lbd)
        new_delta_loss = old_loss - loss
        if new_delta_loss < 0:
            learning_rate *= 0.2
            continue

        U = new_U
        V = new_V
        old_loss = loss
        delta_loss = new_delta_loss

        i += 1
        if 1 % 1 == 0:
            print("#%d: loss=%f, rmse=%f" % (i, loss, rmse(U, V, test_X, test_y)))
            sys.stdout.flush()
    return U, V


