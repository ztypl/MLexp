# coding : utf-8
# create by ztypl on 2018/5/30

import time
import sys
import numpy as np
from scipy import sparse
from sklearn.metrics import mean_squared_error


def cal_time(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print("%s training time: %s secs." % (func.__name__, t2 - t1))
        return result
    return wrapper


def load_data(path='data/'):
    train = sparse.load_npz(path + "train.npz")
    test = sparse.load_npz(path + "test.npz")
    test_X = np.c_[test.tocoo().row, test.tocoo().col]
    test_y = test.tocoo().data
    return train.toarray(), test_X, test_y


# @cal_time
# def trainCF(train, test_X):
#     pred_y = []
#     i = 0
#
#     for user, movie in test_X:
#         i += 1
#         if i % 1000 == 0:
#             print("%d done." % i)
#             sys.stdout.flush()
#         this = train[user,:].T
#         others = train[train[:,movie].toarray().nonzero()[0]]
#         sims = (others @ this).toarray().T / sparse.linalg.norm(this) / sparse.linalg.norm(others, axis=1)
#         score = others[:,movie].toarray().reshape(-1)
#         s = np.sum(sims * score) / np.sum(sims)
#         pred_y.append(s)
#     return pred_y


@cal_time
def get_similarity(M):
    module = np.linalg.norm(M, axis=1).reshape(-1,1)
    return M @ M.T / module.T / module


def rmse(test_y, pred_y):
    return np.sqrt(mean_squared_error(test_y, pred_y))


if __name__ == '__main__':
    train, test_X, test_y = load_data()
    SIM = get_similarity(train)
    np.save("data/sims.npy", SIM)