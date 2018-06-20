# coding : utf-8
# create by ztypl on 2018/5/23

import sys
from scipy import sparse
from sklearn.metrics import roc_auc_score
from data_utils.data_utils_cora import *

class GraRep:
    def __init__(self, **kwargs):
        self.lbd = kwargs['lbd']
        self.d = kwargs['d']
        self.k = kwargs['k']
        self.N = None

    def embedding_sparse(self, S):
        self.N = S.shape[0]
        Di = 1. / np.asarray(S.sum(axis=0)).reshape(-1)
        A = sparse.diags(Di) @ S
        Ak = sparse.identity(self.N)
        W = np.empty((self.N, 0))

        for k in range(1, self.k+1):
            Ak = Ak @ A
            Gamma_k = Ak.sum(axis=0)
            Xk = Ak.multiply(self.N / self.lbd / Gamma_k)
            Xk.data = np.log(Xk.data)
            Xk.data[Xk.data < 0.0] = 0.0
            Xk.data[(Xk.data == np.inf) | (Xk.data == np.nan)] = 0.0
            Uk, sk, _ = sparse.linalg.svds(Xk, k=self.d)
            Wk = Uk @ np.diag(np.sqrt(sk))
            W = np.hstack((W, Wk))
            sys.stdout.flush()
        return W

    def embedding(self, S):
        self.N = S.shape[0]
        Di = 1. / np.asarray(S.sum(axis=0)).reshape(-1)
        A = np.diag(Di) @ S
        Ak = np.identity(self.N)
        W = np.empty((self.N, 0))

        for k in range(1, self.k+1):
            Ak = Ak @ A
            Gamma_k = Ak.sum(axis=0)
            Xk = np.log(Ak / Gamma_k) - np.log(self.lbd / self.N)
            Xk[Xk < 0.0] = 0.0
            Xk[(Xk == np.inf) | (Xk == np.nan)] = 0.0
            uk, sk, _ = np.linalg.svd(Xk)
            Uk = uk[:, :self.d]
            Sk = sk[:self.d]
            Wk = Uk @ np.diag(np.sqrt(Sk))
            W = np.hstack((W, Wk))
        return W

    # def test(self, M):
    #     edges_test, edges_test_false = load_test_data(self.path)
    #     labels = np.concatenate([np.ones(edges_test.shape[0], dtype='int32'),
    #                              np.zeros(edges_test_false.shape[0], dtype='int32')])
    #     sim = []
    #     for u, v in edges_test:
    #         vector_a = M[u, :]
    #         vector_b = M[v, :]
    #         sim.append(np.dot(vector_a, vector_b) /
    #                    (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
    #     for u, v in edges_test_false:
    #         vector_a = M[u, :]
    #         vector_b = M[v, :]
    #         sim.append(np.dot(vector_a, vector_b) /
    #                    (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
    #     auc_score = roc_auc_score(labels, sim)
    #     return auc_score


