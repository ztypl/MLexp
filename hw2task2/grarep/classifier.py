# coding : utf-8
# create by ztypl on 2018/5/28

import sys
from data_utils.data_utils_cora import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .model import GraRep

class GraRepClassifier:
    def __init__(self):
        self.lbd = None
        self.d = None
        self.k = None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None
        self.A = None
        self.X = None
        self.y = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.M = None
        self.embedding_model = None

    def get_data(self, path):
        self.X, self.A, self.y = load_data(path)
        self.y_train, self.y_val, self.y_test, \
            self.idx_train, self.idx_val, self.idx_test = get_splits(self.y)

    def embedding(self, lbd, d, k):
        self.embedding_model = GraRep(lbd=lbd, d=d, k=k).embedding(self.A)

    def validate_grid_search(self):
        lbds = [1]
        ds = np.arange(16, 30, 2)
        ks = np.arange(4, 10)

        best_score = -1.0
        best_params = {}

        for lbd in lbds:
            for d in ds:
                for k in ks:
                    self.embedding(lbd, d, k)

                    X_train = self.embedding_model[self.idx_train, :]
                    y_train = self.y_train[self.idx_train]
                    X_val = self.embedding_model[self.idx_val, :]
                    y_val = self.y_val[self.idx_val]

                    cls_val = LogisticRegression(C=1e-2)
                    cls_val.fit(X_train, y_train)
                    y_pred = cls_val.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                    if score > best_score:
                        best_score = score
                        best_params.update({"lbd": lbd, "d": d, "k": k})
                        print(best_score)
                        print(best_params)
                        sys.stdout.flush()
        print("---------")
        print(best_score)
        print(best_params)
        return best_params

    def test(self):
        lbd = 1
        k = 9
        d= 16

        self.embedding(lbd, d, k)

        X_train = self.embedding_model[self.idx_train, :]
        y_train = self.y_train[self.idx_train]
        X_test = self.embedding_model[self.idx_test, :]
        y_test = self.y_test[self.idx_test]

        cls_val = LogisticRegression(C=1e-2)
        cls_val.fit(X_train, y_train)
        y_pred = cls_val.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        return score