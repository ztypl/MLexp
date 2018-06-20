# coding : utf-8
# create by ztypl on 2018/5/22


from typing import Any
import sys
from multiprocessing import cpu_count
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score

from .Graph import Graph
from data_utils.data_utils_cora import *


class Node2VecClassifier:
    def __init__(self):
        self.graph = Graph()            # type: Graph
        self.embedding_model = None     # type: Word2Vec
        self.classifier = None          # type: Any
        self.workers = cpu_count()      # type: int
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None
        self.X = None
        self.y = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_network(self, obj, way="obj"):
        # type: (Any, str) -> None
        if way == "obj":
            self.graph.load_from_arr(obj)
        elif way == "file":
            self.graph.load_from_numpy(obj)
        else:
            raise ValueError("Unsupported value for parameter way.")

    def embedding(self, w, d, gamma, t, workers=None):
        # type: (int, int, int, int|None) -> None
        if not workers:
            workers = self.workers
        self.embedding_model = self.graph.walk(w, d, gamma, t, workers)

    def embedding_single_worker(self, w, d, gamma, t):
        # type: (int, int, int) -> None
        self.embedding_model = self.graph.walk(w, d, gamma, t, 1)


    def get_data(self, path):
        self.X, A, self.y = load_data(path)
        self.y_train, self.y_val, self.y_test, \
            self.idx_train, self.idx_val, self.idx_test = get_splits(self.y)
        self.load_network(np.array(list(A.todok().keys())), "obj")

    def validate_grid_search(self, gamma=5, w=10, t=15, d=25):
        ps = 2. ** np.arange(-4, 5)
        qs = 2. ** np.arange(-4, 5)

        best_score = -1.0
        best_params = {}

        for p in ps:
            for q in qs:
                self.graph.p = p
                self.graph.q = q

                self.embedding(w, d, gamma, t)
                X_train = self.embedding_model[map(str, self.idx_train)]
                y_train = self.y_train[self.idx_train]
                X_val = self.embedding_model[map(str, self.idx_val)]
                y_val = self.y_val[self.idx_val]

                cls_val = LogisticRegression(C=1e-2)
                cls_val.fit(X_train, y_train)
                y_pred = cls_val.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                if score > best_score:
                    best_score = score
                    best_params.update({'p':p, 'q':q})
                    print(best_score)
                    print(best_params)
                    sys.stdout.flush()
        print("---------")
        print(best_score)
        print(best_params)

    def validate(self):
        p = 1.0
        q = 1.0
        params = {}
        params.update({'p': p, 'q': q})

        gamma = 5
        w = 17
        t = 17
        d = 18

        best_score = -1.0
        best_params = {}

        self.graph.p = p
        self.graph.q = q

        self.embedding(w, d, gamma, t)
        X_train = self.embedding_model[map(str, self.idx_train)]
        y_train = self.y_train[self.idx_train]
        X_val = self.embedding_model[map(str, self.idx_val)]
        y_val = self.y_val[self.idx_val]

        cls_val = LogisticRegression(C=1e-2)
        cls_val.fit(X_train, y_train)
        y_pred = cls_val.predict(X_val)
        score = accuracy_score(y_val, y_pred)

        print("---------")
        print(score)
        print(params)


    def test(self):
        gamma = 5
        w = 17
        t = 17
        d = 18

        self.graph.p = 1.0
        self.graph.q = 0.0625

        self.embedding(w, d, gamma, t)
        X_train = self.embedding_model[map(str, self.idx_train)]
        y_train = self.y_train[self.idx_train]
        X_test = self.embedding_model[map(str, self.idx_test)]
        y_test = self.y_test[self.idx_test]
        cls_val = LogisticRegression(C=1e-2)
        cls_val.fit(X_train, y_train)
        y_pred = cls_val.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        return score

    # 0.81
    # {'gamma': 5, 'w': 17, 't': 17, 'd': 18}

    # 0.7933333333333333
    # {'t': 22, 'd': 20, 'w': 18, 'gamma': 2}