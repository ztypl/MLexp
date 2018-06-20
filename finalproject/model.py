# coding : utf-8
# create by ztypl on 2018/6/11

import sys
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics.pairwise import cosine_distances
from DataProcessor import DataProcessor
from GraRep import GraRep

class Model:
    def __init__(self, filename, embedding_method='deepwalk', **kwargs):
        self.dp = DataProcessor(filename)
        self.workers = cpu_count()
        self.embedding_model = None
        self.embedding_method = embedding_method
        print("Init over.")
        sys.stdout.flush()
        if embedding_method == 'deepwalk':
            self.deepwalk(**kwargs)
        elif embedding_method == 'grarep':
            self.grarep(**kwargs)
        elif embedding_method == "node2vec":
            self.node2vec(**kwargs)
        else:
            raise TypeError("Unsupport type %s" % embedding_method)

    def deepwalk(self, gamma=5, w=10, t=15, d=16):
        self.embedding_model = self.dp.g.deep_walk(w, d, gamma, t, self.workers)

    def grarep(self, lbd=1, k=3, d=16):
        mat = self.dp.get_matrix()
        self.embedding_model = GraRep(lbd=lbd, k=k, d=d).embedding_sparse(mat)

    def node2vec(self, gamma=5, w=10, t=15, d=16, p=1.0, q=0.0625):
        self.embedding_model = self.dp.g.node2vec(w, d, gamma, t, self.workers, p, q)

    def get_vec(self, author_name, i):
        if self.embedding_method == "deepwalk" or self.embedding_method == "node2vec":
            return self.embedding_model[str(self.dp.g.node_dict["%s_%s" % (author_name, i)])]
        else:
            return self.embedding_model[self.dp.g.node_dict["%s_%s" % (author_name, i)]]

    def cluster(self, dampling=0.5, affinity='euclidean'):
        print("Embedding over.")
        sys.stdout.flush()
        authors = self.dp.key_authors
        labels = self.dp.labels
        a_total = 0
        b_total = 0
        c_total = 0
        d_total = 0
        df_out = pd.DataFrame(columns=['name', 'precision', 'recall', 'f1', 'real-labels', 'pred-labels'])
        for author_name in authors:
            sys.stdout.flush()
            ids = authors[author_name]
            vecs = []
            y_true = labels[author_name]

            for i in ids:
                vecs.append(self.get_vec(author_name, i))

            if affinity == 'euclidean':
                cluster_model = AffinityPropagation(damping=dampling)
                y_pred = cluster_model.fit(vecs).labels_.reshape(-1)
            elif affinity == "cosine":
                sim = cosine_distances(vecs)
                cluster_model = AffinityPropagation(affinity='precomputed', damping=dampling)
                y_pred = cluster_model.fit(sim).labels_.reshape(-1)
            else:
                raise TypeError("Unsupported type %s" % affinity)
            if all(np.isnan(y_pred)):
                y_pred = np.arange(len(y_pred))
            a, b, c, d = self.pairwise_metrics(y_pred, y_true)
            a_total += a
            b_total += b
            c_total += c
            d_total += d
            df_out = df_out.append({
                'name': author_name,
                'precision': a / b,
                'recall': c / d,
                'f1': 2 * (a / b) * (c / d) / ((a / b) + (c / d)),
                'real-labels': len(np.unique(y_true)),
                'pred-labels': len(np.unique(y_pred))
            }, ignore_index=True)
        precision = a_total / b_total
        recall = c_total / d_total
        f1_score = 2 * precision * recall / (precision + recall)
        print("Precision = %.3f, Recall = %.3f, F1 = %.3f" % (precision, recall, f1_score))
        df_out.to_csv("out/out.csv", index=False)

    @staticmethod
    def count_group_pred(df):
        pred_group = df.groupby('y_pred')
        count = pred_group.count()
        b = (count * (count - 1) / 2).sum()['y_true']
        return b

    @staticmethod
    def count_group_true(df):
        pred_group = df.groupby('y_true')
        count = pred_group.count()
        b = (count * (count - 1) / 2).sum()['y_pred']
        return b

    def pairwise_metrics(self, y_pred, y_true):
        df = pd.DataFrame(list(zip(y_pred, y_true)), columns=['y_pred', 'y_true'])
        pred_group = df.groupby('y_pred')
        count = pred_group.count()
        b = (count * (count - 1) / 2).sum()['y_true']
        a = pred_group.agg(self.count_group_true).sum()['y_true']

        true_group = df.groupby('y_true')
        count = true_group.count()
        d = (count * (count - 1) / 2).sum()['y_pred']
        c = true_group.agg(self.count_group_pred).sum()['y_pred']
        return a, b, c, d

