# coding : utf-8
# create by ztypl on 2018/6/10

from itertools import combinations
import numpy as np
import pandas as pd
from Graph import Graph

class DataProcessor:
    def __init__(self, filename):
        self.g = Graph()
        self.key_authors = {}
        self.labels = {}
        df = pd.read_csv(filename, na_filter=["", "null"])
        df = df.applymap(lambda x: x.lower().replace("-", "") if type(x)==str else x)
        for _, row in df.iterrows():
            names = row['name'].split(",")
            key_author = row['key-author']
            names.remove(key_author)
            if key_author not in self.key_authors:
                self.key_authors[key_author] = []
                self.labels[key_author] = []
            self.key_authors[key_author].append(row['id'])
            self.labels[key_author].append(row['label'])
            key_author = "%s_%s" % (key_author, row['id'])
            names.append(key_author)
            if len(names) > 1:
                for a, b in combinations(names, 2):
                    self.g.add_edge(a, b)
            else:
                self.g.add_node(names[0])

    def get_graph(self):
        return self.g

    def get_matrix(self):
        n = len(self.g.node_list)
        mat = np.zeros((n, n), dtype='int')
        for i in range(n):
            vs = self.g.get_adjacent_vertexs(i)
            for v in vs:
                mat[i, v] += 1
        return mat
