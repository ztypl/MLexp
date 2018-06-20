# coding : utf-8
# create by ztypl on 2018/5/7

import numpy as np
from gensim.models import Word2Vec


class Graph:
    def __init__(self, filename=None):
        self.node_list = None
        self.head = None
        self.adjTable = None
        if filename:
            self.load_from_numpy(filename)

    def load_from_numpy(self, file):
        # type: (str) -> None

        arr = np.load(file)
        self.load_from_arr(arr)

    def load_from_arr(self, arr):
        # type: (np.array) -> None

        self.node_list = np.unique(arr)
        self.head = {}
        self.adjTable = []
        for node in self.node_list:
            self.head[node] = -1
        for a, b in arr:
            self._add_edge(a, b)

    def _add_edge(self, a, b):
        # type: (int, int) -> None

        self.adjTable.append((a, b, self.head[a]))
        self.head[a] = len(self.adjTable) - 1

    def get_adjacent_vertexs(self, u):
        # type: (int) -> list[int]

        adj_v = []
        next_v_loc = self.head[u]
        while next_v_loc >= 0:
            edge = self.adjTable[next_v_loc]
            adj_v.append(edge[1])
            next_v_loc = edge[2]
        return adj_v

    def random_walk(self, u, t):
        # type: (int, int) -> list[str]

        path = [str(u)]
        cur_node = u
        for i in range(t - 1):
            adj_vertexs = self.get_adjacent_vertexs(cur_node)
            if len(adj_vertexs) == 0:
                break
            next_node = np.random.choice(adj_vertexs)
            path.append(str(next_node))
            cur_node = next_node
        return path

    def deep_walk(self, w, d, gamma, t, workers):
        # type: (int, int, int, int, int) -> Word2Vec

        walks = []
        v_set = self.node_list.copy()
        for i in range(gamma):
            np.random.shuffle(v_set)
            for v in v_set:
                walks.append(self.random_walk(v, t))

        model = Word2Vec(walks, size=d, window=w, min_count=0, hs=1, sg=1, workers=workers)
        return model
