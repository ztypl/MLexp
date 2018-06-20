# coding : utf-8
# create by ztypl on 2018/6/10

import copy
import numpy as np
from gensim.models import Word2Vec


class Graph:
    def __init__(self):
        self.node_list = []
        self.node_dict = {}
        self.head = []
        self.adjTable = []

    def add_node(self, name):
        # type: (str) -> int
        if name not in self.node_dict:
            self.node_list.append(name)
            self.node_dict[name] = len(self.node_list) - 1
            self.head.append(-1)
        return self.node_dict[name]

    def add_edge(self, a, b):
        # type: (int, int) -> None

        a_id = self.add_node(a)
        b_id = self.add_node(b)
        self.adjTable.append((a_id, b_id, self.head[a_id]))
        self.head[a_id] = len(self.adjTable) - 1
        self.adjTable.append((b_id, a_id, self.head[b_id]))
        self.head[b_id] = len(self.adjTable) - 1

    def get_adjacent_vertexs(self, u):
        # type: (int) -> list[int]

        adj_v = []
        next_v_loc = self.head[u]
        while next_v_loc >= 0:
            edge = self.adjTable[next_v_loc]
            adj_v.append(edge[1])
            next_v_loc = edge[2]
        return adj_v

    def have_edge(self, u, v):
        # type: (int, int) -> bool

        next_v_loc = self.head[u]
        while next_v_loc >= 0:
            edge = self.adjTable[next_v_loc]
            if v == edge[1]:
                return True
            next_v_loc = edge[2]
        return False

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
        v_set = copy.deepcopy(self.node_list)
        for i in range(gamma):
            np.random.shuffle(v_set)
            for v in v_set:
                walks.append(self.random_walk(self.node_dict[v], t))

        model = Word2Vec(walks, size=d, window=w, min_count=0, hs=1, sg=1, workers=workers)
        return model

    def node2vec_walk(self, u, t, p, q):
        # type: (int, int) -> list[str]

        path = [str(u)]
        for i in range(t - 1):
            cur_node = int(path[-1])
            adj_vertexes = self.get_adjacent_vertexs(cur_node)
            if len(adj_vertexes) == 0:
                break
            if len(path) == 1:
                path.append(str(np.random.choice(adj_vertexes)))
            else:
                prev_node = int(path[-2])
                w = np.zeros(len(adj_vertexes), dtype='float')
                for i, v in enumerate(adj_vertexes):
                    if v == prev_node:
                        w[i] = 1./ p
                    elif self.have_edge(prev_node, v):
                        w[i] = 1.
                    else:
                        w[i] = 1./ q
                w = w / w.sum()
                next_node = np.random.choice(adj_vertexes, p=w)
                path.append(str(next_node))
        return path

    def node2vec(self, w, d, gamma, t, workers, p, q):
        # type: (int, int, int, int, int) -> Word2Vec

        walks = []
        v_set = copy.deepcopy(self.node_list)
        for i in range(gamma):
            np.random.shuffle(v_set)
            for v in v_set:
                walks.append(self.node2vec_walk(self.node_dict[v], t, p, q))

        model = Word2Vec(walks, size=d, window=w, min_count=0, hs=1, sg=1, workers=workers)
        return model