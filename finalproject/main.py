# coding : utf-8
# create by ztypl on 2018/6/11

import numpy as np
from model import Model

for p in 2. ** np.arange(-4, 5):
    for q in 2. ** np.arange(-4, 5):
        print("p=%f, q=%f" % (p, q))
        m = Model("data/na_truth_value.csv", embedding_method='node2vec', p=p, q=q)
        m.cluster(dampling=0.95, affinity='euclidean')
# m.cluster(dampling=0.95,affinity='cosine')

