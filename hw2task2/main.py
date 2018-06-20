# coding : utf-8
# create by ztypl on 2018/5/7

import tensorflow as tf

from node2vec.classifier import *
from deepwalk.classifier import *
from line.model import LineLinkPrediction
from grarep.classifier import *

print("deep walk")

c0 = DeepWalkClassifier()
c0.get_data("data/cora/")
print("Accuracy = ", c0.test())

print("####################")
print("Line")

model = LineLinkPrediction("data/tencent/", num_epoch=10, embedding_dim=20, learning_rate=1e-3)
M = model.train()
x = model.test(M)
print("AUC = %f" % x)

print("####################")
print("node2vec")

c = Node2VecClassifier()
c.get_data("data/cora/")
# c.validate_grid_search()
print("Accuracy = ", c.test())

print("####################")
print("GraRep")

c = GraRepClassifier()
c.get_data("data/cora/")
print("Accuracy = ", c.test())