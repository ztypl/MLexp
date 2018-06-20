# coding : utf-8
# create by ztypl on 2018/5/15

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from data_utils.data_utils_tencent import *


class Line:
    def __init__(self, **kwargs):
        self.num_epoch = kwargs['num_epoch']

        self.i = tf.placeholder(name='i', dtype=tf.int32)
        self.j = tf.placeholder(name='j', dtype=tf.int32)
        self.u = tf.get_variable('u', (kwargs['num_of_nodes'], kwargs['embedding_dim']),
                                 initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        self.ui = self.u[self.i, :]
        self.uj = self.u[self.j, :]
        self.prod = tf.reduce_sum(self.ui * self.uj)
        self.loss = tf.nn.softplus(-self.prod)
        self.optimizer = tf.train.AdamOptimizer()
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=kwargs['learning_rate'])
        self.train_op = self.optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()

    def train(self, edges_train, print_every=1000):
        i = 0
        total = edges_train.shape[0]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.num_epoch):
                for u, v in edges_train:
                    sess.run(self.train_op, {self.i: u, self.j: v})
                    i += 1
                    if i % print_every == 0:
                        loss = sess.run(self.loss, {self.i: u, self.j: v})
                        print("num(%d/%d), loss=%f" % (i, total, loss))
                print("epoch(%d/%d), loss=%f" % (epoch+1, self.num_epoch, loss))
            self.saver.save(sess, "model/line.ckpt")
            M = sess.run(self.u)
        return M

    def test(self, M, edges_test, edges_test_false):
        labels = np.concatenate([np.ones(edges_test.shape[0], dtype='int32'),
                                 np.zeros(edges_test_false.shape[0], dtype='int32')])
        sim = []
        for u, v in edges_test:
            vector_a = M[u, :]
            vector_b = M[v, :]
            sim.append(np.dot(vector_a, vector_b) /
                       (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
        for u, v in edges_test_false:
            vector_a = M[u, :]
            vector_b = M[v, :]
            sim.append(np.dot(vector_a, vector_b) /
                       (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
        auc_score = roc_auc_score(labels, sim)
        return auc_score


class LineLinkPrediction:
    def __init__(self, path, **kwargs):
        self.edges_train = load_train_data(path)
        self.edges_test, self.edges_test_false = load_test_data(path)

        self.model = Line(num_of_nodes=self.edges_train.max()+1, **kwargs)

    def train(self):
        return self.model.train(self.edges_train)

    def test(self, M):
        return self.model.test(M, self.edges_test, self.edges_test_false)
