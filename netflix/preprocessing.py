# coding : utf-8
# create by ztypl on 2018/5/30

import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

users = np.loadtxt("Project2-data/users.txt", dtype='int')
enc_user = LabelEncoder()
enc_user.fit(users)

train_data = np.loadtxt("Project2-data/netflix_train.txt", dtype='str', delimiter=' ')[:,:-1].astype("int")
A = sparse.coo_matrix((train_data[:,2],
                       (enc_user.transform(train_data[:,0]), train_data[:,1] - 1)))
sparse.save_npz("data/train.npz", A.tocsc())

test_data = np.loadtxt("Project2-data/netflix_test.txt", dtype='str', delimiter=' ')[:,:-1].astype("int")
B = sparse.coo_matrix((test_data[:,2],
                       (enc_user.transform(test_data[:,0]), test_data[:,1] - 1)))
sparse.save_npz("data/test.npz", B.tocsc())