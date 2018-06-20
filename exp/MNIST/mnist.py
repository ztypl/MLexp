# https://gist.github.com/akesling/5358964

import os
import struct
import numpy as np
import matplotlib.pyplot as plt

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "./data/"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (img[idx], lbl[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=plt.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()
    
def ready_data(path="data/"):
    with open(os.path.join(path, 'train-images-idx3-ubyte'), 'rb') as f_train_img:
        magic, num, rows, cols = struct.unpack(">IIII", f_train_img.read(16))
        X_train = np.fromfile(f_train_img, dtype=np.uint8).reshape(num, -1)
    with open(os.path.join(path, 'train-labels-idx1-ubyte'), 'rb') as f_train_lbl:
        magic, num = struct.unpack(">II", f_train_lbl.read(8))
        y_train = np.fromfile(f_train_lbl, dtype=np.int8)
    with open(os.path.join(path, 't10k-images-idx3-ubyte'), 'rb') as f_test_img:
        magic, num, rows, cols = struct.unpack(">IIII", f_test_img.read(16))
        X_test = np.fromfile(f_test_img, dtype=np.uint8).reshape(num, -1)
    with open(os.path.join(path, 't10k-labels-idx1-ubyte'), 'rb') as f_test_lbl:
        magic, num = struct.unpack(">II", f_test_lbl.read(8))
        y_test = np.fromfile(f_test_lbl, dtype=np.int8)
    return X_train, y_train, X_test, y_test
    
    