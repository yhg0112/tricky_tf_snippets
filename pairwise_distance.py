from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def distance_euclidean(x1, x2):
    """
    get euclidean distance between 2 vectors.

    :param x1: D-dim vector
    :param x2: D-dim vector
    :return: a distance between x1 and x2, must be a scalar
    """
    res = tf.square(x1 - x2)
    res = tf.sqrt(tf.reduce_sum(res))

    return res


def distance_binary_xentropy(x1, x2):
    """
    get binary cross entropy between 2 vectors.
    L = -sum_(x_1*log(x_2) + (1 - x_1)*log(1 - x_2))

    :param x1: D-dim vector, each elements must be in [0, 1]
    :param x2: D-dim vector, each elements must be in [0, 1]
    :return: a cross entropy between x1 and x2 must be a scalar
    """
    res = x1*tf.log(1e-10 + x2) + (1 - x1)*tf.log(1e-10 + 1 - x2)
    res = -1.*tf.reduce_sum(res)

    return res


def pairwise_distance_euclidean(X):
    """
    get pairwise euclidean distance between a batched tensor.
    this snippet is based from
        https://www.reddit.com/r/tensorflow/comments/58tq6j/how_to_compute_pairwise_distance_between_points/

    :param X: a tensor with N many D-dim vectors, has a shape of [N, D]
    :return: N by N matrix, each element is a distance between i-th vectors and j-th vectors of X
    """
    batch_size = tf.shape(X)[0] # get N first

    # expand dim of X and X_T
    X = tf.expand_dims(X, 1)
    X_T = tf.expand_dims(X, 0)

    # tile X and X_T as batch size
    X = tf.tile(X, [1, batch_size, 1])
    X_T = tf.tile(X_T, [batch_size, 1, 1])

    # get distance
    res = tf.sqrt(tf.square(X - X_T))
    res = tf.reduce_sum(res, 2)

    return res


def pairwise_distance_binary_xentropy(X):
    """
    get pairwise binary cross entropy between a batched tensor.
    binary cross entropy can be computed as following:
        L = -sum_(x_1*log(x_2) + (1 - x_1)*log(1 - x_2))
    this snippet is based from
        https://www.reddit.com/r/tensorflow/comments/58tq6j/how_to_compute_pairwise_distance_between_points/

    :param X: a tensor with N many D-dim vectors, has a shape of [N, D]
    :return: N by N matrix, each element is a distance between i-th vector and j-th vector of X
    """
    batch_size = tf.shape(X)[0]  # get N first

    # expand dim of X and X_T
    X = tf.expand_dims(X, 1)
    X_T = tf.expand_dims(X, 0)

    # tile X and X_T as batch size
    X = tf.tile(X, [1, batch_size, 1])
    X_T = tf.tile(X_T, [batch_size, 1, 1])

    # get distance
    res = -1.*(X*tf.log(1e-10 + X_T) + (1 - X)*tf.log(1e-10 + 1 - X_T))
    res = tf.reduce_sum(res, 2)

    return res
