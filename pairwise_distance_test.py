from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from pairwise_distance import pairwise_distance_euclidean, pairwise_distance_binary_xentropy


class PairwiseDistantTest(tf.test.TestCase):
    @tf.contrib.eager.run_test_in_graph_and_eager_modes()
    def testPairwiseDistantEuclidean(self):
        x = np.asarray([[1, 2, 3, 4, 5],
                        [2, 3, 4, 5, 6],
                        [3, 4, 5, 6, 7],
                        [4, 5, 6, 7, 8],
                        [0, 0, 0, 0, 0]], dtype=np.float64)
        res = pairwise_distance_euclidean(x)
        self.assertEqual(res.shape, (5, 5))

    @tf.contrib.eager.run_test_in_graph_and_eager_modes()
    def testPairwiseDistantBinaryXentropy(self):
        batch_size = 64
        dim = 256
        x = np.random.uniform(0.0, 1.0, size=[batch_size, dim])
        res = pairwise_distance_binary_xentropy(x)
        self.assertEqual(res.shape, (batch_size, batch_size))


if __name__ == "__main__":
    tf.test.main()
