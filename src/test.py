import unittest
import numpy as np
import numpy.testing as npt
import os
import GBRBM


class DataProcessManagerTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_make_mini_batch(self):
        arr = np.ones((11, 3))
        expected = np.array(
            [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
             ])
        actual = GBRBM.make_mini_batch(arr, 3)

        print(actual.shape)

        npt.assert_array_equal(expected, actual)
