import numpy as np
import random
import unittest
import softmax.q1_softmax as q1

class AssertTest(unittest.TestCase):


    '''
    data is like this [[0,0,0,0,0,0.1],[1]],the forward is x and the last is  y
    '''
    def test_sigmod(self):
        x = np.array([0])
        self.assertEqual(q1.sigmod(x),0.5)

    def test_softmax(self):
        x = np.zeros(5)
        x = q1.sigmod(x)
        self.assertListEqual(list(x),np.array([0.5,0.5,0.5,0.5,0.5]).tolist())


if __name__ == '__main__':
    unittest.main()
