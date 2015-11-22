from __future__ import division, print_function

import numpy as np
import pyBlockSQP

def test_matrix():
    m = pyBlockSQP.PyMatrix(1,1)
    m.Dimension(2,3).Initialize(42)

    assert m.as_array.shape == (2,3)
    assert np.all(m.as_array == 42)

def test_matrix_from_numpy():
    data = np.arange(6).reshape((3,2))

    m = pyBlockSQP.PyMatrix.from_numpy_2d(data)
    assert m.as_array.shape == data.shape
    assert np.all(m.as_array == data)


