from __future__ import division, print_function

import numpy as np
import pyBlockSQP


def test_matrix():
    m = pyBlockSQP.PyMatrix(1, 1)
    m.Dimension(2, 3).Initialize(42)

    assert m.as_array.shape == (2, 3)
    assert np.all(m.as_array == 42)


def test_matrix_from_numpy():
    data = np.arange(6).reshape((3, 2))

    m = pyBlockSQP.PyMatrix.from_numpy_2d(data)
    assert m.as_array.shape == data.shape
    assert np.all(m.as_array == data)

def test_matrix_change_data():
    m = pyBlockSQP.PyMatrix()
    m.Dimension(2,2).Initialize(0.0)

    m.numpy_data[2] = 42
    assert m.numpy_data[2] == 42.0

def test_matrix_change_numpy_array():
    m = pyBlockSQP.PyMatrix()
    m.Dimension(2,2).Initialize(0.0)

    m.as_array[1,1] = 42
    assert m.as_array[1,1] == 42.0

def test_symmatrix():
    m = pyBlockSQP.PySymMatrix(1)
    m.Dimension(3).Initialize(23)

    assert len(m.numpy_data) == 6
    assert np.all(m.numpy_data == 23)


def test_symmatrix_change_data():
    m = pyBlockSQP.PySymMatrix()
    m.Dimension(2,2).Initialize(0.0)

    m.numpy_data[2] = 42
    assert m.numpy_data[2] == 42.0

def test_symmatrix_from_numpy():
    data = np.array([[1., 2., 3.],
                     [2., 4., 5.],
                     [3., 5., 6.]])

    m = pyBlockSQP.PySymMatrix.from_numpy_2d(data)
    assert m.as_array.shape == data.shape
    assert np.all(m.as_array == data)
