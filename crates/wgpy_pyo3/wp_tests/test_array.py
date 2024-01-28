import webgpupy as wp
import pytest
import numpy as np
from test_utils import almost_equals

@pytest.fixture
def input_array():
    return [[1.0], [2.0], [3.0], [4.0]]

@pytest.fixture
def wp_array(input_array):
    return wp.array(input_array)

@pytest.fixture
def np_array(input_array):
    return np.array(input_array)

def test_array(wp_array):
    assert wp_array.shape == [4,1]
    assert wp_array.tolist() == [[1.0], [2.0], [3.0], [4.0]]

def test_array_astype(wp_array):
    arr = wp_array.astype('uint8')
    assert arr.tolist() == [[1], [2], [3], [4]]

def test_indexing(wp_array, np_array):
    almost_equals(wp_array[:, :], np_array[:, :])
    almost_equals(wp_array[:], np_array[:])
    almost_equals(wp_array[1], np_array[1])
    almost_equals(wp_array[:,0], np_array[:,0])

def test_neg(wp_array, np_array):
    almost_equals(-wp_array, -np_array)