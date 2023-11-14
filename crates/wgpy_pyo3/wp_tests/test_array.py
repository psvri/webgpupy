import webgpupy as wp
import pytest

@pytest.fixture
def input_array():
    return [[1.0], [2.0], [3.0], [4.0]]

@pytest.fixture
def wp_array(input_array):
    return wp.array(input_array)

def test_array(wp_array):
    assert wp_array.shape == [4,1]
    assert wp_array.flatten().tolist() == [1.0, 2.0, 3.0, 4.0]

def test_array_astype(wp_array):
    arr = wp_array.astype('uint8')
    assert arr.flatten().tolist() == [1, 2, 3, 4]