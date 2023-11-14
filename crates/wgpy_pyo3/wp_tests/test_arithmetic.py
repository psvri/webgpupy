import webgpupy as wp
import pytest

@pytest.fixture
def input_array():
    return [[1.0], [2.0], [3.0], [4.0]]

@pytest.fixture
def wp_array(input_array):
    return wp.array(input_array)

def test_mul_f32(wp_array):
    arr = wp_array * 10.0
    assert arr.flatten().tolist() == [10.0, 20.0, 30.0, 40.0]