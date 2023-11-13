import numpy as np
import webgpupy as wp
import pytest

@pytest.fixture
def input_array():
    return [0.0, 1.0, 2.0, 3.0]

@pytest.fixture
def wp_array(input_array):
    return wp.array(input_array)

def assert_values(input_array, wp_array, wp_fn, np_fn):
    wp_result = wp_fn(wp_array)
    np.testing.assert_array_almost_equal(np_fn(input_array), wp_result.tolist(), decimal=4)

def test_cos(input_array, wp_array):
    assert_values(input_array, wp_array, wp.cos, np.cos)

def test_sin(input_array, wp_array):
    assert_values(input_array, wp_array, wp.sin, np.sin)