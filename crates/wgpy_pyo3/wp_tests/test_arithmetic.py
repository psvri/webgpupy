import webgpupy as wp
import pytest
import numpy as np

@pytest.fixture
def input_array():
    return [[1.0], [2.0], [3.0], [4.0]]

@pytest.fixture
def wp_array(input_array):
    return wp.array(input_array)

def test_mul_f32(wp_array):
    assert (wp_array * 10.0).tolist() == [[10.0], [20.0], [30.0], [40.0]]
    
def test_rmul_f32(wp_array):
    assert (10.0 * wp_array).tolist() == [[10.0], [20.0], [30.0], [40.0]]

def test_divide_f32(wp_array):
    np.testing.assert_array_almost_equal((wp_array / 10.0).tolist(), [[0.1], [0.2], [0.3], [0.4]])

def test_add_f32(wp_array):
    assert (wp_array + 10.0).tolist() == [[11.0], [12.0], [13.0], [14.0]]

def test_radd_f32(wp_array):
    assert (10.0 + wp_array).tolist() == [[11.0], [12.0], [13.0], [14.0]]

def test_sub_f32(wp_array):
    assert (wp_array - 10.0).tolist() == [[-9.0], [-8.0], [-7.0], [-6.0]]
    
def test_rsub_f32(wp_array):
    assert (10.0 - wp_array).tolist() == [[9.0], [8.0], [7.0], [6.0]]
