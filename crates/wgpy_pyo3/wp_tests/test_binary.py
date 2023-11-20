import webgpupy as wp
import pytest
import numpy as np

@pytest.fixture
def input_array_1():
    return [[1], [2], [0], [4]]

@pytest.fixture
def input_array_2():
    return [[10], [2], [3], [4]]

@pytest.fixture
def wp_array_1(input_array_1):
    return wp.array(input_array_1)

@pytest.fixture
def wp_array_2(input_array_2):
    return wp.array(input_array_2)

def assert_values_nin2(input_array_1, input_array_2, wp_array_1,wp_array_2, wp_fn, np_fn):
    wp_result = wp_fn(wp_array_1, wp_array_2)
    np_result = np_fn(input_array_1, input_array_2)
    np.testing.assert_array_equal(np_result.tolist(), wp_result.tolist())

def assert_values_nin1(input_array_1, wp_array_1, wp_fn, np_fn):
    wp_result = wp_fn(wp_array_1)
    np_result = np_fn(input_array_1)
    np.testing.assert_array_equal(np_result.tolist(), wp_result.tolist())

def test_bitwise_and_u32(input_array_1, input_array_2, wp_array_1, wp_array_2):
    assert_values_nin2(input_array_1, input_array_2, wp_array_1, wp_array_2, wp.bitwise_and, np.bitwise_and)

def test_bitwise_or_u32(input_array_1, input_array_2, wp_array_1, wp_array_2):
    assert_values_nin2(input_array_1, input_array_2, wp_array_1, wp_array_2, wp.bitwise_or, np.bitwise_or)

def test_bitwise_or_u32(input_array_1, wp_array_1):
    assert_values_nin1(input_array_1, wp_array_1, wp.invert, np.invert)
