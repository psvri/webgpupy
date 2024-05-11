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


def test_mul_f32(wp_array, np_array):
    almost_equals(wp_array * 10.0, np_array * 10)


def test_rmul_f32(wp_array, np_array):
    almost_equals(10.0 * wp_array, 10 * np_array)


def test_divide_f32(wp_array, np_array):
    almost_equals(wp_array / 10.0, np_array / 10)


def test_add_f32(wp_array, np_array):
    almost_equals(wp_array + 10.0, np_array + 10)


def test_radd_f32(wp_array, np_array):
    almost_equals(10.0 + wp_array, 10 + np_array)


def test_sub_f32(wp_array, np_array):
    almost_equals(wp_array - 10.0, np_array - 10)


def test_rsub_f32(wp_array, np_array):
    almost_equals(10.0 - wp_array, 10 - np_array)
