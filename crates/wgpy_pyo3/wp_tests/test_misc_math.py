import numpy as np
import webgpupy as wp
import pytest


@pytest.fixture
def input_array_1():
    return [0.0, 1.0, 2.0, 3.0]


@pytest.fixture
def input_array_2():
    return [0.0, -1.0, 20.0, -3.0]


@pytest.fixture
def wp_array_1(input_array_1):
    return wp.array(input_array_1)


@pytest.fixture
def wp_array_2(input_array_2):
    return wp.array(input_array_2)


def assert_values_nin1(input_array, wp_array, wp_fn, np_fn):
    wp_result = wp_fn(wp_array)
    np.testing.assert_array_almost_equal(
        np_fn(input_array), wp_result.tolist(), decimal=4
    )


def assert_values_nin2(
    input_array_1, input_array_2, wp_array_1, wp_array_2, wp_fn, np_fn
):
    wp_result = wp_fn(wp_array_1, wp_array_2)
    np_result = np_fn(input_array_1, input_array_2)
    np.testing.assert_array_almost_equal(np_result, wp_result.tolist(), decimal=4)


def test_sqrt(input_array_1, wp_array_1):
    assert_values_nin1(input_array_1, wp_array_1, wp.sqrt, np.sqrt)


def test_maximum(input_array_1, input_array_2, wp_array_1, wp_array_2):
    assert_values_nin2(
        input_array_1, input_array_2, wp_array_1, wp_array_2, wp.maximum, np.maximum
    )


def test_minimum(input_array_1, input_array_2, wp_array_1, wp_array_2):
    assert_values_nin2(
        input_array_1, input_array_2, wp_array_1, wp_array_2, wp.minimum, np.minimum
    )
