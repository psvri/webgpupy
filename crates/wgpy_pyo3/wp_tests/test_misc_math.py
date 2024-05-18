import numpy as np
import webgpupy as wp
import pytest
from test_utils import assert_values_nin2, assert_values_nin1, almost_equals


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


@pytest.fixture
def np_array_1(input_array_1):
    return np.array(input_array_1)


@pytest.fixture
def np_array_2(input_array_2):
    return np.array(input_array_2)


def test_sqrt(wp_array_1, np_array_1):
    assert_values_nin1(wp_array_1, np_array_1, "sqrt")


def test_maximum(wp_array_1, wp_array_2, np_array_1, np_array_2):
    assert_values_nin2(wp_array_1, wp_array_2, np_array_1, np_array_2, "maximum")


def test_minimum(wp_array_1, wp_array_2, np_array_1, np_array_2):
    assert_values_nin2(wp_array_1, wp_array_2, np_array_1, np_array_2, "minimum")


def test_clip(wp_array_2, np_array_2):
    almost_equals(wp.clip(wp_array_2, 0.0, None), np.clip(np_array_2, 0.0, None))
    almost_equals(wp.clip(wp_array_2, None, 10.0), np.clip(np_array_2, None, 10.0))
    almost_equals(wp.clip(wp_array_2, 0.0, 10.0), np.clip(np_array_2, 0.0, 10.0))


def test_cross():
    def test_cross_results(input1, input2):
        almost_equals(
            wp.cross(wp.array(input1), wp.array(input2)), np.cross(input1, input2)
        )

    test_cross_results(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [0.0, 0.0, 0.0]]
    )
    test_cross_results([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0], [11.0, 12.0]])
    test_cross_results([[7.0, 8.0], [11.0, 12.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    test_cross_results([[1.0, 2.0], [4.0, 5.0]], [[7.0, 8.0], [11.0, 12.0]])


@pytest.mark.skip(reason="Ignoring temporarily")
def test_power(wp_array_1, wp_array_2, np_array_1, np_array_2):
    assert_values_nin2(wp_array_1, wp_array_2, np_array_1, np_array_2, "power")
