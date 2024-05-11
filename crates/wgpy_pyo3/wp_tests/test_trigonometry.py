import numpy as np
import webgpupy as wp
from test_utils import assert_values_nin1
import pytest


@pytest.fixture
def input_array():
    return [0.0, 1.0, 2.0, 3.0]


@pytest.fixture
def wp_array(input_array):
    return wp.array(input_array)


@pytest.fixture
def np_array(input_array):
    return np.array(input_array)


def test_cos(wp_array, np_array):
    assert_values_nin1(wp_array, np_array, "cos")


def test_sin(wp_array, np_array):
    assert_values_nin1(wp_array, np_array, "sin")
