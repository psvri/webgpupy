import numpy as np
import webgpupy as wp
import pytest
from test_utils import *

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
    assert_values_nin1(wp_array_1, np_array_1, 'sqrt')


def test_maximum(wp_array_1, wp_array_2, np_array_1, np_array_2):
    assert_values_nin2(
        wp_array_1, wp_array_2, np_array_1, np_array_2, 'maximum'
    )


def test_minimum(wp_array_1, wp_array_2, np_array_1, np_array_2):
    assert_values_nin2(
        wp_array_1, wp_array_2, np_array_1, np_array_2, 'minimum'
    )
