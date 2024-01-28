import webgpupy as wp
import pytest
import numpy as np
from test_utils import *

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


@pytest.fixture
def np_array_1(input_array_1):
    return np.array(input_array_1)


@pytest.fixture
def np_array_2(input_array_2):
    return np.array(input_array_2)


def test_bitwise_and_u32(wp_array_1, wp_array_2, np_array_1, np_array_2):
    assert_values_nin2(
        wp_array_1, wp_array_2, np_array_1, np_array_2, 'bitwise_and'
    )

def test_bitwise_or_u32(wp_array_1, wp_array_2, np_array_1, np_array_2):
    assert_values_nin2(
        wp_array_1, wp_array_2, np_array_1, np_array_2, 'bitwise_or'
    )

def test_bitwise_or_u32(wp_array_1, np_array_1):
    assert_values_nin1(wp_array_1, np_array_1, 'invert')
