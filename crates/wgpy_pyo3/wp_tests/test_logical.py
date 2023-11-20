import webgpupy as wp
import pytest
import numpy as np

@pytest.fixture
def input_array_1():
    return [[1.0], [2.0], [3.0], [4.0]]

@pytest.fixture
def input_array_2():
    return [[10.0], [2.0], [-3.0], [4.0]]

@pytest.fixture
def wp_array_1(input_array_1):
    return wp.array(input_array_1)

@pytest.fixture
def wp_array_2(input_array_2):
    return wp.array(input_array_2)

def test_gt_f32(wp_array_1, wp_array_2):
    assert (wp_array_1 > wp_array_2).tolist() == [[False], [False], [True], [False]]

def test_lt_f32(wp_array_1, wp_array_2):
    assert (wp_array_1 < wp_array_2).tolist() == [[True], [False], [False], [False]]
