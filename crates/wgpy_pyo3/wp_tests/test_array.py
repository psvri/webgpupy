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


def test_array(wp_array):
    assert wp_array.shape == [4, 1]
    assert wp_array.tolist() == [[1.0], [2.0], [3.0], [4.0]]


def test_array_astype(wp_array):
    arr = wp_array.astype("uint8")
    assert arr.tolist() == [[1], [2], [3], [4]]


def test_indexing(wp_array, np_array):
    almost_equals(wp_array[:, :], np_array[:, :])
    almost_equals(wp_array[:], np_array[:])
    almost_equals(wp_array[1], np_array[1])
    almost_equals(wp_array[:, 0], np_array[:, 0])


def test_neg(wp_array, np_array):
    almost_equals(-wp_array, -np_array)


def test_where(wp_array, np_array):
    bool_array = (np.random.default_rng(0).random([640, 360, 1]) < 0).tolist()
    wp_array = wp.broadcast_to(wp_array.reshape([4]), [640, 360, 4])
    np_array = np.broadcast_to(np_array.reshape([4]), [640, 360, 4])
    wp_where = wp.where(wp.array(bool_array), wp_array, -wp_array)
    np_where = np.where(np.array(bool_array), np_array, -np_array)
    almost_equals(wp_where, np_where)


def test_braodcast_to():
    shape = [1, 160, 1]
    new_shape = [1, 160, 2]
    bool_array = (np.random.default_rng(0).random(shape) < 0).tolist()
    np_array = np.broadcast_to(bool_array, new_shape)
    wp_array = wp.broadcast_to(wp.array(bool_array), new_shape)
    almost_equals(wp_array, np_array)
