import numpy as np
import webgpupy as wp

def almost_equals(x, y, decimal=6):
    np.testing.assert_array_almost_equal(x.tolist(), y.tolist(), decimal=decimal)

def assert_values_nin2(wp_array_1,wp_array_2, np_array_1, np_array_2, fn):
    wp_fn = getattr(wp, fn)
    np_fn = getattr(np, fn)
    almost_equals(wp_fn(wp_array_1, wp_array_2), np_fn(np_array_1, np_array_2))

def assert_values_nin1(wp_arr, np_arr, fn):
    wp_fn = getattr(wp, fn)
    np_fn = getattr(np, fn)
    almost_equals(wp_fn(wp_arr), np_fn(np_arr))