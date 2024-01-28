import numpy as np

def almost_equals(x, y):
    np.testing.assert_array_almost_equal(x.tolist(), y.tolist(), decimal=4)