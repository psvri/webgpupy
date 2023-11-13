import webgpupy as wp

def test_array():
    arr = wp.array([[1], [2], [3]])
    assert arr.shape == [3,1]
    assert arr.flatten().tolist() == [1, 2, 3]