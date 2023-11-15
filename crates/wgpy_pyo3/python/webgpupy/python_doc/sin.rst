sin(x, /, *, where=True, dtype=None)

Trigonometric sine, element-wise.

Parameters
----------
x : array_like
    Input Array
where : array_like, optional
    At locations where the condition is True value is computed, else locations
    where the condition is False will remain uninitialized.

See Also
--------
cos


Examples
--------
>>> import webgpupy as wp
>>> x = wp.array([0.0])
>>> wp.sin(x).tolist()
[0.0]