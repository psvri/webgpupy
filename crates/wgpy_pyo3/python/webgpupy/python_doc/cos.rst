cos(x, /, *, where=True, dtype=None)

Trigonometric cos, element-wise.

Parameters
----------
x : array_like
    Input Array
where : array_like, optional
    At locations where the condition is True value is computed, else locations
    where the condition is False will remain uninitialized.

See Also
--------
sin

Examples
--------
>>> import webgpupy as wp
>>> x = wp.array([0.0])
>>> wp.cos(x).tolist()
[1.0]