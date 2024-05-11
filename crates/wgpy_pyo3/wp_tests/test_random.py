import webgpupy as wp
from jax import random
from test_utils import almost_equals


def test_random():
    key = random.PRNGKey(0)
    wp_rng = wp.random.default_rng()
    jp_rand = random.uniform(key, [100])
    wp_rand = wp_rng.random([100])
    almost_equals(wp_rand, jp_rand)

    key, split_key = random.split(key)
    wp_rand = wp_rng.random([100])
    jp_rand = random.uniform(split_key, [100])
    almost_equals(wp_rand, jp_rand)


def test_random_large():
    wp_rand = wp.random.default_rng().random([1920 * 1080 * 3])
    jp_rand = random.uniform(random.PRNGKey(0), [1920 * 1080 * 3])
    almost_equals(wp_rand, jp_rand)


def test_random_normal():
    wp_rand = wp.random.default_rng().normal([100])
    jp_rand = random.normal(random.PRNGKey(0), [100])
    almost_equals(wp_rand, jp_rand)
