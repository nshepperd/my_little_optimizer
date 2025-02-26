import pytest
import jax
import jax.numpy as jnp
from my_little_optimizer_server.util import fn, StaticFn, Partial

def test_fn_helper():
    # Test regular function wrapping
    def simple_fn(x):
        return x * 2
    
    wrapped = fn(simple_fn)
    assert isinstance(wrapped, StaticFn)
    assert wrapped(3) == 6
    
    # Test wrapping an already wrapped function
    double_wrapped = fn(wrapped)
    assert isinstance(double_wrapped, StaticFn)
    assert double_wrapped is wrapped  # Should return same instance
    
    # Test wrapping a Partial
    partial_fn = Partial(simple_fn, 3)
    wrapped_partial = fn(partial_fn)
    assert isinstance(wrapped_partial, Partial)
    assert wrapped_partial is partial_fn  # Should return same instance

def test_static_fn_basic():
    def add(x, y):
        return x + y
    
    static_add = StaticFn(add)
    assert static_add(2, 3) == 5
    assert static_add(jnp.array([1, 2]), jnp.array([3, 4])).tolist() == [4, 6]

def test_partial_basic():
    def add(x, y, z):
        return x + y + z
    
    # Test basic partial application
    partial_add = Partial(add, 1, z=3)
    assert partial_add(2) == 6
    
    # Test with JAX arrays
    partial_jax = Partial(add, jnp.array([1, 2]), z=jnp.array([5, 6]))
    result = partial_jax(jnp.array([3, 4]))
    assert all(result == jnp.array([9, 12]))

def test_partial_nested():
    def add(w, x, y, z):
        return w + x + y + z
    
    # Test nested partial application
    first_partial = Partial(add, 1, z=4)
    second_partial = Partial(first_partial, 2)
    assert second_partial(3) == 10
    
    # Test with mixed JAX arrays
    first_jax = Partial(add, jnp.array([1, 2]), z=jnp.array([7, 8]))
    second_jax = Partial(first_jax, jnp.array([3, 4]))
    result = second_jax(jnp.array([5, 6]))
    assert all(result == jnp.array([16, 20]))

def test_partial_pytree():
    def add(x, y):
        return x + y
    
    partial_add = Partial(add, 1)
    
    # Test tree_flatten
    flat, aux = jax.tree_util.tree_flatten(partial_add)
    assert len(flat) == 1  # fn, args, kwargs
    
    # Test tree_unflatten
    unflattened = jax.tree_util.tree_unflatten(aux, flat)
    assert isinstance(unflattened, Partial)
    assert unflattened(2) == 3

def test_jax_transformations():
    def multiply(x, y):
        return x * y
    
    # Test JIT compilation
    jitted_mul = jax.jit(StaticFn(multiply))
    assert jitted_mul(2, 3) == 6
    
    # Test gradients
    def square(x):
        return x ** 2
    
    grad_fn = jax.grad(StaticFn(square))
    assert grad_fn(3.0) == 6.0
    
    # Test vmap
    batched_mul = jax.vmap(StaticFn(multiply))
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    result = batched_mul(x, y)
    assert all(result == jnp.array([4, 10, 18]))

def test_error_cases():
    # Test missing required arguments
    def requires_args(x, y):
        return x + y
    
    partial_fn = Partial(requires_args, 1)
    with pytest.raises(TypeError):
        partial_fn()  # Missing required argument y

def test_kwargs_handling():
    def keyword_fn(x, *, y=2, z=3):
        return x + y + z
    
    # Test partial with kwargs
    partial_kw = Partial(keyword_fn, y=5)
    assert partial_kw(1) == 9  # 1 + 5 + 3
    assert partial_kw(1, z=7) == 13  # 1 + 5 + 7
    
    # Test updating kwargs
    partial_update = Partial(partial_kw, z=10)
    assert partial_update(1) == 16  # 1 + 5 + 10