import jax
from typing import Protocol
from functools import partial

class Fn(Protocol):
    """Callable that should be a valid pytree."""
    def __call__(self, *args, **kwargs):
        ...

def fn(f: callable):
    if isinstance(f, (StaticFn, Partial)):
        return f
    return StaticFn(f)

@jax.tree_util.register_pytree_node_class
class StaticFn(Fn):
    fn: callable
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
    def tree_flatten(self):
        return (), (self.fn,)
    @staticmethod
    def tree_unflatten(static, dynamic):
        return StaticFn(*static, *dynamic)

@jax.tree_util.register_pytree_node_class
class Partial(Fn):
    fn: callable
    args: tuple
    kwargs: dict
    def __init__(self, fn, *args, **kwargs):
        if not isinstance(fn, (Partial, StaticFn)):
            fn = StaticFn(fn)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        p = partial(self.fn, *self.args, **self.kwargs)
        return p(*args, **kwargs)

    def tree_flatten(self):
        return (self.fn, self.args, self.kwargs), ()
    @staticmethod
    def tree_unflatten(static, dynamic):
        fn, args, kwargs = dynamic
        return Partial(fn, *args, **kwargs)
