import jax
from functools import partial

def Fn(fn: callable):
    if isinstance(fn, (StaticFn, Partial)):
        return fn
    return StaticFn(fn)

@jax.tree_util.register_pytree_node_class
class StaticFn(object):
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
class Partial(object):
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
