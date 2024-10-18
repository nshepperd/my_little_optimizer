import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T')

@jax.tree_util.register_pytree_node_class
@dataclass
class Static(Generic[T]):
    """Wrapper for a static value that can contain any python object as a pytree node."""
    value: T
    def tree_flatten(self):
        return (), (self.value,)
    @staticmethod
    def tree_unflatten(static, dynamic):
        return Static(*static)

def the(xs):
    assert len(xs) == 1
    return xs[0]

def switchvec(ix: jax.Array, branches: list[callable], x: jax.Array):
    """jax.numpy.where, but handles gradients properly when some of the branches are NaN."""
    return switchvec_(ix, Static(branches), x)

@jax.custom_vjp
def switchvec_(ix: jax.Array, branches: Static[list], x: jax.Array):
    branches = branches.value
    n = len(branches)
    outs = [br(x) for br in branches]
    out = outs[0]
    for i in range(1, n):
        out = jnp.where(ix == i, outs[i], out)
    return out
def switchvec_fwd(ix: jax.Array, branches: Static[list], x: jax.Array):
    branches = branches.value
    n = len(branches)
    outs, vjps = zip(*[jax.vjp(br, x) for br in branches])
    out = outs[0]
    for i in range(1, n):
        out = jnp.where(ix == i, outs[i], out)
    return out, (ix,vjps)
def switchvec_bwd(res: tuple[jax.Array, list[callable]], g: jax.Array):
    ix, vjps = res
    return None, None, switchvec(ix, [lambda g: the(vjp(g)) for vjp in vjps], g)
switchvec_.defvjp(switchvec_fwd, switchvec_bwd)