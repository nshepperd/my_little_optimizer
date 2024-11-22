import jax
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm

@dataclass
class GP:
    kernel: callable
    sigma: jax.Array

    def calc_precision(self, xs: jax.Array):
        n = xs.shape[0]
        k2 = jax.vmap(self.kernel, in_axes=(None, 0))
        k2 = jax.vmap(k2, in_axes=(0, None))
        K = k2(xs, xs)
        C = K + jnp.eye(n)*self.sigma**2
        L = jnp.linalg.cholesky(C)
        return L

    def predictb(self, xs: jax.Array, ys: jax.Array, P: jax.Array, x: jax.Array):
        return jax.vmap(self.predict, in_axes=(None, None, None, 0))(xs, ys, P, x)
    
    def predict(self, xs: jax.Array, ys: jax.Array, P: jax.Array, x: jax.Array):
        assert tuple(xs.shape[1:]) == tuple(x.shape), 'Expected shape of x to be {}, got {}'.format(xs.shape[1:], x.shape)
        kxs = jax.vmap(self.kernel, in_axes=(None, 0))(x, xs)
        u = jax.scipy.linalg.cho_solve((P,True), kxs)
        # mean = jnp.einsum('i,ij,j->', kxs,P,ys)
        # var = self.kernel(x, x) - jnp.einsum('i,ij,j->', kxs, P, kxs)
        mean = u @ ys
        var = self.kernel(x, x) - u @ kxs
        var = jnp.clip(var, min=0.0)
        return (mean, var)

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    xs = jnp.array([[1.0],[0.0],[-1.0]])
    ys = jnp.array([0.5,0.0,-1.0])
    model = GP(lambda x,y: jnp.exp(-0.5*jnp.sum(jnp.square(x-y))), 0.0)

    qs = jnp.linspace(-1, 1, 100)
    prec = model.calc_precision(xs)
    mean, var = model.predictb(xs, ys, prec, qs[:,None])

    plt.plot(qs, mean)
    plt.fill_between(qs, mean-jnp.sqrt(var), mean+jnp.sqrt(var), alpha=0.2)
    plt.show()