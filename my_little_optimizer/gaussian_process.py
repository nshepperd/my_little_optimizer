import jax
import jax.numpy as jnp
import jax.scipy as jsp
from tqdm import tqdm
from typing import Tuple
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm
import einops

@dataclass
class GP:
    kernel: callable
    sigma: jax.Array

    def predictb(self, xs: jax.Array, ys: jax.Array, xstar: jax.Array):
        assert tuple(xs.shape[1:]) == tuple(xstar.shape[1:]), 'Expected shape of xs and xstar to match'
        n = xs.shape[0]
        m = xstar.shape[0]
        k2 = self.kernel
        k2 = jax.vmap(k2, in_axes=(None, 0))
        k2 = jax.vmap(k2, in_axes=(0, None))
        K = k2(xs, xs) # [n, n]
        jitter = 1e-4
        C = K + jnp.eye(n)*(self.sigma**2 + jitter)

        # u = C^-1 * K(x, xstar)

        kxs = k2(xs, xstar) # [n, m]
        u = jax.scipy.linalg.solve(C, kxs, assume_a='pos') # [n, m]
        mean = jnp.einsum('ij,i->j', u, ys)
        var = jax.vmap(self.kernel)(xstar, xstar) - (u * kxs).sum(axis=0)
        var = jnp.clip(var, min=0.0)
        return mean, var
    
    def predict(self, xs: jax.Array, ys: jax.Array, x: jax.Array):
        assert tuple(xs.shape[1:]) == tuple(x.shape), 'Expected shape of x to be {}, got {} (xs.shape={}, ys.shape={})'.format(xs.shape[1:], x.shape, xs.shape, ys.shape)
        return self.predict_masked(xs, ys, xs.shape[0], x)

    def predictb_masked(self, xs: jax.Array, ys: jax.Array, n: int, x: jax.Array):
        return jax.vmap(self.predict_masked, in_axes=(None, None, None, 0))(xs, ys, n, x) 
        
    def predict_masked(self, xs: jax.Array, ys: jax.Array, n: int, x: jax.Array):
        """Predicts the mean and variance of ys given xs and x.

        Uses only the first n points (to be compatible with jit).
        """

        assert tuple(xs.shape[1:]) == tuple(x.shape), 'Expected shape of x to be {}, got {}'.format(xs.shape[1:], x.shape)
        size = xs.shape[0]
        mask = jnp.arange(size) < n

        # # Find unique entries of xs and group them together
        # eq_xs = (xs[None,:,:] == xs[:,None,:]).all(-1)
        # ix_xs = (jnp.cumsum(eq_xs, axis=1) < 1).sum(-1) # index of the first equal element
        # uq = jnp.unique_all(ix_xs, size=size)
        # new_xs = jnp.zeros(xs.shape).at[jnp.where(mask, uq.inverse_indices, -1)].set(xs, mode='drop')
        # new_ys = jnp.zeros(ys.shape).at[jnp.where(mask, uq.inverse_indices, -1)].add(ys, mode='drop')
        # new_cs = jnp.zeros(ys.shape, dtype=jnp.int32).at[jnp.where(mask, uq.inverse_indices, -1)].add(1, mode='drop')
        # new_n = jnp.sum(new_cs>0)

        # cs = jnp.clip(new_cs, 1)
        # xs = new_xs
        # ys = new_ys / cs # average samples for each xs
        # n = new_n
        # mask = jnp.arange(size) < n

        k2 = jax.vmap(self.kernel, in_axes=(None, 0))
        k2 = jax.vmap(k2, in_axes=(0, None))
        K = k2(xs, xs) # [size, size]
        K *= mask[:, None] * mask[None, :] # remove masked entries
        # C = K + jnp.eye(size)*self.sigma**2/cs + jnp.eye(size)*(1-mask)
        jitter = 1e-4
        C = K + jnp.eye(size)*(self.sigma**2 + jitter) + jnp.eye(size)*(1-mask)

        kxs = jax.vmap(self.kernel, in_axes=(None, 0))(x, xs) * mask
        u = jax.scipy.linalg.solve(C, kxs, assume_a='pos') * mask
        mean = u @ ys
        var = self.kernel(x, x) - u @ kxs
        var = jnp.clip(var, min=0.0)
        return mean, var
    
    def marginal_likelihood(self, xs: jax.Array, ys: jax.Array):
        # xs: [n, d]
        # ys: [n]
        assert len(xs.shape) == 2
        assert len(ys.shape) == 1
        n = xs.shape[0]
        k2 = self.kernel
        k2 = jax.vmap(k2, in_axes=(None, 0))
        k2 = jax.vmap(k2, in_axes=(0, None))
        K = k2(xs, xs) # [n, n]
        jitter = 1e-4
        C = K + jnp.eye(n) * (self.sigma**2 + jitter)
        L = jnp.linalg.cholesky(C)
        log_det_half = jnp.log(jnp.diagonal(L, axis1=-2,axis2=-1)).sum()
        ymu_L = jsp.linalg.solve_triangular(L, ys, lower=True) # [n]
        return -0.5 * jnp.square(ymu_L).sum() - log_det_half - (n/2) * jnp.log(2 * jnp.pi)

    def sample_prior(self, key: jax.Array, xs: jax.Array):
        # xs: [n, d]
        assert len(xs.shape) == 2
        n = xs.shape[0]
        k2 = self.kernel
        k2 = jax.vmap(k2, in_axes=(None, 0))
        k2 = jax.vmap(k2, in_axes=(0, None))
        K = k2(xs, xs) # [n, n]
        jitter = 1e-4
        C = K + jnp.eye(n) * (self.sigma**2 + jitter)
        L = jnp.linalg.cholesky(C) # [n, n]
        u = jax.random.normal(key, [n])
        return L @ u

    # def predict_cov(self, xs: jax.Array, ys: jax.Array, xstar: jax.Array):
    #     n = xs.shape[0]
    #     m = xstar.shape[0]
    #     k2 = self.kernel
    #     k2 = jax.vmap(k2, in_axes=(None, 0))
    #     k2 = jax.vmap(k2, in_axes=(0, None))
    #     kxx = k2(xs, xs) + jnp.eye(n)*(self.sigma**2) # [n, n]
    #     jitter = 1e-4
    #     kxs = k2(xs, xstar) # [n, m]
    #     kss = k2(xstar, xstar) # [m, m]


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    xs = jnp.array([[1.0],[0.0],[-1.0]])
    ys = jnp.array([0.5,0.0,-1.0])
    model = GP(lambda x,y: jnp.exp(-0.5*jnp.sum(jnp.square(x-y)/2)), 0.0)

    print(model.marginal_likelihood(xs, ys))

    qs = jnp.linspace(-1, 1, 100)
    mean, var = model.predictb(xs, ys, qs[:,None])
    mean2, var2 = model.predictb_masked(jnp.concatenate([xs, jnp.zeros(xs.shape)],axis=0), 
                                        jnp.concatenate([ys, jnp.zeros(ys.shape)],axis=0),
                                        xs.shape[0], qs[:,None])

    print(mean == mean2)
    print(var == var2)

    plt.plot(qs, mean)
    plt.fill_between(qs, mean-jnp.sqrt(var), mean+jnp.sqrt(var), alpha=0.2)
    plt.show()