import jax
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple, Protocol, Dict, Any, List
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm

from ngp.nuts import nuts_kernel

class Metric(Protocol):
    def whiten(self, x: Any) -> jax.Array: ...
    def unwhiten(self, x: jax.Array) -> Any: ...
    def ndim(self) -> int: ...

Shape = List[int]

@jax.tree_util.register_pytree_node_class
@dataclass
class ScalarMetric(Metric):
    a: float
    shape: Shape
    def whiten(self, array):
        assert list(array.shape) == list(self.shape)
        return array.reshape(-1) / self.a
    def unwhiten(self, array):
        return array.reshape(self.shape) * self.a
    def ndim(self):
        return np.prod(self.shape)

    def tree_flatten(self):
        return (self.shape,), (self.a,)
    @staticmethod
    def tree_unflatten(static, dynamic):
        return ScalarMetric(*dynamic, *static)

# class KroneckerMetric(MetricSpace):
#     def __init__(self, P: Array, Q: Array):
#         self.P = P
#         self.Q = Q
#         # Cache matrix square roots and inverses
#         self.P_sqrt = matrix_sqrt(P)
#         self.Q_sqrt = matrix_sqrt(Q)
#         self.P_invsqrt = matrix_sqrt(jnp.linalg.inv(P))
#         self.Q_invsqrt = matrix_sqrt(jnp.linalg.inv(Q))
    
#     def whiten(self, W: Array) -> Array:
#         return self.P_invsqrt @ W @ self.Q_invsqrt
        
#     def unwhiten(self, W_white: Array) -> Array:
#         return self.P_sqrt @ W_white @ self.Q_sqrt

# class FullCovMetric(MetricSpace):
#     def __init__(self, Sigma: Array):
#         self.L = jnp.linalg.cholesky(Sigma)
#         self.L_inv = jnp.linalg.inv(self.L)
        
#     def whiten(self, x: Array) -> Array:
#         return self.L_inv @ x
        
#     def unwhiten(self, x_white: Array) -> Array:
#         return self.L @ x_white

@dataclass
class DictionaryMetric(Metric):
    metrics: Dict[str, Metric]
        
    def whiten(self, params: Dict[str, Any]) -> jax.Array:
        assert set(params.keys()) == set(self.metrics.keys())
        whitened = {k: self.metrics[k].whiten(v) for k,v in params.items()}
        return jnp.concatenate([whitened[k] for k in sorted(whitened.keys())], axis=0)
        
    def unwhiten(self, white_params: jax.Array) -> Dict[str, Any]:
        parts = {}
        ix = 0
        for k in sorted(self.metrics.keys()):
            parts[k] = self.metrics[k].unwhiten(white_params[ix:ix+self.metrics[k].ndim()])
            ix += self.metrics[k].ndim()
        return parts

def metric_nuts(key, params, logp, ε, metric):
    theta = metric.whiten(params)
    def logp_theta(theta):
        params = metric.unwhiten(theta)
        return logp(params)
    theta_new, alpha, ms = nuts_kernel(key, theta, logp_theta, jax.grad(logp_theta), ε)
    return metric.unwhiten(theta_new), alpha, ms

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from jaxtorch import PRNG
    import einops

    @jax.jit
    def logp(theta):
        return -0.5 * jnp.sum(jnp.square(theta/jnp.array([1,200])))

    rng = PRNG(jax.random.PRNGKey(0))
    theta = jax.random.normal(rng.split(), [2])

    def k_nuts(key, theta, ε):
        return metric_nuts(key, theta, logp, ε, ScalarMetric(jnp.array([1.,200.]), [2]))


    @jax.jit
    def sample(key, theta, ε):
        @scan_tqdm(30000)
        def body_fn(carry, i):
            key, theta = carry
            key, subkey = jax.random.split(key)
            theta, alpha = k_nuts(subkey, theta, ε)
            return (key, theta), (theta, alpha)
        return jax.lax.scan(body_fn, (key, theta), jnp.arange(30000))[1]


    samples, alphas = sample(rng.split(), theta, 1.0)
    samples = jnp.stack(samples)
    alphas = jnp.stack(alphas)
    print(samples.shape)
    print(samples.std(0), jax.random.normal(rng.split(),samples.shape).std(0))
    print(alphas.mean())
    # scatter plot
    plt.scatter(samples[:,0], samples[:,1])
    plt.plot(samples[:,0], samples[:,1], color='red', alpha=0.1)
    plt.show()

