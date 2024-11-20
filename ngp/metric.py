import jax
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple, Protocol, Dict, Any, List
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm

from ngp.nuts import nuts_kernel
from ngp.kfac import kfac

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
        return (self.a,), (self.shape,)
    @staticmethod
    def tree_unflatten(static, dynamic):
        return ScalarMetric(*dynamic, *static)

@jax.tree_util.register_pytree_node_class
@dataclass
class KroneckerMetric(Metric):
    P: jax.Array
    Q: jax.Array
    P_sqrt: jax.Array
    Q_sqrt: jax.Array
    P_invsqrt: jax.Array
    Q_invsqrt: jax.Array
    
    def ndim(self):
        m = self.P.shape[0]
        n = self.Q.shape[0]
        return m*n

    def whiten(self, W: jax.Array) -> jax.Array:
        m = self.P.shape[0]
        n = self.Q.shape[0]
        assert list(W.shape) == [m, n]
        return (self.P_invsqrt @ W @ self.Q_invsqrt).reshape(-1)
        
    def unwhiten(self, W_white: jax.Array) -> jax.Array:
        m = self.P.shape[0]
        n = self.Q.shape[0]
        return self.P_sqrt @ W_white.reshape([m,n]) @ self.Q_sqrt

    def tree_flatten(self):
        return (self.P, self.Q, self.P_sqrt, self.Q_sqrt, self.P_invsqrt, self.Q_invsqrt), ()
    @staticmethod
    def tree_unflatten(static, dynamic):
        return KroneckerMetric(*static, *dynamic)

@jax.tree_util.register_pytree_node_class
@dataclass
class FullCovMetric(Metric):
    M: jax.Array
    L: jax.Array
    L_inv: jax.Array
    # def __init__(self, Sigma: Array):
    #     self.L = jnp.linalg.cholesky(Sigma)
    #     self.L_inv = jnp.linalg.inv(self.L)
    
    def ndim(self):
        return self.M.shape[0]

    def whiten(self, x: jax.Array) -> jax.Array:
        return (self.L_inv @ x).reshape(-1)
        
    def unwhiten(self, x_white: jax.Array) -> jax.Array:
        m = self.M.shape[0]
        return self.L @ x_white.reshape(m)
    
    def tree_flatten(self):
        return (self.M, self.L, self.L_inv), ()
    @staticmethod
    def tree_unflatten(static, dynamic):
        return FullCovMetric(*static, *dynamic)

@jax.tree_util.register_pytree_node_class
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
    
    def tree_flatten(self):
        return (self.metrics,), ()
    @staticmethod
    def tree_unflatten(static, dynamic):
        return DictionaryMetric(*static, *dynamic)

class MetricEstimator(Protocol):
    def __call__(self, samples: jax.Array) -> Metric: ...

@dataclass
class ConstantMetricEstimator(MetricEstimator):
    metric: Metric
    def __call__(self, samples: jax.Array) -> Metric:
        return self.metric

class StdScalarMetricEstimator(MetricEstimator):
    def __call__(self, samples: jax.Array) -> ScalarMetric:
        shape = samples.shape[1:]
        samples = samples.reshape([samples.shape[0], -1])
        samples -= samples.mean(0)
        return ScalarMetric(jnp.sqrt(jnp.mean(jnp.square(samples))), shape)

class CovarianceMetricEstimator(MetricEstimator):
    def __call__(self, samples: jax.Array) -> FullCovMetric:
        assert len(samples.shape) == 2
        cov = jnp.mean(samples[:,None,:] * samples[:,:,None], axis=0)
        L = jnp.linalg.cholesky(cov)
        return FullCovMetric(cov, L, jnp.linalg.inv(L))

class KroneckerMetricEstimator(MetricEstimator):
    def __call__(self, samples: jax.Array) -> KroneckerMetric:
        assert len(samples.shape) == 3
        samples -= samples.mean(0)
        P, Q = kfac(samples)

        # def sqrt(A):
        #     evals, evecs = jnp.linalg.eigh(A)
        #     sqrt_evals = jnp.sqrt(jnp.maximum(evals, 1e-10))
        #     return evecs @ jnp.diag(sqrt_evals) @ evecs.T

        # P_sqrt = sqrt(P)
        # Q_sqrt = sqrt(Q)
        # return KroneckerMetric(P, Q, P_sqrt, Q_sqrt, jnp.linalg.inv(P_sqrt), jnp.linalg.inv(Q_sqrt))
        return KroneckerMetric(P, Q, jnp.linalg.cholesky(P), jnp.linalg.cholesky(Q), jnp.linalg.inv(jnp.linalg.cholesky(P)), jnp.linalg.inv(jnp.linalg.cholesky(Q)))

@dataclass
class DictionaryMetricEstimator(MetricEstimator):
    metrics: Dict[str, MetricEstimator]
    def __call__(self, samples: Dict[str, jax.Array]) -> DictionaryMetric:
        out = {k: self.metrics[k](samples[k]) for k in self.metrics.keys()}
        return DictionaryMetric(out)

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

    kme = KroneckerMetricEstimator()
    Yn = jax.random.normal(jax.random.PRNGKey(0), [1000,2,2])
    Y = Yn.at[:,1,:].add(Yn[:,0,:])
    m = kme(Y)

    def cov(xs):
        return (xs[:,:,None] * xs[:,None,:]).mean(0)

    w = jax.vmap(m.whiten)(Y)
    uwu = jax.vmap(m.unwhiten)(w)
    print('w:', cov(w))
    print('uwu:', cov(uwu.reshape(-1,4)))
    print(cov(Yn.reshape(-1,4)))
    print(cov(Y.reshape(-1,4)))

    # @jax.jit
    # def logp(theta):
    #     return -0.5 * jnp.sum(jnp.square(theta/jnp.array([1,200])))

    # rng = PRNG(jax.random.PRNGKey(0))
    # theta = jax.random.normal(rng.split(), [2])

    # def k_nuts(key, theta, ε):
    #     return metric_nuts(key, theta, logp, ε, ScalarMetric(jnp.array([1.,200.]), [2]))


    # @jax.jit
    # def sample(key, theta, ε):
    #     @scan_tqdm(30000)
    #     def body_fn(carry, i):
    #         key, theta = carry
    #         key, subkey = jax.random.split(key)
    #         theta, alpha = k_nuts(subkey, theta, ε)
    #         return (key, theta), (theta, alpha)
    #     return jax.lax.scan(body_fn, (key, theta), jnp.arange(30000))[1]


    # samples, alphas = sample(rng.split(), theta, 1.0)
    # samples = jnp.stack(samples)
    # alphas = jnp.stack(alphas)
    # print(samples.shape)
    # print(samples.std(0), jax.random.normal(rng.split(),samples.shape).std(0))
    # print(alphas.mean())
    # # scatter plot
    # plt.scatter(samples[:,0], samples[:,1])
    # plt.plot(samples[:,0], samples[:,1], color='red', alpha=0.1)
    # plt.show()

