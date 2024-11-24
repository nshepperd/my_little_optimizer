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
    mean: jax.Array
    
    def ndim(self):
        m = self.P.shape[0]
        n = self.Q.shape[0]
        return m*n

    def whiten(self, W: jax.Array) -> jax.Array:
        m = self.P.shape[0]
        n = self.Q.shape[0]
        assert list(W.shape) == [m, n]
        return (self.P_invsqrt @ (W - self.mean) @ self.Q_invsqrt.T).reshape(-1)
        
    def unwhiten(self, W_white: jax.Array) -> jax.Array:
        m = self.P.shape[0]
        n = self.Q.shape[0]
        return (self.P_sqrt @ W_white.reshape([m,n]) @ self.Q_sqrt.T) + self.mean

    def tree_flatten(self):
        return (self.P, self.Q, self.P_sqrt, self.Q_sqrt, self.P_invsqrt, self.Q_invsqrt, self.mean), ()
    @staticmethod
    def tree_unflatten(static, dynamic):
        return KroneckerMetric(*static, *dynamic)

@jax.tree_util.register_pytree_node_class
@dataclass
class FullCovMetric(Metric):
    M: jax.Array
    L: jax.Array
    # L_inv: jax.Array
    mean: jax.Array
    # def __init__(self, Sigma: Array):
    #     self.L = jnp.linalg.cholesky(Sigma)
    #     self.L_inv = jnp.linalg.inv(self.L)
    
    def ndim(self):
        return self.M.shape[0]

    def whiten(self, x: jax.Array) -> jax.Array:
        return jax.scipy.linalg.solve_triangular(self.L, (x - self.mean), lower=True).reshape(-1)
        # return (self.L_inv @ (x - self.mean)).reshape(-1)
        
    def unwhiten(self, x_white: jax.Array) -> jax.Array:
        m = self.M.shape[0]
        return self.L @ x_white.reshape(m) + self.mean
    
    def tree_flatten(self):
        return (self.M, self.L, self.mean), ()
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
        mean = jnp.mean(samples, axis=0)
        samples -= mean
        cov = jnp.mean(samples[:,None,:] * samples[:,:,None], axis=0)
        L = jnp.linalg.cholesky(cov)
        # L_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(cov.shape[0]), lower=True)
        return FullCovMetric(cov, L, mean)

class KroneckerMetricEstimator(MetricEstimator):
    def __call__(self, samples: jax.Array) -> KroneckerMetric:
        assert len(samples.shape) == 3
        mean = jnp.mean(samples, axis=0)
        samples -= mean
        P, Q = kfac(samples)
        P_sqrt = jnp.linalg.cholesky(P)
        Q_sqrt = jnp.linalg.cholesky(Q)
        P_invsqrt = jax.scipy.linalg.solve_triangular(P_sqrt, jnp.eye(P.shape[0]), lower=True)
        Q_invsqrt = jax.scipy.linalg.solve_triangular(Q_sqrt, jnp.eye(Q.shape[0]), lower=True)
        return KroneckerMetric(P, Q, P_sqrt, Q_sqrt, P_invsqrt, Q_invsqrt, mean)

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

    # check kronecker metric
    kme = KroneckerMetricEstimator()
    Yn = jax.random.normal(jax.random.PRNGKey(0), [1000,2,2])
    A, B = 100*jax.random.normal(jax.random.PRNGKey(2), [2,2,2])
    print('A:', A)
    print('B:', B)
    Y = jax.vmap(lambda Y: A @ Y @ B)(Yn) + 100
    m = kme(Y)

    w = jax.vmap(m.whiten)(Y)
    uwu = jax.vmap(m.unwhiten)(w)
    print('max difference in roundtrip reconstruction:', jnp.max(jnp.abs(Y - uwu)))

    def cov(xs):
        return (xs[:,:,None] * xs[:,None,:]).mean(0)
    
    print('covariance of whitened data:', cov(w))

    # check full covariance metric
    print('CovarianceMetricEstimator:')
    fme = CovarianceMetricEstimator()
    Yn = jax.random.normal(jax.random.PRNGKey(0), [100,4])
    A = jax.random.normal(jax.random.PRNGKey(2), [4,4])
    print('A:', A)
    Y = Yn @ A + 1
    m = fme(Y)

    w = jax.vmap(m.whiten)(Y)
    uwu = jax.vmap(m.unwhiten)(w)
    print('max difference in roundtrip reconstruction:', jnp.max(jnp.abs(Y - uwu)))
    print('covariance of whitened data:', cov(w))
