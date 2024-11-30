import jax
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple, Protocol, Dict, Any, List, TypeVar, Generic
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm

from ngp.nuts import nuts_kernel
from ngp.kfac import kfac

T = TypeVar('T')

class Metric(Generic[T]):
    def whiten(self, x: T) -> T:
        raise NotImplementedError
    
    def unwhiten(self, x: T) -> T:
        raise NotImplementedError

Shape = List[int]

@jax.tree_util.register_pytree_node_class
@dataclass
class ScalarMetric(Metric[jax.Array]):
    a: float
    def whiten(self, array):
        return array / self.a
    def unwhiten(self, array):
        return array * self.a
    
    def tree_flatten(self):
        return (self.a,), ()
    @staticmethod
    def tree_unflatten(static, dynamic):
        return ScalarMetric(*dynamic, *static)

@jax.tree_util.register_pytree_node_class
@dataclass
class KroneckerMetric(Metric[jax.Array]):
    P_sqrt: jax.Array
    Q_sqrt: jax.Array
    P_invsqrt: jax.Array
    Q_invsqrt: jax.Array
    mean: jax.Array

    def whiten(self, W: jax.Array) -> jax.Array:
        return (self.P_invsqrt @ (W - self.mean) @ self.Q_invsqrt.T)
        
    def unwhiten(self, W_white: jax.Array) -> jax.Array:
        return (self.P_sqrt @ W_white @ self.Q_sqrt.T) + self.mean

    def tree_flatten(self):
        return (self.P_sqrt, self.Q_sqrt, self.P_invsqrt, self.Q_invsqrt, self.mean), ()
    @staticmethod
    def tree_unflatten(static, dynamic):
        return KroneckerMetric(*static, *dynamic)

@jax.tree_util.register_pytree_node_class
@dataclass
class FullCovMetric(Metric[jax.Array]):
    L: jax.Array
    mean: jax.Array

    def whiten(self, x: jax.Array) -> jax.Array:
        return jax.scipy.linalg.solve_triangular(self.L, (x - self.mean), lower=True)
        
    def unwhiten(self, x_white: jax.Array) -> jax.Array:
        return self.L @ x_white + self.mean
    
    def tree_flatten(self):
        return (self.L, self.mean), ()
    @staticmethod
    def tree_unflatten(static, dynamic):
        return FullCovMetric(*static, *dynamic)

@jax.tree_util.register_pytree_node_class
@dataclass
class DictionaryMetric(Metric[Dict[str,Any]]):
    metrics: Dict[str, Metric]
    
    def whiten(self, xs: Dict[str, Any]) -> Dict[str, Any]:
        assert set(xs.keys()) == set(self.metrics.keys())
        return {k: self.metrics[k].whiten(v) for k,v in xs.items()}
    
    def unwhiten(self, xs: Dict[str, Any]) -> Dict[str, Any]:
        assert set(xs.keys()) == set(self.metrics.keys())
        return {k: self.metrics[k].unwhiten(v) for k,v in xs.items()}
    
    def tree_flatten(self):
        return (self.metrics,), ()
    @staticmethod
    def tree_unflatten(static, dynamic):
        return DictionaryMetric(*static, *dynamic)

class MetricEstimator(Protocol, Generic[T]):
    """Produces a metric from a set of samples.

    Expects the input to be an array of examples with shape [n, d1, ..., dn] where n is the number of samples.
    Does not support broadcasting.
    """
    def __call__(self, samples: T) -> Metric[T]: ...

@dataclass
class ConstantMetricEstimator(MetricEstimator[T]):
    metric: Metric[T]
    def __call__(self, samples: T) -> Metric[T]:
        return self.metric

class StdScalarMetricEstimator(MetricEstimator[jax.Array]):
    def __call__(self, samples: jax.Array) -> ScalarMetric:
        samples = samples.reshape([samples.shape[0], -1])
        samples -= samples.mean(0)
        return ScalarMetric(jnp.sqrt(jnp.mean(jnp.square(samples))))

class CovarianceMetricEstimator(MetricEstimator[jax.Array]):
    def __call__(self, samples: jax.Array) -> FullCovMetric:
        assert len(samples.shape) == 2
        mean = jnp.mean(samples, axis=0)
        samples -= mean
        cov = jnp.mean(samples[:,None,:] * samples[:,:,None], axis=0)
        L = jnp.linalg.cholesky(cov)
        # L_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(cov.shape[0]), lower=True)
        return FullCovMetric(L, mean)

class KroneckerMetricEstimator(MetricEstimator[jax.Array]):
    def __call__(self, samples: jax.Array) -> KroneckerMetric:
        assert len(samples.shape) == 3
        mean = jnp.mean(samples, axis=0)
        samples -= mean
        P, Q = kfac(samples)
        P_sqrt = jnp.linalg.cholesky(P)
        Q_sqrt = jnp.linalg.cholesky(Q)
        P_invsqrt = jax.scipy.linalg.solve_triangular(P_sqrt, jnp.eye(P.shape[0]), lower=True)
        Q_invsqrt = jax.scipy.linalg.solve_triangular(Q_sqrt, jnp.eye(Q.shape[0]), lower=True)
        return KroneckerMetric(P_sqrt, Q_sqrt, P_invsqrt, Q_invsqrt, mean)

@dataclass
class DictionaryMetricEstimator(MetricEstimator[Dict[str, Any]]):
    metrics: Dict[str, MetricEstimator]
    def __call__(self, samples: Dict[str, Any]) -> DictionaryMetric:
        out = {k: self.metrics[k](samples[k]) for k in self.metrics.keys()}
        return DictionaryMetric(out)

@jax.tree_util.register_static
@dataclass
class TreeFormat(object):
    treedef: jax.tree_util.PyTreeDef
    shapes: List[List[int]]

    def flatten(self, tree):
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        assert treedef == self.treedef
        assert len(leaves) == len(self.shapes)
        return jnp.concatenate([x.reshape(-1) for x in leaves], axis=0)

    def unflatten(self, array):
        ix = 0
        out = []
        for shape in self.shapes:
            out.append(array[ix:ix+np.prod(shape)].reshape(shape))
            ix += np.prod(shape)
        return jax.tree_util.tree_unflatten(self.treedef, out)

def treeformat(tree) -> TreeFormat:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    shapes = [x.shape for x in leaves]
    return TreeFormat(treedef, shapes)

def metric_nuts(key, params, logp, ε, metric):
    tf = treeformat(params)
    theta = tf.flatten(metric.whiten(params))
    def logp_theta(theta):
        params = metric.unwhiten(tf.unflatten(theta))
        return logp(params)
    theta_new, alpha, ms = nuts_kernel(key, theta, logp_theta, jax.grad(logp_theta), ε)
    return metric.unwhiten(tf.unflatten(theta_new)), alpha, ms

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
