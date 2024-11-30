import sys, os

import jax
import jax.numpy as jnp
import optax
import numpy as np
import math
from matplotlib import pyplot as plt
from jax_tqdm import scan_tqdm
from functools import partial
from jax.tree_util import tree_map
from dataclasses import dataclass
import eindex.array_api as EX
from typing import List, Tuple, Any
import einops

from jaxtorch import nn
from jaxtorch import PRNG, Context
import jaxtorch

from ngp.log_h import log_h
from ngp.metric import Metric, MetricEstimator, treeformat
from ngp.nuts import nuts_kernel
from ngp.adapt import warmup_with_dual_averaging, find_reasonable_epsilon
from ngp.ahmc import ahmc_fast, sample_hmc
from ngp.util import Fn, Partial

import optuna

@jax.tree_util.register_pytree_node_class
@dataclass
class TrivialMetric(Metric):
    def whiten(self, x):
        return x
    
    def unwhiten(self, x):
        return x

    def tree_flatten(self):
        return (), ()
    @staticmethod
    def tree_unflatten(static, dynamic):
        return TrivialMetric(*static, *dynamic)

@jax.jit
def treeformat_init(init, key):
    return treeformat(init(key))

def ahmc_with_metric(key: jax.Array, theta: Any, logp: Fn, warmup_steps: int, εmin, εmax, Lmin, Lmax, metric: Metric):
    tf = treeformat(theta)
    def logp_flat(flat_theta):
        theta = metric.unwhiten(tf.unflatten(flat_theta))
        return logp(theta)
    (flat_theta, ε, L, info) = ahmc_fast(key, tf.flatten(metric.whiten(theta)), logp_flat, warmup_steps, εmin, εmax, Lmin, Lmax)
    return (metric.unwhiten(tf.unflatten(flat_theta)), ε, L, info)

def sample_hmc_with_metric(key, theta, logp, n_steps, ε, L, metric):
    tf = treeformat(theta)
    def logp_flat(flat_theta):
        theta = metric.unwhiten(tf.unflatten(flat_theta))
        return logp(theta)
    (flat_theta, flat_chain, ms) = sample_hmc(key, tf.flatten(metric.whiten(theta)), logp_flat, n_steps, ε, L)
    def unflat(x):
        return metric.unwhiten(tf.unflatten(x))
    return (unflat(flat_theta), jax.vmap(unflat)(flat_chain), ms)

def sample_adaptive(key, logp: Fn, init: Fn, n_chains=4, n_samples=1000, metric0: Metric = None, metric_estimator: MetricEstimator = None):
    if metric0 is None:
        metric0 = TrivialMetric()
    metric = tree_map(lambda x: einops.repeat(jnp.asarray(x), '... -> n ...', n=n_chains), metric0)

    def v_ahmc_fast(key, theta, metric, εmin=1e-6, εmax=1.0):
        return jax.vmap(ahmc_with_metric, in_axes=(0,0,None,None,None,None,None,None,0))(key, theta, logp, 100, εmin, εmax, 2, 2000, metric)
    
    def v_sample_hmc(key, theta, metric, n_steps, ε, L):
        return jax.vmap(sample_hmc_with_metric, in_axes=(0,0,None,None,0,0,0))(key, theta, logp, n_steps, ε, L, metric)

    # @partial(jax.vmap, in_axes=(0,0,0,None,None))
    # def v_ahmc_fast_(key, eps, metric, εmin, εmax):
    #     return ahmc_fast(key, eps, Partial(eps_logp, metric), 100, εmin=εmin, εmax=εmax, Lmin = 2, Lmax = 2000)
    
    rng = PRNG(key)
    theta = jax.vmap(init)(rng.split(n_chains))

    (theta, ε, L, info1) = v_ahmc_fast(rng.split(n_chains), theta, metric, 1e-6, 1.0)
    print(f'Finished first warmup with ε={ε} and L={L}, d={info1['d'].mean(1)}, logp={info1['logp'][:,-1]}')

    print('Starting initial chain for mass adaptation...')
    theta, chain, ms = v_sample_hmc(rng.split(n_chains), theta, metric, 100, ε, L)

    metric = jax.vmap(metric_estimator)(chain)
    print('Updated mass metric.')
    print('new metric:', metric)

    (theta, ε, L, info2) = v_ahmc_fast(rng.split(n_chains), theta, metric, 1e-6, 1.0)
    print(f'Finished second warmup with ε={ε} and L={L}, d={info2['d'].mean(1)}, logp={info2['logp'][:,-1]}')

    # print('Third adaption cycle...')
    # eps, chain, ms = jax.vmap(sample_hmc, in_axes=(0,0,0,None,0,0))(rng.split(n_chains), eps, Partial(eps_logp, metric), 100, ε, L)
    # eps, metric, pchain = remetric(metric, eps, chain)
    # (eps, ε, L, info2) = v_ahmc_fast(rng.split(n_chains), eps, metric, 1e-6, 1.0)
    # print(f'Finished third warmup with ε={ε} and L={L}, d={info2['d'].mean(1)}, logp={info2['logp'][:,-1]}')

    print('Finally sampling...')
    theta, chain, ms = v_sample_hmc(rng.split(n_chains), theta, metric, n_samples, ε, L)
    probs = jax.vmap(jax.vmap(logp))(chain)
    return chain, {'probs': probs, 'alphas': ms['alpha'], 'info1': info1, 'info2': info2}