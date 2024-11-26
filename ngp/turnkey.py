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
from typing import List, Tuple
import einops

from jaxtorch import nn
from jaxtorch import PRNG, Context
import jaxtorch

from ngp.log_h import log_h
from ngp.metric import Metric, MetricEstimator
from ngp.nuts import nuts_kernel
from ngp.adapt import warmup_with_dual_averaging, find_reasonable_epsilon
from ngp.ahmc import ahmc_fast, sample_hmc
from ngp.util import Fn, Partial

import optuna

@jax.tree_util.register_pytree_node_class
@dataclass
class TrivialMetric(Metric):
    """Doesn't rescale anything, just flattens a pytree."""
    treedef: jax.tree_util.PyTreeDef
    shapes: List[List[int]]

    def whiten(self, x):
        leaves = jax.tree_util.tree_leaves(x)
        return [leaf.reshape(-1) for leaf in leaves]
    
    def unwhiten(self, x):
        leaves = []
        ix = 0
        for shape in self.shapes:
            leaves.append(x[ix:ix+np.prod(shape)].reshape(shape))
            ix += np.prod(shape)
        return jax.tree_util.tree_unflatten(self.treedef, leaves)

    def tree_flatten(self):
        return (), (self.treedef, self.shapes)
    @staticmethod
    def tree_unflatten(static, dynamic):
        return TrivialMetric(*static, *dynamic)

@jax.jit
def trivial_metric(init: Fn, key: jax.Array):
    xs = init(key)
    leaves, treedef = jax.tree_util.tree_flatten(xs)
    shapes = [x.shape for x in leaves]
    return TrivialMetric(treedef, shapes)

def sample_adaptive(key, logp: Fn, init: Fn, n_chains=4, n_samples=1000, metric0: Metric = None, metric_estimator: MetricEstimator = None):
    if metric0 is None:
        metric0 = trivial_metric(init, key)
    metric = jax.tree_util.tree_map(lambda x: einops.repeat(jnp.asarray(x), '... -> n ...', n=n_chains), metric0)

    def eps_logp(metric, eps):
        params = metric.unwhiten(eps)
        return logp(params)

    def v_ahmc_fast(key, eps, metric, εmin=1e-6, εmax=1.0):
        return v_ahmc_fast_(key, eps, metric, εmin, εmax)
    @partial(jax.vmap, in_axes=(0,0,0,None,None))
    def v_ahmc_fast_(key, eps, metric, εmin, εmax):
        return ahmc_fast(key, eps, Partial(eps_logp, metric), 100, εmin=εmin, εmax=εmax, Lmin = 2, Lmax = 2000)

    @jax.vmap
    def remetric(metric, eps, chain):
        params = metric.unwhiten(eps)
        pchain = jax.vmap(metric.unwhiten)(chain)
        metric = metric_estimator(pchain)
        eps = metric.whiten(params)
        return eps, metric, pchain
    
    rng = PRNG(key)
    params = jax.vmap(init)(rng.split(n_chains))

    eps = jax.vmap(lambda m,c: m.whiten(c))(metric, params)
    # ε, εp = jax.vmap(find_reasonable_epsilon, in_axes=(0,0,None))(rng.split(n_chains), eps, Partial(eps_logp, metric0))
    # print(f'Found step size {ε} with acceptance probability {εp}')
    (eps, ε, L, info1) = v_ahmc_fast(rng.split(n_chains), eps, metric, 1e-6, 1.0)
    print(f'Finished first warmup with ε={ε} and L={L}, d={info1['d'].mean(1)}, logp={info1['logp'][:,-1]}')

    print('Starting initial chain for mass adaptation...')
    eps, chain, ms = jax.vmap(sample_hmc, in_axes=(0,0,0,None,0,0))(rng.split(n_chains), eps, Partial(eps_logp, metric), 100, ε, L)

    eps, metric, pchain = remetric(metric, eps, chain)
    print('Updated mass metric.')

    print('new metric:', metric)

    (eps, ε, L, info2) = v_ahmc_fast(rng.split(n_chains), eps, metric, 1e-6, 1.0)
    print(f'Finished second warmup with ε={ε} and L={L}, d={info2['d'].mean(1)}, logp={info2['logp'][:,-1]}')

    # print('Third adaption cycle...')
    # eps, chain, ms = jax.vmap(sample_hmc, in_axes=(0,0,0,None,0,0))(rng.split(n_chains), eps, Partial(eps_logp, metric), 100, ε, L)
    # eps, metric, pchain = remetric(metric, eps, chain)
    # (eps, ε, L, info2) = v_ahmc_fast(rng.split(n_chains), eps, metric, 1e-6, 1.0)
    # print(f'Finished third warmup with ε={ε} and L={L}, d={info2['d'].mean(1)}, logp={info2['logp'][:,-1]}')

    print('Finally sampling...')
    eps, chain, ms = jax.vmap(sample_hmc, in_axes=(0,0,0,None,0,0))(rng.split(n_chains), eps, Partial(eps_logp, metric), n_samples, ε, L)

    @jax.vmap
    def finish_chain(metric, chain):
        return jax.vmap(metric.unwhiten)(chain)
    chain = finish_chain(metric, chain)
    probs = jax.vmap(jax.vmap(logp))(chain)
    return chain, {'probs': probs, 'alphas': ms['alpha'], 'info1': info1, 'info2': info2, 'pchain': pchain}