import sys, os
sys.path.append('.')

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
from typing import List, Dict
import einops

from jaxtorch import nn
from jaxtorch import PRNG, Context
import jaxtorch

from my_little_optimizer.log_h import log_h
from my_little_optimizer.metric import DictionaryMetric, ScalarMetric, Metric, MetricEstimator
from my_little_optimizer.metric import DictionaryMetricEstimator, CovarianceMetricEstimator, KroneckerMetricEstimator
from my_little_optimizer.util import Partial
from my_little_optimizer.turnkey import sample_adaptive
from my_little_optimizer.gaussian_process import GP

model = nn.Sequential([nn.Linear(1, 10), nn.Tanh(), nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 4)]) 
model.name_everything_()

def init(key):
    rng = PRNG(key)
    params = {}
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            params[mod.weight.name] = jax.random.normal(rng.split(), mod.weight.shape) * 1/jnp.sqrt(np.prod(mod.weight.shape[1:]))
            params[mod.bias.name] = jax.random.normal(rng.split(), mod.bias.shape) * 1
    return params


# params = jax.vmap(init)(jax.random.split(jax.random.PRNGKey(0), 10))
# xs = jnp.linspace(-1, 1, 100)

# def fwd(params, x):
#     cx = Context(params, None)
#     return model(cx, x[:,None]).squeeze(-1)

# out = jax.vmap(fwd, in_axes=(0,None))(params, xs)

xs = jnp.linspace(-1, 1, 100)


def fwd(key):
    rng = PRNG(key)
    α = jax.random.uniform(rng.split(), [])
    σ = 0.1 * jax.random.gamma(rng.split(), shape=[], a=0.5)**-0.5
    δ = 0.01 * jnp.exp(jax.random.normal(rng.split()) + jnp.log(0.1))
    # params = init(key1)
    # cx = Context(params, None)
    # ws = model(cx, xs[:,None])
    
    def rbf(x, y):
        return α * jnp.exp(-0.5 * jnp.sum(jnp.square(x-y)/σ**2)) + (1 - α) * (x*y).sum(axis=-1).square()
    gp = GP(rbf, δ)
    return gp.sample_prior(rng.split(), xs[:,None])


ys = jax.vmap(fwd)(jax.random.split(jax.random.PRNGKey(0), 10))

for i in range(ys.shape[0]):
    plt.plot(xs, ys[i])
plt.show()