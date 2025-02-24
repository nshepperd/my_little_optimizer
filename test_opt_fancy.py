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
import einops

from jaxtorch import nn
from jaxtorch import PRNG, Context
import jaxtorch

from my_little_optimizer.optim import SpaceItem, Optim
from my_little_optimizer.util import Partial

import optuna

if __name__ == '__main__':
    SWEEP_NAME='21-wooden-sweep'
    study = optuna.create_study(
        study_name=SWEEP_NAME,
        load_if_exists=True,
        storage="sqlite:///../vqgan-jax/optuna.db",
        direction="minimize",
        sampler=optuna.samplers.GPSampler(),
    )

    pars = list(study.trials[0].params.keys())
    dists = study.trials[0].distributions
    opt = Optim([SpaceItem(k, dists[k].low, dists[k].high, log=True)for k in pars])

    trials = [trial for trial in study.trials if trial.state==1]
    all_y = jnp.array([trial.value for trial in trials])
    max_y = jnp.max(all_y[jnp.isfinite(all_y)])

    for trial in trials:
        params = trial.params
        value = jnp.where(jnp.isfinite(trial.value), trial.value, max_y)
        opt.notify(params, value)

    # opt.infer()
    # jaxtorch.pt.save(opt.fitted.chains, 'chains.pt')

    from my_little_optimizer.optim import FittedMLP
    chains = jaxtorch.pt.load('chains.pt')
    opt.fitted = FittedMLP(opt.model.model, chains)

    # params = opt.suggest({})
    # print('Suggested params for next trial:', params)
    params = opt.suggestbest({})
    print('Suggested best params:', params)

    key = 'max_lr'

    xs = jnp.linspace(-1, 1, 100)
    xs = opt.space[key].denormalize(xs)
    def pred(x):
        params = opt.suggestbest({key: x}, method='cma-es')
        mean, std = opt.fitted.predict(opt.space.normalize(params)[None])
        return mean.squeeze(), std.squeeze()
    means, stds = jax.vmap(pred)(xs)
    plt.xscale('log')
    plt.plot(xs, means, color='blue')
    plt.fill_between(xs, means - stds, means + stds, alpha=0.2, color='blue')
    plt.fill_between(xs, means - 2*stds, means + 2*stds, alpha=0.2, color='blue')
    plt.fill_between(xs, means - 3*stds, means + 3*stds, alpha=0.2, color='blue')
    plt.scatter([t.params[key] for t in trials], [t.value for t in trials], label='observed')
    plt.scatter([t.params[key] for t in trials], [opt.fitted.predict(opt.space.normalize(t.params))[0] for t in trials], label='prediction(observed)')

    plt.legend()
    plt.show()
