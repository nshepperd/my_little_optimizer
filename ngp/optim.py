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
from typing import List, Dict, Tuple, Protocol
import einops

from jaxtorch import nn
from jaxtorch import PRNG, Context
import jaxtorch

from ngp.log_h import log_h
from ngp.metric import DictionaryMetric, ScalarMetric, Metric, MetricEstimator
from ngp.metric import DictionaryMetricEstimator, CovarianceMetricEstimator, KroneckerMetricEstimator
from ngp.util import Partial
from ngp.turnkey import sample_adaptive
from ngp.ahmc import ahmc_fast

@dataclass
class SpaceItem:
    name: str
    min: float
    max: float
    log: bool = False

    def normalize(self, x):
        if self.log:
            x = jnp.log(x)
            return (x - jnp.log(self.min)) / (jnp.log(self.max) - jnp.log(self.min))
        return (x - self.min) / (self.max - self.min)
    
    def denormalize(self, x):
        if self.log:
            return jnp.exp(x * (jnp.log(self.max) - jnp.log(self.min)) + jnp.log(self.min))
        return x * (self.max - self.min) + self.min


@dataclass
class Space:
    keys: List[str]
    items: Dict[str, SpaceItem]
    n: int

    def normalize(self, params):
        return jnp.stack([self.items[k].normalize(params[k]) for k in self.keys], axis=-1)
    
    def denormalize(self, array):
        return {k:self.items[k].denormalize(array[..., i]) for i,k in enumerate(self.keys)}

@dataclass
class Trial:
    params: Dict[str, float]
    value: float

class FittedModel(Protocol):
    def predict(self, xs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Predict mean and std of the model at the given points."""
        ...

class FittableModel(Protocol):
    def fit(self, xs: jax.Array, ys: jax.Array) -> FittedModel:
        ...

@jax.tree_util.register_pytree_node_class
class FittedMLP(FittedModel):
    def __init__(self, model: nn.Module, chains: dict):
        self.model = model
        self.chains = chains
    def predict(self, xs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        ys = v_fwd(self.model, self.chains, xs)
        return ys.mean(0).squeeze(), ys.std(0).squeeze()
    
    def tree_flatten(self):
        return (self.chains,), (self.model,)
    @staticmethod
    def tree_unflatten(static, dynamic):
        model, = static
        chains, = dynamic
        return FittedMLP(model, chains)

@jax.tree_util.register_static
class MLPWithNoise(FittableModel):
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.name_everything_()
    
    @jax.jit
    def fit(self, key: jax.Array, xs: jax.Array, ys: jax.Array) -> FittedMLP:
        scales = mkscales(self.model)

        def init_state(key):
            return {'params': self.model.init_weights(key), 
                    'log_sigma': jnp.array([jnp.log(0.1)])}

        def logp(xs, ys, state):
            params = state['params']
            log_sigma = state['log_sigma'].squeeze()
            sigma = jnp.exp(log_sigma).squeeze()
            cx = Context(params, None)
            prediction = self.model(cx, xs).squeeze(-1)
            log_p_y = -0.5 * jnp.sum(jnp.square(ys - prediction)/sigma**2)
            log_p_y += -0.5 * np.prod(ys.shape) * (jnp.log(2 * jnp.pi) + log_sigma*2)
            # laplace prior
            log_p_params = sum(-0.5 * jnp.sum(jnp.abs(params[p.name]/scales[p.name])) for p in self.model.parameters())
            mu_log_sigma = jnp.log(0.1)
            std_log_sigma = jnp.log(10.0)
            log_p_sigma = -0.5 * jnp.square((log_sigma - mu_log_sigma)/std_log_sigma)
            return log_p_y + log_p_params + log_p_sigma
        
        def full_hessian_metric(xs, ys, samples: dict) -> Metric:
            f = partial(logp, xs, ys)
            H = jax.vmap(jax.hessian(f))(samples)

            def matmetric(Hpart, shape):
                Hpart = Hpart.reshape(Hpart.shape[0], np.prod(shape), np.prod(shape))
                d = jnp.diagonal(Hpart, axis1=1, axis2=2)
                scale = jnp.mean(jnp.square(d))**-0.25
                return ScalarMetric(scale)

            out = {}
            for p in self.model.parameters():
                out[p.name] = matmetric(H['params'][p.name]['params'][p.name], p.shape)
            return DictionaryMetric({'params': DictionaryMetric(out), 'log_sigma': matmetric(H['log_sigma']['log_sigma'], [1])})
        
        chains, ms = sample_adaptive(key,
                                    logp = Partial(logp, xs, ys),
                                    init = init_state,
                                    n_chains = 4, n_samples = 100,
                                    metric0 = DictionaryMetric({'params': mkmetric(self.model), 'log_sigma': ScalarMetric(jnp.array(jnp.log(2.0)))}),
                                    metric_estimator = Partial(full_hessian_metric, xs, ys))
        
        # Save chains for later use.
        chains = tree_map(lambda x: einops.rearrange(x, 'c n ... -> (c n) ...'), chains)
        accept_probs = ms['alphas'].mean()
        print('acceptance rate:', accept_probs)

        return FittedMLP(self.model, chains)
    
class Optim:
    space: Space
    trials: List[Trial]

    def __init__(self, space: List[SpaceItem]):
        self.space = Space(keys=[item.name for item in space], items={item.name: item for item in space}, n=len(space))
        self.trials = []
        self.in_dim = len(self.space.keys)
        self.model = MLPWithNoise(nn.Sequential([nn.Linear(self.in_dim, 10), nn.LeakyReLU(), nn.Linear(10, 1)]))
        self.fitted = None
        # self.model.name_everything_()
        self.rng = PRNG(jax.random.PRNGKey(0))
        # self.chains = None

    def notify(self, params, value):
        self.trials.append(Trial(params, value))
    
    def infer(self):
        xs = jnp.stack([self.space.normalize(x.params) for x in self.trials])
        ys = jnp.stack([x.value for x in self.trials])

        print('xs/ys:', xs, ys)

        self.fitted = self.model.fit(jax.random.PRNGKey(0), xs, ys)

        # scales = mkscales(self.model)
        # def f_w_beta(x, y, state):
        #     params = state['params']
        #     log_sigma = state['log_sigma'].squeeze()
        #     sigma = jnp.exp(log_sigma).squeeze()
        #     cx = Context(params, None)
        #     prediction = self.model(cx, x).squeeze(-1)
        #     # sigma = 0.05
        #     log_p_y = -0.5 * jnp.sum(jnp.square(y - prediction)/sigma**2)
        #     log_p_y += -0.5 * np.prod(y.shape) * (jnp.log(2 * jnp.pi) + log_sigma*2)
        #     # log_p_params = sum(-0.5 * jnp.sum(jnp.square(params[p.name]/scales[p.name])) for p in self.model.parameters())
        #     # laplace prior
        #     log_p_params = sum(-0.5 * jnp.sum(jnp.abs(params[p.name]/scales[p.name])) for p in self.model.parameters())
        #     mu_log_sigma = jnp.log(0.1)
        #     std_log_sigma = jnp.log(10.0)
        #     log_p_sigma = -0.5 * jnp.square((log_sigma - mu_log_sigma)/std_log_sigma)
        #     return log_p_y + log_p_params + log_p_sigma

        # def full_hessian_metric(x, y, samples: dict) -> Metric:
        #     f = partial(f_w_beta, x, y)
        #     H = jax.vmap(jax.hessian(f))(samples)

        #     def matmetric(Hpart, shape):
        #         Hpart = Hpart.reshape(Hpart.shape[0], np.prod(shape), np.prod(shape))
        #         d = jnp.diagonal(Hpart, axis1=1, axis2=2)
        #         scale = jnp.mean(jnp.square(d))**-0.25
        #         return ScalarMetric(scale)

        #     out = {}
        #     for p in self.model.parameters():
        #         out[p.name] = matmetric(H['params'][p.name]['params'][p.name], p.shape)
        #     return DictionaryMetric({'params': DictionaryMetric(out), 'log_sigma': matmetric(H['log_sigma']['log_sigma'], [1])})

        # chains, ms = sample_adaptive(jax.random.PRNGKey(0),
        #                             logp = Partial(f_w_beta, xs, ys),
        #                             init = Partial(lambda key: {'params': self.model.init_weights(key), 'log_sigma': jnp.array([jnp.log(0.1)])}),
        #                             n_chains = 4, n_samples = 100,
        #                             metric0 = DictionaryMetric({'params': mkmetric(self.model), 'log_sigma': ScalarMetric(jnp.array(jnp.log(2.0)))}),
        #                             metric_estimator = Partial(full_hessian_metric, xs, ys))
        
        # # Save chains for later use.
        # self.chains = tree_map(lambda x: einops.rearrange(x, 'c n ... -> (c n) ...'), chains)
        # accept_probs = ms['alphas'].mean()
        # print('acceptance rate:', accept_probs)

    def suggest(self, params):
        if self.fitted is None:
            # just choose random points
            v = jax.random.uniform(self.rng.split(), [self.space.n], minval=0.0, maxval=1.0)
            return self.space.denormalize(v)

        waterline = max([x.value for x in self.trials])
        self.acf = log_expected_improvement_max(self.fitted, waterline)
        # self.acf = ucb(self.model, self.chains)
        mask = jnp.array([k in params for k in self.space.keys])
        maskval = jnp.array([self.space.items[k].normalize(params[k]) if k in params else 0.0 for k in self.space.keys])
        xmax = findmax_ahmc(self.rng.split(), self.acf, jnp.zeros([self.space.n]), jnp.ones([self.space.n]))
        # xmax, metrics = findmax(self.rng.split(), acf, self.space.n, mask=mask, maskval=maskval, n=4)
        if jnp.isnan(xmax).any():
            print('nans in xmax:', xmax)
            raise AssertionError("AAAA")
        xmax_ac = jnp.exp(self.acf(xmax))
        print('waterline:', waterline.shape, waterline)
        print('xmax_ac:', xmax_ac.shape, xmax_ac)
        mean, std = self.fitted.predict(xmax)
        print('means:', mean.shape, mean)
        print('stds:', std.shape, std)
        print(xmax.shape)

        xvalue = self.space.denormalize(xmax)
        print('xvalue:', xvalue)
        print('expected score:', mean)

        return xvalue

def mkscales(model):
    scales = {}
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            scales[mod.weight.name] = 1.0/math.sqrt(np.prod(mod.weight.shape[1:]))
            scales[mod.bias.name] = 1.0
    return scales

def mkmetric(model):
    scales = mkscales(model)
    metrics = {}
    for p in model.parameters():
        metrics[p.name] = ScalarMetric(scales[p.name])
    return DictionaryMetric(metrics)

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None,0,None))
def v_fwd(model, chains, x: jax.Array) -> jax.Array:
    cx = Context(chains['params'], None)
    return model(cx, x)

def log_expected_improvement(model, chains, waterline, eps=0.01):
    """Log expected improvement acquisition function (minimization)."""
    def f(chains, waterline, eps, xs):
        dist = v_fwd(model, chains, xs).squeeze(-1)
        mean = dist.mean(0) # [N]
        std = dist.std(0) # [N]
        ei = log_h((waterline - mean)/(std+eps)) + jnp.log(std+eps)
        return ei
    return Partial(f, chains, waterline, eps)

def lcb(model, chains, γ=1.0):
    """Acquisition function for lowest expected score."""
    def f(chains, γ, xs):
        dist = v_fwd(model, chains, xs)
        mean = jnp.mean(dist, axis=0).squeeze(-1)
        std = jnp.std(dist, axis=0).squeeze(-1)
        return -(mean - γ*std)
    return Partial(f, chains, γ)

def log_expected_improvement_max(fit: FittedModel, waterline, eps=0.01):
    """Log expected improvement acquisition function (maximization)."""
    def f(fit, waterline, eps, xs):
        mean, std = fit.predict(xs)
        ei = log_h((mean - waterline)/(std+eps)) + jnp.log(std+eps)
        return ei
    return Partial(f, fit, waterline, eps)

def ucb(model, chains, γ=1.0):
    """Acquisition function for highest expected score."""
    def f(chains, γ, xs):
        dist = v_fwd(model, chains, xs)
        mean = jnp.mean(dist, axis=0).squeeze(-1)
        std = jnp.std(dist, axis=0).squeeze(-1)
        return mean + γ*std
    return Partial(f, chains, γ)

def findmax(key, acquisition_function, in_dim, mask=None, maskval=None, n=101):
    if mask is None:
        mask = jnp.full(in_dim, False)
    xs, metrics = jax.vmap(findmax_ac_, in_axes=(0,None,None,None))(jax.random.split(key,n), acquisition_function, mask, maskval)
    rs = acquisition_function(xs)
    metrics['rs'] = rs
    return xs[jnp.argmax(rs)], metrics

ac_steps = 10

@jax.jit
def findmax_ac_(key, acquisition_function, mask, maskval):
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey, [mask.shape[0]], minval=0, maxval=1)

    opt = optax.adam(0.05, 0.9, 0.9)
    opt_state = opt.init(x)

    def f_loss(x, key, p):
        x += jax.random.normal(key, (50,)+x.shape) * 0.5 * p
        ac = acquisition_function(x)
        loss = -ac.mean()
        return loss, (loss, {'ac':ac})
    f_grad = jax.grad(f_loss, has_aux=True)

    @scan_tqdm(ac_steps)
    def step(carry, i):
        x, opt_state, key = carry
        key, subkey = jax.random.split(key)
        grad, value = f_grad(x, subkey, (1-i/ac_steps))
        jax.debug.print('grad: xs={} grad={}', jnp.isnan(x).any(), jnp.isnan(grad).any())
        updates, opt_state = opt.update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, 0, 1)
        # if maskval is not None:
        #     x = jnp.where(mask, maskval, x)
        return (x, opt_state, key), {'value': value}
    
    (x, _, _), metrics = jax.lax.scan(step, (x, opt_state, key), jnp.arange(ac_steps))
    return x, metrics

def findmax_ahmc(key, f, minbound, maxbound):
    minbound = jnp.asarray(minbound)
    maxbound = jnp.asarray(maxbound)
    x0 = jax.random.uniform(key, [], minval=minbound, maxval=maxbound)
    xsamples = jax.random.uniform(key, (100,)+minbound.shape, minval=minbound, maxval=maxbound)
    ysamples = jax.vmap(f)(xsamples)
    ysamples = jnp.nan_to_num(ysamples, nan=0.0, posinf=0.0, neginf=0.0)
    Tmax = jnp.clip(ysamples.std(), 10, 1000)
    def logp(x, i):
        Tmin = 0.01
        T = jnp.exp((100-i)/100 * (jnp.log(Tmax) - jnp.log(Tmin)) + jnp.log(Tmin))
        return jnp.where(jnp.logical_and(jnp.all(minbound<x), jnp.all(x<maxbound)), f(x).mean() / T, -jnp.inf)
    x, ε, L, info = ahmc_fast(key, x0, logp, 100, εmin=1e-8, εmax=1.0, Lmin = 2, Lmax = 2000, pass_i=True)
    print('thetas during optimization:', info['theta'])
    print('values:', jax.vmap(f)(info['theta']))
    print(f'final ε,L={ε},{L}')
    return x

if __name__ == '__main__':

    def true_f(x):
        return jnp.sin(x) + jnp.cos(2*x)#.squeeze(-1)
    
    opt = Optim([SpaceItem('x', -1, 1)])

    params = opt.suggest({})

    val = true_f(params['x'])
    opt.notify(params, val)
    opt.infer()

    for i in range(20):
        params = opt.suggest({})

        # print('sigmas:', jnp.exp(opt.chains['log_sigma']))

        xs = jnp.linspace(-1, 1, 100)
        ys = true_f(xs)

        xs_ = jax.vmap(opt.space.normalize)({'x':xs})
        mean, std = opt.fitted.predict(xs_)

        # print('aaa:', (v_fwd(opt.model, opt.chains, opt.space.normalize(opt.trials[0].params)[None]) - true_f(opt.trials[0].params['x'])).square())

        print('means:', mean.shape, mean)
        print('stds:', std.shape, std)

        fig, ax1 = plt.subplots()
        ax1.plot(xs, ys, label='true')
        ax1.scatter([t.params['x'] for t in opt.trials], [t.value for t in opt.trials], label='observed')
        ax1.plot(xs, mean, label='mean')
        ax1.fill_between(xs, mean - 1*std, mean + 1*std, alpha=0.2)
        ax1.fill_between(xs, mean - 2*std, mean + 2*std, alpha=0.2)
        ax1.fill_between(xs, mean - 3*std, mean + 3*std, alpha=0.2)

        ax2 = ax1.twinx()
        ax2.scatter(params['x'][None], opt.acf(opt.space.normalize(params))[None], label='acq')
        ax2.plot(xs, jax.vmap(lambda x: opt.acf(opt.space.normalize({'x': x})))(xs), label='acquisition function')
        
        ax1.legend()
        ax2.legend()
        plt.show()

        val = true_f(params['x'])
        opt.notify(params, val)
        opt.infer()
