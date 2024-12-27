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
from typing import List, Dict, Tuple, Protocol, Callable
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

    def __getitem__(self, key):
        return self.items[key]

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
        # self.model.name_everything_()
    
    # @jax.jit
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

    def __init__(self, space: List[SpaceItem], objective='min', heuristic='lcb'):
        self.space = Space(keys=[item.name for item in space], items={item.name: item for item in space}, n=len(space))
        self.trials = []
        self.in_dim = len(self.space.keys)
        self.model = MLPWithNoise(nn.Sequential([nn.Linear(self.in_dim, 10), nn.Tanh(), nn.Linear(10, 1)]))
        if hasattr(self.model.model, 'name_everything_'):
            self.model.model.name_everything_()
        self.fitted = None
        self.rng = PRNG(jax.random.PRNGKey(0))
        self.objective = objective
        self.heuristic = heuristic

    def notify(self, params, value):
        self.trials.append(Trial(params, value))
    
    def infer(self):
        xs = jnp.stack([self.space.normalize(x.params) for x in self.trials])
        if self.objective == 'min':
            yworst = max([x.value for x in self.trials if jnp.isfinite(x.value)])
        else:
            yworst = min([x.value for x in self.trials if jnp.isfinite(x.value)])
        ys = jnp.stack([x.value if jnp.isfinite(x.value) else yworst for x in self.trials])
        self.fitted = self.model.fit(jax.random.PRNGKey(0), xs, ys)

    def suggestbest(self, params, method='cma-es', key=None, psize=50):
        assert self.fitted is not None

        mask = jnp.array([k in params for k in self.space.keys])

        if self.objective == 'min':
            # pessimistic bound: try to find a point that's definitely good
            def acf(fit, x):
                mean, std = fit.predict(x)
                return -(mean + std)
            acf = Partial(acf, self.fitted)
        elif self.objective == 'max':
            def acf(fit, x):
                mean, std = fit.predict(x)
                return mean - std
            acf = Partial(acf, self.fitted)

        if key is None:
            key = jax.random.PRNGKey(0)

        if method == 'cma-es':
            maskval = jnp.array([self.space.items[k].normalize(params[k]) if k in params else 0.0 for k in self.space.keys])
            maskv_indices = jnp.array([i for i,k in enumerate(self.space.keys) if k not in params])
            def unmask(x_var):
                return maskval.at[maskv_indices].set(x_var)
            def acf_mask(x_var):
                return acf(unmask(x_var))
            # xmax = findmax_ahmc(self.rng.split(), acf_mask, jnp.zeros(maskv_indices.shape), jnp.ones(maskv_indices.shape))
            xmax, _ = findmax_cmaes(key, acf_mask, jnp.zeros(maskv_indices.shape), jnp.ones(maskv_indices.shape), population_size = psize)
            xmax = unmask(xmax)
        elif method == 'ac':
            maskval = jnp.array([self.space.items[k].normalize(params[k]) if k in params else 0.0 for k in self.space.keys])
            xmax, _ = findmax_ac_(key, acf, mask, maskval)
        return self.space.denormalize(jnp.where(mask, maskval, xmax))

    def suggest(self, params, method = 'cma-es'):
        if self.fitted is None:
            # just choose random points
            v = jax.random.uniform(self.rng.split(), [self.space.n], minval=0.0, maxval=1.0)
            return self.space.denormalize(v)

        if self.heuristic == 'lcb':
            if self.objective == 'min':
                acf = lcb(self.fitted, 3.0)
            elif self.objective == 'max':
                acf = ucb(self.fitted, 3.0)
        elif self.heuristic == 'ei':
            if self.objective == 'min':
                waterline = min([x.value for x in self.trials if jnp.isfinite(x.value)])
                acf = log_expected_improvement_min(self.fitted, waterline)
            elif self.objective == 'max':
                waterline = max([x.value for x in self.trials if jnp.isfinite(x.value)])
                acf = log_expected_improvement_max(self.fitted, waterline)
        
        self.acf = acf
        mask = jnp.array([k in params for k in self.space.keys])

        if method == 'cma-es':
            maskval = jnp.array([self.space.items[k].normalize(params[k]) if k in params else 0.0 for k in self.space.keys])
            maskv_indices = jnp.array([i for i,k in enumerate(self.space.keys) if k not in params])
            def unmask(x_var):
                return maskval.at[maskv_indices].set(x_var)
            def acf_mask(x_var):
                return acf(unmask(x_var))
            # xmax = findmax_ahmc(self.rng.split(), acf_mask, jnp.zeros(maskv_indices.shape), jnp.ones(maskv_indices.shape))
            xmax, _ = findmax_cmaes(self.rng.split(), acf_mask, jnp.zeros(maskv_indices.shape), jnp.ones(maskv_indices.shape))
            xmax = unmask(xmax)
        elif method == 'ac':
            maskval = jnp.array([self.space.items[k].normalize(params[k]) if k in params else 0.0 for k in self.space.keys])
            xmax, _ = findmax_ac_(self.rng.split(), acf, mask, maskval)
        return self.space.denormalize(jnp.where(mask, maskval, xmax))

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

def log_expected_improvement_min(fit: FittedModel, waterline, eps=0.01):
    """Log expected improvement acquisition function (minimization)."""
    def f(fit, waterline, eps, xs):
        mean, std = fit.predict(xs)
        ei = log_h((waterline - mean)/(std+eps)) + jnp.log(std+eps)
        return ei
    return Partial(f, fit, waterline, eps)

def lcb(fit: FittedModel, γ=1.0):
    """Acquisition function for lowest expected score."""
    def f(fit, γ, xs):
        mean, std = fit.predict(xs)
        return -mean + γ*std
    return Partial(f, fit, γ)

def log_expected_improvement_max(fit: FittedModel, waterline, eps=0.01):
    """Log expected improvement acquisition function (maximization)."""
    def f(fit, waterline, eps, xs):
        mean, std = fit.predict(xs)
        ei = log_h((mean - waterline)/(std+eps)) + jnp.log(std+eps)
        return ei
    return Partial(f, fit, waterline, eps)

def ucb(fit: FittedModel, γ=1.0):
    """Acquisition function for highest expected score."""
    def f(fit, γ, xs):
        mean, std = fit.predict(xs)
        return mean + γ*std
    return Partial(f, fit, γ)

def findmax(key, acquisition_function, in_dim, mask=None, maskval=None, n=101):
    if mask is None:
        mask = jnp.full(in_dim, False)
    xs, metrics = jax.vmap(findmax_ac_, in_axes=(0,None,None,None))(jax.random.split(key,n), acquisition_function, mask, maskval)
    rs = acquisition_function(xs)
    metrics['rs'] = rs
    return xs[jnp.argmax(rs)], metrics

ac_steps = 1000

@jax.jit
def findmax_ac_(key, acquisition_function, mask, maskval):
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey, [mask.shape[0]], minval=0, maxval=1)

    opt = optax.adam(0.05, 0.9, 0.9)
    opt_state = opt.init(x)

    def f_loss(x, key, p):
        key1, key2 = jax.random.split(key)
        x += jax.random.normal(key1, (50,)+x.shape) * 2.0 * p
        x = jnp.clip(x, 0, 1)
        ac = acquisition_function(x)
        loss = -ac.mean()
        return loss, (loss, {'ac':ac})
    f_grad = jax.grad(f_loss, has_aux=True)

    @scan_tqdm(ac_steps)
    def step(carry, i):
        x, opt_state, key = carry
        key, subkey = jax.random.split(key)
        grad, value = f_grad(x, subkey, (1-i/ac_steps))
        updates, opt_state = opt.update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, 0, 1)
        if maskval is not None:
            x = jnp.where(mask, maskval, x)
        return (x, opt_state, key), {'value': value}
    
    (x, _, _), metrics = jax.lax.scan(step, (x, opt_state, key), jnp.arange(ac_steps))
    return x, metrics

def findmax_ahmc(key, f, minbound, maxbound):
    key1, key2, key3 = jax.random.split(key, 3)
    minbound = jnp.asarray(minbound)
    maxbound = jnp.asarray(maxbound)
    x0 = jax.random.uniform(key1, minbound.shape, minval=minbound, maxval=maxbound)
    xsamples = jax.random.uniform(key2, (100,)+minbound.shape, minval=minbound, maxval=maxbound)
    ysamples = jax.vmap(f)(xsamples)
    ysamples = jnp.nan_to_num(ysamples, nan=0.0, posinf=0.0, neginf=0.0)
    Tmax = jnp.clip((ysamples.max()-ysamples.min())*10, 10, 1000)
    steps = 100
    def logp(x, i):
        Tmin = 0.01
        # T = jnp.exp((1-i/steps) * (jnp.log(Tmax) - jnp.log(Tmin)) + jnp.log(Tmin))
        T = Tmax
        # return f(x).mean() / T
        return jnp.where(jnp.logical_and(jnp.all(minbound<x), jnp.all(x<maxbound)), f(x).mean() / T, -jnp.inf)
    x, ε, L, info = ahmc_fast(key3, x0, logp, steps, εmin=1e-8, εmax=1.0, Lmin = 2, Lmax = 2000, pass_i=True)
    print(f'Tmax={Tmax}')
    print('thetas during optimization:', info['theta'])
    print('values:', jax.vmap(f)(info['theta']))
    print('accept probs:', info['α'])
    print('ds:', info['d'])
    print('ε:', info['ε'])
    print('L:', info['L'])
    print(f'final ε,L={ε},{L}')
    return x

def findmax_cmaes(key: jax.random.PRNGKey,
                 f: Callable[[jax.Array], float],
                 minbound: jax.Array,
                 maxbound: jax.Array,
                 population_size: int = None,
                 max_iterations: int = 1000) -> Tuple[jax.Array, float]:
    """
    Maximizes function f using CMA-ES in JAX.
    
    Args:
        key: JAX PRNG key
        f: Function to maximize
        minbound: Lower bounds for each dimension, shape [n]
        maxbound: Upper bounds for each dimension, shape [n]
        population_size: Population size (if None, uses 4 + floor(3 * log(n)))
        max_iterations: Maximum number of iterations
    
    Returns:
        Tuple of (best_x, best_value)
    """
    n = minbound.shape[0]  # dimension
    if population_size is None:
        population_size = 4 + int(3 * math.log(n))
    
    # Strategy parameters
    mu = population_size // 2  # number of parents
    weights = jnp.log(mu + 0.5) - jnp.log(jnp.arange(mu) + 1)
    weights = weights / jnp.sum(weights)  # normalized weights for recombination
    mueff = 1 / jnp.sum(weights ** 2)  # variance-effective size of mu
    
    # Adaptation parameters
    cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)  # time constant for cumulation path
    cs = (mueff + 2) / (n + mueff + 5)  # time constant for sigma path
    c1 = 2 / ((n + 1.3)**2 + mueff)  # learning rate for rank-one update
    cmu = jnp.minimum(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))  # learning rate for rank-mu update
    damps = 1 + 2*jnp.maximum(0, jnp.sqrt((mueff - 1)/(n + 1)) - 1) + cs  # damping for sigma
    
    # Initialize strategy parameters
    mean = (minbound + maxbound) / 2  # start in middle of bounds
    sigma = jnp.max(maxbound - minbound) / 2  # initial step size
    
    # Initialize covariance matrix and evolution paths
    C = jnp.eye(n)  # covariance matrix
    pc = jnp.zeros(n)  # evolution path for C
    ps = jnp.zeros(n)  # evolution path for sigma
    
    def body_fun(state):
        i, key, mean, sigma, C, pc, ps, best_x, best_value = state
        
        # Generate population
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, (population_size, n))
        D, B = jnp.linalg.eigh(C)  # eigendecomposition of C
        D = jnp.sqrt(jnp.maximum(0, D))  # D contains standard deviations now
        X = mean + sigma * (Z @ (B * D))
        
        # Clip to bounds
        X = jnp.clip(X, minbound, maxbound)
        
        # Evaluate and sort
        values = jax.vmap(f)(X)
        indices = jnp.argsort(-values)  # negative for maximization
        values = values[indices]
        X = X[indices]
        Z = Z[indices]
        
        # Update mean
        old_mean = mean
        mean = jnp.sum(X[:mu] * weights[:, None], axis=0)
        
        # Update evolution paths
        y = (mean - old_mean) / sigma
        ps = (1 - cs) * ps + jnp.sqrt(cs * (2 - cs) * mueff) * (B @ (B.T @ y))
        hsig = jnp.linalg.norm(ps) / jnp.sqrt(1 - (1-cs)**(2*(i+1))) < (1.4 + 2/(n+1)) * jnp.sqrt(n)
        pc = (1 - cc) * pc + hsig * jnp.sqrt(cc * (2 - cc) * mueff) * y
        
        # Update covariance matrix
        artmp = (1/sigma) * (X[:mu] - old_mean)
        C = (1 - c1 - cmu) * C + \
            c1 * (pc[:, None] @ pc[None, :] + (1-hsig) * cc*(2-cc) * C) + \
            cmu * (artmp.T @ (weights[:, None] * artmp))
        
        # Update sigma
        sigma = sigma * jnp.exp((cs/damps) * (jnp.linalg.norm(ps) / jnp.sqrt(n) - 1))
        
        # Update best solution if needed
        best_x = jnp.where(values[0] > best_value, X[0], best_x)
        best_value = jnp.maximum(best_value, values[0])
        
        return i + 1, key, mean, sigma, C, pc, ps, best_x, best_value
    
    # Initial state
    best_x = mean
    best_value = f(mean)
    init_state = (0, key, mean, sigma, C, pc, ps, best_x, best_value)
    
    # Run optimization loop
    final_state = jax.lax.while_loop(
        lambda state: state[0] < max_iterations,
        body_fun,
        init_state
    )
    
    return final_state[-2], final_state[-1]  # best_x, best_value

if __name__ == '__main__':

    def true_f(x):
        return jnp.sin(x) + jnp.cos(2*x)#.squeeze(-1)
    
    opt = Optim([SpaceItem('x', -1, 1)], objective='max')

    params = opt.suggest({})

    val = true_f(params['x'])
    opt.notify(params, val)
    opt.infer()

    for i in range(20):
        params = opt.suggest({})

        print(opt.suggestbest({}))

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
        ax2.scatter(params['x'][None], opt.acf(opt.space.normalize(params))[None], label='acq', color='red')
        ax2.plot(xs, jax.vmap(lambda x: opt.acf(opt.space.normalize({'x': x})))(xs), label='acquisition function', color='red')
        
        ax1.legend()
        ax2.legend()
        plt.show()

        val = true_f(params['x'])
        opt.notify(params, val)
        opt.infer()
