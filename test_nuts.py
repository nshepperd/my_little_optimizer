import sys, os

import jax
import jax.numpy as jnp
import jaxtorch
import optax
import numpy as np
import math
from matplotlib import pyplot as plt
from jax_tqdm import scan_tqdm
from functools import partial
from jax.tree_util import tree_map
import jaxopt
from dataclasses import dataclass

from jaxtorch import nn
from jaxtorch import PRNG, Context

from ngp.log_h import log_h
from ngp.metric import DictionaryMetric, ScalarMetric
from ngp.metric import DictionaryMetricEstimator, CovarianceMetricEstimator, KroneckerMetricEstimator
from ngp.nuts import nuts_kernel
from ngp.adapt import warmup_with_dual_averaging, find_reasonable_epsilon
import hmc

if __name__ == '__main__':
    in_dim = 1

    def true_f(x):
        return jnp.sin(x) - x**2/10 #.squeeze(-1)

    x = jnp.array([[1.0],[2.0],[-1.0]])
    y = true_f(x) #jnp.array([0.5,0.0, 0.4, 0.5]).unsqueeze(-1)
    model = nn.Sequential([nn.Linear(x.shape[-1], 10), nn.SiLU(), nn.Linear(10, 10), nn.SiLU(), nn.Linear(10, 1)])
    model.name_everything_()

    n_steps = 1000
    aq_steps = 100

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
            metrics[p.name] = ScalarMetric(scales[p.name], p.shape)
        return DictionaryMetric(metrics)

    def mkmetric_estimator(model):
        part = {}
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                part[mod.weight.name] = KroneckerMetricEstimator()
                part[mod.bias.name] = CovarianceMetricEstimator()
        return DictionaryMetricEstimator(part)

    scales = mkscales(model)
    metric0 = mkmetric(model)
    estimator = mkmetric_estimator(model)
    σ_y = 0.01

    def f_loss(metric, eps):
        params = metric.unwhiten(eps)
        cx = Context(params, None)
        prediction = model(cx, x)
        log_2_pi = jnp.log(2 * jnp.pi)
        log_p_y = -0.5 * jnp.sum(jnp.square((y - prediction)/σ_y)) - y.shape[0] * (0.5 * log_2_pi + jnp.log(σ_y))
        log_p_eps = sum(-0.5 * jnp.sum(jnp.square(params[p.name]/scales[p.name])) - np.prod(p.shape) * (0.5 * log_2_pi) for p in model.parameters())
        return log_p_y + log_p_eps

    @jax.tree_util.register_pytree_node_class
    @dataclass
    class MCMC:
        n_steps: int
        n_chains: int

        def tree_flatten(self):
            return (), (self.n_steps, self.n_chains)
        @staticmethod
        def tree_unflatten(static, dynamic):
            return MCMC(*static, *dynamic)
        
        @jax.jit
        def sample(self, key: jax.Array, x: jax.Array, y: jax.Array):
            def nuts_sample(key, theta_init, logp, n_steps, step_size):
                @scan_tqdm(self.n_steps)
                def step_fn(carry, i):
                    theta, key = carry
                    key, subkey = jax.random.split(key)
                    theta_new, alpha, ms = nuts_kernel(subkey, theta, logp, jax.grad(logp), step_size)
                    return (theta, key), (theta_new, alpha, ms)
                _, (thetas, alphas, ms) = jax.lax.scan(step_fn, (theta_init, key), jnp.arange(n_steps)) 
                return thetas, alphas, ms
            
            keys = jax.random.split(key,8)
            params = model.init_weights(keys[0])
            
            eps = metric0.whiten(params)
            ε, εp = find_reasonable_epsilon(keys[1], eps, partial(f_loss, metric0))
            jax.debug.print('Found step size {} with acceptance probability {}', ε, εp)
            eps, ε, metrics = warmup_with_dual_averaging(keys[2], eps, partial(f_loss, metric0), ε)
            jax.debug.print('Warmup finished with step size {} and log probs {}', ε, metrics['probs'])
            chain, accept_prob, ms = nuts_sample(keys[3], eps, partial(f_loss, metric0), 25, ε)
            jax.debug.print('Finished first adaptation interval - accept_prob = {}', accept_prob.mean())
            pchain = jax.vmap(metric0.unwhiten)(chain)
            params = jax.tree_util.tree_map(lambda x:x[-1], pchain)
            metric = estimator(pchain)
            eps = metric.whiten(params)
            eps, ε, metrics = warmup_with_dual_averaging(keys[6], eps, partial(f_loss, metric), ε, warmup_steps=20)
            chain, accept_prob, ms = nuts_sample(keys[4], eps, partial(f_loss, metric), 50, ε)
            jax.debug.print('Finished second adaptation interval - accept_prob = {}', accept_prob.mean())
            pchain = jax.vmap(metric.unwhiten)(chain)
            params = jax.tree_util.tree_map(lambda x:x[-1], pchain)
            metric = estimator(pchain)
            eps = metric.whiten(params)
            eps, ε, metrics = warmup_with_dual_averaging(keys[6], eps, partial(f_loss, metric), ε, warmup_steps=20)
            chain, accept_prob, ms = nuts_sample(keys[5], eps, partial(f_loss, metric), 100, ε)
            jax.debug.print('Finished third adaptation interval - accept_prob = {}', accept_prob.mean())
            pchain = jax.vmap(metric.unwhiten)(chain)
            params = jax.tree_util.tree_map(lambda x:x[-1], pchain)
            metric = estimator(pchain)
            eps = metric.whiten(params)

            eps, ε, metrics = warmup_with_dual_averaging(keys[6], eps, partial(f_loss, metric), ε)

            chain, accept_prob, ms = nuts_sample(keys[7], eps, partial(f_loss, metric), self.n_steps, ε)
            # chain, accept_prob = hmc.sample(key2, eps_init, f_loss, n_steps=self.n_steps, n_leapfrog_steps=self.n_leapfrog_steps, step_size=self.step_size)

            probs = jax.vmap(partial(f_loss, metric))(chain)
            chain = jax.vmap(metric.unwhiten)(chain)
            return chain, accept_prob, {'probs':probs}

    @jax.jit
    @partial(jax.vmap, in_axes=(0,None))
    @partial(jax.vmap, in_axes=(0,None))
    def v_fwd(chains, x: jax.Array) -> jax.Array:
        cx = Context(chains, None)
        return model(cx, x)

    def expected_improvement(waterline: jax.Array, mean: jax.Array, std: jax.Array, eps=0.0):
        return log_h((mean - waterline)/(std+eps)) + jnp.log(std+eps)

    @jax.jit
    def acquisition_function(chains, xs, waterline):
        # best: [] scalar
        dist = v_fwd(chains, xs).squeeze(-1)
        mean = dist.mean((0,1)) # [N]
        std = dist.std((0,1)) # [N]
        return expected_improvement(waterline, mean, std)
        # return std

    @jax.jit
    def findmax(key, chains, waterline):
        return jax.vmap(findmax_ac_, in_axes=(0,None,None))(jax.random.split(key,101),chains,waterline)

    ac_steps = 500
    def findmax_ac_(key, chains, waterline):
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, [in_dim], minval=-4, maxval=4)

        opt = optax.adam(0.1, 0.9, 0.9)
        opt_state = opt.init(x)

        def f_loss(x, key, p):
            x += jax.random.normal(key, (100,) + x.shape) * 4.0 * p
            ac = acquisition_function(chains, x, waterline)
            loss = -ac.mean()
            return loss, (loss, {'ac':ac})
        f_grad = jax.grad(f_loss, has_aux=True)

        @scan_tqdm(ac_steps)
        def step(carry, i):
            x, opt_state, key = carry
            key, subkey = jax.random.split(key)
            grad, value = f_grad(x, subkey, (1-i/ac_steps)**2)
            # updates, opt_state = opt.update(grad, opt_state, x, value=value, grad=grad, value_fn=lambda x:f_loss(x)[0])
            updates, opt_state = opt.update(grad, opt_state, x)
            x = optax.apply_updates(x, updates)
            x = jnp.clip(x, -4, 4)
            return (x, opt_state, key), {'value': value}
        
        (x, _, _), metrics = jax.lax.scan(step, (x, opt_state, key), jnp.arange(ac_steps))
        return x, metrics

    def gelman_rubin(query_ys):
        J = query_ys.shape[0]
        L = query_ys.shape[1]
        xj = jnp.mean(query_ys, axis=1) # c x ..
        xm = jnp.mean(xj, axis=0) # x ..
        B = L/(J-1) * jnp.sum(jnp.square(xj - xm), axis=0)
        W = query_ys.var(1).mean(0)
        R = ((L-1)/L * W + 1/L * B)/W
        return R

    for i in range(10):
        rng = PRNG(jax.random.PRNGKey(0))
        mcmc = MCMC(n_steps=200, n_chains=4)

        chains, accept_probs, ms = jax.vmap(mcmc.sample, in_axes=(0,None,None))(rng.split(4), x, y)
        jaxtorch.pt.save(chains, 'testchains.pt')
        query_xs = jax.random.uniform(jax.random.PRNGKey(1), [10, in_dim], dtype=jnp.float32, minval=-5, maxval=5)
        query_ys = v_fwd(chains, query_xs) # c n x ...
        print(ms)
        # plt.plot(ms['probs'].mean(0))
        # plt.fill_between(jnp.arange(ms['probs'].shape[-1]), ms['probs'].min(0), ms['probs'].max(0), alpha=0.2)
        # plt.show()
        # exit()
        R = gelman_rubin(query_ys)
        print('gelman-rubin:', R)
        print('acceptance rate:', accept_probs.mean(1))

        # chains = jaxtorch.pt.load('testchains.pt')

        xs = jnp.linspace(-5, 5, 100)[:, None]
        ys = v_fwd(chains, xs)
        print(xs.shape, ys.shape)
        mean = jnp.mean(ys, axis=(0,1)).squeeze(-1)
        std = jnp.std(ys, axis=(0,1)).squeeze(-1)
        lo1, hi1 = jnp.quantile(ys, jnp.array([0.158655, 0.841345]), axis=(0,1))
        lo2, hi2 = jnp.quantile(ys, jnp.array([1-0.97725, 0.97725]), axis=(0,1))
        lo3, hi3 = jnp.quantile(ys, jnp.array([1-0.99865, 0.99865]), axis=(0,1))

        print('aaa:', jnp.mean(jnp.square(v_fwd(chains, x) - y)))
        print('mean = ', mean)
        print('std = ', std)
        # plot mean and shaded confidence interval
        plt.plot(xs.squeeze(-1), mean, label='mean')
        plt.fill_between(xs.squeeze(-1), lo1.squeeze(-1),hi1.squeeze(-1), alpha=0.2, color='lightblue')
        plt.fill_between(xs.squeeze(-1), lo2.squeeze(-1),hi2.squeeze(-1), alpha=0.2, color='lightblue')
        plt.fill_between(xs.squeeze(-1), lo3.squeeze(-1),hi3.squeeze(-1), alpha=0.2, color='lightblue')
        # plt.fill_between(xs.squeeze(-1), mean - 3*std, mean + 3*std, alpha=0.2, color='lightblue')
        # plt.fill_between(xs.squeeze(-1), mean - 2*std, mean + 2*std, alpha=0.2, color='lightblue')
        # plt.fill_between(xs.squeeze(-1), mean - 1*std, mean + 1*std, alpha=0.2, color='lightblue')

        # plot acquisition function
        xs = jnp.linspace(-5, 5, 100)[:, None]
        pred = acquisition_function(chains, xs, jnp.max(y))
        # preg = jax.vmap(jax.grad(lambda xs: acquisition(params, xs, jnp.max(y))))(xs)
        plt.plot(xs, pred.clamp(-20), label='acquisition')
        # plt.plot(xs, preg, label='acquisition (gradient)')

        xmax, metrics = findmax(jax.random.PRNGKey(1), chains, jnp.max(y))
        ymax = v_fwd(chains, xmax)
        mean = jnp.mean(ymax, axis=(0,1)).squeeze(-1)
        std = jnp.std(ymax, axis=(0,1)).squeeze(-1)
        # plot mean and shaded confidence interval
        plt.scatter(xmax.squeeze(-1), mean, label='max', color='red', marker='x')

        # plot true function
        plt.plot(xs.squeeze(-1), true_f(xs.squeeze(-1)), label='true', color='blue')

        # plot data points
        plt.scatter(x.squeeze(-1), y.squeeze(-1), label='data')
        plt.legend()
        plt.show()

        
        ymax = acquisition_function(chains, xmax, jnp.max(y))
        ix = jnp.argmax(ymax)
        xmax = xmax[ix]
        x = jnp.concatenate([x, xmax[None]], axis=0)
        y = true_f(x)
