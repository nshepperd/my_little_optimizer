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
import hmc

if __name__ == '__main__':
    in_dim = 1

    def true_f(x):
        return jnp.sin(x) + jnp.cos(2*x)#.squeeze(-1)

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

    @jax.tree_util.register_pytree_node_class
    @dataclass
    class MCMC:
        n_steps: int
        n_leapfrog_steps: int
        step_size: float

        def tree_flatten(self):
            return (self.step_size,), (self.n_steps, self.n_leapfrog_steps)
        @staticmethod
        def tree_unflatten(static, dynamic):
            return MCMC(*static, *dynamic)
        
        @jax.jit
        def sample(self, key: jax.Array, x: jax.Array, y: jax.Array):
            scales = mkscales(model)
            def f_loss(eps):
                params = dict()
                for p in model.parameters():
                    params[p.name] = eps[p.name] * scales[p.name]

                cx = Context(params, None)
                prediction = model(cx, x)
                log_2_pi = jnp.log(2 * jnp.pi)
                y_loss = jnp.mean(jnp.square(y - prediction))
                σ_y = 0.01
                log_p_y = -0.5 * jnp.sum(jnp.square(y - prediction)/σ_y**2) - y.shape[0] * (0.5 * log_2_pi + jnp.log(σ_y))
                log_p_eps = sum(-0.5 * jnp.sum(jnp.square(eps[p.name])) - np.prod(p.shape) * (0.5 * log_2_pi) for p in model.parameters())
                return log_p_y + log_p_eps

            key1,key2 = jax.random.split(key)
            eps_init = {p:v/scales[p] for (p,v) in model.init_weights(key1).items()}
            chain, accept_prob = hmc.sample(key2, eps_init, f_loss, n_steps=self.n_steps, n_leapfrog_steps=self.n_leapfrog_steps, step_size=self.step_size)
            chain = {p:v*scales[p] for (p,v) in chain.items()}
            return chain, accept_prob

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
        # return expected_improvement(waterline, mean, std)
        return std

    @jax.jit
    def findmax(key, chains, waterline):
        x = jax.random.normal(key, [101, in_dim])

        opt = optax.lbfgs(0.001)
        # opt = optax.adam(0.1, 0.99, 0.999)
        opt_state = opt.init(x)

        # def f_mean(params, x):
        #     dist = v_fwd(params, x).squeeze(-1)
        #     return dist.mean(0), dist.std(0) # [101]

        def f_loss(x):
            ac = acquisition_function(chains, x, waterline)
            loss = -ac.mean() + jnp.maximum(0, jnp.abs(x)-4).mean()
            return loss, (loss, {'ac':ac})
            # mean, std = f_mean(params, x)
            # waterline = jnp.max(y)
            # return (-eiv.mean()), {'x':x,'mean':mean,'std':std, 'ei':eiv}
            # return -f_mean(params, x).mean()
        f_grad = jax.grad(f_loss, has_aux=True)

        @scan_tqdm(aq_steps)
        def step(carry, i):
            x, opt_state = carry
            grad, (value, metrics) = f_grad(x)
            updates, opt_state = opt.update(grad, opt_state, x, value=value, grad=grad, value_fn=lambda x:f_loss(x)[0])
            x = optax.apply_updates(x, updates)
            x = jnp.clip(x, -5, 5)
            return (x, opt_state), metrics
        
        (x, _), metrics = jax.lax.scan(step, (x, opt_state), jnp.arange(aq_steps))
        return x, metrics

    def gelman_rubin(statistics):
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
        mcmc = MCMC(n_steps=1000, n_leapfrog_steps=200, step_size=0.001)
        chains, accept_probs = jax.vmap(mcmc.sample, in_axes=(0,None,None))(rng.split(4), x, y)
        chains = {p:v[:,(mcmc.n_steps//10):] for (p,v) in chains.items()} # discard burn-in samples

        query_xs = jax.random.uniform(jax.random.PRNGKey(1), [10, in_dim], dtype=jnp.float32, minval=-5, maxval=5)
        query_ys = v_fwd(chains, query_xs) # c n x ...

        R = gelman_rubin(query_ys)
        while accept_probs.mean() < 0.5 or R.max() > 1.1:
            print('gelman-rubin:', R)
            print('acceptance rate:', accept_probs.mean())
            if accept_probs.mean() < 0.5:
                print('rejecting: acceptance rate too low')
                mcmc = MCMC(n_steps=mcmc.n_steps, n_leapfrog_steps=mcmc.n_leapfrog_steps, step_size=mcmc.step_size/2)
            else:
                print('rejecting: gelman-rubin too high')
                mcmc = MCMC(n_steps=mcmc.n_steps*2, n_leapfrog_steps=mcmc.n_leapfrog_steps, step_size=mcmc.step_size)
            chains, accept_probs = jax.vmap(mcmc.sample, in_axes=(0,None,None))(rng.split(4), x, y)
            chains = {p:v[:,(mcmc.n_steps//10):] for (p,v) in chains.items()} # discard burn-in samples
            query_ys = v_fwd(chains, query_xs) # c n x ...
            R = gelman_rubin(query_ys)
        
        # while R.max() > 1.1:
        #     print('gelman-rubin:', R)
        #     print('rejecting: gelman-rubin too high')
        #     mcmc = MCMC(n_steps=mcmc.n_steps, n_leapfrog_steps=mcmc.n_leapfrog_steps*2, step_size=mcmc.step_size)
        #     chains, accept_probs = jax.vmap(mcmc.sample, in_axes=(0,None,None))(rng.split(4), x, y)
        #     chains = {p:v[:,(mcmc.n_steps//10):] for (p,v) in chains.items()} # discard burn-in samples
        #     query_ys = v_fwd(chains, query_xs) # c n x ...
        #     R = gelman_rubin(query_ys)

        print('gelman-rubin:', R)
        print('acceptance rate:', accept_probs.mean())

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
