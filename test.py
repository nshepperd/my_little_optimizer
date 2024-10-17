import sys, os

import jax
import jax.numpy as jnp
import jaxtorch
import optax
import math
from matplotlib import pyplot as plt
from jax_tqdm import scan_tqdm
from functools import partial
from jax.tree_util import tree_map
import jaxopt

from jaxtorch import nn
from jaxtorch import PRNG, Context

if __name__ == '__main__':
    in_dim = 1

    def true_f(x):
        return jnp.sin(x) + jnp.cos(2*x)#.squeeze(-1)

    x = jnp.array([[1.0],[2.0], [-1.0]])
    y = true_f(x) #jnp.array([0.5,0.0, 0.4, 0.5]).unsqueeze(-1)
    model = nn.Sequential([nn.Linear(x.shape[-1], 10), nn.GELU(), nn.Linear(10, 10), nn.GELU(), nn.Linear(10, 1)])

    steps = 1000

    @jax.jit
    @partial(jax.vmap, in_axes=(0,None,None))
    def train(key: jax.random.PRNGKey, x: jax.Array, y: jax.Array):
        params = model.init_weights(key)
        y += jax.random.normal(key, y.shape) * 0.1
        opt = optax.adam(1e-3, 0.9, 0.95)
        opt_state = opt.init(params)

        def f_loss(params):
            cx = Context(params, None)
            return jnp.mean(jnp.square(model(cx, x) - y))
        f_grad = jax.value_and_grad(f_loss)

        @scan_tqdm(steps)
        def step(carry, i):
            params, opt_state = carry
            loss, grads = f_grad(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss
    
        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), jnp.arange(steps))
        return params, losses

    @jax.jit
    @partial(jax.vmap, in_axes=(0,None))
    def v_fwd(params, x):
        cx = Context(params, None)
        return model(cx, x)
    
    from log_h import log_h

    def prob_improve(waterline, mean, std, eps=0.01):
        # return 1/2 * (1 + jax.lax.erf((mean - waterline)/(jnp.sqrt(2) * std+eps)))
        # return jax.scipy.stats.norm.logcdf(mean, waterline, std+eps)
        return log_h((mean - waterline)/(std+eps)) + jnp.log(std+eps)
        # return jnp.log(h((mean - waterline)/(std+eps)) * std+0.01)

    # def ei(waterline, mean, std):
    #     return 1/2 * (1 + jax.lax.erf((mean - waterline)/(jnp.sqrt(2) * std+1e-1)))
    #     # return 0.5 * (waterline + mean +
    #     #                  jnp.exp(-((waterline - mean)**2/(2 * std**2))) * jnp.sqrt(2 / math.pi) * std +
    #     #                   (waterline - mean) * jax.lax.erf((waterline - mean)/(jnp.sqrt(2.0) * std))) - waterline

    @jax.jit
    def acquisition(params, xs, best):
        # Acquisition function.
        # params: [M, *]  M is ensemble size
        # xs: [N, D] N is number of points
        # best: [] scalar
        dist = v_fwd(params, xs).squeeze(-1) #[M, N]
        mean = dist.mean(0) # [N]
        std = dist.std(0) # [N]
        return prob_improve(best, mean, std)
        # return std

    @jax.jit
    def findmax(key, params, waterline):
        x = jax.random.normal(key, [101, in_dim])

        opt = optax.lbfgs(0.001)
        # opt = optax.adam(0.1, 0.99, 0.999)
        opt_state = opt.init(x)

        # def f_mean(params, x):
        #     dist = v_fwd(params, x).squeeze(-1)
        #     return dist.mean(0), dist.std(0) # [101]

        def f_loss(x):
            ac = acquisition(params, x, waterline)
            loss = -ac.mean() + jnp.maximum(0, jnp.abs(x)-4).mean()
            return loss, (loss, {'ac':ac})
            # mean, std = f_mean(params, x)
            # waterline = jnp.max(y)
            # return (-eiv.mean()), {'x':x,'mean':mean,'std':std, 'ei':eiv}
            # return -f_mean(params, x).mean()
        f_grad = jax.grad(f_loss, has_aux=True)

        @scan_tqdm(steps)
        def step(carry, i):
            x, opt_state = carry
            grad, (value, metrics) = f_grad(x)
            updates, opt_state = opt.update(grad, opt_state, x, value=value, grad=grad, value_fn=lambda x:f_loss(x)[0])
            x = optax.apply_updates(x, updates)
            x = jnp.clip(x, -5, 5)
            return (x, opt_state), metrics
        
        (x, _), metrics = jax.lax.scan(step, (x, opt_state), jnp.arange(steps))
        return x, metrics

    for _ in range(10):
        keys = jax.random.split(jax.random.PRNGKey(0), 100)
        params, losses = train(keys, x, y)
        xs = jnp.linspace(-5, 5, 100)[:, None]
        ys = v_fwd(params, xs)
        print(xs.shape, ys.shape)
        xs = xs.squeeze(-1)
        mean = jnp.mean(ys, axis=0).squeeze(-1)
        std = jnp.std(ys, axis=0).squeeze(-1)
        # plot mean and shaded confidence interval
        plt.plot(xs, mean, label='mean')
        plt.fill_between(xs, mean - 1*std, mean + 1*std, alpha=0.2)

        # plot acquisition function
        xs = jnp.linspace(-5, 5, 100)[:, None]
        pred = acquisition(params, xs, jnp.max(y))
        preg = jax.vmap(jax.grad(lambda xs: acquisition(params, xs, jnp.max(y))))(xs)
        plt.plot(xs, jnp.exp(pred), label='acquisition')
        # plt.plot(xs, preg, label='acquisition (gradient)')

        xmax, metrics = findmax(jax.random.PRNGKey(1), params, jnp.max(y))
        ymax = v_fwd(params, xmax)
        mean = jnp.mean(ymax, axis=0).squeeze(-1)
        std = jnp.std(ymax, axis=0).squeeze(-1)
        # plot mean and shaded confidence interval
        plt.scatter(xmax.squeeze(-1), mean, label='max', color='red', marker='x')

        # plot true function
        plt.plot(xs[:, 0], true_f(xs[:, 0]), label='true', color='blue')

        # plot data points
        plt.scatter(x[:, 0], y[:, 0], label='data')
        plt.legend()
        plt.show()

        
        ymax = acquisition(params, xmax, jnp.max(y))
        ix = jnp.argmax(ymax)
        xmax = xmax[ix]
        x = jnp.concatenate([x, xmax[None]], axis=0)
        y = true_f(x)