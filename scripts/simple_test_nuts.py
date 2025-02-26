import jax
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm

from my_little_optimizer.opt.nuts import nuts_kernel
from my_little_optimizer.opt.adapt import find_reasonable_epsilon, warmup_with_dual_averaging

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from jaxtorch import PRNG
    import einops

    @jax.jit
    def logp(theta):
        return -0.5 * jnp.sum(jnp.square(theta[0]/theta[1])) - 0.5 * jnp.square(theta[1])
    logp_grad = jax.grad(logp)

    rng = PRNG(jax.random.PRNGKey(0))
    theta = jax.random.normal(rng.split(), [2])+10
    r = jax.random.normal(rng.split(), [2])

    def k_nuts(key, theta, ε):
        return nuts_kernel(key, theta, logp, logp_grad, 1.0)

    n_steps = 2000

    @jax.jit
    def sample(key, theta, ε):
        @scan_tqdm(n_steps)
        def body_fn(carry, i):
            key, theta = carry
            key, subkey = jax.random.split(key)
            theta, alpha, metrics = nuts_kernel(subkey, theta, logp, logp_grad, ε)
            return (key, theta), (theta, alpha, metrics)
        return jax.lax.scan(body_fn, (key, theta), jnp.arange(n_steps))[1]


    # samples = []
    # alphas = []
    #     theta, alpha, metrics = nuts_kernel(rng.split(), theta, logp, logp_grad, 1.0)
    #     samples.append(theta)
    #     alphas.append(alpha)
    ε, εp = find_reasonable_epsilon(rng.split(), theta, jax.jit(logp))
    print(f'Found step size: {ε} with acceptance probability: {εp}')

    theta, ε, metrics = warmup_with_dual_averaging(rng.split(), theta, logp, ε)
    print(f'Finished warmup phase with ε={ε}, metrics={metrics}')


    samples, alphas, metrics = sample(rng.split(), theta, ε)
    samples = jnp.stack(samples)
    alphas = jnp.stack(alphas)
    print(samples.shape)
    print(metrics)
    # print(einops.rearrange(samples, '(x n) d -> x n d', x=20).std(1))
    print(samples.std(0), jax.random.normal(rng.split(),samples.shape).std(0))
    print('alphas:', alphas)
    # scatter plot
    plt.scatter(samples[:,0], samples[:,1])
    plt.plot(samples[:,0], samples[:,1], color='red', alpha=0.1)
    plt.show()

