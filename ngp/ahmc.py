import jax
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm
import einops
import jax.scipy as jsp

from ngp.gaussian_process import GP
from ngp.util import Fn, fn

def leapfrog(theta: jax.Array, r: jax.Array, logp_grad, eps: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Single leapfrog step"""
    r_half = r + 0.5 * eps * logp_grad(theta)
    theta_new = theta + eps * r_half
    r_new = r_half + 0.5 * eps * logp_grad(theta_new)
    return theta_new, r_new

def compute_hamiltonian(theta, r, logp):
    """Compute Hamiltonian (smaller is better)"""
    return -logp(theta) + 0.5 * jnp.sum(r**2)

def finiteor(x, y):
    return jnp.where(jnp.isfinite(x), x, y)

# @partial(jax.jit, static_argnums=(2,))
def hmc_kernel(key, theta, logp, ε, L):
    subkeys = jax.random.split(key, 3)
    r0 = jax.random.normal(subkeys[0], theta.shape)
    H0 = compute_hamiltonian(theta, r0, logp)
    u0 = jax.random.uniform(subkeys[1], []) # u = u0 * exp(-H0)
    n0 = 1

    grad = jax.grad(logp)

    def step_fn(i, carry):
        key, theta, r, n, theta_new, alpha_sum = carry
        theta, r = leapfrog(theta, r, grad, ε)  
        H = compute_hamiltonian(theta, r, logp)
        alpha_sum += finiteor(jnp.exp(jnp.clip(H0 - H, max=0.0)), 0.0)
        n_prime = jnp.log(u0) <= H0 - H
        n += n_prime

        key, subkey = jax.random.split(key)
        choose = jax.random.bernoulli(subkey, n_prime / n)
        theta_new = jnp.where(choose, theta, theta_new)
        
        return (key, theta, r, n, theta_new, alpha_sum)
    
    (key, _, _, n, theta_new, alpha_sum) = jax.lax.fori_loop(0, L, step_fn, (key, theta, r0, n0, theta, 0.0))

    return theta_new, {'alpha': alpha_sum/L, 'n': (n-1)/L, 'd': jnp.sum(jnp.square(theta_new - theta))/jnp.sqrt(L)}

def matern52(x, y, σ=0.1):
    d = jnp.sqrt(jnp.sum(jnp.square(x-y)))
    return (1 + jnp.sqrt(5)*d/σ + 5/3 * d**2/σ**2) * jnp.exp(-jnp.sqrt(5)*d/σ)

# @partial(jax.jit, static_argnums=(2,3))
def ahmc(key, theta, logp, warmup_steps, εmin=1e-5, εmax=1.0, Lmin = 2, Lmax = 100):
    γ0 = jnp.array([0.5, 0.5])
    A = jnp.array([jnp.log(εmax) - jnp.log(εmin), 
                   Lmax - Lmin])
    B = jnp.array([jnp.log(εmin), Lmin])
    logε0, L0 = γ0*A+B
    ε0 = jnp.exp(logε0)
    L0 = jnp.clip(L0.astype(jnp.int32), Lmin, Lmax)

    α = 1.0

    key, subkey = jax.random.split(key)
    theta, metrics = hmc_kernel(subkey, theta, logp, ε0, L0)
    xs = [γ0]
    ds = [metrics['d']]


    gp = GP(partial(matern52, σ=0.2), 0.2)
    gp_alpha = GP(partial(matern52, σ=0.1), 0.2)

    array_ε = jnp.linspace(0.0, 1.0, 100)
    array_L = jnp.linspace(0.0, 1.0, 100)
    array_γ = einops.rearrange(jnp.stack(jnp.meshgrid(array_ε, array_L), axis=-1),
                               'a b c -> (a b) c')

    info = []

    for i in tqdm(range(warmup_steps)):
        rs = jnp.stack(ds)
        s = α/jnp.max(rs)
        rs = s * rs

        # upper confidence bound
        beta = jnp.sqrt(2 * jnp.log((i+1)**3 * jnp.pi**2/(3*0.1)))
        mean, var = gp.predictb(jnp.stack(xs), rs, array_γ)
        u = var #mean + beta * jnp.sqrt(var)
        γ = array_γ[jnp.argmax(u)]
        # if not jnp.all(jnp.isfinite(u)):
        #     exit('nyan')

        logε, L = γ * A + B
        ε = jnp.exp(logε)
        L = jnp.clip(L.astype(jnp.int32), Lmin, Lmax)

        key, subkey = jax.random.split(key)
        # theta, metrics = hmc_kernel(subkey, theta, logp, ε, L)
        theta, _, metrics = sample_hmc(subkey, theta, logp, 10, ε, L, False)
        xs.append(γ)
        ds.append(metrics['d'].mean())
        metrics = {'d': metrics['d'].mean()}
        metrics['ε'] = ε
        metrics['L'] = L
        metrics['γ'] = γ
        metrics['logp'] = logp(theta)
        info.append(metrics)

    xs = jnp.stack(xs)
    ds = jnp.stack(ds)

    s = α/jnp.max(ds)
    rs = s * ds

    # lower confidence bound (conservative estimate of ε and L)
    mean, var = gp.predictb(xs, rs, array_γ)
    u = mean - 2*jnp.sqrt(var)
    γ = array_γ[jnp.argmax(u)]
    logε, L = γ * A + B
    ε = jnp.exp(logε)
    L = jnp.clip(L.astype(jnp.int32), Lmin, Lmax)

    # print('final ε:', ε)
    # print('final L:', L)

    # import json
    # with open('stuff.json', 'w') as fp:
    #     json.dump({'xs': xs.tolist(), 'rs': rs.tolist()}, fp)

    keys = list(info[0].keys())
    info = {k:jnp.stack([info[i][k] for i in range(warmup_steps)]) for k in keys}

    return (theta, ε, L, info)

def normalcdf(x, mean, var):
    return 0.5 * jsp.special.erfc((mean - x)/(jnp.sqrt(2 * var)))

def ahmc_fast(key: jax.Array, theta: jax.Array, logp: Fn, warmup_steps: int, εmin=1e-5, εmax=1.0, Lmin = 2, Lmax = 100, tqdm=True):
    return ahmc_fast_(key, theta, fn(logp), warmup_steps, εmin, εmax, Lmin, Lmax, tqdm)

@partial(jax.jit, static_argnums=(3,8))
def ahmc_fast_(key, theta, logp: Fn, warmup_steps, εmin, εmax, Lmin, Lmax, tqdm):
    # γ0 = jnp.array([0.5, 0.5])
    A = jnp.array([jnp.log(εmax) - jnp.log(εmin), 
                   Lmax - Lmin])
    B = jnp.array([jnp.log(εmin), Lmin])
    # logε0, L0 = γ0*A+B
    # ε0 = jnp.exp(logε0)
    # L0 = jnp.clip(L0.astype(jnp.int32), Lmin, Lmax)

    α = 4.0

    # key, subkey = jax.random.split(key)
    # theta, metrics = hmc_kernel(subkey, theta, logp, ε0, L0)
    xs = jnp.zeros([warmup_steps, 2])#.at[0].set(γ0)
    ds = jnp.zeros([warmup_steps])#.at[0].set(metrics['d'])
    ts = jnp.linspace(0, 1, warmup_steps)
    alphas = jnp.zeros([warmup_steps])

    gp = GP(partial(matern52, σ=0.05), 0.1)
    gp_alpha = GP(partial(matern52, σ=0.2), 0.2)

    array_ε = jnp.linspace(0.0, 1.0, 100)
    array_L = jnp.linspace(0.0, 1.0, 100)
    array_γ = einops.rearrange(jnp.stack(jnp.meshgrid(array_ε, array_L), axis=-1),
                               'a b c -> (a b) c')

    key, subkey = jax.random.split(key)
    carry = (subkey, theta, xs, ds, alphas)

    @(scan_tqdm(warmup_steps, desc='Adaptive HMC warmup') if tqdm else identity)
    def scan_fn(carry, i):
        key, theta, xs, ds, alphas = carry
        
        rs = ds * α/jnp.clip(jnp.max(ds), 1e-8)

        alpha_pred = gp_alpha.predictb_masked(jnp.stack([xs[:,0], ts], axis=-1), alphas, i, jnp.stack([array_γ[:,0], jnp.broadcast_to(ts[i],[10000])], axis=-1))
        p_middle = normalcdf(0.90, alpha_pred[0], alpha_pred[1]) - normalcdf(0.10, alpha_pred[0], alpha_pred[1])

        # upper confidence bound
        beta = 2 * jnp.sqrt(2 * jnp.log((i+1)**3 * jnp.pi**2/(3*0.1)))
        mean, var = gp.predictb_masked(xs, rs, i, array_γ)
        u = mean + beta * jnp.sqrt(var)
        ix = jnp.argmax(u * (p_middle+0.0001))
        γ = array_γ[ix]

        logε, L = γ * A + B
        ε = jnp.exp(logε)
        L = jnp.clip(L.astype(jnp.int32), Lmin, Lmax)

        # jax.debug.print('choosing next (ε,L): ({},{}) - prediction = {}±{} - (logp = {})', ε, L, mean[ix], jnp.sqrt(var[ix]), logp(theta))
        key, subkey = jax.random.split(key)
        theta, metrics_ = hmc_kernel(subkey, theta, logp, ε, L)
        # theta, _, metrics_ = sample_hmc(subkey, theta, logp, 10, ε, L, False)
        xs = xs.at[i].set(γ)
        ds = ds.at[i].set(metrics_['d'].mean())
        alphas = alphas.at[i].set(metrics_['alpha'].mean())
        metrics = {}
        metrics['d'] = metrics_['d'].mean()
        metrics['ε'] = ε
        metrics['α'] = metrics_['alpha'].mean()
        metrics['L'] = L
        metrics['γ'] = γ
        metrics['logp'] = logp(theta)
        return (key, theta, xs, ds, alphas), (metrics)

    (key, theta, xs, ds, alphas), (info) = jax.lax.scan(scan_fn, carry, jnp.arange(warmup_steps))

    s = α/jnp.max(ds)
    rs = s * ds

    # lower confidence bound (conservative estimate of ε and L)
    alpha_pred = gp_alpha.predictb(jnp.stack([xs[:,0], ts], axis=-1), alphas, jnp.stack([array_γ[:,0], jnp.broadcast_to(1.0,[10000])], axis=-1))
    p_middle = normalcdf(0.90, alpha_pred[0], alpha_pred[1]) - normalcdf(0.10, alpha_pred[0], alpha_pred[1])
    mean, var = gp.predictb(xs, rs, array_γ)
    u = mean - 2*jnp.sqrt(var)
    ix = jnp.argmax(u * (p_middle+0.0001))
    γ = array_γ[ix]
    logε, L = γ * A + B
    ε = jnp.exp(logε)
    L = jnp.clip(L.astype(jnp.int32), Lmin, Lmax)

    # import json
    # with open('stuff.json', 'w') as fp:
    #     json.dump({'xs': xs.tolist(), 'rs': rs.tolist()}, fp)

    return (theta, ε, L, info)

identity = lambda x: x

def sample_hmc(key, theta, logp, n_steps, ε, L, tqdm=True):
    return sample_hmc_(key, theta, fn(logp), n_steps, ε, L, tqdm)

# @partial(jax.jit, static_argnums=(3,6))
def sample_hmc_(key, theta, logp: Fn, n_steps, ε, L, tqdm):
    @(scan_tqdm(n_steps) if tqdm else identity)
    def step_fn(carry, i):
        key, theta = carry
        key, subkey = jax.random.split(key)
        theta, metrics = hmc_kernel(subkey, theta, logp, ε, L)
        return (key, theta), (theta, metrics)
    (_, theta_new), (chain, metrics) = jax.lax.scan(step_fn, (key, theta), jnp.arange(n_steps))
    return theta_new, chain, metrics

if __name__ == '__main__':
    import tabulate
    def logp(θ):
        # return -0.5 * (jnp.square(θ[0]/θ[1]) + jnp.square(θ[1]) + jnp.square(θ[0]))
        x = θ[0]
        y = θ[1]
        return -0.5 * (jnp.sqrt(x**2+y**2)-1)**2/0.001
    
    key = jax.random.PRNGKey(0)
    theta = jax.random.normal(jax.random.PRNGKey(1), [2])
    warmup_steps = 100
    (theta, ε, L, info) = ahmc_fast(key, theta, logp, warmup_steps, εmin=1e-8, εmax=1.0, Lmin = 2, Lmax = 100)

    import json
    with open('stuff.json', 'w') as fp:
        json.dump({'xs': info['γ'].tolist(), 'ds': info['d'].tolist(), 'αs': info['α'].tolist()}, fp)

    keys = list(info.keys())
    print(tabulate.tabulate([[info[k][i] for k in keys] for i in range(warmup_steps)], headers=keys))
    print(f'ε={ε} L={L}')

    _, chain, metrics = sample_hmc(key, theta, logp, 1000, ε, L)
    print({k:v.mean() for (k,v) in metrics.items()})
    
    from matplotlib import pyplot as plt
    plt.scatter(chain[:,0], chain[:,1])
    plt.show()

    