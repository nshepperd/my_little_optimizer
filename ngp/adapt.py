import jax
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple, Protocol, Dict, Any, List
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm

from ngp.nuts import nuts_kernel, leapfrog, compute_hamiltonian

def find_reasonable_epsilon(key, theta, logp, target=0.5, max_steps=50):
    logp_grad = jax.grad(logp)
    r = jax.random.normal(key, theta.shape)
    H0 = compute_hamiltonian(theta, r, logp)
    # p = exp(-H)
    ε = 1.0
    theta_new, r_new = leapfrog(theta, r, logp_grad, ε)
    def calc_a(theta_new, r_new):
        H = compute_hamiltonian(theta_new, r_new, logp)
        H = jnp.where(jnp.isnan(H), -jnp.inf, H)
        return 2 * (H0 - H > jnp.log(target)) - 1
    a = calc_a(theta_new, r_new)

    init_state = (theta_new, r_new, ε)
    def cond_fn(state):
        (theta_new, r_new, ε) = state
        return calc_a(theta_new, r_new) == a
    def body_fn(state):
        (theta_new, r_new, ε) = state
        ε *= 2.0**a
        theta_new, r_new = leapfrog(theta, r, logp_grad, ε)
        jax.debug.print('ε: {}, a: {}, p: {}', ε, calc_a(theta_new, r_new), jnp.exp(H0 - compute_hamiltonian(theta_new, r_new, logp)))
        return (theta_new, r_new, ε)

    state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    (theta_new, r_new, ε) = state
    ε *= 0.4
    theta_new, r_new = leapfrog(theta, r, logp_grad, ε)
    return ε, jnp.exp(H0 - compute_hamiltonian(theta_new, r_new, logp))

def warmup_with_dual_averaging(key, theta, logp, ε0, δ=0.5, warmup_steps=20, pbar=True):
    return warmup_with_dual_averaging_(key, theta, logp, ε0, δ, warmup_steps, pbar)

@partial(jax.jit,static_argnums=(2,5,6))
def warmup_with_dual_averaging_(key, theta, logp, ε0, δ, warmup_steps, pbar):
    logp_grad = jax.grad(logp)

    γ = 0.1
    t0 = 10
    κ = 0.75
    µ = jnp.log(10 * ε0)

    hbar0 = 0.0
    log_ε0 = jnp.log(ε0)
    log_εbar0 = jnp.log(1.0)

    init_state = (key, theta, hbar0, log_ε0, log_εbar0)

    @(scan_tqdm(warmup_steps, desc='Step size adaptation...') if pbar else lambda x: x)
    def scan_fn(carry, i):
        (key, theta, hbar, log_ε, log_εbar) = carry
        prob = logp(theta)
        key, subkey = jax.random.split(key)
        theta, alpha, ms = nuts_kernel(subkey, theta, logp, logp_grad, jnp.exp(log_ε))
        m = i*jnp.clip(25/warmup_steps, max=1.0) + 1.0
        # jax.debug.print('warmup step {}: hbar={}, log_ε={}, log_εbar={}, alpha={}', i, hbar, log_ε, log_εbar, alpha)
        hbar = (1 - 1/(m+t0)) * hbar + 1/(m+t0) * jnp.mean(δ-alpha)
        log_ε = µ - jnp.sqrt(m)/γ * hbar
        log_εbar = m**(-κ) * log_ε + (1-m**(-κ)) * log_εbar

        return (key, theta, hbar, log_ε, log_εbar), {'alphas': alpha, 'probs':prob, 'log_ε':log_ε, 'hbar':hbar}
    
    state, metrics = jax.lax.scan(scan_fn, init_state, jnp.arange(warmup_steps))
    (key, theta, hbar, log_ε, log_εbar) = state
    return theta, jnp.exp(log_εbar), metrics
