import jax
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm


def leapfrog(theta: jax.Array, r: jax.Array, logp_grad, eps: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Single leapfrog step"""
    r_half = r + 0.5 * eps * logp_grad(theta)
    theta_new = theta + eps * r_half
    r_new = r_half + 0.5 * eps * logp_grad(theta_new)
    return theta_new, r_new

def compute_hamiltonian(theta, r, logp):
    """Compute Hamiltonian (smaller is better)"""
    return -logp(theta) + 0.5 * jnp.sum(r**2)

def no_uturn(theta_minus, theta_plus, r_minus, r_plus, axis=-1):
    """Check U-turn condition"""
    diff = theta_plus - theta_minus
    return jnp.logical_and(jnp.sum(diff * r_minus, axis=axis) >= 0, 
                           jnp.sum(diff * r_plus, axis=axis) >= 0)

def rademacher_int(key, shape):
    # returns -1 or 1
    return jax.random.bernoulli(key, 0.5, shape) * 2 - 1

MAX_J=10
@partial(jax.jit, static_argnums=(2,3))
def nuts_kernel(key, theta, logp, logp_grad, ε):
    key, *subkeys = jax.random.split(key, 3)
    r0 = jax.random.normal(subkeys[0], theta.shape)
    H0 = compute_hamiltonian(theta, r0, logp)
    u0 = jax.random.uniform(subkeys[1], []) # u = u0 * exp(-H0)
    theta_m = theta
    theta_p = theta
    r_m = r0
    r_p = r0
    j = 0
    theta_new = theta
    n = 1
    s = jnp.array(True)

    state = (key, theta_m, theta_p, r_m, r_p, j, theta_new, n, s, 0.0)

    def cond_fn(state):
        (key, theta_m, theta_p, r_m, r_p, j, theta_new, n, s, alpha) = state
        return jnp.logical_and(s, j<=MAX_J)

    def body_fn(state):
        (key, theta_m, theta_p, r_m, r_p, j, theta_new, n, s, alpha) = state
        key, *subkeys = jax.random.split(key, 4)
        
        v = rademacher_int(subkeys[0], [])
        theta_n, r_n, theta_prime, n_prime, s_prime, alpha, metrics = build_tree_fwd(
            subkeys[1], jnp.where(v==1, theta_p, theta_m), jnp.where(v==1, r_p, r_m), logp, logp_grad, H0, u0, v, MAX_J, j, ε)
        theta_p = jnp.where(v==1, theta_n, theta_p)
        r_p = jnp.where(v==1, r_n, r_p)
        theta_m = jnp.where(v==-1, theta_n, theta_m)
        r_m = jnp.where(v==-1, r_n, r_m)

        theta_new = jnp.where(jnp.logical_and(s_prime, jax.random.bernoulli(subkeys[2], jnp.minimum(1, n_prime / n))),
                              theta_prime, theta_new)
        n += n_prime
        s = jnp.logical_and(s_prime, no_uturn(theta_m, theta_p, r_m, r_p))
        j += 1
        return (key, theta_m, theta_p, r_m, r_p, j, theta_new, n, s, alpha)

    state = jax.lax.while_loop(cond_fn, body_fn, state)
    (key, theta_m, theta_p, r_m, r_p, j, theta_new, n, s, alpha) = state

    metrics = {}
    metrics['j'] = j-1
    return theta_new, alpha, metrics

def build_tree(key, theta, r, logp, logp_grad, H0, u0, v, j, ε):
    """Recursively build tree"""
    # print('build_tree', v, j, theta, r, jnp.log(u0) - H0)
    subkeys = jax.random.split(key, 3)
    if j == 0:
        # Base case - take one leapfrog step
        theta_new, r_new = leapfrog(theta, r, logp_grad, v * ε)
        
        # Check slice constraint
        # If the new point is below slice constraint, reject it
        # If it's so far below slice constraint that we're probably doomed beyond this point from integration error, stop the whole sampling loop
        H = compute_hamiltonian(theta_new, r_new, logp)
        # u <= exp(-H)
        # u0 <= exp(H0-H)
        delta_max = 1000.0
        n_prime = jnp.log(u0) <= H0 - H
        s_prime = jnp.log(u0) <= H0 - H + delta_max

        return theta_new, r_new, theta_new, r_new, theta_new, n_prime, s_prime
    else:
        # Recursion - build left and right subtrees
        theta_m, r_m, theta_p, r_p, theta_prime, n_prime, s_prime = \
            build_tree(subkeys[0], theta, r, logp, logp_grad, H0, u0, v, j-1, ε)
        
        if s_prime:
            if v == -1:
                theta_m, r_m, _, _, theta_pp, n_pp, s_pp = \
                    build_tree(subkeys[1], theta_m, r_m, logp, logp_grad, H0, u0, v, j-1, ε)
            else:
                _, _, theta_p, r_p, theta_pp, n_pp, s_pp = \
                    build_tree(subkeys[1], theta_p, r_p, logp, logp_grad, H0, u0, v, j-1, ε)
            
            if jax.random.bernoulli(subkeys[2], n_pp / (n_prime + n_pp)):
                theta_prime = theta_pp
            s_prime = jnp.logical_and(s_pp, no_uturn(theta_m, theta_p, r_m, r_p))
            n_prime = n_prime + n_pp
        return theta_m, r_m, theta_p, r_p, theta_prime, n_prime, s_prime

def broadcast_right(x, shape):
    while len(x.shape) < len(shape):
        x = x[...,None]
    return x

delta_max = 1000.0
@partial(jax.jit, static_argnums=(3,4,8))
def build_tree_fwd(key: jax.Array, theta: jax.Array, r: jax.Array, logp: callable, logp_grad: callable, 
                   H0: jax.Array, u0: jax.Array, v: jax.Array, max_j: int, j: jax.Array, ε: jax.Array): 
    """Sequential implementation of build_tree.

    To make this work without recursion, we use a "stack" (really just an array) to keep track of all current subtrees.
    When we start a subtree at any level, we set the start point of that level to the current point.
    When we end a subtree, we check the u turn condition wrt the beginning and end (ie. the current point) of the subtree.
    This can be fully vectorized, removing all data dependent control flow.

    It is probably possible to do early stopping (stop the loop when s=0) with a jax.lax.while_loop, but this is difficult to support with vmap.
    """

    def step_fn(i, carry):
        key, theta, r, result, n, s, stack, alpha_sum, (count_slice_fail, count_energy_fail, count_uturn_fail) = carry
        key, subkey = jax.random.split(key)

        # Leapfrog step
        theta, r = leapfrog(theta, r, logp_grad, +ε)
        H = compute_hamiltonian(theta, r, logp)

        # Step size adaptation statistic
        # logp(θ,r) = exp(-H)
        alpha_sum += jnp.exp(jnp.clip(H0 - H, max=0.0))

        # Check slice sampling and energy condition
        n_prime = jnp.log(u0) <= H0 - H
        s_prime = jnp.log(u0) <= H0 - H + delta_max
        n = n_prime + n
        s = jnp.logical_and(s_prime, s)

        count_slice_fail += n_prime == 0
        count_energy_fail += s_prime == 0

        # If the new point passes slice sampling, accept it with p=1/n (ensures uniform sampling from all valid points)
        select = jax.random.bernoulli(subkey, n_prime / jnp.clip(n,1))
        result = jnp.where(select, theta, result)

        # === Check u turn condition for all subtrees ===
        ks = jnp.arange(max_j)

        subtree_start = jnp.logical_and(i % (2**(ks+1)) == 0, ks<j)
        stack = jnp.where(broadcast_right(subtree_start, stack.shape), jnp.stack([theta, r]), stack)

        subtree_end = jnp.logical_and((i+1) % (2**(ks+1)) == 0, ks<j)
        nou = no_uturn(stack[:,0], theta, stack[:,1], r)
        nou = jnp.where(broadcast_right(subtree_end, nou.shape), nou, 1)
        nou = jnp.all(nou)
        s = jnp.logical_and(s, nou)
        count_uturn_fail += nou == 0
        # === End of u turn condition ===

        return (key, theta, r, result, n, s, stack, alpha_sum, (count_slice_fail, count_energy_fail, count_uturn_fail))
    
    stack = jnp.zeros((max_j, 2, *theta.shape))
    result = theta
    n = 0
    s = jnp.array(True)
    r = v*r
    count_slice_fail = 0
    count_energy_fail = 0
    count_uturn_fail = 0
    alpha_sum = 0.0
    carry = (key, theta, r, result, n, s, stack, alpha_sum, (count_slice_fail, count_energy_fail, count_uturn_fail))
    carry = jax.lax.fori_loop(0, 2**j, step_fn, carry)
    (key, theta, r, result, n, s, stack, alpha_sum, (count_slice_fail, count_energy_fail, count_uturn_fail)) = carry

    metrics = {'count_slice_fail':count_slice_fail, 'count_energy_fail':count_energy_fail, 'count_uturn_fail':count_uturn_fail}

    return theta, v*r, result, n, s, alpha_sum/(2**j), metrics


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from jaxtorch import PRNG
    import einops

    @jax.jit
    def logp(theta):
        return -0.5 * jnp.sum(jnp.square(theta))
    logp_grad = jax.grad(logp)

    rng = PRNG(jax.random.PRNGKey(0))
    theta = jax.random.normal(rng.split(), [2])
    r = jax.random.normal(rng.split(), [2])

    def k_nuts(key, theta, ε):
        return nuts_kernel(key, theta, logp, logp_grad, 1.0)

    # print(jax.jit(jax.vmap(k_nuts, in_axes=(0,0,None))).lower(rng.split(3), jax.random.normal(rng.split(), [3,2]), 1.0).as_text())
    # exit()

    @jax.jit
    def sample(key, theta, ε):
        @scan_tqdm(30000)
        def body_fn(carry, i):
            key, theta = carry
            key, subkey = jax.random.split(key)
            theta, alpha, metrics = nuts_kernel(subkey, theta, logp, logp_grad, ε)
            return (key, theta), (theta, alpha, metrics)
        return jax.lax.scan(body_fn, (key, theta), jnp.arange(30000))[1]


    # samples = []
    # alphas = []
    # for _ in tqdm(range(30000)):
    #     theta, alpha, metrics = nuts_kernel(rng.split(), theta, logp, logp_grad, 1.0)
    #     samples.append(theta)
    #     alphas.append(alpha)
    samples, alphas, metrics = sample(rng.split(), theta, 1.0)
    samples = jnp.stack(samples)
    alphas = jnp.stack(alphas)
    print(samples.shape)
    # print(metrics)
    # print(einops.rearrange(samples, '(x n) d -> x n d', x=20).std(1))
    print(samples.std(0), jax.random.normal(rng.split(),samples.shape).std(0))
    print(alphas.mean())
    # scatter plot
    plt.scatter(samples[:,0], samples[:,1])
    plt.plot(samples[:,0], samples[:,1], color='red', alpha=0.1)
    plt.show()

