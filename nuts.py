import jax
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple
from functools import partial


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

def nuts_kernel(key, theta, logp, logp_grad, ε):
    key, *subkeys = jax.random.split(key, 3)
    r0 = jax.random.normal(subkeys[0], theta.shape)
    H0 = compute_hamiltonian(theta, r0, logp)
    u0 = jax.random.uniform(subkeys[1], []) # u = u0 * exp(-H0)
    theta_minus = theta
    theta_plus = theta
    r_minus = r0
    r_plus = r0
    j = 0
    theta_new = theta
    n = 1
    s = 1
    while s:
        key, *subkeys = jax.random.split(key, 4)
        v = rademacher_int(subkeys[0], [])
        if v == -1:
            # theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime = build_tree(subkeys[1], theta_minus, r_minus, logp, logp_grad, H0, u0, v, j, ε)
            theta_minus, r_minus, theta_prime, n_prime, s_prime, metrics = build_tree_fwd(subkeys[1], theta_minus, r_minus, logp, logp_grad, H0, u0, v, j, ε)
        else:
            theta_plus, r_plus, theta_prime, n_prime, s_prime, metrics = build_tree_fwd(subkeys[1], theta_plus, r_plus, logp, logp_grad, H0, u0, v, j, ε)
        if s_prime:
            if jax.random.bernoulli(subkeys[2], jnp.minimum(1, n_prime / n)):
                theta_new = theta_prime
        n += n_prime
        s = jnp.logical_and(s_prime, no_uturn(theta_minus, theta_plus, r_minus, r_plus))
        j += 1
    # if metrics['count_uturn_fail'] > 0:
    #     print('count_uturn_fail', metrics['count_uturn_fail'])
    #     print('count_slice_fail', metrics['count_slice_fail'])
    #     print('count_energy_fail', metrics['count_energy_fail'])
    #     print('log', metrics['log'])
    #     # plot the points in the log with the velocity direction as arrows
    #     log_thetas, log_rs = zip(*metrics['log'])
    #     log_thetas = jnp.stack(log_thetas)
    #     log_rs = jnp.stack(log_rs)
    #     plt.scatter(log_thetas[:,0], log_thetas[:,1], color='blue')
    #     plt.quiver(log_thetas[:,0], log_thetas[:,1], log_rs[:,0], log_rs[:,1], color='red', angles='xy')
    #     plt.show()
    #     exit()
    return theta_new

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
@partial(jax.jit, static_argnums=(3,4,8,10))
def build_tree_fwd(key: jax.Array, theta: jax.Array, r: jax.Array, logp: callable, logp_grad: callable, 
                   H0: jax.Array, u0: jax.Array, v: jax.Array, j: int, ε: jax.Array, return_log=False):
    """Recursively build tree"""
    rng = PRNG(key)
    stack = jnp.zeros((j+1, 2, *theta.shape))

    result = theta

    n = 0
    s = 1

    r = v*r

    count_slice_fail = 0
    count_energy_fail = 0
    count_uturn_fail = 0

    log = []

    for i in range(2**j):
        theta, r = leapfrog(theta, r, logp_grad, +ε)
        H = compute_hamiltonian(theta, r, logp)
        n_prime = jnp.log(u0) <= H0 - H
        s_prime = jnp.log(u0) <= H0 - H + delta_max
        n = n_prime + n
        s = jnp.logical_and(s_prime, s)

        count_slice_fail += n_prime == 0
        count_energy_fail += s_prime == 0

        select = jax.random.bernoulli(rng.split(), n_prime / jnp.clip(n,1))
        result = jnp.where(select, theta, result)

        theta_m, r_m, theta_p, r_p = theta, r, theta, r

        if j > 0:
            ks = jnp.arange(j+1)
            subtree_start = (i % (2**(ks+1)) == 0)
            stack = jnp.where(broadcast_right(subtree_start, stack.shape), jnp.stack([theta_m, r_m]), stack)

            # for k in range(j):
            #     if i % (2**(k+1)) == 0:
            #         # starts the subtree at this level
            #         stack = stack.at[k].set(jnp.stack([theta_m, r_m]))

            subtree_end = (i+1) % (2**(ks+1)) == 0
            nou = no_uturn(stack[:,0], theta_p, stack[:,1], r_p)
            nou = jnp.where(broadcast_right(subtree_end, nou.shape), nou, 1)
            nou = jnp.all(nou)
            s = jnp.logical_and(s, nou)
            count_uturn_fail += nou == 0

            # for k in range(j):
            #     if (i+1) % (2**(k+1)) == 0:
            #         # completed the subtree, combine two parts and check u turn
            #         theta_m, r_m = stack[k]
            #         nou = no_uturn(theta_m, theta_p, r_m, r_p)
            #         s = jnp.logical_and(s, nou)
            #         count_uturn_fail += nou == 0
            #     else:
            #         break

        if return_log:
            log.append((theta, r))
        # if not s:
        #     break
    
    metrics = {'count_slice_fail':count_slice_fail, 'count_energy_fail':count_energy_fail, 'count_uturn_fail':count_uturn_fail, 'log':log}

    return theta, v*r, result, n, s, metrics


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

    samples = []
    for _ in tqdm(range(1000)):
        theta = nuts_kernel(rng.split(), theta, logp, logp_grad, 0.05)
        samples.append(theta)
    samples = jnp.stack(samples)
    # print(einops.rearrange(samples, '(x n) d -> x n d', x=20).std(1))
    print(samples.std(0), jax.random.normal(rng.split(),samples.shape).std(0))
    # scatter plot
    plt.scatter(samples[:,0], samples[:,1])
    plt.plot(samples[:,0], samples[:,1], color='red', alpha=0.1)
    plt.show()

