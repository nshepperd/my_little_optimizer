import jax
import jax.numpy as jnp


def leapfrog(theta, r, logp_grad, eps):
    """Single leapfrog step"""
    r_half = r + 0.5 * eps * logp_grad(theta)
    theta_new = theta + eps * r_half
    r_new = r_half + 0.5 * eps * logp_grad(theta_new)
    return theta_new, r_new


def compute_hamiltonian(theta, r, logp):
    """Compute Hamiltonian (smaller is better)"""
    return -logp(theta) + 0.5 * jnp.sum(r**2)

def no_uturn(theta_minus, theta_plus, r_minus, r_plus):
    """Check U-turn condition"""
    diff = theta_plus - theta_minus
    return jnp.logical_and(jnp.dot(diff, r_minus) >= 0, jnp.dot(diff, r_plus) >= 0)

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
            theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime = build_tree(subkeys[1], theta_minus, r_minus, logp, logp_grad, H0, u0, v, j, ε)
        else:
            _, _, theta_plus, r_plus, theta_prime, n_prime, s_prime = build_tree(subkeys[1], theta_plus, r_plus, logp, logp_grad, H0, u0, v, j, ε)
        if s_prime:
            if jax.random.bernoulli(subkeys[2], jnp.minimum(1, n_prime / n)):
                theta_new = theta_prime
        n += n_prime
        s = jnp.logical_and(s_prime, no_uturn(theta_minus, theta_plus, r_minus, r_plus))
        j += 1
    return theta_new

def build_tree(key, theta, r, logp, logp_grad, H0, u0, v, j, ε):
    """Recursively build tree"""
    print('build_tree', v, j, theta, r, jnp.log(u0) - H0)
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
        key, *subkeys = jax.random.split(key, 4)
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
    samples = []
    for _ in range(100):
        theta = nuts_kernel(rng.split(), theta, logp, logp_grad, 0.1)
        samples.append(theta)
    samples = jnp.stack(samples)
    # print(einops.rearrange(samples, '(x n) d -> x n d', x=20).std(1))
    print(samples.std(0), jax.random.normal(rng.split(),samples.shape).std(0))
    # scatter plot
    plt.scatter(samples[:,0], samples[:,1])
    plt.plot(samples[:,0], samples[:,1], color='red', alpha=0.1)
    plt.show()

