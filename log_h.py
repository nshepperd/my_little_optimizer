from num import erfcx
import jax
import jax.numpy as jnp
import math

def h(z):
    pdf = jax.scipy.stats.norm.pdf
    cdf = jax.scipy.stats.norm.cdf
    return pdf(z) + z * cdf(z)

# def erfcx(z):
#     return jnp.exp(z*z) * jax.lax.erfc(z)
def log_h_case1(z):
    pdf = jax.scipy.stats.norm.pdf
    cdf = jax.scipy.stats.norm.cdf
    return jnp.log(pdf(z) + z * cdf(z)) # z > -1
def log_h_case2(z):
    # c1 = jnp.log(2*math.pi)/2
    # c2 = jnp.log(math.pi/2)/2
    # return -z*z/2 - c1 + jnp.log1p(-jnp.exp(c2)*erfcx(-z/jnp.sqrt(2))*jnp.abs(z)) # z > -1/sqrt(eps)
    sqrt_2pi = math.sqrt(2 * math.pi)
    r = math.sqrt(0.5 * math.pi) * erfcx(-z * math.sqrt(0.5))
    return -0.5 * z**2 + jnp.log1p(z * r) - jnp.log(sqrt_2pi)
def log_h_case3(z):
    c1 = jnp.log(2*math.pi)/2
    return -z*z/2 - c1 - 2 * jnp.log(jnp.abs(z)) # z < -1/sqrt(eps)

@jax.custom_vjp
def log_h(z):
    return jnp.where(z>-1, log_h_case1(z), 
                        jnp.where(z>-1000, log_h_case2(z), log_h_case3(z)))

def log_h_fwd(z):
    return log_h(z), (z,)
def log_h_bwd(res, g):
    z, = res
    grad_case1 = jax.vjp(log_h_case1, z)[1](g)[0]
    grad_case2 = jax.vjp(log_h_case2, z)[1](g)[0]
    grad_case3 = jax.vjp(log_h_case3, z)[1](g)[0]
    return jnp.where(z>-1, grad_case1, 
                        jnp.where(z>-1000, grad_case2, grad_case3)),
log_h.defvjp(log_h_fwd, log_h_bwd)