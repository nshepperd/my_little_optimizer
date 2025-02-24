import jax
import jax.numpy as jnp
import math
from functools import partial
from dataclasses import dataclass
from typing import Generic, TypeVar

from my_little_optimizer.switchvec import switchvec
from my_little_optimizer.num import erfcx

def h(z):
    pdf = jax.scipy.stats.norm.pdf
    cdf = jax.scipy.stats.norm.cdf
    return pdf(z) + z * cdf(z)

# @jax.custom_vjp
def log_h(z):
    pdf = jax.scipy.stats.norm.pdf
    cdf = jax.scipy.stats.norm.cdf
    def log_h_case1(z): # -1 < z
        return jnp.log(pdf(z) + z * cdf(z))
    def log_h_case2(z): # -∞ < z < -1
        sqrt_2pi = math.sqrt(2 * math.pi)
        r = math.sqrt(0.5 * math.pi) * erfcx(-z * math.sqrt(0.5))
        return -0.5 * z**2 + jnp.log1p(z * r) - jnp.log(sqrt_2pi)
    # def log_h_case3(z):
    #     c1 = jnp.log(2*math.pi)/2
    #     return -z*z/2 - c1 - 2 * jnp.log(jnp.abs(z)) # z < -1/sqrt(eps)
    ix = jnp.where(z>-1, 0, 1)
    return switchvec(ix, [log_h_case1, log_h_case2], z)
