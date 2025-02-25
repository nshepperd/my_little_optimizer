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

def kfac(Y):
    k = len(Y)

    m, n = Y[0].shape
    P = jnp.identity(m)
    Q = jnp.identity(n)

    def step_fn(i, carry):
        P, Q = carry
        Q_inv = jnp.linalg.inv(Q)
        Ph = jax.vmap(lambda Yj: Yj @ Q_inv @ Yj.T)(Y).mean(0) / n
        Ph_inv = jnp.linalg.inv(Ph)
        Qh = jax.vmap(lambda Yj: Yj.T @ Ph_inv @ Yj)(Y).mean(0) / m
        # Ph = sum(Y[j] @ Y[j].T for j in range(k)) / (n*k)
        # Qh = sum(Y[j].T @ Y[j] for j in range(k)) / (m*k)
        A = Ph.square().sum().sqrt()
        B = Qh.square().sum().sqrt()
        T = (A*B).sqrt()
        P = Ph * (T/A)
        Q = Qh * (T/B)
        return (P, Q)
    P, Q = jax.lax.fori_loop(0, 4, step_fn, (P,Q))
    return P, Q

if __name__ == '__main__':
    Y = jax.random.normal(jax.random.PRNGKey(0), [1000, 2,2])
    A, B = jax.random.normal(jax.random.PRNGKey(2), [2,2,2])**2
    print('A:', A)
    print('B:', B)
    Y = jax.vmap(lambda Y: A @ Y @ B)(Y)
    P, Q = kfac(Y)
    print(jnp.linalg.inv(jnp.linalg.cholesky(P)))
    print(jnp.linalg.inv(jnp.linalg.cholesky(Q)).T)
    dat = jnp.linalg.inv(jnp.linalg.cholesky(P)) @ Y @ jnp.linalg.inv(jnp.linalg.cholesky(Q)).T
    print(kfac(dat))
# Y = 10*jnp.array([jax.random.normal(jax.random.PRNGKey(i), [2,2]) for i in range(100)])
# P, Q = kfac(Y)
# print(P)
# print(Q)
# print(P[:,:,None,None] * Q[None,None,:,:])