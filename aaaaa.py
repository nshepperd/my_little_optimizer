import jax
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm
import einops
import json

from ngp.gaussian_process import GP

stuff = json.load(open('stuff.json', 'r'))
xs = jnp.array(stuff['xs'])
rs = jnp.array(stuff['rs'])

# vals = dict()
# for (γ, r) in zip(xs, rs):
#     γ = tuple(γ.tolist())
#     if γ not in vals:
#         vals[γ] = []
#     vals[γ].append(r)
# xs = jnp.array([k for k in vals.keys()])
# rs = jnp.array([sum(vals[k])/len(vals[k]) for k in vals.keys()])

# print(xs, rs)

def kernel(x, y):
    σ = 0.2
    return jnp.exp(-0.5 * jnp.sum(jnp.square(x-y)/σ**2))
gp = GP(kernel, 0.1)

array_ε = jnp.linspace(0.0, 1.0, 100)
array_L = jnp.linspace(0.0, 1.0, 100)
array_γ = einops.rearrange(jnp.stack(jnp.meshgrid(array_ε, array_L), axis=-1),
                            'a b c -> (a b) c')

prec = gp.calc_precision(jnp.stack(xs))
print(xs.shape, prec.shape)
mean, var = gp.predictb(xs, rs, prec, array_γ)
u = mean - 2*jnp.sqrt(var)
γ = array_γ[jnp.argmax(u)]

# plot mean on a heatmap
from matplotlib import pyplot as plt
# Reshape array_γ predictions back into 2D grid
mean_2d = einops.rearrange(mean, '(a b) -> a b', a=100, b=100)
var_2d = einops.rearrange(var, '(a b) -> a b', a=100, b=100)

# Create figure with two subplots - mean and variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot mean
im1 = ax1.imshow(mean_2d, origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(im1, ax=ax1)
ax1.set_title('GP Mean')
ax1.set_xlabel('ε')
ax1.set_ylabel('L')

# Plot variance
im2 = ax2.imshow(var_2d, origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(im2, ax=ax2)
ax2.set_title('GP Variance') 
ax2.set_xlabel('ε')
ax2.set_ylabel('L')

# Plot observation points
xs_array = jnp.stack(xs)
ax1.scatter(xs_array[:, 0], xs_array[:, 1], c='red', s=20, label='Observations')
ax2.scatter(xs_array[:, 0], xs_array[:, 1], c='red', s=20, label='Observations')

# plot argmax
ax1.scatter(γ[None, 0], γ[None, 1], c='blue', s=20, label='Argmax')
ax2.scatter(γ[None, 0], γ[None, 1], c='blue', s=20, label='Argmax')


plt.tight_layout()
plt.show()
