import jax
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple
from functools import partial
from dataclasses import dataclass
from jax_tqdm import scan_tqdm
import einops
import json
import jax.scipy as jsp

from ngp.gaussian_process import GP

stuff = json.load(open('stuff.json', 'r'))
xs = jnp.array(stuff['xs'])
ds = jnp.array(stuff['ds'])
alphas = jnp.array(stuff['αs'])

print(xs, ds)
print('Maximum of ds:', jnp.max(ds))

rs = ds


# vals = dict()
# for (γ, r) in zip(xs, rs):
#     γ = tuple(γ.tolist())
#     if γ not in vals:
#         vals[γ] = []
#     vals[γ].append(r)
# xs = jnp.array([k for k in vals.keys()])
# rs = jnp.array([sum(vals[k])/len(vals[k]) for k in vals.keys()])

# print(xs, rs)

def normalcdf(x, mean, var):
    return 0.5 * jsp.special.erfc((mean - x)/(jnp.sqrt(2 * var)))

def matern52(x, y, σ=0.1):
    d = jnp.sqrt(jnp.sum(jnp.square(x-y)))
    return (1 + jnp.sqrt(5)*d/σ + 5/3 * d**2/σ**2) * jnp.exp(-jnp.sqrt(5)*d/σ)
gp = GP(matern52, 0.5)
gp_alpha = GP(partial(matern52, σ=0.2), 0.2)

array_ε = jnp.linspace(0.0, 1.0, 100)
array_L = jnp.linspace(0.0, 1.0, 100)
array_γ = einops.rearrange(jnp.stack(jnp.meshgrid(array_ε, array_L), axis=-1),
                            'a b c -> (a b) c')

ts = jnp.linspace(0,1, xs.shape[0])
array_γα = einops.rearrange(jnp.stack(jnp.meshgrid(array_ε, jnp.linspace(0,1, 100)), axis=-1),
                            'a b c -> (a b) c')

alpha_pred = gp_alpha.predictb(jnp.stack([xs[:,0], ts],axis=-1), 
                               alphas, 
                               array_γα)
p_middle = normalcdf(0.95, alpha_pred[0], alpha_pred[1]) - normalcdf(0.05, alpha_pred[0], alpha_pred[1])

# jnp.stack([xs[:,0], jnp.linspace(0,1, xs.shape[0])],axis=-1)
xs_t = jnp.concatenate([xs, ts[:,None]],axis=-1)
array_γ_t = jnp.concatenate([array_γ, jnp.full([100*100,1], 1.0)], axis=-1)
mean, var = gp.predictb(xs_t, rs, array_γ_t)
u = mean - 2*jnp.sqrt(var)
γ = array_γ[jnp.argmax(u)]

# plot mean on a heatmap
from matplotlib import pyplot as plt
# Reshape array_γ predictions back into 2D grid
mean_2d = einops.rearrange(mean, '(a b) -> a b', a=100, b=100)
var_2d = einops.rearrange(var, '(a b) -> a b', a=100, b=100)

# Create figure with two subplots - mean and variance
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

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

im3 = ax3.imshow(einops.rearrange(p_middle, '(a b) -> a b', a=100, b=100), origin='lower', extent=[0, 1, 0, 1])
ax3.scatter(xs_array[:, 0], ts, c='red', s=20, label='Observations')
plt.colorbar(im3, ax=ax3)


# ax3.plot(ds)

plt.tight_layout()
plt.show()
