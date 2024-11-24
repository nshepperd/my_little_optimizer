import sys, os

import jax
import jax.numpy as jnp
import optax
import numpy as np
import math
from matplotlib import pyplot as plt
from jax_tqdm import scan_tqdm
from functools import partial
from jax.tree_util import tree_map
from dataclasses import dataclass
import eindex.array_api as EX
import einops

from jaxtorch import nn
from jaxtorch import PRNG, Context
import jaxtorch

from ngp.log_h import log_h
from ngp.metric import DictionaryMetric, ScalarMetric
from ngp.metric import DictionaryMetricEstimator, CovarianceMetricEstimator, KroneckerMetricEstimator
from ngp.nuts import nuts_kernel
from ngp.adapt import warmup_with_dual_averaging, find_reasonable_epsilon
from ngp.ahmc import ahmc_fast, sample_hmc

import optuna

if __name__ == '__main__':
    SWEEP_NAME='21-wooden-sweep'
    study = optuna.create_study(
        study_name=SWEEP_NAME,
        load_if_exists=True,
        storage="sqlite:///../vqgan-jax/optuna.db",
        direction="minimize",
        sampler=optuna.samplers.GPSampler(),
    )

    pars = list(study.trials[0].params.keys())
    dists = study.trials[0].distributions

    # FrozenTrial(number=0, state=1, values=[5.536185545195277], 
    #             datetime_start=datetime.datetime(2024, 9, 11, 20, 40, 32, 883724), 
    #             datetime_complete=datetime.datetime(2024, 9, 11, 20, 49, 32, 328135), 
    #             params={'max_lr': 0.7452580557427365, 
    #                     'wpe': 0.37182103810908307, 
    #                     'wae': 0.22778681549779659, 
    #                     '.attn.q_proj': 3.32734447768913, 
    #                     '.attn.k_proj': 0.26484074814870834,
    #                     '.attn.v_proj': 0.1379646667507215, 
    #                     '.attn.c_proj': 2.209878139013646,
    #                     '.cross_attn.q_proj': 1.199168619833259, 
    #                     '.cross_attn.o_proj': 0.12690310853632134, 
    #                     '.cross_attn.k_clip': 0.418888424747393, 
    #                     '.cross_attn.v_clip': 3.5323477153753364, 
    #                     '.cross_attn.k_tags': 2.139220814603158, 
    #                     '.cross_attn.v_tags': 3.761321312960273, 
    #                     '.ffn.up_proj': 6.065056101613057, 
    #                     '.ffn.up_gate': 1.126805132037476, 
    #                     '.ffn.down_proj': 0.34775468285277833, 
    #                     'lm_head': 0.1956715618027044}, 
    #             user_attrs={}, system_attrs={}, intermediate_values={}, 
    #             distributions={'max_lr': FloatDistribution(high=1.0, log=True, low=0.01, step=None), 'wpe': FloatDistribution(high=10.0, log=True, low=0.1, step=None), 'wae': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.attn.q_proj': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.attn.k_proj': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.attn.v_proj': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.attn.c_proj': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.cross_attn.q_proj': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.cross_attn.o_proj': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.cross_attn.k_clip': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.cross_attn.v_clip': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.cross_attn.k_tags': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.cross_attn.v_tags': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.ffn.up_proj': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.ffn.up_gate': FloatDistribution(high=10.0, log=True, low=0.1, step=None), '.ffn.down_proj': FloatDistribution(high=10.0, log=True, low=0.1, step=None), 'lm_head': FloatDistribution(high=10.0, log=True, low=0.1, step=None)}, 
    #             trial_id=1516, value=None)

    in_dim = len(pars)
    out_dim = 1

    dhigh = jnp.array([dists[p].high for p in pars])
    dlow = jnp.array([dists[p].low for p in pars])
    dmid = 0.5 * (jnp.log(dhigh) + jnp.log(dlow))
    dscale = 0.5 * (jnp.log(dhigh) - jnp.log(dlow))

    x = jnp.array([[trial.params[p] for p in pars] for trial in study.trials if trial.state==1])
    y = jnp.array([[trial.values[0]] for trial in study.trials if trial.state==1])

    x = (jnp.log(x) - dmid) / dscale

    print('data:')
    print(' x:', x.shape)
    print(' y:', y.shape)

    y = jnp.where(jnp.isfinite(y), y, jnp.max(y[jnp.isfinite(y)]))
    model = nn.Sequential([nn.Linear(in_dim, 20), nn.SiLU(), nn.Linear(20, 20), nn.SiLU(), nn.Linear(20, out_dim)])
    model.name_everything_()

    aq_steps = 100

    def mkscales(model):
        scales = {}
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                scales[mod.weight.name] = 1.0/math.sqrt(np.prod(mod.weight.shape[1:]))
                scales[mod.bias.name] = 1.0
        return scales

    def mkmetric(model):
        scales = mkscales(model)
        metrics = {}
        for p in model.parameters():
            metrics[p.name] = ScalarMetric(scales[p.name], p.shape)
        return DictionaryMetric(metrics)

    def mkmetric_estimator(model):
        part = {}
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                part[mod.weight.name] = KroneckerMetricEstimator()
                part[mod.bias.name] = CovarianceMetricEstimator()
        return DictionaryMetricEstimator(part)

    scales = mkscales(model)
    metric0 = mkmetric(model)
    estimator = mkmetric_estimator(model)

    def f_loss(metric, x, y, eps, σ_y=0.1):
        params = metric.unwhiten(eps)
        cx = Context(params, None)
        prediction = model(cx, x)
        log_p_y = -0.5 * jnp.sum(jnp.square(y - prediction)/σ_y**2)
        log_p_eps = sum(-0.5 * jnp.sum(jnp.square(params[p.name]/scales[p.name])) for p in model.parameters())
        return log_p_y + log_p_eps

    @jax.tree_util.register_pytree_node_class
    @dataclass
    class MCMC:
        n_steps: int
        n_chains: int

        def tree_flatten(self):
            return (), (self.n_steps, self.n_chains)
        @staticmethod
        def tree_unflatten(static, dynamic):
            return MCMC(*static, *dynamic)
        
        @jax.jit
        def find_reasonable_epsilon(self, key, eps, metric, x, y):
            ε, εp = find_reasonable_epsilon(key, eps, partial(f_loss, metric, x, y))
            return ε, εp

        def warmup_with_dual_averaging(self, key, eps, metric, x, y, ε, warmup_steps=100):
            eps, ε, metrics = warmup_with_dual_averaging(key, eps, partial(f_loss, metric, x, y), ε, warmup_steps=warmup_steps)
            jax.debug.print('Warmup finished with step size {}\n log probs {}\n log_ε {}\n log_εbar {}\n hbar {}', 
                            ε, metrics['probs'], metrics['log_ε'], metrics['alphas'], metrics['hbar'])
            return eps, ε, metrics

        @partial(jax.jit, static_argnums=6)
        def nuts_sample(self, key, eps, metric, x, y, n_steps, ε):
            def nuts_sample(key, theta_init, logp, n_steps, step_size):
                @scan_tqdm(n_steps)
                def step_fn(carry, i):
                    theta, key = carry
                    key, subkey = jax.random.split(key)
                    theta_new, alpha, ms = nuts_kernel(subkey, theta, logp, jax.grad(logp), step_size)
                    return (theta, key), (theta_new, alpha, ms)
                _, (thetas, alphas, ms) = jax.lax.scan(step_fn, (theta_init, key), jnp.arange(n_steps)) 
                return thetas, alphas, ms
            chain, accept_prob, ms = nuts_sample(key, eps, partial(f_loss, metric, x, y), n_steps, ε)
            return chain, accept_prob, ms

        @jax.jit
        def sample(self, key: jax.Array, x: jax.Array, y: jax.Array):
            keys = jax.random.split(key,10)
            params = model.init_weights(keys[0])

            eps = metric0.whiten(params)
            ε, εp = find_reasonable_epsilon(keys[9], eps, partial(f_loss, metric0, x, y))
            jax.debug.print('Found step size {} with acceptance probability {}', ε, εp)
            # eps, ε, metrics = warmup_with_dual_averaging(keys[2], eps, partial(f_loss, metric0), ε)
            # jax.debug.print('Warmup finished with step size {} and log probs {}', ε, metrics['probs'])
            (eps, ε, L, info1) = ahmc_fast(keys[1], eps, partial(f_loss, metric0, x, y), 100, εmin=ε/100, εmax=ε*100, Lmin = 2, Lmax = 2000)
            jax.debug.print('Warmup finished with ε={} and L={}, d={}, logp={}', ε, L, info1['d'].mean(), info1['logp'][-1])
            # chain, accept_prob, ms = nuts_sample(keys[3], eps, partial(f_loss, metric0), 25, ε)
            eps, chain, ms = sample_hmc(keys[2], eps, partial(f_loss, metric0, x, y), 1000, ε, L)
            jax.debug.print('Finished first adaptation interval - accept_prob = {}', ms['alpha'].mean())
            params = metric0.unwhiten(eps)
            pchain = jax.vmap(metric0.unwhiten)(chain)
            metric = estimator(pchain)
            echain = jax.vmap(metric.whiten)(pchain)
            eps = metric.whiten(params)
            (eps, ε, L, info2) = ahmc_fast(keys[3], eps, partial(f_loss, metric, x, y), 100, εmin=1e-4, εmax=1.0, Lmin = 2, Lmax = 2000)
            jax.debug.print('Warmup finished with ε={} and L={}, d={}, logp={}', ε, L, info2['d'].mean(), info2['logp'][-1])
            eps, chain, ms = sample_hmc(keys[8], eps, partial(f_loss, metric, x, y), self.n_steps, ε, L)

            chain = einops.rearrange(chain, '(x n) ... -> x n ...', x=10)[-1]
            probs = jax.vmap(partial(f_loss, metric, x, y))(chain)
            chain = jax.vmap(metric.unwhiten)(chain)
            return chain, ms['alpha'], {'probs':probs, 'info1':info1, 'info2':info2}, echain

    @jax.jit
    @partial(jax.vmap, in_axes=(0,None))
    @partial(jax.vmap, in_axes=(0,None))
    def v_fwd(chains, x: jax.Array) -> jax.Array:
        cx = Context(chains, None)
        return model(cx, x)

    def expected_improvement(waterline: jax.Array, mean: jax.Array, std: jax.Array, eps=0.0):
        return log_h((mean - waterline)/(std+eps)) + jnp.log(std+eps)

    @jax.jit
    def acquisition_function(chains, xs, waterline):
        # best: [] scalar
        dist = v_fwd(chains, xs).squeeze(-1)
        mean = dist.mean((0,1)) # [N]
        std = dist.std((0,1)) # [N]
        # return expected_improvement(waterline, mean, std)
        return -(mean + std)

    @jax.jit
    def findmax(key, chains, waterline):
        return jax.vmap(findmax_ac_, in_axes=(0,None,None))(jax.random.split(key,101),chains,waterline)

    ac_steps = 500
    def findmax_ac_(key, chains, waterline):
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, [in_dim], minval=-1, maxval=1)

        opt = optax.adam(0.05, 0.9, 0.9)
        opt_state = opt.init(x)        

        def f_loss(x, key, p):
            x += jax.random.normal(key, (10,)+x.shape) * 0.5 * p
            ac = acquisition_function(chains, x, waterline)
            loss = -ac.mean()
            return loss, (loss, {'ac':ac})
        f_grad = jax.grad(f_loss, has_aux=True)

        @scan_tqdm(ac_steps)
        def step(carry, i):
            x, opt_state, key = carry
            key, subkey = jax.random.split(key)
            grad, value = f_grad(x, subkey, (1-i/findmin_steps))
            # updates, opt_state = opt.update(grad, opt_state, x, value=value, grad=grad, value_fn=lambda x:f_loss(x)[0])
            updates, opt_state = opt.update(grad, opt_state, x)
            x = optax.apply_updates(x, updates)
            x = jnp.clip(x, -1, 1)
            return (x, opt_state, key), {'value': value}
        
        (x, _, _), metrics = jax.lax.scan(step, (x, opt_state, key), jnp.arange(ac_steps))
        return x, metrics

    findmin_steps = 500
    @jax.jit
    def findmin_v(key, chains, j, vj):
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, [in_dim], minval=-1, maxval=1)
        x = x.at[j].set(vj)

        opt = optax.adam(0.05, 0.9, 0.9)
        opt_state = opt.init(x)

        def f_loss(x, key, p):
            x += jax.random.normal(key, (50,)+x.shape) * 0.5 * p
            dist = v_fwd(chains, x[None]).squeeze(-1) # [n_chains, n_samples, 3]
            mean = dist.mean((0,1)).mean() + dist.std((0,1)).mean() # mean predicted training loss according to the posterior
            return mean, mean # try to minimize mean
        f_grad = jax.grad(f_loss, has_aux=True)

        @scan_tqdm(findmin_steps)
        def step(carry, i):
            x, opt_state, key = carry
            key, subkey = jax.random.split(key)
            grad, value = f_grad(x, subkey, (1-i/findmin_steps))
            # updates, opt_state = opt.update(grad, opt_state, x, value=value, grad=grad, value_fn=lambda x:f_loss(x)[0])
            updates, opt_state = opt.update(grad, opt_state, x)
            x = optax.apply_updates(x, updates)
            x = jnp.clip(x, -1, 1)
            x = x.at[j].set(vj)
            return (x, opt_state, key), {'value': value}
        
        (x, _, _), metrics = jax.lax.scan(step, (x, opt_state, key), jnp.arange(findmin_steps))
        return x, metrics

    def gelman_rubin(query_ys):
        J = query_ys.shape[0]
        L = query_ys.shape[1]
        xj = jnp.mean(query_ys, axis=1) # c x ..
        xm = jnp.mean(xj, axis=0) # x ..
        B = L/(J-1) * jnp.sum(jnp.square(xj - xm), axis=0)
        W = query_ys.var(1).mean(0)
        R = ((L-1)/L * W + 1/L * B)/W
        return R

    for i in range(1):
        rng = PRNG(jax.random.PRNGKey(0))
        mcmc = MCMC(n_steps=2000, n_chains=4)

        # params = model.init_weights(rng.split())
        # eps = metric0.whiten(params)
        # ε, εp = find_reasonable_epsilon(rng.split(), eps, partial(f_loss, metric0, x, y))
        # jax.debug.print('Found step size {} with acceptance probability {}', ε, εp)
        # (eps, ε, L, info) = ahmc_fast(rng.split(), eps, partial(f_loss, metric0, x, y), 100, εmin=1e-8, εmax=1.0, Lmin = 2, Lmax = 1000)
        # # (eps, ε, L, info) = ahmc_fast(rng.split(), eps, partial(f_loss, metric0, x, y), 1000, εmin=1e-4, εmax=1.0, Lmin = 2, Lmax = 100)
        # print(f'Found ε={ε}, L={L}, d={info["d"].mean()}, logp={info["logp"][-1]}')
        # import json
        # with open('stuff.json', 'w') as fp:
        #     json.dump({'xs': info['γ'].tolist(), 'ds': info['d'].tolist(), 'αs': info['α'].tolist()}, fp)
        # exit()

        chains, accept_probs, ms, ec = jax.vmap(mcmc.sample, in_axes=(0,None,None))(rng.split(4), x, y)
        jaxtorch.pt.save(chains, 'chains.pt')
        import json
        with open('stuff.json', 'w') as fp:
            json.dump({'xs': ms['info2']['γ'][0].tolist(), 'ds': ms['info2']['d'][0].tolist(), 'αs': ms['info2']['α'][0].tolist()}, fp)

        print(jnp.square(ec).mean(1))

        # chains, accept_probs, ms = mcmc.sample(rng.split(), x, y)
        # chains = {p:v[None] for (p,v) in chains.items()}
        print('acceptance rate:', accept_probs.mean())
        jaxtorch.pt.save(chains, 'chains.pt')

        # chains = jaxtorch.pt.load('chains.pt')
        query_xs = jax.random.uniform(jax.random.PRNGKey(1), [10, in_dim], dtype=jnp.float32, minval=-1, maxval=1)
        query_ys = v_fwd(chains, query_xs) # c n x ...
        R = gelman_rubin(query_ys)
        print('gelman-rubin:', R)

        xmax, metrics = findmax(jax.random.PRNGKey(1), chains, jnp.max(y))
        ymax = v_fwd(chains, xmax)
        mean = jnp.mean(ymax, axis=(0,1)).squeeze(-1) # 101
        imax = jnp.argmax(mean, axis=0)
        print(xmax.shape)
        print({k:v for (k,v) in zip(pars, jnp.exp(xmax[imax]*dscale+dmid))})
        print('expected score:', mean[imax])
        print('a')

        j = pars.index('max_lr')
        vj = jnp.linspace(-1, 1, 100)
        keys = jax.random.split(jax.random.PRNGKey(1), 4)
        findmin_v_ = jax.vmap(findmin_v, in_axes=(None,None,None,0))
        findmin_v_ = jax.vmap(findmin_v_, in_axes=(0,None,None,None))
        xmin, metrics = findmin_v_(keys, chains, j, vj)
        yvals = v_fwd(chains, xmin)
        ymean = yvals.mean((0,1)).squeeze(-1)
        ystd = yvals.std((0,1)).squeeze(-1)
        ix = ymean.argmin(0)
        print(ymean.shape, ystd.shape)
        ymean = EX.gather(ymean, ix[None], 'k x, [k] x -> x')
        ystd = EX.gather(ystd, ix[None], 'k x, [k] x -> x')
        print(metrics)
        plt.xscale('log')
        plt.plot(jnp.exp(vj*dscale[j]+dmid[j]), ymean, label='min', color='blue')
        plt.fill_between(jnp.exp(vj*dscale[j]+dmid[j]), ymean - ystd[ix], ymean + ystd[ix], alpha=0.2, color='blue')
        plt.show()
        # print(xmin, metrics)