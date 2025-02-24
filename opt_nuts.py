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

from my_little_optimizer.log_h import log_h
from my_little_optimizer.metric import DictionaryMetric, ScalarMetric, Metric, MetricEstimator
from my_little_optimizer.metric import DictionaryMetricEstimator, CovarianceMetricEstimator, KroneckerMetricEstimator
from my_little_optimizer.nuts import nuts_kernel
from my_little_optimizer.adapt import warmup_with_dual_averaging, find_reasonable_epsilon
from my_little_optimizer.ahmc import ahmc_fast, sample_hmc
from my_little_optimizer.util import Partial
from my_little_optimizer.turnkey import sample_adaptive

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
    # model = nn.Sequential([nn.Linear(in_dim, 10), nn.Tanh(), nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, out_dim)])
    model = nn.Sequential([nn.Linear(in_dim, 10), nn.LeakyReLU(), nn.Linear(10, out_dim)])
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
            metrics[p.name] = ScalarMetric(scales[p.name])
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

    def f_loss(x, y, params, σ_y=0.1):
        cx = Context(params, None)
        prediction = model(cx, x)
        log_p_y = -0.5 * jnp.sum(jnp.square(y - prediction)/σ_y**2)
        # log_p_eps = sum(-0.5 * jnp.sum(jnp.square(params[p.name]/scales[p.name])) for p in model.parameters())
        # laplace prior
        log_p_eps = sum(-0.5 * jnp.sum(jnp.abs(params[p.name]/scales[p.name])) for p in model.parameters())
        return log_p_y + log_p_eps
    

    @jax.jit
    @partial(jax.vmap, in_axes=(0,None))
    @partial(jax.vmap, in_axes=(0,None))
    def v_fwd(chains, x: jax.Array) -> jax.Array:
        cx = Context(chains, None)
        return model(cx, x)

    def log_expected_improvement(chains, waterline, eps=0.01):
        """Log expected improvement acquisition function (minimization)."""
        def f(chains, waterline, eps, xs):
            dist = v_fwd(chains, xs).squeeze(-1)
            mean = dist.mean((0,1)) # [N]
            std = dist.std((0,1)) # [N]
            return log_h((waterline - mean)/(std+eps)) + jnp.log(std+eps)
        return Partial(f, chains, waterline, eps)
    
    # @jax.jit
    # def acquisition_function(chains, xs, waterline):
    #     # best: [] scalar
    #     dist = v_fwd(chains, xs).squeeze(-1)
    #     mean = dist.mean((0,1)) # [N]
    #     std = dist.std((0,1)) # [N]
    #     return expected_improvement(waterline, mean, std)
    #     # return -(mean + std)

    def findmax(key, acquisition_function, mask=None, maskval=None, n=101):
        xs, metrics = jax.vmap(findmax_ac_, in_axes=(0,None,None,None))(jax.random.split(key,n), acquisition_function, mask, maskval)
        rs = acquisition_function(xs)
        metrics['rs'] = rs
        return xs[jnp.argmax(rs)], metrics

    ac_steps = 500

    @jax.jit
    def findmax_ac_(key, acquisition_function, mask, maskval):
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, [in_dim], minval=-1, maxval=1)

        opt = optax.adam(0.05, 0.9, 0.9)
        opt_state = opt.init(x)        

        def f_loss(x, key, p):
            x += jax.random.normal(key, (50,)+x.shape) * 0.5 * p
            ac = acquisition_function(x)
            loss = -ac.mean()
            return loss, (loss, {'ac':ac})
        f_grad = jax.grad(f_loss, has_aux=True)

        @scan_tqdm(ac_steps)
        def step(carry, i):
            x, opt_state, key = carry
            key, subkey = jax.random.split(key)
            grad, value = f_grad(x, subkey, (1-i/findmin_steps))
            updates, opt_state = opt.update(grad, opt_state, x)
            x = optax.apply_updates(x, updates)
            x = jnp.clip(x, -1, 1)
            if mask is not None:
                x = jnp.where(mask, maskval, x)
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
        rng = PRNG(jax.random.PRNGKey(2))

        if True:
            def f_w_beta(x, y, state):
                params = state['params']
                log_sigma = state['log_sigma'].squeeze()
                sigma = jnp.exp(log_sigma).squeeze()
                cx = Context(params, None)
                prediction = model(cx, x)
                # sigma = 0.05
                log_p_y = -0.5 * jnp.sum(jnp.square(y - prediction)/sigma**2)
                log_p_y += -0.5 * np.prod(y.shape) * (jnp.log(2 * jnp.pi) + log_sigma*2)
                # log_p_params = sum(-0.5 * jnp.sum(jnp.square(params[p.name]/scales[p.name])) for p in model.parameters())
                # laplace prior
                log_p_params = sum(-0.5 * jnp.sum(jnp.abs(params[p.name]/scales[p.name])) for p in model.parameters())
                mu_log_sigma = jnp.log(0.1)
                std_log_sigma = jnp.log(2.0)
                log_p_sigma = -0.5 * jnp.square((log_sigma - mu_log_sigma)/std_log_sigma)
                return log_p_y + log_p_params + log_p_sigma
            
            @jax.jit
            def full_hessian(x, y, samples: dict) -> Metric:
                f = partial(f_w_beta, x, y)
                H = jax.vmap(jax.hessian(f))(samples)

                def matmetric(Hpart, shape):
                    Hpart = Hpart.reshape(Hpart.shape[0], np.prod(shape), np.prod(shape))
                    d = jnp.diagonal(Hpart, axis1=1, axis2=2)
                    scale = jnp.mean(jnp.square(d))**-0.25
                    return ScalarMetric(scale)

                out = {}
                for p in model.parameters():
                    out[p.name] = matmetric(H['params'][p.name]['params'][p.name], p.shape)
                return DictionaryMetric({'params': DictionaryMetric(out), 'log_sigma': matmetric(H['log_sigma']['log_sigma'], [1])})


            chains, ms = sample_adaptive(jax.random.PRNGKey(0),
                                        logp = Partial(f_w_beta, x, y),
                                        init = Partial(lambda key: {'params': model.init_weights(key), 'log_sigma': jnp.array([jnp.log(0.1)])}),
                                        n_chains = 4, n_samples = 2000,
                                        metric0 = DictionaryMetric({'params': mkmetric(model), 'log_sigma': ScalarMetric(jnp.array(jnp.log(2.0)))}),
                                        metric_estimator = partial(full_hessian, x, y)) #DictionaryMetricEstimator({'params': mkmetric_estimator(model), 'log_sigma': CovarianceMetricEstimator()}))
            # decimate to save memory
            chains = tree_map(lambda x: einops.rearrange(x, 'c (n x) ... -> x c n ...', x=10)[-1], chains)
            accept_probs = ms['alphas'].mean()

            jaxtorch.pt.save(chains, 'chains.pt')
            import json
            with open('stuff.json', 'w') as fp:
                json.dump({'xs': ms['info1']['γ'][0].tolist(), 'ds': ms['info1']['d'][0].tolist(), 'αs': ms['info1']['α'][0].tolist()}, fp)

            # chains, accept_probs, ms = mcmc.sample(rng.split(), x, y)
            # chains = {p:v[None] for (p,v) in chains.items()}
            print('acceptance rate:', accept_probs.mean())
        else:
            chains = jaxtorch.pt.load('chains.pt')
        print('sigmas:', jax.nn.sigmoid(chains['log_sigma']))
        chains = chains['params']
        query_xs = jax.random.uniform(jax.random.PRNGKey(1), [10, in_dim], dtype=jnp.float32, minval=-1, maxval=1)
        query_ys = v_fwd(chains, query_xs) # c n x ...
        R = gelman_rubin(query_ys)
        print('gelman-rubin:', R)

        waterline = jnp.min(y)
        acf = log_expected_improvement(chains, waterline)
        xmax, metrics = findmax(jax.random.PRNGKey(1), acf)
        ymax = v_fwd(chains, xmax)
        xmax_ac = jnp.exp(acf(xmax))
        print('waterline:', waterline.shape, waterline)
        print('xmax_ac:', xmax_ac.shape, xmax_ac)
        mean = jnp.mean(ymax, axis=(0,1)).squeeze(-1) # 101
        std = jnp.std(ymax, axis=(0,1)).squeeze(-1)
        print('means:', mean.shape, mean)
        print('stds:', std.shape, std)
        print(xmax.shape)
        print({k:v for (k,v) in zip(pars, jnp.exp(xmax*dscale+dmid))})
        print('expected score:', mean)
        # print('metrics:', metrics)

        def lowest_mean(chains):
            """Acquisition function for lowest expected score."""
            def f(chains, xs):
                return -jnp.mean(v_fwd(chains, xs), axis=(0,1)).squeeze(-1)
            return Partial(f, chains)
        
        def conservative_lowest_mean(chains, γ=1.0):
            """Acquisition function for lowest expected score."""
            def f(chains, γ, xs):
                dist = v_fwd(chains, xs)
                mean = jnp.mean(dist, axis=(0,1)).squeeze(-1)
                std = jnp.std(dist, axis=(0,1)).squeeze(-1)
                return -(mean + γ*std)
            return Partial(f, chains, γ)


        j = pars.index('max_lr')
        vj = jnp.linspace(-1, 1, 100)
        xmin, metrics = jax.vmap(lambda vj: findmax(jax.random.PRNGKey(1), conservative_lowest_mean(chains, 1.0), n=4, mask=jnp.arange(len(pars))==j, maskval=vj))(vj)
        # keys = jax.random.split(jax.random.PRNGKey(1), 4)
        # findmin_v_ = jax.vmap(findmin_v, in_axes=(None,None,None,0))
        # findmin_v_ = jax.vmap(findmin_v_, in_axes=(0,None,None,None))
        # xmin, metrics = findmin_v_(keys, chains, j, vj)
        yvals = v_fwd(chains, xmin)
        ymean = yvals.mean((0,1)).squeeze(-1)
        ystd = yvals.std((0,1)).squeeze(-1)
        print(ymean.shape, ystd.shape)
        print(metrics)
        plt.xscale('log')
        plt.plot(jnp.exp(vj*dscale[j]+dmid[j]), ymean, label='min', color='blue')
        plt.fill_between(jnp.exp(vj*dscale[j]+dmid[j]), ymean - ystd, ymean + ystd, alpha=0.2, color='blue')
        plt.show()
        # print(xmin, metrics)