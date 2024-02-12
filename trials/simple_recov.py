#!/usr/local/opt/python/libexec/bin/python
import numpy as np
import pymc3 as pm
import arviz as az
import theano.tensor as tt
from matplotlib import pyplot as plt
import corner

def px(x, b, c):
    return (1 + c) * b**(-1 - c) * (b - x)**c

def px_sample(b, c, shape=1):
    u = np.random.rand(shape)
    c1 = c + 1
    return b - (b**c1 * (1 - u))**(1 / c1)

def plot_comp_samples(b, c):
    n = 10000
    samples = px_sample(b, c, shape=n)

    x_range = np.linspace(0, b, num=1000)
    pxs = px(x_range, b, c)

    fig, ax = plt.subplots()
    ax.plot(x_range, pxs)
    ax.hist(samples, density=True)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$p(x)$')
    plt.show()

def loglike(samples, b, c):
    m = len(samples)
    c1 = c + 1
    return m * (np.log(c1) -c1 * np.log(b)) + c * np.sum(np.log(b - samples))

def notloglike(samples, b, c):
    return np.product(b**(-1 - c) * (c + 1) * (b - samples)**c)

def recov_2d(true_b, true_c, n=200):
    np.random.seed(1)
    samples = px_sample(true_b, true_c, shape=n)
    
    print(samples.max(), samples.min())
    b_range = np.linspace(np.max(samples)+0.001, 2.4, num=300)
    c_range = np.linspace(1, 2.5, num=300)

    lls = np.empty(shape=(len(b_range), len(c_range)))
    for i, b in enumerate(b_range):
        for j, c in enumerate(c_range):
            lls[i, j] = loglike(samples, b, c)

    lls = np.exp(lls)
    lls /= np.max(lls)
    # lls = lls - np.max(lls)
    ind = np.unravel_index(np.argmax(lls, axis=None), lls.shape)
    print(ind, b_range[ind[0]], c_range[ind[1]])
    print(np.shape(lls))

    b_marg = np.trapz(lls, x=c_range, axis=1)
    b_marg = b_marg / np.sum(b_marg)
    c_marg = np.trapz(lls, x=b_range, axis=0)
    c_marg = c_marg / np.sum(c_marg)

    fig = plt.figure(figsize=(6, 3.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 5)
    ax2d = fig.add_subplot(gs[1:, :-1])
    ax1db = fig.add_subplot(gs[0, :-1])
    ax1dc = fig.add_subplot(gs[1:, -1])

    im = ax2d.imshow(np.swapaxes(lls, 0, 1), extent=(b_range.min(), b_range.max(), c_range.min(), c_range.max()), origin='lower', vmin=1e-10, aspect='auto')
    ax2d.scatter(true_b, true_c, marker='x', color='red')
    ax2d.set_ylabel(r'$c$')
    ax2d.set_xlabel(r'$b$')
    # fig.colorbar(im, ax=ax2d, label=r'$\mathcal{L} /\textrm{max} \mathcal{L}$')

    ax1db.plot(b_range, b_marg)
    print(np.argmax(b_marg))
    b_cumsum = np.cumsum(b_marg) * np.diff(b_range)[0]
    b_cumsum /= np.max(b_cumsum)
    b_90pcred_lower = b_range[np.argmin(np.abs(b_cumsum - 0.05))]
    b_90pcred_upper = b_range[np.argmin(np.abs(b_cumsum - 0.95))]
    b_90pcred_mid = b_range[np.argmin(np.abs(b_cumsum - 0.5))]
    print(b_90pcred_lower, b_90pcred_upper)
    ax1db.axvline(b_90pcred_lower, ls='dashed', color='grey', alpha=0.6)
    ax1db.axvline(b_90pcred_upper, ls='dashed', color='grey', alpha=0.6)
    ax1db.set_title(r'$b=%0.2f^{+%0.2f}_{-%0.2f}$'%(b_90pcred_mid, b_90pcred_upper - b_90pcred_mid, b_90pcred_mid - b_90pcred_lower), size=11)
    ax1db.axvline(true_b, ls='dashed', color='red', alpha=0.6)
    ax1db.set_xlim(b_range.min(), b_range.max())
    ax1db.xaxis.set_ticklabels([])
    ax1db.yaxis.set_ticks([])
    # ax1db.set_ylabel(r'$p(b)$')

    ax1dc.plot(c_marg, c_range)
    print(np.argmax(c_marg))
    c_cumsum = np.cumsum(c_marg) * np.diff(c_range)[0]
    c_cumsum /= np.max(c_cumsum)
    c_90pcred_lower = c_range[np.argmin(np.abs(c_cumsum - 0.05))]
    c_90pcred_upper = c_range[np.argmin(np.abs(c_cumsum - 0.95))]
    c_90pcred_mid = c_range[np.argmin(np.abs(c_cumsum - 0.5))]
    print(c_90pcred_lower, c_90pcred_upper)
    ax1dc.axhline(c_90pcred_lower, ls='dashed', color='grey', alpha=0.6)
    ax1dc.axhline(c_90pcred_upper, ls='dashed', color='grey', alpha=0.6)
    ax1dc.set_ylabel(r'$c=%0.2f^{+%0.2f}_{-%0.2f}$'%(c_90pcred_mid, c_90pcred_upper - c_90pcred_mid, c_90pcred_mid - c_90pcred_lower), size=11, rotation=360, labelpad=35)
    ax1dc.yaxis.set_label_position('right')
    ax1dc.axhline(true_c, ls='dashed', color='red', alpha=0.6)
    ax1dc.set_ylim(c_range.min(), c_range.max())
    ax1dc.yaxis.set_ticklabels([])
    ax1dc.xaxis.set_ticks([])
    # ax1dc.set_title(r'$p(c)$', size=10)

    plt.savefig('figs/2d_grid.png', dpi=300)
    plt.show()
            

def loglike_tt(samples, b, c):
    m = tt.shape(samples)
    c1 = c + 1
    return m * tt.log(c1 * tt.pow(b, -c1)) + c * tt.sum(tt.log(b - samples))

def pymc_trial():
    np.random.seed(1)
    true_b = 2
    true_c = 1.5
    n = 200
    samples = px_sample(true_b, true_c, shape=n)

    with pm.Model() as model:
        b = pm.Uniform('b', lower=samples.max(), upper=2.4)
        c = pm.Uniform('c', lower=1, upper=2.5)
        likelihood = pm.DensityDist('ll', loglike_tt, observed=dict(samples=samples, b=b, c=c))

        # trace = pm.sample(10000, tune=2000, target_accept=0.9, return_inferencedata=True, idata_kwargs={"density_dist_obs": False})
        # trace.to_netcdf('2dtrial.nc')

        trace = az.from_netcdf('2dtrial.nc')
        summary = az.summary(trace, kind="all", hdi_prob=0.90)
        print(summary)
        print(summary['hdi_5%']['b'])
        # axarr = az.plot_trace(trace)
        ax = az.plot_pair(trace, figsize=(6, 3.5), kind='kde', marginals=True,
                kde_kwargs={'contour_kwargs':{'alpha':0}, 'hdi_probs':[0.1, 0.3, 0.5, 0.7, 0.9], 'fill_last':True},
                )

        axb = ax[0][0]
        axc = ax[1][1]
        ax2d = ax[1][0]
        axb.axvline(summary['hdi_5%']['b'], ls='dashed', color='grey', alpha=0.7)
        axb.axvline(summary['hdi_95%']['b'], ls='dashed', color='grey', alpha=0.7)
        axb.axvline(true_b, ls='dashed', color='red', alpha=0.7)
        axc.axhline(summary['hdi_5%']['c'], ls='dashed', color='grey', alpha=0.7)
        axc.axhline(summary['hdi_95%']['c'], ls='dashed', color='grey', alpha=0.7)
        axc.axhline(true_c, ls='dashed', color='red', alpha=0.7)
        axb.set_title(r'$b=%0.2f^{+%0.2f}_{-%0.2f}$'%(
            summary['mean']['b'], 
            summary['hdi_95%']['b'] - summary['mean']['b'],
            summary['mean']['b'] - summary['hdi_5%']['b']),
            fontsize=11)
        axc.set_ylabel(r'$c=%0.2f^{+%0.2f}_{-%0.2f}$'%(
            summary['mean']['c'], 
            summary['hdi_95%']['c'] - summary['mean']['c'],
            summary['mean']['c'] - summary['hdi_5%']['c']),
            fontsize=11, rotation=0, labelpad=35)
        axc.yaxis.set_label_position('right')

        ax2d.scatter(true_b, true_c, marker='x', color='red')
        ax2d.set_xlabel(r'$b$', size=11)
        ax2d.set_ylabel(r'$c$', size=11)
        ax2d.tick_params(labelsize=11)

        plt.savefig('figs/pymc_2dtrial.png', dpi=300)
        plt.show()


def hierarch_trial():
    true_b_mean = 2
    true_b_sd = 0.5
    true_c_mean = 3
    true_c_sd = 0.5
    ns = np.random.normal(loc=100, scale=30, size=10).astype(int)
    assert all(ns > 1)
    samples = []
    bs_used = np.random.normal(true_b_mean, true_b_sd, size=len(ns))
    cs_used = np.random.normal(true_c_mean, true_c_sd, size=len(ns))
    for n, b_used, c_used in zip(ns, bs_used, cs_used):
        samples.append(px_sample(b_used, c_used, shape=n))
    print(ns)
    
    with pm.Model() as hierarchical_model:
        b_mean = pm.Lognormal('b_mean', mu=2, sigma=1)
        b_sd = pm.HalfCauchy('b_sd', beta=2)
        c_mean = pm.Lognormal('c_mean', mu=2, sigma=1)
        c_sd = pm.HalfCauchy('c_sd', beta=2)
        
        for sample in samples:
            b_est = pm.Normal('b_normal', mu=b_mean, sigma=b_sd)
            c_est = pm.Normal('c_normal', mu=c_mean, sigma=c_sd)
            observed = dict(samples=samples, b=b_est, c=c_est)
            likelihood = pm.DensityDist('ll', loglike_tt, observed=observed)

        trace = pm.sample(10000, tune=2000, target_accept=0.9, return_inferencedata=True, idata_kwargs={"density_dist_obs": False})

        print(az.summary(trace, kind='all', hdi_prob=0.9))

        axarr = az.plot_trace(trace)
        plt.show()


if __name__ == '__main__':
    b = 2
    c = 1.5
    # plot_comp_samples(b, c)
    # recov_2d(b, c)
    pymc_trial()
    # hierarch_trial()
