import numpy as np
import sys, os
import scipy.optimize as opt
from scipy.special import erf, erfc
from scipy.stats import spearmanr, gaussian_kde
import matplotlib.pyplot as plt

from read_flare_db import keep_regions_min_num, return_all_flares
from flare_classes import Flare, ActiveRegion
from fitting_distributions import Gaussian, Exponential, PowerLaw, Seperable, LogNormal, Weibull, ExponentialTrunc, PowerLawTrunc, LogNormalTrunc

path_to_working = '/Users/julian/Documents/phd/solar_flares/'

def min_ln_l(data, distribution, p0s='default', bounds='default'):
    p0s, bounds = distribution.p0s_bounds(distribution, data, p0s=p0s, bounds=bounds)
    opts = {'eps': 1e-12}
    result = opt.minimize(neg_ln_l, p0s, jac='3-point', args=(data, distribution), bounds=bounds, options=opts)
    try:
        assert result.success
    except AssertionError:
        print(f'Something went wrong in the minimsation for {distribution.__name__}')
        if distribution != Weibull and distribution != Seperable:
            print(result)
    return result.x

def ln_l(params, data, distribution):
    _distribution = distribution(*params)
    ln_l = np.sum(_distribution.ln_pdf(data))
    return ln_l

def neg_ln_l(params, data, distribution):
    _distribution = distribution(*params)
    ln_l = np.sum(_distribution.ln_pdf(data))
    return -ln_l

# return the empirical cdf of array 'a', sort_a is 'x', norm_a is 'y' for plotting
def rank_cdf(a):
    sort_a = np.sort(a)
    ranks = np.arange(len(a))
    norm_a = ranks / float(max(ranks))
    return sort_a, norm_a

def aic(data, verbose=False, justthree=False, distributions=None):
    if distributions==None:
        if not justthree:
            distributions = [Gaussian, Exponential, PowerLaw, Seperable, LogNormal, Weibull]
        elif justthree:
            distributions = [Exponential, PowerLaw, LogNormal]
    aics = []
    params_list = []
    for distribution in distributions:
        if 'mle_params' in distribution.__dict__:
            params = distribution.mle_params(distribution, data)
        else:
            params = min_ln_l(data, distribution, p0s='default', bounds='default')
        lnl = ln_l(params, data, distribution)
        p = len(params)
        n = len(data)
        aic = 2 * p - 2 * lnl
        try:
            aicc = aic + 2 * p * (p + 1) / (len(data) - p - 1)
        except ZeroDivisionError:
            print(data)
            return
        if verbose:
            print('Shape: ', distribution.__name__)
            print('Best-fit params: ', params)
            print('AICc base score: %.3e'%aicc)

        aics.append(aicc)
        params_list.append(params)

    aics = np.array(aics)
    del_aics = aics - np.min(aics)
    rel_probs = np.exp(-del_aics / 2)
    if verbose:
        for distribution, rel_prob, del_aic in zip(distributions, rel_probs, del_aics):
            print(f'{distribution.__name__} \t relative prob {rel_prob:.4e} \t log rel. prob {-del_aic/2:.2f}')

    return distributions, params_list, del_aics

def plot_pdf_overall_fits(data, distributions, params_list, loglog=False, units='h', which='pdelt'):
    fig, ax = plt.subplots(figsize=(3.3, 2))

    nbins = 50
    if loglog:
        bins = np.logspace(np.log10(min(data)), np.log10(max(data)), num=nbins)
    else:
        bins = np.linspace(min(data), max(data), num=nbins)
    ax.hist(data, bins=bins, log=True, color='black', histtype='step', density=True, label='Data', lw=1.5)
    ylims = ax.get_ylim()
    if loglog:
        x_range = np.logspace(np.log10(min(data)), np.log10(max(data)), num=1000)
    else:
        x_range = np.linspace(min(data), max(data), num=1000)
    if which=='pdelt':
        if units=='h':
            unit_str = 'hours'
        elif units=='s':
            unit_str = 's'
        elif units=='D':
            unit_str = 'days'
        ax.set_xlabel(r'$\Delta t$ (%s)'%unit_str)
        ax.set_ylabel(r'$p(\Delta t)$')
        if loglog:
            legend_loc = 'lower left'
        else:
            lengend_loc = 'upper right'
    elif which=='psize':
        if units=='h':
            unit_str = 'arb. units'
        elif units=='s':
            unit_str = r'J/m$^2$'
        elif units=='D':
            unit_str = 'arb. units'
        ax.set_xlabel(r'$\Delta s$ (%s)'%unit_str)
        ax.set_ylabel(r'$p(\Delta s)$')
        if loglog:
            legend_loc = 'lower left'
        else:
            legend_loc = 'best'
    if loglog:
        ax.set_xscale('log')
        ax.set_xlim(xmin=min(bins), xmax=max(bins))
        if which=='pdelt':
            ax.set_xlim(xmin=0.1)
        name = '%s_overall_loglog_fits.pdf'%(which)
    else:
        ax.set_xlim(xmin=0)
        name = '%s_overall_fits.pdf'%(which)

    if len(distributions) == 3:
        linestyles = ['--', ':', '-.']
    else:
        linestyles = ['-' for i in range(len(distributions))]
    for distribution, params, ls in zip(distributions, params_list, linestyles):
        dist = distribution(*params)
        y = dist.pdf(x_range)
        # if dist.name == 'Seperable':
        #     ls = 'dashed'
        #     z = 10
        #     lw = 1.5
        # else:
        # ls = 'solid'
        z = 3
        lw = 1.5
        ax.plot(x_range, y, label=dist.name, alpha=1, ls=ls, zorder=z, lw=lw)

    ax.set_ylim(ylims[0], ylims[1]*2)

    ax.legend(loc=legend_loc, handlelength=2)
    plt.savefig(os.path.join(path_to_working, 'figs/data', name), dpi=450)
    plt.savefig(os.path.join(path_to_working, 'paper_tex', name), dpi=450)
    plt.show()

def plot_pdelt_overall(flares, units='h', loglog=False, justthree=False):
    regions = np.unique(flares['region_num'])
    all_waiting_times = []
    for region in regions:
        region_flares = [Flare(*f, units=units) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)
        waiting_times = active.waiting_times(units=units)
        all_waiting_times.append(waiting_times)
    all_waiting_times = np.concatenate(all_waiting_times).ravel()

    distributions, params_list, aics = aic(all_waiting_times, verbose=True, justthree=justthree)

    plot_pdf_overall_fits(all_waiting_times, distributions=distributions, params_list=params_list, loglog=loglog, units=units, which='pdelt')

def pdelt_individual_aic(flares, min_waiting_times=6, units='h', justthree=False, verbose=False, c1masking=False, dels_masking=False, which_wt='start'):
    regions = np.unique(flares['region_num'])

    best_dist = []
    num_delt = []
    actives = []
    for region in regions:
        region_flares = [Flare(*f, units=units) for f in flares[flares['region_num']==region]]

        if c1masking:
            masked_flares = [f for f in region_flares if f.peak_flux >= 1e-6]
        if dels_masking:
            masked_flares = [f for f in region_flares if f.energy_proxy >= 9.6e-4]
        elif not (c1masking or dels_masking):
            masked_flares = region_flares

        if len(masked_flares) < min_waiting_times:
            continue

        active = ActiveRegion(region, masked_flares)
        waiting_times = active.waiting_times(units=units, which=which_wt)
        if which_wt=='peak':
            if any([w <= 0 for w in waiting_times]) or any(np.isnan(waiting_times)):
                continue
        if len(waiting_times) < min_waiting_times:
            continue

        distributions, params_list, aics = aic(waiting_times, justthree=justthree, verbose=False)
        best = distributions[np.argmin(aics)]
        best_dist.append(best.__name__)
        num_delt.append(len(waiting_times))
        actives.append(active)
    bests, nums = np.unique(best_dist, return_counts=True)
    if verbose:
        print(f'Total number of active regions considered: {len(actives)}')
        for best, num in zip(bests, nums):
            print(f'{best} best-fit for {num} ({100 * num / len(actives):.2f}%)')
    return np.array(best_dist), np.array(num_delt), np.array(actives)

def plot_propbestdist_numdelt(flares, min_num=20, max_num=51, units='h'):
    best_dist, num_delt, actives = pdelt_individual_aic(flares, min_waiting_times=min_num, units=units, verbose=True)
    mask = (num_delt >= min_num) & (num_delt < max_num)
    best_dist = best_dist[mask]
    num_delt = num_delt[mask]
    bin_width = 1
    bins = np.arange(min(num_delt), max(num_delt)+1, step=bin_width)
    bin_mids = (bins[1:] + bins[:-1]) / 2
    total_distnames = np.sort(np.unique(best_dist))
    proportions_ordered = []
    for i, bin in enumerate(bins[:-1]):
        mask = (num_delt >= bin) & (num_delt < bins[i+1])
        dists = best_dist[mask]
        distnames_in_bin, counts = np.unique(dists, return_counts=True)
        proportions = counts / len(dists)

        ordered = []
        for distname in total_distnames:
            if distname in distnames_in_bin:
                ordered.append(proportions[np.where(distnames_in_bin==distname)[0]][0])
            else:
                ordered.append(0)
        proportions_ordered.append(ordered)

    proportions_full = np.transpose(proportions_ordered)
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    for (i, dist), prop in zip(enumerate(total_distnames), proportions_full):
        if i==0:
            height = np.zeros(len(prop))
        else:
            height = np.sum(proportions_full[:i], axis=0)

        ax.bar(bin_mids, prop, bin_width, bottom=height, label=dist)

    ax.legend()

    ax.set_xlim(min(bins), max(bins))
    ax.set_xlabel('Number of waiting times in an active region')
    ax.set_ylabel('Proportion of regions that are best-fit\nto each distribution')
    plt.savefig(os.path.join(path_to_working, 'figs/data', 'prop_delt_numwaiting.png'), dpi=600)

    plt.show()

def plot_propbestdist_rate(flares, min_num=10, max_num=100, which='pdelt', units='h', justthree=False):
    if which == 'delt':
        best_dist, nums, actives = pdelt_individual_aic(flares, min_waiting_times=min_num, units=units, verbose=True, justthree=justthree)
    if which == 'size':
        best_dist, nums, actives = psizes_individual_aic(flares, min_sizes=min_num, units=units, verbose=True, justthree=justthree)
    mask = (nums >= min_num) & (nums < max_num)
    best_dist = best_dist[mask]
    nums = nums[mask]
    actives = actives[mask]
    rates = []
    for active in actives:
        rate, _ = active.rate()
        rates.append(rate)
    rates = np.array(rates)
    assert all(rates > 0)

    cdf_x, cdf_y = rank_cdf(rates)

    # bins = np.linspace(0, max(rates), num=15)
    bins = np.linspace(0.025, 0.3, num=20)
    bin_width = np.diff(bins)[0]
    bin_mids = (bins[1:] + bins[:-1]) / 2
    total_distnames = np.sort(np.unique(best_dist))
    proportions_ordered = []
    for i, bin in enumerate(bins[:-1]):
        mask = (rates >= bin) & (rates < bins[i+1])
        dists = best_dist[mask]
        distnames_in_bin, counts = np.unique(dists, return_counts=True)
        proportions = counts / len(dists)

        ordered = []
        for distname in total_distnames:
            if distname in distnames_in_bin:
                ordered.append(proportions[np.where(distnames_in_bin==distname)[0]][0])
            else:
                ordered.append(0)
        proportions_ordered.append(ordered)

    proportions_full = np.transpose(proportions_ordered)
    fig, ax = plt.subplots(figsize=(3.3, 2))
    for (i, dist), prop in zip(enumerate(total_distnames), proportions_full):
        if i==0:
            height = np.zeros(len(prop))
        else:
            height = np.sum(proportions_full[:i], axis=0)

        ax.bar(bin_mids, prop, bin_width, bottom=height, label=dist)

    cdf_ax = ax.twinx()
    cdf_ax.plot(cdf_x, cdf_y, color='black', label='CDF', zorder=1, alpha=0.7)
    cdf_ax.set_ylabel('CDF')
    cdf_ax.set_ylim(0, 1)

    ax.legend(loc='lower right', fontsize=8)

    ax.set_xlim(min(bins), max(bins))
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$\lambda_k$ (hours$^{-1}$)')
    if which=='delt':
        ax.set_ylabel(r'Proportion with $p(\Delta t)$' + '\nbest-fit to each shape')
    if which=='size':
        ax.set_ylabel(r'Proportion with $p(\Delta s)$' + '\nbest-fit to each shape')
    # plt.savefig(os.path.join(path_to_working, 'figs/data', 'prop_%s_rate.png'%which), dpi=600)
    plt.savefig(os.path.join(path_to_working, 'paper_tex', 'prop_%s_rate_min%d.pdf'%(which, min_num)), dpi=450)

    plt.show()

def plot_bestdists_rate_violin(flares, min_num=10, which='pdelt'):
    if which == 'pdelt':
        best_dist, nums, actives = pdelt_individual_aic(flares, min_waiting_times=min_num, units=units, verbose=True)
    if which == 'size':
        best_dist, nums, actives = psizes_individual_aic(flares, min_sizes=min_num, verbose=True)
    mask = (nums >= min_num)
    best_dist = best_dist[mask]
    nums = nums[mask]
    actives = actives[mask]
    rates = []
    rates_err = []
    for active in actives:
        rate, rate_err = active.rate()
        rates.append(rate)
        rates_err.append(rate_err)

    order = np.sort(np.unique(best_dist))

    fig, ax = plt.subplots(figsize=(4.5, 3.3))


    ax = sns.violinplot(x=rates, y=best_dist, inner='box', linewidth=0.5, order=order)

    ax.set_xlim(xmin=0)
    ax.set_xlabel(r'Rate (hours$^{-1}$)')
    if which=='pdelt':
        ax.set_ylabel('Best-fit shape for waiting time distribution')
    if which=='size':
        ax.set_ylabel('Best-fit shape for size distribution')
    plt.savefig(os.path.join(path_to_working, 'figs/data', 'violin_rate_%s.png'%which), dpi=600)
    plt.show()

def plot_psizes_overall(flares, loglog=True, justthree=False, units='s'):
    flares = [Flare(*f, units=units) for f in flares]
    sizes = np.array([f.energy_proxy for f in flares if not np.isnan(f.energy_proxy)])
    sizes = sizes[sizes > 0]

    distributions, params_list, aics = aic(sizes, verbose=True, justthree=justthree)
    plot_pdf_overall_fits(sizes, distributions=distributions, params_list=params_list, loglog=loglog, which='psize', units=units)

def psizes_individual_aic(flares, min_sizes=6, justthree=False, verbose=False, units='h', c1masking=False, dels_masking=False):
    regions = np.unique(flares['region_num'])

    best_dist = []
    num_sizes = []
    actives = []
    for region in regions:
        region_flares = [Flare(*f, units=units) for f in flares[flares['region_num']==region]]

        if c1masking:
            masked_flares = [f for f in region_flares if f.peak_flux >= 1e-6]
        if dels_masking:
            masked_flares = [f for f in region_flares if f.energy_proxy >= 9.6e-4]
        elif not (c1masking or dels_masking):
            masked_flares = region_flares

        if len(masked_flares) < min_sizes:
            continue

        active = ActiveRegion(region, masked_flares)
        try:
            sizes = np.array([f.energy_proxy for f in masked_flares if not np.isnan(f.energy_proxy)])
        except AttributeError:
            print([f.peak_flux for f in region_flares])
            print(region)
            continue
        sizes = sizes[sizes > 0]
        if len(sizes) < min_sizes:
            continue

        distributions, params_list, aics = aic(sizes, justthree=justthree, verbose=False)
        best = distributions[np.argmin(aics)]
        best_dist.append(best.__name__)
        num_sizes.append(len(sizes))
        actives.append(active)
    bests, nums = np.unique(best_dist, return_counts=True)
    if verbose:
        print(f'Total number of active regions considered: {len(actives)}')
        for best, num in zip(bests, nums):
            print(f'{best} best-fit for {num} ({100 * num / len(actives):.2f}%)')
    return np.array(best_dist), np.array(num_sizes), np.array(actives)

def match_actives(x_actives, t_actives, x_dists=None, t_dists=None):
    region_nums_with_sizes = np.array([a.region_number for a in x_actives])
    region_nums_with_delt = np.array([a.region_number for a in t_actives])
    print(len(region_nums_with_delt), len(region_nums_with_sizes))

    # check region numbers are at least ordered
    for a in [region_nums_with_sizes, region_nums_with_delt]:
        assert np.all(a[:-1] <= a[1:])

    # only consider regions with both sizes and waiting times
    indices = []
    for i, region_with_delt in enumerate(region_nums_with_delt):
        if region_with_delt in region_nums_with_sizes:
            indices.append(i)
    t_actives = t_actives[indices]
    region_nums_with_delt = np.array([a.region_number for a in t_actives])

    if t_dists is not None:
        t_dists = t_dists[indices]

    indices = []
    for i, region_with_size in enumerate(region_nums_with_sizes):
        if region_with_size in region_nums_with_delt:
            indices.append(i)
    x_actives = x_actives[indices]
    region_nums_with_sizes = np.array([a.region_number for a in x_actives])

    if x_dists is not None:
        x_dists = x_dists[indices]

    assert all(region_nums_with_sizes == region_nums_with_delt)

    if t_dists is not None and x_dists is not None:
        return x_dists, x_actives, t_dists, t_actives
    else:
        return x_actives, t_actives


def plot_best_dists(flares, verbose=True):
    if verbose:
        print('When fitting size distributions:')
    x_dists, _, x_actives = psizes_individual_aic(flares, min_sizes=6, verbose=verbose)
    if verbose:
        print('When fitting waiting time distributions:')
    t_dists, _, t_actives = pdelt_individual_aic(flares, min_waiting_times=6, verbose=verbose)

    x_dists, x_actives, t_dists, t_actives = match_actives(x_actives, t_actives, x_dists=x_dists, t_dists=t_dists)

    order = np.sort(np.unique(x_dists))
    for trim in ['Seperable', 'Weibull']:
        if trim in order:
            order = np.delete(order, np.where(order==trim)[0])

    num_counts = np.zeros(shape=(len(order), len(order)))
    for i, distcomp_pdelt in enumerate(order):
        for j, distcomp_size in enumerate(order):
            delt_mask = t_dists==distcomp_pdelt
            size_mask = x_dists==distcomp_size
            num_counts[i, j] = np.sum(delt_mask & size_mask)

    total = np.sum(num_counts)
    
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ax.imshow(np.transpose(num_counts), interpolation='none', origin='upper', cmap='gray')
    for i, _ in enumerate(order):
        for j, __ in enumerate(order):
            num = num_counts[i, j]
            s = f'{int(num):d}\n({100*num/total:.0f}\%)'
            ax.text(i, j, s, ha='center', va='center', bbox=dict(edgecolor='none', facecolor='white', alpha=0.5))

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels(order[::-1])

    ax.set_xlabel('Best-fit distribution for waiting times')
    ax.set_ylabel('Best-fit distribution for sizes')

    plt.savefig(os.path.join(path_to_working, 'figs/data', 'best_dists_grid.png'), dpi=600)
    plt.show()

def plot_correlations_hist(flares):
    regions, num_per = np.unique(flares['region_num'], return_counts=True)

    forwards = []
    backwards = []
    bins = np.linspace(-1, 1, num=30)
    for region in regions:
        region_flares = [Flare(*f) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)

        try:
            forward, ci = active.forward_correlation()
            backward, ci = active.backward_correlation()
        except TypeError:
            continue
        if forward is not None:
            forwards.append(forward)
        if backward is not None:
            backwards.append(backward)

    print(f'Median rho+ is {np.median(forwards)}')
    print(f'Median rho- is {np.median(backwards)}')
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ax.hist(forwards, histtype='step', color='black', bins=bins, label=r'$\rho_+$')
    ax.hist(backwards, histtype='step', color='black', bins=bins, linestyle='dashed', label=r'$\rho_-$')
    ax.set_xlabel('Spearman correlation coefficient')
    ax.set_ylabel('Counts')
    ax.legend()
    plt.savefig(os.path.join(path_to_working, 'figs/data', 'correlations_hist.png'), dpi=600)
    plt.show()

def plot_colatitudes_hist(flares):
    regions = np.unique(flares['region_num'])
    colats = []
    for region in regions:
        region_flares = [Flare(*f) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)

        colat = active.colatitude()
        if colat is not None:
            colats.append(active.colatitude())

    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ax.hist(colats, histtype='step', color='black', bins=30)
    ax.set_xlabel('Co-latitude')
    ax.set_ylabel('Counts')
    plt.savefig(os.path.join(path_to_working, 'figs/data', 'colatitudes_hist.png'), dpi=600)
    plt.show()

def plot_correl_rate(flares, latitudes=False, min_num=5, which_wt='start'):
    regions = np.unique(flares['region_num'])

    forwards = []
    rates = []
    colats = []
    matching = []

    corr_err = []
    rate_err = []
    for region in regions:
        region_flares = [Flare(*f) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)

        # forward, ci = active.forward_correlation()
        forward, ci = active.fb_correlation(which='forward', which_wt=which_wt)
        if np.isnan(forward):
            continue
        rate, cred_int = active.rate()
        append_flag = False
        if latitudes:
            colat = active.colatitude()
            if forward is not None and rate is not None and colat is not None:
                append_flag = True
        else:
            if forward is not None and rate is not None and len(region_flares)>=min_num:
                append_flag = True

        if append_flag:
            if latitudes:
                colats.append(colat)
            forwards.append(forward)
            rates.append(rate)
            corr_err.append(ci)
            rate_err.append(cred_int)

            sizes = np.array([f.energy_proxy for f in region_flares if not np.isnan(f.energy_proxy)])
            sizes = sizes[sizes > 0]
            # distributions, params_list, aics = aic(sizes, verbose=False)
            # best_psize = distributions[np.argmin(aics)]
            # distributions, params_list, aics = aic(active.waiting_times(), verbose=False)
            # best_pdelt = distributions[np.argmin(aics)]
            # if best_pdelt == best_psize:
            #     matching.append(True)
            # else:
            #     matching.append(False)

    matching = np.array(matching)
    rates = np.array(rates)
    forwards = np.array(forwards)
    if latitudes:
        colats = np.array(colats)
        driving = rates * colats
    else:
        driving = rates

    fig, ax = plt.subplots(figsize=(3.3, 2.))

    ax.scatter(driving, forwards, marker='x', s=3, lw=0.5, color='black')
    # ax.scatter(driving[matching], forwards[matching], marker='x', s=5, lw=0.5, color='red', label=r'Size and $\Delta t$ PDFs ``match"')
    # ax.scatter(driving[~matching], forwards[~matching], marker='x', s=5, lw=0.5, color='black', label=r'Size and $\Delta t$ PDFs not ``matching"')
    
    # ax.legend(loc='lower right')
    # rate_err = np.abs(np.transpose(rate_err) - rates)
    # corr_err = np.abs(np.transpose(corr_err) - forwards)
    # ax.errorbar(driving, forwards, xerr=rate_err, yerr=corr_err, fmt='none', lw=0.5, color='black', alpha=0.2)

    ax.axhline(0, ls='dashed', lw=0.8, color='black', alpha=0.5)
    ax.set_ylabel(r'$\rho_{+,\,k}$')
    if latitudes:
        ax.set_xlabel(r'Driving rate [cos(latitude)$\,\times\,$rate] ($\textrm{hours}^{-1})$')
        print(spearmanr(driving, forwards))
        name = 'correl_driving.png'
    else:
        ax.set_xlabel(r'$\lambda_k$ ($\textrm{hours}^{-1})$')
        print(spearmanr(rates, forwards))
        name = 'correl_rate_min%d_%s.pdf'%(min_num, which_wt)

    ax.set_xlim(xmin=0)
    ax.set_ylim(-1, 1)
    plt.savefig(os.path.join(path_to_working, 'figs/data', name), dpi=600)
    plt.savefig(os.path.join(path_to_working, 'paper_tex', name), dpi=450)
    plt.show()

def plot_best_dists_correl(flares, verbose=True):
    if verbose:
        print('When fitting size distributions:')
    x_dists, _, x_actives = psizes_individual_aic(flares, min_sizes=6, justthree=True, verbose=verbose)
    if verbose:
        print('When fitting waiting time distributions:')
    t_dists, _, t_actives = pdelt_individual_aic(flares, min_waiting_times=6, justthree=True, verbose=verbose)

    x_dists, x_actives, t_dists, t_actives = match_actives(x_actives, t_actives, x_dists=x_dists, t_dists=t_dists)

    order = np.sort(np.unique(x_dists))
    for trim in ['Seperable', 'Weibull']:
        if trim in order:
            order = np.delete(order, np.where(order==trim)[0])

    num_counts = np.zeros(shape=(len(order), len(order)))
    correls = np.zeros(shape=(len(order), len(order)))
    for i, distcomp_pdelt in enumerate(order):
        for j, distcomp_size in enumerate(order):
            delt_mask = t_dists==distcomp_pdelt
            size_mask = x_dists==distcomp_size
            num_counts[i, j] = np.sum(delt_mask & size_mask)
            actives = x_actives[delt_mask & size_mask]
            temp_correl_list = []
            for active in actives:
                correl, correl_err = active.forward_correlation()
                if correl is not None:
                    temp_correl_list.append(correl)
            correls[i, j] = np.mean(temp_correl_list)

    total = np.sum(num_counts)
    
    fig, ax = plt.subplots(figsize=(5, 3.3))
    im = ax.imshow(np.transpose(correls), interpolation='none', origin='upper', cmap='gray', vmin=-.3, vmax=0.3)
    for i, _ in enumerate(order):
        for j, __ in enumerate(order):
            num = num_counts[i, j]
            correl_mean = correls[i, j]
            s = r'$N=\,$' + f'{int(num):d}'
            rho = r'$\bar{\rho}_+=$\,' + f'{correl_mean:.3f}'
            ax.text(i, j, rho, ha='center', va='center', bbox=dict(edgecolor='none', facecolor='white', alpha=0.5))
            ax.text(i-0.42, j+0.35, s, ha='left', va='center', bbox=dict(edgecolor='none', facecolor='white', alpha=0.5), fontsize=6)

    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label=r'$\bar{\rho}_+$')

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels(order[::-1])

    ax.set_xlabel('Best-fit distribution for waiting times')
    ax.set_ylabel('Best-fit distribution for sizes')

    plt.savefig(os.path.join(path_to_working, 'figs/data', 'best_dists_correl.png'), dpi=600)
    plt.show()

def _swarmplot_code():
    ax = sns.swarmplot(x=rates, y=best_dist, size=1.5, ax=ax)
    
    ax = sns.stripplot(x=rates, y=best_dist, size=2, ax=ax, order=order, color='black')
    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    colors = []
    alpha = 0.2
    errs_reordered = []
    for point_pair in ax.collections:
        fcs = point_pair.get_fc()
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)
            # print(np.where(rates==x)[0][0])
            errs_reordered.append(rates_err[np.where(rates==x)[0][0]])
            fc = fcs[0]
            fc[-1] = alpha
            colors.append(fc)
    ax.errorbar(x_coords, y_coords, xerr=np.transpose(errs_reordered), fmt='none', ecolor=colors, zorder=-1)
    pass

def lnnorm_rates(flares):
    regions = np.unique(flares['region_num'])
    rates = []
    for region in regions:
        region_flares = [Flare(*f, units=units) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)
        rates.append(active.rate()[0])

    fit = LogNormal.mle_params(LogNormal, rates)
    print(fit)

def plot_pdfs_2panel(flares, min_n=5, delt_units='h', dels_units='s', c1masking=False, dels_masking=False, which_wt='start', loglin_pdelt=False):
    fig, axarr = plt.subplots(nrows=2, figsize=(3.3, 3.5))

    regions = np.unique(flares['region_num'])
    all_waiting_times = []
    all_sizes = []
    masked_waiting_times = []
    masked_sizes = []
    for region in regions:
        region_flares = [Flare(*f, units=dels_units) for f in flares[flares['region_num']==region]]
        if c1masking:
            masked_flares = [f for f in region_flares if f.peak_flux >= 1e-6]
        if dels_masking:
            masked_flares = [f for f in region_flares if f.energy_proxy >= 9.6e-4]
        elif not (c1masking or dels_masking):
            masked_flares = region_flares
        active = ActiveRegion(region, masked_flares)
        full_active = ActiveRegion(region, region_flares)
        if which_wt=='peak':
            wt = full_active.waiting_times(units=delt_units,                         which=which_wt)
            if any([w <= 0 for w in wt]) or any(np.isnan(wt)):
                continue

        if len(active.flares) < min_n:
            continue
        masked_waiting_times.append(active.waiting_times(units=delt_units, which=which_wt))
        masked_sizes.append([f.energy_proxy for f in masked_flares])
        all_waiting_times.append(full_active.waiting_times(units=delt_units, which=which_wt))
        all_sizes.append([f.energy_proxy for f in region_flares])
    masked_waiting_times = np.concatenate(masked_waiting_times).ravel()
    try:
        assert(all([m > 0 for m in masked_waiting_times]))
    except:
        print(np.where(masked_waiting_times < 0))
    masked_sizes = np.concatenate(masked_sizes).ravel()
    masked_sizes = masked_sizes[masked_sizes > 0]
    all_waiting_times = np.concatenate(all_waiting_times).ravel()
    all_sizes = np.concatenate(all_sizes).ravel()
    all_sizes = all_sizes[all_sizes > 0]

    nbins = 50
    for data, all_data, which, ax in zip([masked_waiting_times, masked_sizes], [all_waiting_times, all_sizes], ['pdelt', 'psize'], axarr):
        distributions, params_list, aics = aic(data, verbose=True, justthree=True)
        bins = np.logspace(np.log10(min(data)), np.log10(max(data)), num=nbins)
        if loglin_pdelt and which=='pdelt':
            bins = np.linspace(min(data), max(data), num=nbins)
        data_heights, data_bins, _ = ax.hist(data, bins=bins, log=True, color='black', histtype='step', density=True, label='Data', lw=1.5, alpha=1)
        ylims = ax.get_ylim()

        all_bins = np.logspace(np.log10(min(all_data)), np.log10(max(all_data)), num=nbins)
        if which=='psize' and (c1masking or dels_masking):
            all_data_heights, all_data_bins, _ = ax.hist(all_data, bins=all_bins, log=True, color='red', histtype='step', density=True, lw=1, alpha=0)
            scale = 1.5 if c1masking else 2
            all_data_heights *= scale

        x_range = np.logspace(np.log10(min(all_data)), np.log10(max(all_data)), num=1000)
        if loglin_pdelt and which=='pdelt':
            x_range = np.linspace(min(all_data), max(all_data), num=1000)
        linestyles = ['--', ':', '-.']
        for distribution, params, ls in zip(distributions, params_list, linestyles):
            dist = distribution(*params)
            y = dist.pdf(x_range)
            z = 3
            lw = 1.5
            ax.plot(x_range, y, label=dist.name, alpha=1, ls=ls, zorder=z, lw=lw)

        if which=='pdelt':
            unit_str = 'hours'
            ax.set_xlabel(r'$\Delta t$ (%s)'%unit_str)
            ax.set_ylabel(r'$p(\Delta t)$')
            if loglin_pdelt:
                legend_loc = 'upper right'
            else:
                legend_loc = 'lower left'
            ax.legend(loc=legend_loc, handlelength=2, fontsize=8)
        elif which=='psize':
            unit_str = r'Jm$^{-2}$'
            ax.set_xlabel(r'$\Delta s$ (%s)'%unit_str)
            ax.set_ylabel(r'$p(\Delta s)$')
            legend_loc = 'lower left'
        if which=='psize':
            ax.set_xscale('log')
        elif not loglin_pdelt:
            ax.set_xscale('log')
            
        ax.set_xlim(xmin=min(all_bins), xmax=max(all_bins))
        if which=='pdelt':
            ax.set_xlim(xmin=0.1)
        if which=='psize' and (c1masking or dels_masking):
            comp_x_range = np.logspace(np.log10(2e-5), np.log10(1e-3), num=10000)
            all_data_y = [all_data_heights[np.argmin(abs(all_data_bins - x))] for x in comp_x_range]
            masked_y = [data_heights[np.argmin(abs(data_bins - x))] if x > min(bins) else ylims[0] for x in comp_x_range]
            ax.fill_between(comp_x_range*1.06, all_data_y, masked_y, fc='black',  ec='None', alpha=0.3)

        if which=='psize':
            ax.set_xlim(xmin=2.5e-5)

        ax.set_ylim(ylims[0], ylims[1]*2)
        ax.tick_params(axis='x', which='minor', bottom=False)
    
    if not (c1masking or dels_masking):
        name = '%s_overall_loglog_fits_%s.pdf'%('pdfs', which_wt)
    elif c1masking:
        name = '%s_overall_loglog_fits_c1mask_%s.pdf'%('pdfs', which_wt)
    elif dels_masking:
        name = '%s_overall_loglog_fits_delsmask_%s.pdf'%('pdfs', which_wt)
    if loglin_pdelt:
        name = '%s_overall_loglog_fits_delsmask_%s_loglin.pdf'%('pdfs', which_wt)

    plt.savefig(os.path.join(path_to_working, 'paper_tex', name), dpi=450)
    plt.show()

def plot_proportions_2panel(flares, min_num=5, c1masking=False, dels_masking=False, which_wt='start'):
    fig, axarr = plt.subplots(nrows=2, figsize=(3.3, 4))

    for ax, which in zip(axarr, ['delt', 'size']):
        if which == 'delt':
            best_dist, nums, actives = pdelt_individual_aic(flares, min_waiting_times=min_num-1, units='s', verbose=True, justthree=True, c1masking=c1masking, dels_masking=dels_masking, which_wt=which_wt)
        if which == 'size':
            best_dist, nums, actives = psizes_individual_aic(flares, min_sizes=min_num, units='s', verbose=True, justthree=True, c1masking=c1masking, dels_masking=dels_masking)
        rates = []
        for active in actives:
            rate, _ = active.rate()
            rates.append(rate)
        rates = np.array(rates)
        assert all(rates > 0)

        cdf_x, cdf_y = rank_cdf(rates)

        # bins = np.linspace(0, max(rates), num=15)
        bins = np.linspace(0.025, 0.3, num=20)
        bin_width = np.diff(bins)[0]
        bin_mids = (bins[1:] + bins[:-1]) / 2
        total_distnames = np.sort(np.unique(best_dist))
        proportions_ordered = []
        for i, bin in enumerate(bins[:-1]):
            mask = (rates >= bin) & (rates < bins[i+1])
            dists = best_dist[mask]
            distnames_in_bin, counts = np.unique(dists, return_counts=True)
            proportions = counts / len(dists)

            ordered = []
            for distname in total_distnames:
                if distname in distnames_in_bin:
                    ordered.append(proportions[np.where(distnames_in_bin==distname)[0]][0])
                else:
                    ordered.append(0)
            proportions_ordered.append(ordered)

        proportions_full = np.transpose(proportions_ordered)
        for (i, dist), prop in zip(enumerate(total_distnames), proportions_full):
            if i==0:
                height = np.zeros(len(prop))
            else:
                height = np.sum(proportions_full[:i], axis=0)

            ax.bar(bin_mids, prop, bin_width, bottom=height, label=dist)

        cdf_ax = ax.twinx()
        cdf_ax.plot(cdf_x, cdf_y, color='black', label='CDF', zorder=1, alpha=0.7)
        cdf_ax.set_ylabel('CDF')
        cdf_ax.set_ylim(0, 1)

        if which=='delt':
            ax.legend(loc='lower right', fontsize=8)

        ax.set_xlim(min(bins), max(bins))
        ax.set_ylim(0, 1)
        ax.set_xlabel(r'$\lambda_k$ (hours$^{-1}$)')
        if which=='delt':
            ax.set_ylabel(r'Proportion with $p(\Delta t)$' + '\nfitted best by each shape')
        if which=='size':
            ax.set_ylabel(r'Proportion with $p(\Delta s)$' + '\nfitted best by each shape')
    # plt.savefig(os.path.join(path_to_working, 'figs/data', 'prop_%s_rate.png'%which), dpi=600)
    if not (c1masking or dels_masking):
        name = 'props_2panel_rate_min%d_%s.pdf'%(min_num, which_wt)
    elif c1masking:
        name = 'props_2panel_rate_min%d_c1mask_%s.pdf'%(min_num, which_wt)
    elif dels_masking:
        name = 'props_2panel_rate_min%d_delsmask_%s.pdf'%(min_num, which_wt)
    plt.savefig(os.path.join(path_to_working, 'paper_tex', name), dpi=450)

    plt.show()

def plot_perar_cdfs(all_flares, c1masking=False, dels_masking=True, which_wt='peak'):
    # regions, num_per = np.unique(all_flares['region_num'], return_counts=True)
    # zip(*sorted(zip(regions, num_per)))
    # print(num_per[-20:-10])
    # print(regions[-20:-10])

    # region_nums = [2645, 4975, 6993, 11967]
    region_nums = [2645, 4975, 11967, 13014]
    gridspec = dict(hspace=0.0, height_ratios=[1, 1, 0.2, 1, 1])
    fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(5.5, 6.5), gridspec_kw=gridspec)
    topleft_ax = [axarr[0, 0], axarr[1, 0]]
    topright_ax = [axarr[0, 1], axarr[1, 1]]
    botleft_ax = [axarr[3, 0], axarr[4, 0]]
    botright_ax = [axarr[3, 1], axarr[4, 1]]
    axarr[2, 0].set_visible(False)
    axarr[2, 1].set_visible(False)
    for region, ax_quadrant in zip(region_nums, [topleft_ax, topright_ax, botleft_ax, botright_ax]):
        all_sizes = []
        masked_waiting_times = []
        masked_sizes = []
        region_flares = [Flare(*f, units='s') for f in all_flares[all_flares['region_num']==region]]
        if c1masking:
            masked_flares = [f for f in region_flares if f.peak_flux >= 1e-6]
        if dels_masking:
            masked_flares = [f for f in region_flares if f.energy_proxy >= 9.6e-4]
        elif not (c1masking or dels_masking):
            masked_flares = region_flares
        active = ActiveRegion(region, masked_flares)
        full_active = ActiveRegion(region, region_flares)
        masked_waiting_times.append(active.waiting_times(units='h', which=which_wt))
        masked_sizes.append([f.energy_proxy for f in masked_flares])
        all_sizes.append([f.energy_proxy for f in region_flares])
        masked_waiting_times = np.concatenate(masked_waiting_times).ravel()
        masked_sizes = np.concatenate(masked_sizes).ravel()
        masked_sizes = masked_sizes[masked_sizes > 0]
        all_sizes = np.concatenate(all_sizes).ravel()
        all_sizes = all_sizes[all_sizes > 0]

        print('Active region %d, number of flares: %d'%(region, len(all_sizes)))
        for data, which, ax in zip([masked_waiting_times, masked_sizes], ['pdelt', 'psize'], ax_quadrant):
            distributions, params_list, aics = aic(data, verbose=False, justthree=True)
            print(which)
            print(aics)

            if which=='psize':
                minx = 1e-3
                maxx = max(data) * 1.5
            else:
                minx = min(data) / 1.5
                maxx = max(data) * 1.5
            if which=='psize':
                x_range = np.logspace(np.log10(minx), np.log10(maxx), num=1000)
            else:
                x_range = np.linspace(minx, maxx, num=1000)
            rank_x, rank_y = rank_cdf(data)
            rank_y = 1 - rank_y
            ax.step(rank_x, rank_y, where='post', color='black', lw=1.2, zorder=4)

            ax.set_yscale('log')
            if which=='psize':
                ax.set_xscale('log')
            ylims = ax.get_ylim()

            linestyles = ['--', ':', '-.']
            ymins = []
            for distribution, params, ls, a in zip(distributions, params_list, linestyles, aics):
                dist = distribution(*params)
                y = dist.cdf(x_range)
                y = 1 - y
                z = 3
                if a == min(aics):
                    label = r'\textbf{%s}'%dist.name
                    lw = 1.5
                else:
                    label = dist.name
                    lw = 1.5
                ax.plot(x_range, y, label=label, alpha=1, ls=ls, zorder=z, lw=lw)
                ymins.append(min(y))
            ymins = np.sort(ymins)

            ax.minorticks_off()
            if which=='pdelt':
                # if region==4975:
                #     ax.set_xticks([1, 1e1, 1e2])
                ax.set_xlim(xmin=minx)
                ax.set_ylim(ymin=ymins[1])
                unit_str = 'hours'
                ax.set_xlabel(r'$\Delta t$ (%s)'%unit_str)
                # ax.set_ylabel('Fraction with\n'+r"$\Delta t' < \Delta t$")
                ax.set_ylabel('CCDF')
                legend_loc = 'upper right'
                ax.legend(loc=legend_loc, handlelength=2, fontsize=7)
                ax.set_title(r'AR %d, $N_k = %d$'%(region, len(all_sizes)))
            elif which=='psize':
                ax.set_xlim(xmin=5e-4)
                ax.set_ylim(ymin=ymins[1], ymax=1.5)
                ax.fill_betweenx([ymins[1], 1.5], [1e-4, 1e-4], [1e-3, 1e-3], alpha=0.2, fc='black', ec='none')
                unit_str = r'Jm$^{-2}$'
                ax.set_xlabel(r'$\Delta s$ (%s)'%unit_str)
                # ax.set_ylabel('Fraction with\n'+r"$\Delta s' < \Delta s$")
                ax.set_ylabel('CCDF')
                legend_loc = 'lower left'
                ax.legend(loc=legend_loc, handlelength=2, fontsize=7)

                # ax.set_ylim(ymins[-2], ylims[1]*2)
            ax.tick_params(axis='x', which='minor', bottom=False)

        
    # if not (c1masking or dels_masking):
    #     name = '%s_ar%d_loglog_fits_%s.pdf'%('pdfs', region, which_wt)
    # elif c1masking:
    #     name = '%s_ar%d_loglog_fits_c1mask_%s.pdf'%('pdfs', region, which_wt)
    # elif dels_masking:
    #     name = '%s_ar%d_loglog_fits_delsmask_%s.pdf'%('pdfs', region, which_wt)
    plt.savefig(os.path.join(path_to_working, 'paper_tex', 'individual_ars_ccdf.pdf'), dpi=450)
    plt.show()

if __name__=="__main__":
    units = 'h'

    total_data = return_all_flares() 
    flares = keep_regions_min_num(total_data, min_per_region=5, verbose=False)
    # plot_pdelt_overall(flares, loglog=True, justthree=True)
    # plot_psizes_overall(flares, loglog=True, justthree=True, units='s')
    # psizes_individual_aic(flares, min_sizes=5, verbose=True, justthree=True)

    # pdelt_overall_aic(flares, units=units, loglog=True)
    # best_dist, num_delt, actives = pdelt_individual_aic(flares, min_waiting_times=4, units=units, verbose=True, justthree=True)
    # plot_propbestdist_numdelt(flares, min_num=10, max_num=60, units=units)

    # plot_correlations_hist(flares)
    # plot_colatitudes_hist(flares)
    # plot_correl_rate(flares, latitudes=False, min_num=10, which_wt='peak')
    # plot_best_dists_correl(flares)
    # plot_propbestdist_rate(flares, min_num=10, max_num=1000, which='delt', units='h', justthree=True)
    # plot_bestdists_rate_violin(flares, min_num=5, which='delt')

    # lnnorm_rates(flares)
    plot_pdfs_2panel(flares, c1masking=False, dels_masking=True, which_wt='peak', loglin_pdelt=True)
    # plot_proportions_2panel(flares, min_num=5, c1masking=False, dels_masking=True, which_wt='peak')
    # plot_perar_cdfs(flares, c1masking=False, dels_masking=True)
