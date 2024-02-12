import numpy as np
import sys, os
import matplotlib.pyplot as plt
seq_4colors = ["#e66101", "#fdb863", "#b2abd2", "#5e3c99"]
qual_5colors = ["#e41a1c", "#4daf4a", "#984ea3", "#377eb8", "#ff7f00"]
from scipy.stats import gaussian_kde, spearmanr, ks_2samp
fig_dir = os.path.join('/Users/julian/Documents/phd/solar_flares/figs', 'data')
tex_dir = os.path.join('/Users/julian/Documents/phd/solar_flares/', 'paper_tex')
# from extract_xrs_summary import read_years_old, keep_regions_min_num_old, Flare_old, ActiveRegion_old
from read_flare_db import return_all_flares, keep_regions_min_num
from flare_classes import Flare, ActiveRegion

# plot num of flares by active region number
def plot_flares_by_region(flares):
    fig, ax = plt.subplots(figsize=(8, 3))

    regions = np.unique(flares['region_num'])
    num_flares = []
    for region in regions:
        flares_in_region = flares[flares['region_num']==region]
        num_flares.append(len(flares_in_region))

    ax.scatter(regions, num_flares, marker='s', s=5, fc='black', ec='none', alpha=0.5)
    ax.set_xlabel('Active region number')
    ax.set_ylabel('Number of flares')

    plt.savefig(os.path.join(fig_dir, 'flares_by_region.png'), dpi=600)
    plt.show()

# histogram of num of flares in active regions
def plot_hist_numflares(flares, log=True):
    fig, ax = plt.subplots(figsize=(3.3, 1.5))

    regions, num_per_region = np.unique(flares['region_num'], return_counts=True)
    print(f'The median number of flares per region is {np.median(num_per_region)}, the average is {np.mean(num_per_region)}')

    ax.hist(num_per_region, bins=np.arange(0.5, max(num_per_region)+1.5), histtype='step', align='mid', log=log, color='black', density=True)
    # ax.set_xlim(0, 160)
    ax.set_xlim(xmin=0)
    ax.set_xlabel(r'$N_k$')
    # ax.set_title('Histogram of number of flares per active region')
    ax.set_ylabel(r'$p(N_k)$')

    if not log:
        ax.set_ylim(ymax=25)
        # plt.savefig(os.path.join(fig_dir, 'hist_numflares_notlog.png'), dpi=600)
    elif log:
        # plt.savefig(os.path.join(fig_dir, 'hist_numflares.png'), dpi=600)
        ax.minorticks_off()
        plt.savefig(os.path.join(tex_dir, 'hist_num.pdf'), dpi=450)
        # ax.set_yticks([1,10,100])
    plt.show()    

def plot_average_waitingtimes(flares, norm_by_dur=False):
    regions, num_per_region = np.unique(flares['region_num'], return_counts=True)
    regions = [regions for _, regions in sorted(zip(num_per_region, regions))]
    num_per_region = np.sort(num_per_region)

    means = []
    stds = []
    unique_nums_per_region, multiplicities = np.unique(num_per_region, return_counts=True)

    for unique_num_per_region, multiplicity in zip(unique_nums_per_region, multiplicities):
        region_subset = []
        for region, num_per in zip(regions, num_per_region):
            if num_per==unique_num_per_region:
                region_subset.append(region)

        all_waiting_times = []
        for region in region_subset:
            region_flares = [Flare(*f) for f in flares[flares['region_num']==region]]
            active = ActiveRegion(region, region_flares)
            waiting_times = active.waiting_times()
            if norm_by_dur:
                waiting_times = waiting_times / active.duration
            all_waiting_times.append(waiting_times)
        all_waiting_times = np.concatenate(all_waiting_times).ravel()
        means.append(np.mean(all_waiting_times))
        stds.append(np.std(all_waiting_times))
    
    fig, ax = plt.subplots(figsize=(3.5, 2.3))

    ax.errorbar(unique_nums_per_region, means, yerr=stds, fmt='none', color='black', alpha=1, lw=0.7)
    ax.scatter(unique_nums_per_region, means, color='black', alpha=1, s=3)
    ax.set_xlim(xmin=2)
    ax.set_xlabel('Number of flares in the active region')
    if norm_by_dur:
        ax.set_ylabel('Average waiting time /\n duration of active region')
        name = 'ave_waiting_time_normed.png'
        ax.set_ylim(0, 0.4)
    else:
        ax.set_ylabel('Average waiting time (hours)')
        name = 'ave_waiting_time_by_numper.png'
        ax.set_ylim(ymin=0, ymax=65)

    plt.savefig(os.path.join(fig_dir, name), dpi=600)
    plt.show()


def plot_pdelt_overall(flares, loglog=False):
    regions = np.unique(flares['region'])
    all_waiting_times = []
    for region in regions:
        region_flares = [Flare(*f) for f in flares[flares['region']==region]]
        active = ActiveRegion(region, region_flares)
        waiting_times = active.waiting_times()
        all_waiting_times.append(waiting_times)
    all_waiting_times = np.concatenate(all_waiting_times).ravel()

    print(f'Number of waiting times included in histogram: {len(all_waiting_times)}')
    fig, ax = plt.subplots(figsize=(4.5, 3.3))

    if loglog:
        bins = np.logspace(np.log10(min(all_waiting_times)), np.log10(max(all_waiting_times)), num=50)
    else:
        bins = np.linspace(min(all_waiting_times), max(all_waiting_times), num=50)
    ax.hist(all_waiting_times, bins=bins, log=True, color='black', histtype='step', density=True)
    ax.set_title('Histogram of waiting times\nacross all active regions', size=10)
    ax.set_xlabel('Waiting time (hours)')
    ax.set_ylabel('Probability density')
    if loglog:
        ax.set_xscale('log')
        ax.set_xlim(xmin=min(bins), xmax=max(bins))
        name = 'pdelt_overall_loglog.png'
        ax.set_ylim(ymin=1e-5)
    else:
        ax.set_xlim(xmin=0)
        name = 'pdelt_overall.png'

    plt.savefig(os.path.join(fig_dir, name), dpi=600)
    plt.show()    


def plot_pdelt_eachregionkde(flares, topn=20):
    regions, num_per_region = np.unique(flares['region'], return_counts=True)
    regions = [regions for _, regions in sorted(zip(num_per_region, regions))]
    num_per_region = np.sort(num_per_region)

    fig, ax = plt.subplots(figsize=(5, 3))
    xplot = np.linspace(0, 20, num=1000)

    regions = regions[::-1]
    regions = regions[:topn]

    for region in regions:
        region_flares = [Flare(*f) for f in flares[flares['region']==region]]
        active = ActiveRegion(region, region_flares)
        waiting_times = active.waiting_times()

        kde = gaussian_kde(waiting_times, bw_method='scott')
        pdf = kde.pdf(xplot)
        ax.plot(xplot, pdf, color='black', alpha=0.5)
    ax.set_xlim(xmin=0, xmax=max(xplot))
    ax.set_xlabel('Waiting time (hours)')
    ax.set_ylabel('Probability density')
    ax.set_title('KDE estimates of waiting time PDFs\nfor the top %d most active regions'%topn, size=11)

    plt.savefig(os.path.join(fig_dir, 'pdelt_top%d.png'%topn), dpi=600)
    plt.show()


def plot_hist_rates(flares, units='h'):
    fig, ax = plt.subplots(figsize=(3.3, 1.5))

    regions = np.unique(flares['region_num'])
    rates = []
    for region in regions:
        region_flares = [Flare(*f, units=units) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)
        rates.append(active.rate()[0])

    rates = np.sort(rates)
    print(f'95% of regions have lambda > {rates[int(0.05*len(rates))]}')
    print(f'95% of regions have lambda < {rates[int(0.95*len(rates))]}')
    print(f'central 80% of regions have {rates[int(0.1*len(rates))]} < lambda < {rates[int(0.9*len(rates))]}')

    ax.hist(rates, bins=50, histtype='step', density=True, color='black', range=(0, 0.5))
    ax.set_xlim(xmin=0, xmax=0.5)
    ax.set_xlabel(r'$\lambda_k$ (hours$^{-1}$)')
    ax.set_ylabel(r'$p(\lambda_k)$')

    plt.savefig(os.path.join(tex_dir, 'hist_rates.pdf'), dpi=450)
    plt.show()

def plot_hist_numrates_2panel(flares, units='h'):
    fig, (ax_nums, ax_rates) = plt.subplots(nrows=2, figsize=(3.3, 2.8))

    regions, num_per_region = np.unique(flares['region_num'], return_counts=True)
    print(f'The median number of flares per region is {np.median(num_per_region)}, the average is {np.mean(num_per_region)}')

    ax_nums.hist(num_per_region, bins=np.arange(0.5, max(num_per_region)+1.5), histtype='step', align='mid', log=True, color='black', density=True)
    ax_nums.set_xlim(xmin=0)
    ax_nums.set_xlabel(r'$N_k$')
    ax_nums.set_ylabel(r'${\rm Pr}(N_k)$')

    kept_regions = regions[num_per_region >= 5]
    rates = []
    for region in kept_regions:
        region_flares = [Flare(*f, units=units) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)
        rates.append(active.rate()[0])

    rates = np.sort(rates)
    print(f'95% of regions have lambda > {rates[int(0.05*len(rates))]}')
    print(f'95% of regions have lambda < {rates[int(0.95*len(rates))]}')
    print(f'The mean flare rate is {np.mean(rates)}')

    ax_rates.hist(rates, bins=50, histtype='step', density=True, color='black', range=(0, 0.5))
    ax_rates.set_xlim(xmin=0, xmax=0.5)
    ax_rates.set_xlabel(r'$\lambda_k$ (hours$^{-1}$)')
    ax_rates.set_ylabel(r'$p(\lambda_k)$')

    ax_nums.minorticks_off()
    plt.savefig(os.path.join(tex_dir, 'hist_numrates.pdf'), dpi=450)
    plt.show()    

def plot_hist_corr(flares):
    regions, num_per_region = np.unique(flares['region_num'], return_counts=True)

    forwards = []
    backwards = []
    bins = np.linspace(-1, 1, num=30)
    for region in regions:
        region_flares = [Flare(*f) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)

        try:
            forward, ci = active.fb_correlation(which='forward')
            backward, ci = active.fb_correlation(which='backward')
        except TypeError:
            continue
        if forward is not None:
            forwards.append(forward)
        if backward is not None:
            backwards.append(backward)

    print(f'Mean rho_+ = {np.mean(forwards)}, mean rho_- = {np.mean(backwards)}')
    print(f'Median rho_+ = {np.median(forwards)}, median rho_- = {np.median(backwards)}')
    print(f'Median N_k = {np.median(num_per_region)}')
    fig, ax = plt.subplots(figsize=(3.3, 1.5))
    ax.hist(forwards, histtype='step', color='black', bins=bins, label=r'$\rho_{+,\,k}$', density=True)
    ax.hist(backwards, histtype='step', color='black', bins=bins, linestyle='dashed', label=r'$\rho_{-,\,k}$', density=True)
    ax.set_xlabel(r'$\rho_{\pm,\,k}$')
    ax.set_ylabel(r'$p\left(\rho_{\pm,\,k}\right)$')
    ax.legend()
    ax.set_xlim(-1, 1)
    plt.savefig(os.path.join(tex_dir, 'hist_corr.pdf'), dpi=450)
    plt.show()


def plot_corr_2panel(flares, hist_min=5, scatter_min=10, which_wt='start'):
    regions, num_per_region = np.unique(flares['region_num'], return_counts=True)

    forwards = []
    backwards = []
    rates = []
    bins = np.linspace(-1, 1, num=30)
    for region in regions[num_per_region >= hist_min]:
        region_flares = [Flare(*f) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)

        try:
            forward, ci = active.fb_correlation(which='forward', which_wt=which_wt)
            backward, ci = active.fb_correlation(which='backward', which_wt=which_wt)
            if np.isnan(forward) or np.isnan(backward):
                continue
        except TypeError:
            continue
        if forward is not None:
            forwards.append(forward)
        if backward is not None:
            backwards.append(backward)

    print(f'Mean rho_+ = {np.mean(forwards)}, mean rho_- = {np.mean(backwards)}')
    print(f'Median rho_+ = {np.median(forwards)}, median rho_- = {np.median(backwards)}')
    print(f'Median N_k = {np.median(num_per_region)}')
    print(ks_2samp(forwards, backwards))
    fig, axarr = plt.subplots(nrows=2, figsize=(3.3, 3.5))
    axarr[0].hist(forwards, histtype='step', color='black', bins=bins, label=r'$\rho_{+,\,k}$', density=True)
    axarr[0].hist(backwards, histtype='step', color='black', bins=bins, linestyle='dashed', label=r'$\rho_{-,\,k}$', density=True)
    axarr[0].set_xlabel(r'$\rho_{\pm,\,k}$')
    axarr[0].set_ylabel(r'$p\left(\rho_{\pm,\,k}\right)$')
    axarr[0].legend()
    axarr[0].set_xlim(-1, 1)

    forwards = []
    rates = []
    for region in regions[num_per_region >= scatter_min]:
        region_flares = [Flare(*f) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)

        try:
            forward, ci = active.fb_correlation(which='forward', which_wt=which_wt)
            if np.isnan(forward):
                continue
            rate, ci = active.rate()
        except TypeError:
            continue
        if forward is not None:
            forwards.append(forward)
        if rate is not None:
            rates.append(rate)

    print(spearmanr(forwards, rates))
    axarr[1].scatter(rates, forwards, marker='x', s=3, lw=0.5, color='black')
    axarr[1].axhline(0, ls='dashed', lw=0.8, color='black', alpha=0.5)
    axarr[1].set_ylabel(r'$\rho_{+,\,k}$')
    axarr[1].set_xlabel(r'$\lambda_k$ ($\textrm{hours}^{-1})$')
    axarr[1].set_xlim(xmin=0)
    axarr[1].set_ylim(-1, 1)

    plt.savefig(os.path.join(tex_dir, 'hist_corr_2panel_%s.pdf'%(which_wt)), dpi=450)
    plt.show()

def autocorrelation(flares):
    regions, num_per_region = np.unique(flares['region_num'], return_counts=True)
    delt_ac = []
    size_ac = []
    bins = np.linspace(-1, 1, num=30)
    for region in regions:
        region_flares = [Flare(*f) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)

        delt_ac.append(active.autocorrelation(which='delt', ci=0)[0])
        size_ac.append(active.autocorrelation(which='size', ci=0)[0])

    
    print(f'Mean rho_delt = {np.mean(delt_ac)}, mean rho_size = {np.mean(size_ac)}')
    print(f'Median rho_delt = {np.median(delt_ac)}, median rho_size = {np.median(size_ac)}')

    fig, ax = plt.subplots(figsize=(3.3, 1.5))

    ax.hist(delt_ac, histtype='step', color='black', bins=bins, label=r'$\rho_{\Delta t,\,k}$', density=True)
    ax.hist(size_ac, histtype='step', color='black', bins=bins, linestyle='dashed', label=r'$\rho_{\Delta s,\,k}$', density=True)

    ax.set_xlabel(r'$\rho_{\Delta t,\,k}$, $\rho_{\Delta s,\,k}$')
    ax.set_ylabel(r'$p(\rho_{\Delta t,\,k})$, $p(\rho_{\Delta s,\,k})$')
    ax.legend()
    ax.set_xlim(-1, 1)
    plt.savefig(os.path.join(tex_dir, 'hist_ac.pdf'), dpi=450)
    plt.show()

def plot_all_deltas(flares):
    flares = [Flare(*f, units='s') for f in flares]
    all = np.array([(f.start_time, f.peak_flux, f.duration) for f in flares if f.peak_flux > 0 and f.duration > 0])
    dates = all[:, 0]
    fluxes = all[:, 1]
    durations = all[:, 2]
    dels = fluxes * durations

    print(f'Average duration: {np.mean(durations)}\tMedian duration: {np.median(durations)}')
    print(f'Fraction of flares with pflux < C1.0: {np.sum(fluxes<1e-6) / len(fluxes)}')
    print(f'Fraction of flares with dels < C1.0*mean dur: {np.sum(dels<(1e-6*np.mean(durations))) / len(dels)}')
    print(f'Fraction of flares with dels < C1.0*median dur: {np.sum(dels<(1e-6*np.median(durations))) / len(dels)}')
    return

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.scatter(dates, fluxes * durations, s=2, alpha=0.1, fc='blue', ec='None')
    ax.axhline(1e-6 * np.mean(durations), color='blue', ls='--', label=r'$C1.0\times\langle\tau_k\rangle$')
    ax.axhline(1e-6 * np.median(durations), color='blue', ls='-.', label=r'$C1.0\times\,\textrm{median}(\tau_k)$')
    ax2 = ax.twinx()
    ax2.scatter(dates, fluxes, s=2, alpha=0.1, fc='red', ec='None')
    ax2.axhline(1e-6, color='red', ls='--', label='$C1.0$')
    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax.set_ylabel(r'$\Delta s$')
    ax2.set_ylabel(r'Peak flux')
    ax.set_xlabel('Date')
    ax.legend()
    plt.savefig(os.path.join(fig_dir, 'dels_pflux_date.pdf'), dpi=450)
    plt.show()

def how_many_near(flares):
    regions, num_per_region = np.unique(flares['region_num'], return_counts=True)
    num_total = 0
    num_start_before_end = 0
    for region in regions:
        region_flares = [Flare(*f) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)
        starts = np.array([f.start_time for f in region_flares])
        ends = np.array([f.end_time for f in region_flares])
        
        diff_end_start = (starts[1:] - ends[:-1]) / np.timedelta64(1, 'h')
        num_total += len(starts)
        num_start_before_end += sum(diff_end_start < 0)

    print(f'{num_total} flares\n{num_start_before_end} started before last flare ended\n{num_start_before_end / num_total * 100:0.2f}%')


def duration(flares):
    regions, num_per_region = np.unique(flares['region_num'], return_counts=True)
    num_total = 0
    num_start_before_end = 0
    all_durs = []
    for region in regions:
        region_flares = [Flare(*f, units='s') for f in flares[flares['region_num']==region]]
        durations = [f.duration for f in region_flares]
        all_durs.append(durations)

    all_durs = np.concatenate(all_durs).ravel()

    print(np.median(all_durs), np.mean(all_durs))

        


if __name__ == "__main__":
    # total_data = read_years()
    # flares = keep_regions_min_num(total_data, min_per_region=3, verbose=False)
    all_flares = return_all_flares()
    flares = keep_regions_min_num(all_flares, min_per_region=5, verbose=False)
    # plot_all_deltas(flares)

    # plot_flares_by_region(flares)
    # plot_hist_numflares(flares, log=True)
    # plot_average_waitingtimes(flares, norm_by_dur=True)
    # plot_pdelt_overall(flares, loglog=True)
    # plot_pdelt_eachregionkde(flares)    
    # plot_hist_rates(flares)
    # plot_hist_corr(flares)
    # plot_hist_numrates_2panel(flares)
    # plot_corr_2panel(flares, which_wt='peak')
    # autocorrelation(flares)

    # how_many_near(flares)
    # duration(flares)
