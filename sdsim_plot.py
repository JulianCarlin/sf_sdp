import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

from scipy.stats import spearmanr, ks_2samp

from sdsim import glitches

path_to_working = '/Users/julian/Documents/phd/sdp_alpha/'
tex_dir = '/Users/julian/Documents/phd/solar_flares/paper_tex/'

import sdsim as sd

def avet_alpha_plot(n=1e4):
    etas = [s.split('_')[1] for s in os.listdir(os.path.join(path_to_working, 'picklejar'))]
    fig, ax = plt.subplots(figsize=(5, 4))
    for jumps in etas:
        folder = os.path.join(path_to_working, 'picklejar', 'eta_%s'%jumps)
        files = os.listdir(folder)
        alphas = np.sort([float(s.split('_')[1][1:]) for s in files])
        avets = []
        for alpha in alphas:
            file_name = 'lists_a%0.2f_%0.0e.pickle'%(alpha, int(n))
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'rb') as infile:
                list_read = pickle.load(infile)
            times = list_read['time_list']
            avet = np.mean(times)
            avets.append(avet)
        ax.plot(alphas, avets, label=jumps)
    ax.set_xscale('log')
    ax.set_xlim(min(alphas), max(alphas))
    ax.set_ylim(ymin=0)
    ax.set_ylabel(r'$\langle \Delta t \rangle$')
    ax.set_xlabel(r'$\alpha$')
    ax.legend()
    plt.savefig('avet_alpha.png', dpi=600)
    plt.show()
    
def plot_sdp_eg():
    np.random.seed(5) # for alpha=1, jumps=pow
    start_x = 0.3
    n = 20
    npdf = 1e7
    time_res = 10000

    # alphas = [0.1, 10]
    alphas = [10, 0.1]
    beta = 1e-2
    jumps = 'pow'

    fig = plt.figure(figsize=(6.5, 3))
    gs = GridSpec(nrows=2, ncols=4, figure=fig, height_ratios=[1.5, 1])

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    for alpha, ax in zip(alphas, [ax1, ax2]):
        _, time_list, size_list = sd.glitches(n, alpha, beta, jumps=jumps, start_x=start_x)
        t_range = np.linspace(0., np.sum(time_list)+0.1, time_res)
        cum_times = np.cumsum(time_list)
        time_indices = [abs(t_range - times).argmin() for times in cum_times]
        stress = start_x + np.array(range(time_res)) * np.diff(t_range)[0]
        for i, size in enumerate(size_list):
            stress[time_indices[i]:] += -size
        rate = 1. / (1. - stress)

        ax.plot(t_range, stress, color='black')
        ax.set_xlim(0, max(t_range))

        ax.set_xlabel('Time (arb. units)')
        ax.set_ylim(0, 1.1)
        # ax.axhline(1, ls='dashed', lw=0.7, color='red')
        ax.tick_params(labelbottom=False, axis='x', length=0, pad=10)
        ax.set_title(r'$\alpha=%g$'%alpha)

        ax.set_ylabel(r'Stress, $X$')

    folder = os.path.join(path_to_working, 'picklejar', 'eta_%s'%jumps)

    for alpha, axes in zip(alphas, [bottom_axes[:2], bottom_axes[2:]]):
        file = os.path.join(folder, f'lists_a{alpha:0.2f}_{npdf:1.0e}.pickle')
        print(file)
        if not os.path.isfile(file):
            print(f'Generating lists for PDFs (alpha={alpha:g}, eta={jumps}, n={npdf})')
            sd.gen_pdfs(alpha, n=npdf, jumps=jumps)
            assert os.path.isfile(file)

        with open(file, 'rb') as infile:
            list_read = pickle.load(infile)
        times = list_read['time_list']
        sizes = list_read['size_list']
        if alpha == 10:
            xmin = 1e-3
        elif alpha == 0.1:
            xmin = 5e-3
        logx_bins = np.logspace(np.log10(xmin), np.log10(max(sizes)), 100)
    

        if alpha==0.1:
            axes[0].hist(times, bins=logx_bins, histtype='step', density=True, log=True, color='black') 
            axes[0].set_xscale('log')
            axes[0].set_xlim(min(logx_bins), max(logx_bins))
            axes[0].set_xticks([1e-2, 1e-1, 1])
            
        else:
            axes[0].hist(times, bins=100, histtype='step', density=True, log=True, color='black') 
            axes[0].set_xlim(0, 1)

        axes[1].hist(sizes, bins=logx_bins, histtype='step', density=True, log=True, color='black') 
        axes[1].set_xscale('log')
        axes[1].set_xlim(min(logx_bins), max(logx_bins))

    for i, ax in enumerate(bottom_axes):
        # ax.tick_params(labelleft=False, which='both', axis='y', length=0, pad=10)
        if i > 1:
            ax.set_yticks([0.1, 1, 10])
            ax.tick_params(which='minor', axis='y', length=0, pad=10)
        else:
            ax.set_yticks([1e-3, 1e-1, 1e1])
        if i % 2:
            ax.set_xlabel(r'$\Delta X$')
            ax.set_ylabel(r'$p(\Delta X)$')
            if i == 3:
                ax.set_xticks([1e-2, 1e-1, 1])
        else:
            ax.set_xlabel(r'$\Delta t$')
            ax.set_ylabel(r'$p(\Delta t)$')
        

    plt.savefig(tex_dir + 'sdp_eg.pdf', format='pdf', dpi=450)
    plt.show()

def plot_correl(n=1e4):
    jumps = 'pow'
    folder = os.path.join(path_to_working, 'picklejar', 'eta_%s'%jumps)
    files = os.listdir(folder)
    # alphas = np.sort([float(s.split('_')[1][1:]) for s in files])
    alphas = np.logspace(-2, 2, num=100)
    fcorrels = []
    bcorrels = []
    # ms = []
    for alpha in alphas:
        file_name = 'lists_a%0.2f_%0.0e.pickle'%(alpha, int(n))
        file_path = os.path.join(folder, file_name)
        with open(file_path, 'rb') as infile:
            list_read = pickle.load(infile)
        times = list_read['time_list']
        sizes = list_read['size_list']
        fcorrel = spearmanr(times[1:], sizes[:-1])[0]
        fcorrels.append(fcorrel)
        bcorrel = spearmanr(times[:-1], sizes[1:])[0]
        bcorrels.append(bcorrel)
        # delt_normed = np.diff(times) / np.mean(np.diff(times))
        # size_normed = sizes / np.mean(sizes)
        # ms.append(ks_2samp(delt_normed, size_normed)[0])

    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    ax.plot(alphas, fcorrels, color='black', label=r'$\rho_+$')
    ax.plot(alphas, bcorrels, color='black', ls='dashed', label=r'$\rho_-$')
    # ax.plot(alphas, ms, color='red', label=r'$M$')
    # print(ms)

    ax.set_xscale('log')
    ax.set_xlim(min(alphas), max(alphas))
    ax.set_ylim(ymax=1)
    ax.set_ylabel('Spearman\ncross-correlation')
    ax.set_xlabel(r'$\alpha$')
    ax.legend()
    plt.savefig(tex_dir + 'sdp_correl.pdf', format='pdf', dpi=450)
    plt.show()

def match_test():
    alpha_range = np.logspace(-3, 2, num=100)
    ms = []
    n = 1e4
    for alpha in alpha_range:
        lag_list, time_list, size_list = glitches(int(n), alpha, beta=1e-2, pbar=False, jumps='pow')
        delts = np.array(time_list) / np.mean(time_list)
        sizes = np.array(size_list) / np.mean(size_list)
        ms.append(ks_2samp(delts, sizes)[1])

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(alpha_range, ms, color='black')
    # logx_bins = np.logspace(np.log10(min(sizes)), np.log10(1), 100)
    # logt_bins = np.logspace(np.log10(min(delts)), np.log10(1), 100)


    # ax.hist(delts, bins=logt_bins, histtype='step', density=True, log=True, color='red', ls='dashed') 
    # ax.hist(sizes, bins=logx_bins, histtype='step', density=True, log=True, color='green') 
    ax.set_xscale('log')

    plt.show()


def plot_ac(n=1e4):
    jumps = 'lognorm_fixed'
    folder = os.path.join(path_to_working, 'picklejar', 'eta_%s'%jumps)
    files = os.listdir(folder)
    # alphas = np.sort([float(s.split('_')[1][1:]) for s in files])
    alphas = np.logspace(-2, 2, num=100)
    delt_acs = []
    size_acs = []
    # ms = []
    for alpha in alphas:
        file_name = 'lists_a%0.2f_%0.0e.pickle'%(alpha, int(n))
        file_path = os.path.join(folder, file_name)
        with open(file_path, 'rb') as infile:
            list_read = pickle.load(infile)
        times = list_read['time_list']
        sizes = list_read['size_list']
        delt_ac = spearmanr(times[1:], times[:-1])[0]
        delt_acs.append(delt_ac)
        size_ac = spearmanr(sizes[:-1], sizes[1:])[0]
        size_acs.append(size_ac)

    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    ax.plot(alphas, delt_acs, color='black', label=r'$\rho_{\Delta t}$')
    ax.plot(alphas, size_acs, color='black', ls='dashed', label=r'$\rho_{\Delta X}$')

    ax.set_xscale('log')
    ax.set_xlim(min(alphas), max(alphas))
    ax.set_ylim(ymax=1)
    ax.set_ylabel('Autocorrelation')
    ax.set_xlabel(r'$\alpha$')
    ax.legend()
    plt.savefig(tex_dir + 'ac_correl_%s.pdf'%jumps, format='pdf', dpi=450)
    plt.show()

if __name__ == '__main__':
    # avet_alpha_plot()
    plot_sdp_eg()
    # plot_correl(n=1e5)
    # plot_ac(n=1e4)
    # match_test()
