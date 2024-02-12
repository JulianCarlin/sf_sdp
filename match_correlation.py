import numpy as np
import os, sys
import pickle
from scipy.stats import spearmanr, pearsonr, ks_2samp, gaussian_kde

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
seq_4colors = ["#e66101", "#fdb863", "#b2abd2", "#5e3c99"]
qual_5colors = ["#e41a1c", "#4daf4a", "#984ea3", "#377eb8", "#ff7f00"]

path_to_working = '/Users/julian/Documents/phd/solar_flares'

from sdsim import glitches
from read_flare_db import return_all_flares, keep_regions_min_num
from flare_classes import Flare, ActiveRegion

# mirrors data inside bounding_box into the nine adjacent boxes, then trims to just perc_extra on each length
def data_nineway_mirror(data, bounding_box, perc_extra=0.2):
    xmin, xmax = bounding_box[0]
    ymin, ymax = bounding_box[1]
    # Mirror points
    data_center = data
    data_left = np.copy(data_center)
    data_left[:, 0] = xmin - (data_left[:, 0] - xmin)
    data_right = np.copy(data_center)
    data_right[:, 0] = xmax + (xmax - data_right[:, 0])
    data_down = np.copy(data_center)
    data_down[:, 1] = ymin - (data_down[:, 1] - ymin)
    data_up = np.copy(data_center)
    data_up[:, 1] = ymax + (ymax - data_up[:, 1])
    data_upleft = np.copy(data_left)
    data_upleft[:, 1] = ymax + (ymax - data_upleft[:, 1])
    data_upright = np.copy(data_right)
    data_upright[:, 1] = ymax + (ymax - data_upright[:, 1])
    data_downleft = np.copy(data_left)
    data_downleft[:, 1] = ymin - (data_downleft[:, 1] - ymin)
    data_downright = np.copy(data_right)
    data_downright[:, 1] = ymin - (data_downright[:, 1] - ymin)
    points = np.vstack([data_center, data_left, data_right, data_up, data_down, data_upleft, data_upright, data_downleft, data_downright])

    # Trim mirrored frame to withtin a 'perc' pad
    xr, yr = np.ptp(data.T[0]) * perc_extra, np.ptp(data.T[1]) * perc_extra
    xmin, xmax = bounding_box[0][0] - xr, bounding_box[0][1] + xr
    ymin, ymax = bounding_box[1][0] - yr, bounding_box[1][1] + yr
    msk = (points[:, 0] > xmin) & (points[:, 0] < xmax) &\
        (points[:, 1] > ymin) & (points[:, 1] < ymax)
    points = points[msk]

    return points.T    

def reflect_in(x_grid, y_grid, bounding_box, pdf):
    xmin, xmax = bounding_box[0]
    ymin, ymax = bounding_box[1]
    
    x_range = np.unique(x_grid)
    y_range = np.unique(y_grid)
    ylen = len(y_range)
    xlen = len(x_range)
    up_index = np.argmin(abs(y_range - ymin))
    down_index = np.argmin(abs(y_range - ymax))
    left_index = np.argmin(abs(x_range - xmin))
    right_index = np.argmin(abs(x_range - xmax))

    
    pdf[up_index:2*up_index, :] += pdf[:up_index, :]
    pdf[-(ylen-2*down_index):-(ylen-down_index), :] += pdf[down_index:, :]
    pdf[:, left_index:2*left_index] += pdf[:, :left_index]
    pdf[:, -(xlen-2*right_index):-(xlen-right_index)] += pdf[:, right_index:]

    pdf = pdf[up_index:down_index, left_index:right_index]
    
    return pdf


def kde_2d_edgecorr(ks, rho, perc_extra=0.05, num_grid=200):
    bounding_box = [(0, 1), (-1, 1)]
    extras = data_nineway_mirror(np.vstack([ks, rho]).T, bounding_box, perc_extra=perc_extra)

    kernel = gaussian_kde(np.vstack([extras[0], extras[1]]))

    x_range = np.linspace(*bounding_box[0], num=num_grid)
    y_range = np.linspace(*bounding_box[1], num=num_grid)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

    pdf = np.reshape(kernel(positions).T, x_grid.shape)
    norm = np.trapz(np.trapz(pdf, x=x_range, axis=1), x=y_range, axis=0)
    pdf = pdf / norm
    return pdf, x_range, y_range
    
def kde_2d_levels(pdf, x_range, y_range, levels, resolution):
    heights = np.linspace(0, np.max(pdf), num=resolution)
    dx = np.diff(x_range)[0]
    dy = np.diff(y_range)[0]

    integrals = np.empty(resolution)
    for i, h in enumerate(heights):
        integrals[i] = np.trapz(np.trapz(np.where(pdf > h, pdf, 0), x=x_range, axis=1), x=y_range, axis=0)
    
    pdf_vals = []
    pdf_flat = pdf.flatten()
    for l in levels[::-1]:
        height = heights[np.argmin(abs(integrals - l))]
        pdf_val = pdf_flat[np.argmin(np.abs(pdf_flat - height))]
        pdf_vals.append(pdf_val)

    return pdf_vals


def goes_pn(flares):
    _, num_per_region = np.unique(flares['region_num'], return_counts=True)
    return num_per_region

def goes_rho_ks(min_per=5, which_wt='start'):
    flares = return_all_flares()
    flares = keep_regions_min_num(flares, min_per_region=min_per)
    regions = np.unique(flares['region_num'])

    forwards = []
    backwards = []
    ks = []
    for region in regions:
        region_flares = [Flare(*f) for f in flares[flares['region_num']==region]]
        active = ActiveRegion(region, region_flares)
        f = active.fb_correlation(which='forward', which_wt=which_wt)[0]
        b = active.fb_correlation(which='backward', which_wt=which_wt)[0]
        if np.isnan(f) or np.isnan(b):
            continue
        forwards.append(f)
        backwards.append(b)
        delts = active.waiting_times(which=which_wt)
        sizes = active.sizes(which='energy_proxy')
        delts_norm = delts / np.mean(delts)
        sizes_norm = sizes / np.mean(sizes)
        ks.append(ks_2samp(delts_norm, sizes_norm)[1])

    return forwards, backwards, ks

def plot_just_goes(min_per, which_wt='start'):
    forwards, backwards, ks = goes_rho_ks(min_per, which_wt=which_wt)
    forward_ks_pdf, x_range, y_range = kde_2d_edgecorr(ks, forwards, num_grid=300)
    backward_ks_pdf, x_range, y_range = kde_2d_edgecorr(ks, backwards, num_grid=300)

    fig, axarr = plt.subplots(nrows=2, figsize=(3.3, 4))

    levels = np.array([0.1, 0.5, 0.9])
    f_levels = kde_2d_levels(forward_ks_pdf, x_range, y_range, levels, 200)
    b_levels = kde_2d_levels(backward_ks_pdf, x_range, y_range, levels, 200)
    axarr[0].scatter(ks, forwards, s=3, c='black', marker='x', lw=0.5)
    axarr[0].contour(forward_ks_pdf, levels=f_levels, extent=[0, 1, -1, 1], colors='red') 
    axarr[0].set_ylabel(r'$\rho_{+,\,k}$')
    axarr[1].scatter(ks, backwards, s=3, c='black', marker='x', lw=0.5)
    axarr[1].contour(backward_ks_pdf, levels=b_levels, extent=[0, 1, -1, 1], colors='red') 
    axarr[1].set_ylabel(r'$\rho_{-,\,k}$')

    for ax in axarr:
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(r'$\mathcal{M}_k$')

    plt.savefig(os.path.join(path_to_working, 'paper_tex', f'goes_rhoks_corr_{min_per}min_{which_wt}.pdf'), dpi=450)
    plt.show()

def plot_just_sdp(min_per):
    nregions = 10000
    alphas = [0.01, 0.1, 1, 10]
    # alphas = [0.1]
    pn = goes_pn(keep_regions_min_num(return_all_flares(), min_per_region=min_per))
    # alpha = 0.01
    fig, axarr = plt.subplots(nrows=2, figsize=(3.3, 4))
    # colors = ['red', 'blue', 'green']
    legend_lines = []
    for alpha, c in zip(alphas, seq_4colors):
        forwards = []
        backwards = []
        ks = []
        for i in range(nregions):
            n = np.random.choice(pn)
            _, delts, sizes = glitches(n + 100, alpha, beta=1e-2, jumps='pow', start_x=0.5)
            delts = delts[100:]
            sizes = sizes[100:]
            forwards.append(spearmanr(delts[1:], sizes[:-1])[0])
            backwards.append(spearmanr(delts[:-1], sizes[1:])[0])
            ks.append(ks_2samp(delts[1:] / np.mean(delts[1:]), sizes / np.mean(sizes))[1])

        forward_ks_pdf, x_range, y_range = kde_2d_edgecorr(ks, forwards, num_grid=300)
        backward_ks_pdf, x_range, y_range = kde_2d_edgecorr(ks, backwards, num_grid=300)
        levels = np.array([0.1, 0.5, 0.9])
        f_levels = kde_2d_levels(forward_ks_pdf, x_range, y_range, levels, 500)
        b_levels = kde_2d_levels(backward_ks_pdf, x_range, y_range, levels, 500)
        axarr[0].contour(forward_ks_pdf, levels=f_levels, extent=[0, 1, -1, 1], colors=c) 
        axarr[1].contour(backward_ks_pdf, levels=b_levels, extent=[0, 1, -1, 1], colors=c) 
        legend_lines.append(Line2D([0], [0], color=c, lw=1))
        # axarr[0].scatter(ks, forwards, s=2, c='black', marker='o', alpha=0.3)
        # axarr[1].scatter(ks, backwards, s=2, c='black', marker='o', alpha=0.3)

    axarr[0].set_ylabel(r'$\rho_+$')
    axarr[1].set_ylabel(r'$\rho_-$')
    alpha_str = [r'$\alpha=%g$'%alpha for alpha in alphas]
    axarr[0].legend(legend_lines, alpha_str, loc='lower right')
    # axarr[1].legend(legend_lines, alpha_str, loc='lower left')
    for ax in axarr:
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(r'$\mathcal{M}$')
        # ax.set_xlabel(r'$M$')
    plt.savefig(os.path.join(path_to_working, 'paper_tex', f'sdp_rhoks_corr_{min_per}min.pdf'), dpi=450)
    plt.show()

def plot_test_2dkde():
    def measure(n):
        m1 = np.random.normal(size=n)
        m2 = np.random.normal(scale=0.2, size=n)
        return m1+m2, m1-m2
    # fake_data = measure(2000)
    def simple_measure(n):
        m1 = np.random.normal(size=n)
        m2 = np.random.normal(size=n)
        return m1, m2
    fake_data = simple_measure(2000)

    # bounding_box = [(-2, 2), (-1, 1.5)]
    # def trim(fake_data, bounding_box):
    #     xmin, xmax = bounding_box[0]
    #     ymin, ymax = bounding_box[1]
    #     mask = (fake_data[0] > xmin) & (fake_data[0] < xmax) &\
    #             (fake_data[1] > ymin) & (fake_data[1] < ymax)
    #     trimmed_x = fake_data[0][mask]
    #     trimmed_y = fake_data[1][mask]
    #     return (trimmed_x, trimmed_y)

    # trimmed = trim(fake_data, bounding_box)
    # extras = data_nineway_mirror(np.vstack([trimmed[0], trimmed[1]]).T, bounding_box, perc_extra=0.05)

    # biased_kernel = gaussian_kde(np.vstack([trimmed[0], trimmed[1]])) 
    # kernel = gaussian_kde(np.vstack([extras[0], extras[1]]))
    untrimmed = gaussian_kde(np.vstack([fake_data[0], fake_data[1]]))

    # x_grid, y_grid = np.meshgrid(np.linspace(*bounding_box[0], num=100), np.linspace(*bounding_box[1], num=100))
    x_range = np.linspace(-3, 3, num=200)
    y_range = np.linspace(-3, 3, num=200)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # biased_pdf = np.reshape(biased_kernel(positions).T, x_grid.shape)
    # better_pdf = np.reshape(kernel(positions).T, x_grid.shape)
    untrimmed_pdf = np.reshape(untrimmed(positions).T, x_grid.shape)

    levels = kde_2d_levels(untrimmed_pdf, x_range, y_range, [0.1, 0.5, 0.9], 500)

    fake_data = np.array(fake_data)
    distances = np.sqrt(fake_data[0]**2 + fake_data[1]**2)

    fake_data[0] = fake_data[0][np.argsort(distances)]
    fake_data[1] = fake_data[1][np.argsort(distances)]
    in_int = int(len(fake_data[0])*0.1)
    mid_int = int(len(fake_data[0])*0.5)
    out_int = int(len(fake_data[0])*0.9)
    inner = (fake_data[0][:in_int], fake_data[1][:in_int])
    middle = (fake_data[0][in_int:mid_int], fake_data[1][in_int:mid_int])
    outer = (fake_data[0][mid_int:out_int], fake_data[1][mid_int:out_int])


    # extra_perc = 0.2
    # extra_x = (bounding_box[0][1] - bounding_box[0][0]) * extra_perc
    # big_x = np.linspace(bounding_box[0][0] - extra_x, bounding_box[0][1] + extra_x, num=1000)
    # print(big_x)
    # extra_y = (bounding_box[1][1] - bounding_box[1][0]) * extra_perc
    # big_y = np.linspace(bounding_box[1][0] - extra_y, bounding_box[1][1] + extra_y, num=1000)
    # big_x_grid, big_y_grid = np.meshgrid(big_x, big_y)
    # big_positions = np.vstack([big_x_grid.ravel(), big_y_grid.ravel()])
    # big_kernel = gaussian_kde(np.vstack([trimmed[0], trimmed[1]]))
    # big_pdf = np.reshape(big_kernel(big_positions).T, big_x_grid.shape)
    # big_pdf_reflect = reflect_in(big_x_grid, big_y_grid, bounding_box, big_pdf)

    # plt.scatter(extras[0], extras[1], s=2, alpha=0.3, color='red')
    # plt.scatter(trimmed[0], trimmed[1], s=2, alpha=0.3, color='black')

    plt.scatter(inner[0], inner[1], s=2, alpha=0.3, color='red')
    plt.scatter(middle[0], middle[1], s=2, alpha=0.3, color='orange')
    plt.scatter(outer[0], outer[1], s=2, alpha=0.3, color='yellow')

    # plt.contour(biased_pdf, levels=np.max(biased_pdf) * np.array([0.1, 0.5, 0.9]), extent=[*bounding_box[0], *bounding_box[1]], colors='black')
    # plt.contour(better_pdf, levels=np.max(better_pdf) * np.array([0.1, 0.5, 0.9]), extent=[*bounding_box[0], *bounding_box[1]], colors='red')
    plt.contour(untrimmed_pdf, levels=levels, extent=[-3, 3, -3, 3], colors='blue')
    # plt.contour(big_pdf, levels=np.max(big_pdf) * np.array([0.1, 0.5, 0.9]), extent=[min(big_x), max(big_x), min(big_y), max(big_y)], colors='green')

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()



if __name__=='__main__':
    plot_just_goes(min_per=10, which_wt='peak')
    # plot_just_sdp(min_per=10)
    # plot_test_2dkde()
