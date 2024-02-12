import numpy as np
import os
import arviz as az
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel
from stan_plot import plot_pair
cwd = '/Users/julian/Documents/phd/solar_flares/trials'
figdir = '/Users/julian/Documents/phd/solar_flares/figs'

def px_sample(b, c, shape=1):
    u = np.random.rand(shape)
    c1 = c + 1
    return b - (b**c1 * (1 - u))**(1 / c1)

def plot_2d():
    stanfile = os.path.join(cwd, 'powlaw.stan')

    model = CmdStanModel(stan_file=stanfile)

    np.random.seed(1)
    true_b = 2
    true_c = 1.5
    n = 200
    samples = px_sample(true_b, true_c, shape=n)

    data_dic = {"N": n,
        "y": samples}

    fit = model.sample(data=data_dic, seed=1)

    summary = fit.summary()
    print(summary)

    draws = fit.draws()
    print(np.shape(draws))
    vars = fit.stan_variables()
    for (k,v) in vars.items():
        print(k, v.shape)

    ax = az.plot_pair(vars, figsize=(6, 3.5), kind='kde', marginals=True,
            kde_kwargs={'contour_kwargs':{'alpha':0}, 'hdi_probs':[0.1, 0.3, 0.5, 0.7, 0.9], 'fill_last':True})

    axb = ax[0][0]
    axc = ax[1][1]
    ax2d = ax[1][0]
    axb.axvline(summary['5%']['b'], ls='dashed', color='grey', alpha=0.7)
    axb.axvline(summary['95%']['b'], ls='dashed', color='grey', alpha=0.7)
    axb.axvline(true_b, ls='dashed', color='red', alpha=0.7)
    axc.axhline(summary['5%']['c'], ls='dashed', color='grey', alpha=0.7)
    axc.axhline(summary['95%']['c'], ls='dashed', color='grey', alpha=0.7)
    axc.axhline(true_c, ls='dashed', color='red', alpha=0.7)
    axb.set_title(r'$b=%0.2f^{+%0.2f}_{-%0.2f}$'%(
        summary['50%']['b'],
        summary['95%']['b'] - summary['50%']['b'],
        summary['50%']['b'] - summary['5%']['b']),
        fontsize=11)
    axc.set_ylabel(r'$c=%0.2f^{+%0.2f}_{-%0.2f}$'%(
        summary['50%']['c'],
        summary['95%']['c'] - summary['50%']['c'],
        summary['50%']['c'] - summary['5%']['c']),
        fontsize=11, rotation=0, labelpad=35)
    axc.yaxis.set_label_position('right')

    ax2d.scatter(true_b, true_c, marker='x', color='red')
    ax2d.set_xlabel(r'$b$', size=11)
    ax2d.set_ylabel(r'$c$', size=11)
    ax2d.tick_params(labelsize=11)

    plt.savefig(os.path.join(figdir, 'pystan_2dtrial.png'), dpi=300)

    plt.show()

def corner_powlaw():
    stanfile = os.path.join(cwd, 'powlaw.stan')

    model = CmdStanModel(stan_file=stanfile)

    np.random.seed(1)
    true_b = 2
    true_c = 1.5
    n = 200
    samples = px_sample(true_b, true_c, shape=n)

    data_dic = {"N": n,
        "y": samples}

    fit = model.sample(data=data_dic, seed=1)
    
    var_names = ['b', 'c']
    latex_labels = [r'$b$', r'$c$']
    truth_dic = {'true_b': true_b, 'true_c': true_c}
    fig, ax = plot_pair(fit, var_names, latex_labels, truth_dic, figsize=(5, 3))
    plt.show()

if __name__ == '__main__':
    # plot_2d()
    corner_powlaw()
