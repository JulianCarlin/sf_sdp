import numpy as np
from scipy.stats import cauchy
import os
import arviz as az
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel
from stan_plot import plot_pair, plot_ppc
cwd = '/Users/julian/Documents/phd/solar_flares/trials'
figdir = '/Users/julian/Documents/phd/solar_flares/figs'

# import logging
# logging.getLogger("cmdstanpy").setLevel(logging.DEBUG)

def px_sample(b, c, shape=1):
    u = np.random.rand(shape)
    c1 = c + 1
    return b - (b**c1 * (1 - u))**(1 / c1)

def generate_truth_dic():
    truth_dic = {'true_b_mu': 2, 
            'true_b_sig': 0.2,
            'true_c_mu': 3,
            'true_c_sig': 0.3
            }
    return truth_dic

def generate_data(truth_dic, seed=1):
    np.random.seed(seed)
    ns = np.random.normal(loc=200, scale=30, size=20).astype(int)
    assert all(ns > 1)
    bs_used = np.random.normal(truth_dic['true_b_mu'], truth_dic['true_b_sig'], size=len(ns))
    cs_used = np.random.normal(truth_dic['true_c_mu'], truth_dic['true_c_sig'], size=len(ns))
    assert all(cs_used > 1)
    assert all(bs_used > 0)
    samples = []
    lower_bounds = []
    for n, b_used, c_used in zip(ns, bs_used, cs_used):
        sample = px_sample(b_used, c_used, shape=n)
        samples.append(sample)
        lower_bounds.append(np.max(sample))
    flat_samples = [s for each_list in samples for s in each_list]

    data_dic = {"N": np.sum(ns),
            "K": len(ns),
            "y": flat_samples,
            "s": ns,
            "lower_bounds": lower_bounds
            }

    return data_dic

def corner_hpl(data_dic, truth_dic, seed=0):
    stanfile = os.path.join(cwd, 'hierarch_powlaw.stan')

    model = CmdStanModel(stan_file=stanfile)

    if seed=='none':
        seed = np.random.randint(1000000)
    fit = model.sample(data=data_dic, seed=seed)

    var_names = ['b_mu', 'b_sig', 'c_mu', 'c_sig']
    latex_labels = [ r'$b_\mu$', r'$b_\sigma$', r'$c_\mu$', r'$c_\sigma$']
    priors = {'b_mu': {'type': 'gaussian', 'params': [2, 1], 'lower_constraint': 0},
            'b_sig': {'type': 'cauchy', 'params': [0, 1], 'lower_constraint': 0},
            'c_mu': {'type': 'gaussian', 'params': [2, 1], 'lower_constraint': 0},
            'c_sig': {'type': 'cauchy', 'params': [0, 1], 'lower_constraint': 0}}

    fig, ax = plot_pair(fit, var_names, latex_labels, truth_dic, priors=priors, num_prior=int(1e5), figsize=(6, 4))
    plt.savefig(os.path.join(figdir, 'hierarch_powlaw.png'), dpi=600)
    plt.show()

def ppc_hpl(data_dic, truth_dic, seed=0):
    stanfile = os.path.join(cwd, 'hierarch_powlaw.stan')
    model = CmdStanModel(stan_file=stanfile)
    if seed=='none':
        seed = np.random.randint(1000000)
    data_dic = generate_data(truth_dic, seed=seed)
    fit = model.sample(data=data_dic, seed=seed)

    var_names = ['b_mu', 'b_sig', 'c_mu', 'c_sig']
    latex_labels = [r'$b$', r'$c$']
    priors = {'b': {'mean': {'type': 'gaussian', 'params': [2, 1], 'lower_constraint': 0},
                    'sd': {'type': 'cauchy', 'params': [0, 1], 'lower_constraint': 0}},
              'c': {'mean': {'type': 'gaussian', 'params': [2, 1], 'lower_constraint': 0},
                    'sd': {'type': 'cauchy', 'params': [0, 1], 'lower_constraint': 0}}
             }

    fig, ax = plot_ppc(fit, var_names, latex_labels, truth_dic, priors, num_hist=int(1e5), figsize=(5, 1.5))
    plt.savefig(os.path.join(figdir, 'ppc_powlaw.png'), dpi=600)
    plt.show()

if __name__ == '__main__':
    truth_dic = generate_truth_dic()
    data_dic = generate_data(truth_dic)
    print(data_dic['s'])
    # corner_hpl(data_dic, truth_dic, seed=1)
    ppc_hpl(data_dic, truth_dic, seed=1)
