import numpy as np
import os
import arviz as az
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel
from stan_plot import plot_pair
cwd = '/Users/julian/Documents/phd/solar_flares/trials'
figdir = '/Users/julian/Documents/phd/solar_flares/figs'

# import logging
# logging.getLogger("cmdstanpy").setLevel(logging.DEBUG)

def px_sample(b, c, shape=1):
    u = np.random.rand(shape)
    c1 = c + 1
    return b - (b**c1 * (1 - u))**(1 / c1)

def generate_truth_dic():
    truth_dic = {'true_lam': 2, 
            'true_delta': 3,
            'true_beta': 2.5,
            }
    return truth_dic

def generate_data(truth_dic, seed=1):
    if seed=='none':
        seed = np.random.randint(100000)
    np.random.seed(seed)
    ns = 1000
    b = truth_dic['true_beta']
    c = truth_dic['true_lam'] * truth_dic['true_beta'] + truth_dic['true_delta']
    sample = px_sample(b, c, shape=ns)
    lower_bound = np.max(sample)

    data_dic = {"N": ns,
            "delt": sample,
            "lower_bound": lower_bound
            }

    return data_dic

def corner_noh(data_dic, truth_dic, seed=0):
    stanfile = os.path.join(cwd, 'pdelt_noh.stan')

    model = CmdStanModel(stan_file=stanfile)

    if seed=='none':
        seed = np.random.randint(100000)
    fit = model.sample(data=data_dic, seed=seed, iter_sampling=2000)

    latex_labels = [ r'$\lambda_0\,$', r'$\delta\,$', r'$\beta\,$']
    var_names = ['lam', 'delta', 'bet']
    priors = {'lam': {'type': 'gaussian', 'params': [2, 2], 'lower_constraint': 0},
            'delta': {'type': 'gaussian', 'params': [3, 1], 'lower_constraint': 1},
              'bet': {'type': 'gaussian', 'params': [2, 2], 'lower_constraint': 0}}
    fig, ax = plot_pair(fit, var_names, latex_labels, truth_dic=truth_dic, priors=priors,
            num_prior=int(1e5), figsize=(5, 3))
    plt.savefig(os.path.join(figdir, 'pdelt_noh_seed%d_beta%0.1f.png'%(seed, truth_dic['true_beta'])), dpi=600)
    plt.show()

def lower_lim_test(data_dic):
    stanfile = os.path.join(cwd, 'pdelt_noh.stan')

    model = CmdStanModel(stan_file=stanfile)

    fit = model.sample(data=data_dic, seed=1)

    summary = fit.summary(sig_figs=2)
    draws = fit.draws()
    vars = fit.stan_variables()
    
    print(summary)
    print(fit.diagnose())

if __name__ == '__main__':
    truth_dic = generate_truth_dic()
    print(truth_dic)
    data_dic = generate_data(truth_dic, seed=0)
    print(data_dic['lower_bound'])
    # plot_2d(data_dic, truth_dic, seed=0)
    corner_noh(data_dic, truth_dic, seed=0)
