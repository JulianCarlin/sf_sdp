import numpy as np
import os
import arviz as az
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel, from_csv
from stan_plot import plot_pair
cwd = '/Users/julian/Documents/phd/solar_flares/trials'
figdir = '/Users/julian/Documents/phd/solar_flares/figs'

import logging
# logging.getLogger("cmdstanpy").setLevel(logging.DEBUG)

def px_sample(b, c, shape=1):
    u = np.random.rand(shape)
    c1 = c + 1
    return b - (b**c1 * (1 - u))**(1 / c1)

def generate_truth_dic():
    truth_dic = {'true_lam': 2, 
            'true_beta': 2.5,
            'true_delta': 4,
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

def marg_corner(data_dic, truths, newsamples=False, seed=1):
    if newsamples:
        stanfile = os.path.join(cwd, 'pdelt_marg.stan')
        model = CmdStanModel(stan_file=stanfile)
        if seed=='none':
            seed = np.random.randint(100000)
        inits = {'lam': 2, 'bet': 2}
        outdir = os.path.join(cwd, 'pdelt_marg_draws', 'seed_%d'%seed)
        fit = model.sample(data=data_dic, seed=seed, iter_sampling=2000, inits=inits, output_dir=outdir)
        print(fit.diagnose())

    else:
        outdir = os.path.join(cwd, 'pdelt_marg_draws', 'seed_%d'%seed)
        fit = from_csv(outdir)

    var_names = ['lam', 'bet']
    latex_labels = [r'$\lambda_0$', r'$\beta$']
    priors = {'lam': {'type': 'gaussian', 'params': [2, 2], 'lower_constraint': 0},
              'bet': {'type': 'gaussian', 'params': [2, 2], 'lower_constraint': 0}}
    fig, ax = plot_pair(fit, var_names, latex_labels, truth_dic=truths, priors=priors, num_prior=int(1e4), figsize=(4, 3))

    plt.savefig(os.path.join(figdir, 'pdelt_marg.png'), dpi=600)
    plt.show()


def test_ll(data_dic):
    delt = data_dic['delt']
    beta = 2
    lam = 2
    def term1(delt, beta):
        return (2/3 - np.log(1 - delt / beta))**(-2)
    def term2(delt, beta):
        return (3/2 * np.log(1 - delt / beta) - 1)**(-1)
    def term3(delt, beta, lam):
        return 3/2 * (2 + beta * lam) * np.log(1 - delt / beta) - beta * lam - 5
    # print(np.log(1 - delt / beta))
    print(term1(delt, beta))
    print(term2(delt, beta) * term3(delt, beta, lam))
    


if __name__ == '__main__':
    truth_dic = generate_truth_dic()
    print(truth_dic)
    data_dic = generate_data(truth_dic, seed=0)
    print(data_dic['lower_bound'])
    # plot_2d(data_dic, truth_dic, seed=0, newsamples=True)
    # test_ll(data_dic)
    marg_corner(data_dic, truth_dic, newsamples=True, seed=0)
