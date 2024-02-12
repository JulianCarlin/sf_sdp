import numpy as np
from scipy.stats import cauchy, expon
import os
import arviz as az
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel, CmdStanMCMC, from_csv
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
    truth_dic = {'true_lam_mu': 3, 
            'true_lam_sig': 0.3,
            'true_delta_mu': 2,
            'true_delta_sig': 0.2,
            'true_xcrs_mu': 2.5,
            'true_xcrs_sig': 0.1
            }
    return truth_dic

def generate_data(truth_dic, num_ns=10, ns_mean=100, ns_sig=30, seed=1):
    if seed=='none':
        seed = np.random.randint(100000)
    np.random.seed(seed)
    ns = np.random.normal(loc=ns_mean, scale=ns_sig, size=num_ns).astype(int)
    assert all(ns > 1)
    lams_used = np.random.normal(truth_dic['true_lam_mu'], truth_dic['true_lam_sig'], size=num_ns)
    deltas_used = np.random.normal(truth_dic['true_delta_mu'], truth_dic['true_delta_sig'], size=num_ns)
    xcrss_used = np.random.normal(truth_dic['true_xcrs_mu'], truth_dic['true_xcrs_sig'], size=num_ns)
    lengths_used = np.random.normal(5, 1, size=num_ns)
    assert all(deltas_used > 1)
    assert all(lams_used > 0)
    assert all(xcrss_used > 0)
    assert all(lengths_used > 0)
    samples = []
    lower_bounds = []
    for n, lam_used, delta_used, xcrs_used, length_used in zip(ns, lams_used, deltas_used, xcrss_used, lengths_used):
        b_used = length_used * xcrs_used
        c_used = lam_used * length_used * xcrs_used + delta_used
        sample = px_sample(b_used, c_used, shape=n)
        samples.append(sample)
        lower_bounds.append(np.max(sample) / length_used)
    flat_samples = [s for each_list in samples for s in each_list]

    data_dic = {"N": np.sum(ns),
            "K": num_ns,
            "delts": flat_samples,
            "ndelts": ns,
            "lengths": lengths_used,
            "lower_bounds": lower_bounds
            }

    return data_dic

def corner_hpdelt(data_dic, truth_dic, seed=1, new_samples=True):
    if new_samples:
        stanfile = os.path.join(cwd, 'fake_pdelt.stan')
        model = CmdStanModel(stan_file=stanfile)
        if seed=='none':
            seed = np.random.randint(100000)
        data_dic = generate_data(truth_dic, num_ns=10, ns_mean=100, ns_sig=30, seed=seed)
        fit = model.sample(data=data_dic, iter_sampling=3000, seed=seed, adapt_delta=0.99)

    else:
        fit = from_csv(os.path.join(cwd, 'draws'))

    var_names = ['lam_mu', 'lam_sig', 'delta_mu', 'delta_sig', 'xcrs_mu', 'xcrs_sig']
    latex_labels = [ r'$\lambda_\mu$', r'$\lambda_\sigma$', r'$\delta_\mu$', r'$\delta_\sigma$', r'$(x_{\rm cr}/S)_\mu$', r'$(x_{\rm cr}/S)_\sigma$']
    
    priors = {'lam_mu': {'type': 'gaussian', 'params': [2, 1], 'lower_constraint': 0},
              'lam_sig': {'type': 'exponential', 'params': [0, 1], 'lower_constraint': 0},
              'delta_mu': {'type': 'gaussian', 'params': [2, 1], 'lower_constraint': 1},
              'delta_sig': {'type': 'exponential', 'params': [0, 1], 'lower_constraint': 0},
              'xcrs_mu': {'type': 'gaussian', 'params': [2, 1], 'lower_constraint': 0},
              'xcrs_sig': {'type': 'exponential', 'params': [0, 1], 'lower_constraint': 0}}
    
    fig, ax = plot_pair(fit, var_names, latex_labels, truth_dic=truth_dic, priors=priors, num_prior=int(1e4), figsize=(9, 6))
    plt.savefig(os.path.join(figdir, 'fake_pdelt_centered.png'), dpi=600)
    plt.show()

def plot_posterior_pred(truth_dic, new_samples=False, seed='none', fit='none'):
    if new_samples:
        stanfile = os.path.join(cwd, 'fake_pdelt.stan')
        model = CmdStanModel(stan_file=stanfile)
        if seed=='none':
            seed = np.random.randint(100000)
        data_dic = generate_data(truth_dic, num_ns=10, ns_mean=100, ns_sig=30, seed=seed)
        fit = model.sample(data=data_dic, iter_sampling=3000, seed=seed, adapt_delta=0.99)

    elif fit=='none':
        fit = from_csv(os.path.join(cwd, 'draws'))

    vars = fit.stan_variables()

    # print(vars.keys())

    # priors_constraints = [[(2, 1), (0, 5), 0], [(2, 1), (0, 5), 1], [(2, 1), (0, 5), 0]] 
    priors_constraints = [[(2, 1), 1, 0], [(2, 1), 1, 1], [(2, 1), 1, 0]] 
    var_names = [k for k in vars.keys() if 'est' not in k and 'shifted' not in k]
    samples = fit.draws_xr(vars=var_names)
    # print(truth_dic)
    # print(fit.summary(sig_figs=2))
    # print(fit.diagnose())
    latex_labels = [r'$\lambda_0$', r'$\delta$', r'$x_{\rm cr} / S$'] 

    fig, axarr = plt.subplots(ncols=3, figsize=(6, 1.5))

    n_per_posterior = 1000
    n_per_prior = 1000
    n_prior = 10000
    n_true = int(1e6)

    for (i, ax), prior_constraint, label in zip(enumerate(axarr), priors_constraints, latex_labels):
        var_mean_str = var_names[2*i]
        var_sig_str = var_names[2*i+1]
        true_mean_key = 'true_' + var_mean_str.split('_')[0] + '_mean'
        true_sig_key = 'true_' + var_sig_str.split('_')[0] + '_sd'

        true_mean = truth_dic[true_mean_key]
        true_sig = truth_dic[true_sig_key]

        mu_prior = prior_constraint[0]
        sig_prior = prior_constraint[1]
        constraint = prior_constraint[2]
        r_plot = (constraint, mu_prior[0] + 3 * mu_prior[1])

        posterior_means = np.ravel(samples[var_mean_str].to_numpy())
        posterior_sigs = np.ravel(samples[var_sig_str].to_numpy())

        posterior_samples = np.ravel([np.random.normal(p_mean, p_sig, size=n_per_posterior) for (p_mean, p_sig) in zip(posterior_means, posterior_sigs)])
        ax.hist(posterior_samples, density=True, bins=100, histtype='step', ec='blue', fc='none', range=r_plot, label='Posterior')
        
        true_samples = np.random.normal(true_mean, true_sig, size=n_true)
        ax.hist(true_samples, density=True, bins=100, histtype='step', ec='red', fc='none', range=r_plot, zorder=0, label='True')

        mu_priors = np.random.normal(*mu_prior, size=n_prior)
        mu_priors = mu_priors[mu_priors>0]
        # sig_priors = cauchy.rvs(*sig_prior, size=2*n_prior)
        # sig_priors = sig_priors[sig_priors>0]
        sig_priors = expon.rvs(sig_prior, size=n_prior)
        prior_samples = np.ravel([np.random.normal(mean, sig, size=n_per_prior) for (mean, sig) in zip(mu_priors, sig_priors)])
        ax.hist(prior_samples, density=True, bins=100, histtype='step', ec='green', fc='none', range=r_plot, zorder=-1, label='Prior')

        ax.set_xlim(constraint, mu_prior[0] + 3 * mu_prior[1]) 
        ax.set_xlabel(label)
        ax.set_yticks([])
    axarr[0].set_ylabel('PDF')    
    axarr[0].legend(loc='upper left', prop={'size': 8})

    plt.savefig(os.path.join(figdir, 'ppc_fakepdelt_centered.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    truth_dic = generate_truth_dic()
    data_dic = generate_data(truth_dic, seed=1, num_ns=10, ns_mean=100, ns_sig=30)
    # fit = plot_2d(data_dic, truth_dic, seed='none')
    # plot_posterior_pred(truth_dic, new_samples=False, fit=fit)
    corner_hpdelt(data_dic, truth_dic, seed=1, new_samples=False)
