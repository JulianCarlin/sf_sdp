import numpy as np
from scipy.stats import cauchy, expon
import os
import arviz as az
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel, CmdStanMCMC, from_csv
from stan_plot import plot_pair, plot_ppc
cwd = '/Users/julian/Documents/phd/solar_flares/trials'
figdir = '/Users/julian/Documents/phd/solar_flares/figs'

import logging
logging.getLogger("cmdstanpy").setLevel(logging.DEBUG)

def psample(index, scale, shape=1):
    u = np.random.rand(shape)
    c1 = index + 1
    return scale * (1 - u**(1 / c1))

def generate_truth_dic():
    truth_dic = {'true_tau_mu': 3, 
            'true_tau_sig': 0.1,
            'true_beta_mu': 3,
            'true_beta_sig': 0.1,
            }
    return truth_dic

def generate_data(truth_dic, num_ns=50, ns_each=100, seed=1):
    if seed=='none':
        seed = np.random.randint(100000)
    np.random.seed(seed)
    tauk_used = np.random.normal(truth_dic['true_tau_mu'], truth_dic['true_tau_sig'], size=num_ns)
    betak_used = np.random.normal(truth_dic['true_beta_mu'], truth_dic['true_beta_sig'], size=num_ns)
    assert all(betak_used > 0)
    while not all(tauk_used > 0):
        tauk_used = np.where(tauk_used > 0, tauk_used, np.random.normal(truth_dic['true_tau_mu'], truth_dic['true_tau_sig'])) 

    delt_samples = []
    lower_bounds_tau = []
    for tau, beta in zip(tauk_used, betak_used):
        c_used = beta
        delt_sample = psample(c_used, tau, shape=ns_each)
        delt_samples.append(delt_sample)
        lower_bounds_tau.append(np.max(delt_sample))
    delt_flat_samples = [s for each_list in delt_samples for s in each_list]

    data_dic = {"NT": len(delt_flat_samples),
            "K": num_ns,
            "delt": delt_flat_samples,
            "st": ns_each * np.ones(num_ns).astype(int),
            "lower_bounds_tau": lower_bounds_tau,
            }

    return data_dic

def corner_combined(truth_dic, fit):
    var_names = ['tau_mu', 'tau_sig', 'beta_mu', 'beta_sig']
    latex_labels = [r'$\tau_\mu$', r'$\tau_\sigma$', r'$\beta_\mu$', r'$\beta_\sigma$']
    
    priors = {
              'tau_mu': {'type': 'gaussian', 'params': [3, 1], 'lower_constraint': 0},
              'tau_sig': {'type': 'exponential', 'params': [0, 1], 'lower_constraint': 0},
              'beta_mu': {'type': 'gaussian', 'params': [3, 1], 'lower_constraint': 0},
              'beta_sig': {'type': 'exponential', 'params': [0, 1], 'lower_constraint': 0},
              }
    
    fig, ax = plot_pair(fit, var_names, latex_labels, truth_dic=truth_dic, priors=priors, num_prior=int(1e4), figsize=(9, 6))
    plt.savefig(os.path.join(figdir, 'simplest_hier_synthetic.png'), dpi=600)
    plt.show()

def plot_posterior_pred(truth_dic, fit):

    vars = fit.stan_variables()

    print(vars.keys())

    priors_constraints = [[(3, 1), 1, 0], [(3, 1), 1, 0]] 
    var_names = [k for k in vars.keys() if 'est' not in k and 'shifted' not in k]
    samples = fit.draws_xr(vars=var_names)
    # print(truth_dic)
    # print(fit.summary(sig_figs=2))
    # print(fit.diagnose())
    # latex_labels = [r'$\tau$', r'$\xi$', r'$\lambda_0$', r'$\delta$'] 
    latex_labels = [r'$\tau$', r'$\beta$'] 

    # fig, axarr = plt.subplots(ncols=4, figsize=(6, 1.5))
    fig, axarr = plt.subplots(ncols=2, figsize=(6, 1.5))

    n_per_posterior = 1000
    n_per_prior = 1000
    n_prior = 10000
    n_true = int(1e6)

    for (i, ax), prior_constraint, label in zip(enumerate(axarr), priors_constraints, latex_labels):
        var_mean_str = var_names[2*i]
        var_sig_str = var_names[2*i+1]
        true_mean_key = 'true_' + var_mean_str.split('_')[0] + '_mu'
        true_sig_key = 'true_' + var_sig_str.split('_')[0] + '_sig'

        true_mean = truth_dic[true_mean_key]
        true_sig = truth_dic[true_sig_key]

        mu_prior = prior_constraint[0]
        sig_prior = prior_constraint[1]
        constraint = prior_constraint[2]
        r_plot = (constraint, mu_prior[0] + 3 * mu_prior[1])

        posterior_means = np.ravel(samples[var_mean_str].to_numpy())
        posterior_sigs = np.ravel(samples[var_sig_str].to_numpy())

        posterior_samples = np.ravel([np.random.normal(p_mean, abs(p_sig), size=n_per_posterior) for (p_mean, p_sig) in zip(posterior_means, posterior_sigs)])
        print(posterior_samples)
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

    plt.savefig(os.path.join(figdir, 'ppc_simplest_hier.pdf'), dpi=450)

    plt.show()

def get_samples(truth_dic, seed=1, num_ns=10, ns_each=100, new_samples=True, save=False):
    if seed=='none':
        seed = np.random.randint(100000)
    data_dic = generate_data(truth_dic, num_ns=num_ns, ns_each=ns_each, seed=seed)

    if new_samples:
        stanfile = os.path.join(cwd, 'simplest_hier.stan')
        model = CmdStanModel(stan_file=stanfile)
        fit = model.sample(data=data_dic, iter_sampling=1000, seed=seed, adapt_delta=0.95, show_console=False)

        if save:
            fit.save_csvfiles(os.path.join(cwd, 'simplest_hier_draws'))

    else:
        fit = from_csv(os.path.join(cwd, 'simplest_hier_draws'))

    return data_dic, fit

def sample_testing(nsamp=10000):
    beta = 5
    tau = 1
    delt_range = np.linspace(0, tau, num=10000)
    def analytic_pdf(beta, tau):
        return (beta + 1) / tau * (1 - delt_range / tau)**(beta)

    fig, ax = plt.subplots(figsize=(4,3))

    ax.plot(delt_range, analytic_pdf(beta, tau))

    samples = psample(beta, tau, shape=nsamp)

    ax.hist(samples, density=True)

    plt.show()


if __name__ == '__main__':
    truth_dic = generate_truth_dic()
    data_dic, fit = get_samples(truth_dic, seed=1, num_ns=50, ns_each=100, new_samples=False, save=True)
    plot_posterior_pred(truth_dic, fit)
    # corner_combined(truth_dic, fit)
    # sample_testing()
