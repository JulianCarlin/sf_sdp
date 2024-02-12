import numpy as np
import os, string, math, csv, sys
from time import time
import warnings
import pickle

from numpy.random import rand
from scipy.integrate import quad, quadrature
from scipy.optimize import fsolve, fmin
import scipy.optimize as opt
from scipy.special import erf, erfinv
from scipy.stats import pearsonr, spearmanr, truncnorm
from scipy.stats import ks_2samp, kstest

import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams
from matplotlib import gridspec
import matplotlib.patches as mpatches

path_to_working = '/Users/julian/Documents/phd/sdp_alpha'

# extra general functions from funcs.py
from funcs import gss, time_to_timestring, progress_bar, figsize

# power-law jump size distribution, power=-1.5
def g_pow(xi, y, beta):
    if xi >= beta * y and xi <= y:
        return xi**(-1.5)
    else:
        return 0

# power-law jump size distribution, power=1.5
def g_powinc(xi, y, beta):
    if xi >= beta * y and xi <= y:
        return xi**(1.5)
    else:
        return 0

# broad gaussian shaped jump size distribution, mean is always centred in allowed domain, standard deviation set relative to the mean
def g_gauss(xi, y, beta):
    if xi >= beta * y and xi <= y:
        mean = y * (1 + beta) / 2.
        sig = mean / 1.5
        return np.exp(-(xi - mean)**2. / sig**2.) 
    else:
        return 0

# uniform jump size distribution
def g_uniform(xi, y, beta):
    if xi >= beta * y and xi <= y:
        return 1
    else:
        return 0

# 'weird' arbitrary jump size distribtuion used for testing
def g_weird(xi, y, beta):
    if xi >= beta * y and xi <= y:
        return np.exp(-np.pi * xi**(1.1)) + (xi / y)**2.
    else:
        return 0

def g_gauss_fixed(xi, y, beta):
    if xi >= beta * y and xi <= y:
        mean = 0.5
        sig = mean / 1.5
        return np.exp(-(xi - mean)**2. / sig**2.)
    else:
        return 0
    
def g_sep(xi, y, beta, d=2.):
    if xi <= y:
        return (y - xi)**d
    else:
        return 0

def g_lognorm(xi, y, beta):
    if xi >= beta * y and xi <= y:
        mean = -2
        sig = 0.5
        return np.exp(-(np.log(xi) - np.log(y) - mean)**2. / (2*sig**2.))
    else:
        return 0

def g_lognorm_fixed(xi, y, beta):
    if xi >= beta * y and xi <= y:
        mean = -2
        sig = 0.5
        return np.exp(-(np.log(xi) - mean)**2. / (2*sig**2.))
    else:
        return 0

# generates n glitches, using the jump size distribution defined by jumps, with parameters alpha and beta passed through. start_x defines the starting lag of the system. pbar turns on/off a terminal-printed progress bar
def glitches(n, alpha, beta, jumps='pow', start_x=0, pbar=False):
    x = start_x
    n=int(n)
    lag_list = np.empty(n)
    time_list = np.empty(n)
    size_list = np.empty(n)
    
    start_time = time()
    cum_time = 0
    
    g_input = eval('g_' + jumps) # sets g_input function to be one specified by 'jumps' input. Note: yes eval() calls are bad in general, but here we are literally just searching the local for a certain function name.
    
    if 'gauss' in jumps:
        lag_update = lag_update_gauss
    elif 'pow' in jumps:
        lag_update = lag_update_pow
    elif 'uniform' in jumps:
        lag_update = lag_update_uniform
    else:
        lag_update = lag_update_gen

    for i in range(n):
        if pbar:
            current_time = time()
            progress_bar(i, n, cum_time)
            
        new_x, t, size = lag_update(x, alpha, beta, g_input, jumps, d=-1.5)
        # new_x, t, size = lag_update(x, alpha, beta, g_input, jumps, d=-3)
        x = new_x
        lag_list[i] = x
        time_list[i] = t
        size_list[i] = size
       
        if pbar:
            cum_time += time() - current_time

    if pbar:
        print('\nTotal time to generate %d glitches: %s' %(n, time_to_timestring(time() - start_time)))
    return lag_list, time_list, size_list

# updates the lag x by finding a waiting time (involves alpha, x), then a jump size (involves, jump distribution 'jumps', beta, and x)
# uses analytic inverse cumulative method to find waiting time (as we don't change functional form of lambda), rejection-acceptance method for the jump size, to allow for plug-and-play testing of different functional forms
def lag_update_gen(x, alpha, beta, g_input, jumps, d=0):    
    time_taken = time_gen_std(x, alpha) # finds random waiting time
    lag_prior = x + time_taken # calculates lag just prior to glitch   
    glitch_size = rej_size(g_input, lag_prior, beta) # finds random glitch size
    new_lag = lag_prior - glitch_size # calculates lag just after glitch

    return new_lag, time_taken, glitch_size

def lag_update_gauss(x, alpha, beta, g_input, jumps, d=0):
    time_taken = time_gen_std(x, alpha) # finds random waiting time
    lag_prior = x + time_taken # calculates lag just prior to glitch
    if 'fixed' in jumps:
        mean = 0.5
    else:
        mean = 0.5 * lag_prior * (1 + beta)
    sig = 0.75 * mean / np.sqrt(2.)
    glitch_size = truncnorm_size(lag_prior, beta, mean, sig) # finds random glitch size from truncated gaussian
    new_lag = lag_prior - glitch_size # calculates lag just after glitch

    return new_lag, time_taken, glitch_size

def lag_update_pow(x, alpha, beta, g_input, jumps, d=0):
    time_taken = time_gen_std(x, alpha) # finds random waiting time
    lag_prior = x + time_taken # calculates lag just prior to glitch   
    glitch_size = pow_invcum_size(lag_prior, beta, d=d) # finds random glitch size
    new_lag = lag_prior - glitch_size # calculates lag just after glitch'
    return new_lag, time_taken, glitch_size

def lag_update_uniform(x, alpha, beta, g_input, jumps, d=0):
    time_taken = time_gen_std(x, alpha) # finds random waiting time
    lag_prior = x + time_taken # calculates lag just prior to glitch   
    glitch_size = np.random.uniform(low=beta*lag_prior, high=lag_prior)
    new_lag = lag_prior - glitch_size # calculates lag just after glitch'
    return new_lag, time_taken, glitch_size

# find random waiting time from analytic form of inverse cumulative distribution, given simple alpha/(1-x) shape for lambda
def time_gen_std(x, alpha):
    u = rand()
    return (1. - x) * (1. - (1. - u)**(1. / alpha))

# find random jump size using acceptance-rejection algorithm, given g jump size distribution, 'y' lag prior to jump
def rej_size(g, y, beta):
    # first normalise jump distribution g in appropriate domain
    norm = quad(lambda xi: g(xi, y, beta), beta*y, y)[0]
    if norm==0:
        eta = lambda x, y, beta: g(x, y, beta)
    else:
        eta = lambda x, y, beta: g(x, y, beta) / norm
    
    # find maximum of eta using the gss method (requires eta to be bounded above and unimodal)
    c = eta(gss(lambda xi: -eta(xi, y, beta), y * beta, y), y, beta)

    # rejection-acceptance method finds two points uniformly placed in rectangle defined by domain/range of eta, then accepts the 'x' value if it lies below the curve defined by eta, otherwise it generates two new points
    v1 = y * (beta + (1. - beta) * rand())
    v2 = c * rand()
    while v2 > eta(v1, y, beta):
        v1 = y * (beta + (1. - beta) * rand())
        v2 = c * rand()
    
    return v1

def truncnorm_size(y, beta, mean, sig):
    a = (beta * y - mean) / sig
    b = (y - mean) / sig
    return truncnorm.rvs(a, b, loc=mean, scale=sig)

def pow_invcum_size(y, beta, d):
    by = y * beta
    d1 = d + 1.
    return (rand() * (y**d1 - by**d1) + by**d1)**(1/d1)

def gen_set(n=1e4, jumps='pow'):
    alphas = np.logspace(-2, 2, num=100)
    beta = 0.
    if jumps=='pow':
        beta=1e-2
    folder = os.path.join(path_to_working,'picklejar/eta_%s'%(jumps))
    
    if not(os.path.isdir(folder)):
        os.makedirs(folder)

    for i, alpha in enumerate(alphas):
        list_file = folder + '/lists_a%0.2f_%1.e.pickle' %(alpha, n)
        lag_list, time_list, size_list = glitches(n, alpha, beta, pbar=False, jumps=jumps)
        list_dic = {'lag_list':lag_list, 'time_list':time_list, 'size_list':size_list}
        with open(list_file, 'wb') as outfile:
            pickle.dump(list_dic, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stdout.write('\ralpha = %0.2f done, %d/%d completed' %(alpha, i+1, len(alphas)))
        sys.stdout.flush()

def gen_pdfs(alpha, n=1e4, jumps='pow'):
    beta = 0.
    if jumps=='pow':
        beta=1e-2
    folder = os.path.join(path_to_working,'picklejar/eta_%s'%(jumps))
    
    if not(os.path.isdir(folder)):
        os.makedirs(folder)
    list_file = folder + '/lists_a%0.2f_%1.e.pickle' %(alpha, n)
    lag_list, time_list, size_list = glitches(n, alpha, beta, pbar=False, jumps=jumps)
    list_dic = {'lag_list':lag_list, 'time_list':time_list, 'size_list':size_list}
    with open(list_file, 'wb') as outfile:
        pickle.dump(list_dic, outfile, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    gen_set(n=1e4, jumps='lognorm_fixed')
