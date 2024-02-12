import numpy as np
from scipy.special import logsumexp, erfcinv
from scipy.stats import spearmanr
import warnings

class ActiveRegion:
    def __init__(self, region_num, flares, units='h', c1masking=False):
        assert all([f.region_num == region_num for f in flares])
        self.region_num = region_num
        flares.sort(key=lambda f: f.start_time)
        if c1masking:
            flares = [f for f in flares if f.peak_flux > 1e-6]
        self.flares = flares
        if len(flares) > 0:
            self.duration = self.duration(units=units)

    def waiting_times(self, which='start', units='h'):
        if which not in ['start', 'end', 'peak']:
            raise AssertionError('Need to have "which" param as one of "start", "end", or "peak".')
        if units not in ['h', 's', 'D']:
            raise NotImplementedError(f"Units have to be 's', 'h', or 'D' for Flare class")
        if which=='start':
            epochs = [flare.start_time for flare in self.flares]
        elif which=='end':
            epochs = [flare.end_time for flare in self.flares]
        elif which=='peak':
            epochs = [flare.peak_time for flare in self.flares]
        epochs = np.sort(epochs)
        delta_times = np.diff(epochs)

        try:
            waiting_times = delta_times / np.timedelta64(1, units)
        except np.core._exceptions.UFuncTypeError:
            print(delta_times)
            return np.nan
        if units == 'h':
            two_weeks = 14 * 24
            thirty_sec = 0.5 / 60
        elif units == 's':
            two_weeks = 14 * 24 * 60 * 60
            thirty_sec = 30
        elif units == 'D':
            two_weeks = 14
            thirty_sec = 0.5 / 60 / 24
        if any(waiting_times > two_weeks):
            print(f'Region {self.region_num} as at least one >14day waiting time, check!')
            # waiting_times = waiting_times[waiting_times < two_weeks]
        if any(waiting_times == 0):
            # print(f'Region {self.region_num} as at least one zero waiting time, check!')
            # waiting_times = waiting_times[waiting_times > 0]
            waiting_times = np.where(waiting_times==0, thirty_sec, waiting_times)
        return waiting_times

    def duration(self, units='h'):
        if units not in ['h', 's', 'D']:
            raise NotImplementedError(f"Units have to be 's', 'h', or 'D' for Flare class")
        try:
            dur_td = max([f.end_time for f in self.flares]) - min([f.start_time for f in self.flares])
        except:
            print(f'Something odd with the duration for Region {self.region_num}')

        return dur_td / np.timedelta64(1, units)

    def rate(self, ci=0.95, prior='default_gamma', resolution=10000):
        try:
            assert prior=='default_gamma'
        except AssertionError:
            raise NotImplementedError('Only default_gamma prior implemented at this stage, sorry')

        n = len(self.flares)
        t = self.duration
        if t == 0:
            return None, None

        # theta param in gamma prior set such that mean of prior is MLE of lambda
        theta = n / (2 * t)

        # 0.1% -- 99.9% of the gamma prior, analytically solved for k = 2
        minprior_thousandth = 0.045402 * theta
        maxprior_thousandth = 9.23341 * theta
        lam_range = np.linspace(minprior_thousandth, maxprior_thousandth, num=resolution)
        lam_bin = lam_range[1] - lam_range[0]

        lam_logpost_nonorm = (n + 1) * np.log(lam_range) - lam_range * t * (1 + 2 / n)

        lognorm = logsumexp(lam_logpost_nonorm, b=lam_bin)
        lam_logpost = lam_logpost_nonorm - lognorm

        cdf = np.cumsum(np.exp(lam_logpost) * lam_bin)

        lam_median = lam_range[np.argmin(abs(cdf - 0.5))]
        lower = (1 - ci) / 2
        upper = 1 - lower
        lam_lower = lam_range[np.argmin(abs(cdf - lower))]
        lam_upper = lam_range[np.argmin(abs(cdf - upper))]

        return lam_median, (lam_lower, lam_upper)

    def sizes(self, which):
        if which not in ['energy_proxy', 'peak_flux', 'integ_flux']:
            raise NotImplementedError('Can only return sizes using energy_proxy, peak_flux or integ_flux')
        if which=='energy_proxy':
            s = np.array([f.energy_proxy for f in self.flares])
        if which=='peak_flux':
            s = np.array([f.peak_flux for f in self.flares])
        if which=='integ_flux':
            s = np.array([f.integ_flux for f in self.flares])
        return s

    def fb_correlation(self, which, size_which='energy_proxy', ci=0.95, which_wt='start'):
        if which not in ['forward', 'backward']:
            raise AssertionError(f"Cross-correlation must have key-word 'forward' or 'backward'")
        delt = self.waiting_times(which=which_wt)

        if any([wt <= 0 for wt in delt]) or any(np.isnan(delt)):
            return np.nan, (0, 0)

        sizes = self.sizes(which=size_which)
        try:
            assert len(delt) + 1 == len(sizes)
        except AssertionError:
            return None, None
        if which == 'forward':
            rc = spearmanr(sizes[:-1], delt)[0]
        elif which == 'backward':
            rc = spearmanr(sizes[1:], delt)[0]
        if ci==0:
            return rc, None
        z = np.sqrt(2) * erfcinv(1 - ci)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            if rc == -1:
                ci_down = -1
                ci_up = -1
            elif rc == 1:
                ci_down = 1
                ci_up = 1
            else:
                ci_down = np.tanh(np.arctanh(rc) - z * (1 + rc**2/2)/np.sqrt(len(delt) - 3))
                ci_up = np.tanh(np.arctanh(rc) + z * (1 + rc**2/2)/np.sqrt(len(delt) - 3))
        return rc, (ci_down, ci_up)

    def colatitude(self):
        locs = [f.location for f in self.flares]
        lats = [l[:3] for l in locs]
        lats_conv = [int(l[1:]) if l.startswith('N') else -int(l[1:]) if l.startswith('S') else np.nan for l in lats]
        if len(lats_conv) == 0:
            return None
        try:
            lats_conv = np.array(lats_conv)
            lats_conv = lats_conv[~np.isnan(lats_conv)]
            if len(lats_conv) > 0:
                lat_mean = np.mean(lats_conv)
            else:
                return None
        except RuntimeWarning:
            print(locs)
        colat_mean = np.cos(lat_mean * np.pi / 180)
        return colat_mean

    def autocorrelation(self, which, size_which='energy_proxy', lag=1, ci=0.95):
        if which == 'delt':
            data = self.waiting_times()
        elif which == 'size':
            data = self.sizes(which=size_which)
        else:
            print(which)
            raise AssertionError('which needs to be delt or size for autocorrelation')
        try:
            assert len(data) - lag > 4
        except AssertionError:
            return None, None
        rc = spearmanr(data[:-lag], data[lag:])[0]
        if ci==0:
            return rc, None
        z = np.sqrt(2) * erfcinv(1 - ci)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            if rc == -1:
                ci_down = -1
                ci_up = -1
            elif rc == 1:
                ci_down = 1
                ci_up = 1
            else:
                ci_down = np.tanh(np.arctanh(rc) - z * (1 + rc**2/2)/np.sqrt(len(data) - 4))
                ci_up = np.tanh(np.arctanh(rc) + z * (1 + rc**2/2)/np.sqrt(len(data) - 4))
        return rc, (ci_down, ci_up)


class Flare:
    def __init__(self, id, start_time, peak_time, end_time, location, fclass, peak_flux, integ_flux, region_num, units='h', dels_units='jm2'):
        if units not in ['h', 's', 'D']:
            raise NotImplementedError(f"Units have to be 's', 'h', or 'D' for Flare class, but specified as {units}")
        self.start_time = np.datetime64(start_time)
        self.end_time = np.datetime64(end_time)
        if peak_time == 'None':
            self.peak_time = np.datetime64(None)
        else:
            self.peak_time = np.datetime64(peak_time)
        try:
            assert self.end_time > self.start_time
        except AssertionError:
            # if end_time not after start_time, set end_time to peak_time if that's after start_time
            if self.peak_time > self.start_time:
                self.end_time = self.peak_time

        self.id = id
        self.location = location
        self.fclass = fclass
        self.peak_flux = peak_flux
        self.integ_flux = integ_flux
        self.region_num = region_num
        self.duration = (self.end_time - self.start_time) / np.timedelta64(1, units)
        if self.duration > 0:
            self.anomalous_flag = False
        else:
            self.anomalous_flag = True
        self.energy_proxy = self.peak_flux * self.duration

