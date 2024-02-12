import numpy as np
from scipy.special import erf, erfc, lambertw
from scipy.optimize import root, root_scalar
pi = np.pi

class Gaussian:
    def __init__(self, mu, sig):
        self.mu = mu
        self.sig = sig
        self.name = 'Gaussian'

    def pdf(self, x):
        norm = np.sqrt(2 / pi) / (self.sig * (1 + erf(self.mu / (np.sqrt(2) * self.sig))))
        return norm * np.exp(-(x - self.mu)**2 / (2 * self.sig**2))

    def ln_pdf(self, x):
        norm = np.log(np.sqrt(2 / pi)) - np.log(self.sig) - np.log(1 + erf(self.mu / (np.sqrt(2) * self.sig)))
        return norm - (x - self.mu)**2 / (2 * self.sig**2)

    def mle_params(self, data):
        n = len(data)
        mu = np.mean(data)
        sig = 1 / (n + 1) * np.sum((data - mu)**2)
        return mu, sig

    def cdf(self, x):
        return 1 + erfc((x - self.mu) / (np.sqrt(2) * self.sig)) / (-2 + erfc(self.mu / (np.sqrt(2) * self.sig)))

    def p0s_bounds(self, data, p0s='default', bounds='default', for_opt=True):
        if p0s=='default':
            self.p0s = [np.mean(data), np.std(data)]
        else:
            self.p0s = p0s
        if bounds=='default':
            self.lower_bounds = [0, 0]
            if for_opt:
                self.upper_bounds = [None, None]
            else:
                self.upper_bounds = [np.inf, np.inf]
            self.bounds = (self.lower_bounds, self.upper_bounds)
        else:
            self.bounds = bounds
        if for_opt:
            self.bounds = np.transpose(self.bounds)
        return self.p0s, self.bounds

class PowerLaw:
    def __init__(self, d, xm):
        self.d = d
        self.xm = xm
        self.name = 'Power Law'

    def pdf(self, x):
        norm = (self.d - 1) * self.xm**(self.d - 1)
        return norm * x**(-self.d)

    def ln_pdf(self, x):
        npl = np.log(self.d - 1) + (self.d - 1) * np.log(self.xm)
        return npl - self.d * np.log(x)

    def mle_params(self, data):
        xm = min(data)
        d = 1 + 1 / (np.mean(np.log(data)) - np.log(xm))
        return d, xm

    def cdf(self, x):
        return 1 - (self.xm / x)**(self.d - 1)

    def p0s_bounds(self, data, p0s='default', bounds='default', for_opt=True):
        if p0s=='default':
            self.p0s = [3, min(data) / 2]
        else:
            self.p0s = p0s
        if bounds=='default':
            self.lower_bounds = [1 * (1 + 1e-3 * min(data)), 1e-3 * min(data)]
            if for_opt:
                self.upper_bounds = [None, min(data)]
            else:
                self.upper_bounds = [np.inf, min(data)]
            self.bounds = (self.lower_bounds, self.upper_bounds)
        else:
            self.bounds = bounds
        if for_opt:
            self.bounds = np.transpose(self.bounds)
        return self.p0s, self.bounds

class PowerLawTrunc:
    def __init__(self, d, xm, tm=9.6e-4):
        self.d = d
        self.xm = xm
        self.tm = tm
        self.name = 'Power Law'

    def pdf(self, x):
        norm = (self.d - 1) * self.xm**(self.d - 1) * (1 - self.xm / self.tm)**(self.d - 1)
        return norm * x**(-self.d)

    def ln_pdf(self, x):
        npl = np.log(self.d - 1) + (self.d - 1) * (np.log(self.xm * (1 - self.xm / self.tm)))
        return npl - self.d * np.log(x)

    # def mle_params(self, data):
    #     xm = min(self.tm / 2, min(data))
    #     d = 1 + len(data) * (np.sum(np.log(data)) - len(data) * (np.log(xm) + np.log(1 - xm / self.tm)))**(-1)
    #     return d, xm

    def cdf(self, x):
        pass

    def p0s_bounds(self, data, p0s='default', bounds='default', for_opt=True):
        if p0s=='default':
            self.p0s = [3, min(data) / 2]
        else:
            self.p0s = p0s
        if bounds=='default':
            self.lower_bounds = [1.01, 9.6e-4]
            if for_opt:
                self.upper_bounds = [None, None]
            else:
                self.upper_bounds = [np.inf, np.inf]
            self.bounds = (self.lower_bounds, self.upper_bounds)
        else:
            self.bounds = bounds
        if for_opt:
            self.bounds = np.transpose(self.bounds)
        return self.p0s, self.bounds

class Seperable:
    def __init__(self, b, c):
        self.b = b
        self.c = c
        self.name = 'Seperable'

    def pdf(self, x):
        norm = (1 + self.c) / self.b
        return norm * (1 - x / self.b)**self.c

    def ln_pdf(self, x):
        norm = np.log(1 + self.c) - np.log(self.b)
        return norm + self.c * np.log(1 - x / self.b)

    def mle_params(self, data):
        n = len(data)
        fun = lambda b: b * np.sum(1 / (b - data)) - 2 * n 
        sol = root_scalar(fun, bracket=[max(data)*(1+1e-5), max(data)*(1e5)])
        if sol.converged:
            b = sol.root
        else:
            print(sol)
            return
        c = -n / (np.sum(np.log(1 - data / b))) - 1
        return b, c

    def cdf(self, x):
        return 1 - ((self.b - x) / self.b)**(1 + self.c)

    def p0s_bounds(self, data, p0s='default', bounds='default', for_opt=True):
        if p0s=='default':
            self.p0s = [np.max(data)*(2 + 3), 3]
        else:
            self.p0s = p0s
        if bounds=='default':
            self.lower_bounds = [max(data) * (1 + 1e-3 * min(data)/max(data)), 1]
            if for_opt:
                self.upper_bounds = [None, None]
            else:
                self.upper_bounds = [np.inf, np.inf]
            self.bounds = (self.lower_bounds, self.upper_bounds)
        else:
            self.bounds = bounds
        if for_opt:
            self.bounds = np.transpose(self.bounds)
        return self.p0s, self.bounds

class LogNormal:
    def __init__(self, mu, sig):
        self.mu = mu
        self.sig = sig
        self.name = 'Log Normal'

    def pdf(self, x):
        norm = 1 / (np.sqrt(2 * pi) * self.sig)
        return norm / x * np.exp(-(np.log(x) - self.mu)**2 / (2 * self.sig**2))

    def ln_pdf(self, x):
        norm = -np.log(np.sqrt(2 * pi)) - np.log(self.sig)
        return norm - np.log(x) - (np.log(x) - self.mu)**2 / (2 * self.sig**2)

    def mle_params(self, data):
        n = len(data)
        lndata = np.log(data)
        mu = np.sum(lndata) / n
        sig = np.sqrt(np.sum((lndata - mu)**2) / (n-1))
        return mu, sig

    def cdf(self, x):
        return 1 / 2 * erfc((self.mu - np.log(x)) / (np.sqrt(2) * self.sig))

    def p0s_bounds(self, data, p0s='default', bounds='default', for_opt=True):
        if p0s=='default':
            mean = np.mean(data)
            var = np.var(data)
            self.p0s = [np.log(mean**2 / np.sqrt(var + mean**2)), np.sqrt(np.log(var / mean**2 + 1))]
        else:
            self.p0s = p0s
        if bounds=='default':
            self.lower_bounds = [0, np.std(data) * 1e-3]
            if for_opt:
                self.upper_bounds = [None, None]
            else:
                self.upper_bounds = [np.inf, np.inf]
            self.bounds = (self.lower_bounds, self.upper_bounds)
        else:
            self.bounds = bounds
        if for_opt:
            self.bounds = np.transpose(self.bounds)
        return self.p0s, self.bounds

class LogNormalTrunc:
    def __init__(self, mu, sig, tm=9.6e-4):
        self.mu = mu
        self.sig = sig
        self.tm = tm
        self.name = 'Log Normal'

    def pdf(self, x):
        norm = 1 / (np.sqrt(2 * pi) * self.sig) * 2 / erfc((self.mu - self.tm)**2 / (np.sqrt(2) * self.sig))
        return norm / x * np.exp(-(np.log(x) - self.mu)**2 / (2 * self.sig**2))

    def ln_pdf(self, x):
        norm = np.log(np.sqrt(2 / pi)) - np.log(self.sig / erfc((self.mu - self.tm)**2 / (np.sqrt(2) * self.sig)))
        return norm - np.log(x) - (np.log(x) - self.mu)**2 / (2 * self.sig**2)

    # def mle_params(self, data):
    #     pass

    def cdf(self, x):
        pass

    def p0s_bounds(self, data, p0s='default', bounds='default', for_opt=True):
        if p0s=='default':
            mean = np.mean(data)
            var = np.var(data)
            self.p0s = [np.log(mean**2 / np.sqrt(var + mean**2)), np.sqrt(np.log(var / mean**2 + 1))]
        else:
            self.p0s = p0s
        if bounds=='default':
            self.lower_bounds = [np.log(9.6e-4), np.std(data) * 1e-3]
            if for_opt:
                self.upper_bounds = [None, None]
            else:
                self.upper_bounds = [np.inf, np.inf]
            self.bounds = (self.lower_bounds, self.upper_bounds)
        else:
            self.bounds = bounds
        if for_opt:
            self.bounds = np.transpose(self.bounds)
        return self.p0s, self.bounds

class Exponential:
    def __init__(self, lam):
        self.lam = lam
        self.name = 'Exponential'

    def pdf(self, x):
        norm = self.lam
        return norm * np.exp(-self.lam * x)

    def ln_pdf(self, x):
        norm = np.log(self.lam)
        return norm - self.lam * x

    def mle_params(self, data):
        return [1 / np.mean(data)]

    def cdf(self, x):
        return 1 - np.exp(-self.lam * x)

    def p0s_bounds(self, data, p0s='default', bounds='default', for_opt=True):
        if p0s=='default':
            self.p0s = [1 / np.mean(data)]
        else:
            self.p0s = p0s
        if bounds=='default':
            self.lower_bounds = [min(data) * 1e-3]
            if for_opt:
                self.upper_bounds = [None]
            else:
                self.upper_bounds = [np.inf]
            self.bounds = (self.lower_bounds, self.upper_bounds)
        else:
            self.bounds = bounds
        if for_opt:
            self.bounds = np.transpose(self.bounds)
        return self.p0s, self.bounds

class ExponentialTrunc:
    def __init__(self, lam, tm=9.6e-4):
        self.lam = lam
        self.tm = tm
        self.name = 'Exponential'

    def pdf(self, x):
        norm = self.lam / (1 - np.exp(-self.tm * self.lam))
        return norm * np.exp(-self.lam * x)

    def ln_pdf(self, x):
        norm = np.log(self.lam / (1 - np.exp(-self.tm * self.lam)))
        return norm - self.lam * x

    def mle_params(self, data):
        n = len(data)
        m = 9.6e-4
        mnx = m * n * np.sum(data)
        print(mnx)
        lam = (mnx - lambertw(-np.exp(mnx - 1))) / m
        return [1 / np.mean(data)]

    def cdf(self, x):
        pass

    def p0s_bounds(self, data, p0s='default', bounds='default', for_opt=True):
        if p0s=='default':
            self.p0s = [1 / np.mean(data)]
        else:
            self.p0s = p0s
        if bounds=='default':
            self.lower_bounds = [1.2e-16]
            if for_opt:
                self.upper_bounds = [None]
            else:
                self.upper_bounds = [np.inf]
            self.bounds = (self.lower_bounds, self.upper_bounds)
        else:
            self.bounds = bounds
        if for_opt:
            self.bounds = np.transpose(self.bounds)
        return self.p0s, self.bounds

class Weibull:
    def __init__(self, lam, k):
        self.lam = lam
        self.k = k
        self.name = 'Weibull'

    def pdf(self, x):
        norm = self.k * self.lam**(-self.k)
        return norm * x**(self.k - 1) * np.exp(-(x / self.lam)**(self.k))

    def ln_pdf(self, x):
        norm = np.log(self.k) - self.k * np.log(self.lam)
        return norm + (self.k - 1) * np.log(x) - (x / self.lam)**(self.k)

    def mle_params(self, data):
        lndata = np.log(data)
        sumlndata = np.sum(lndata)
        n = len(data)
        fun = lambda k: np.sum(data**k * lndata) / np.sum(data**k) - 1 / k - sumlndata / n
        sol = root_scalar(fun, bracket=[0.001, 10])
        if sol.converged:
            k = sol.root
        else:
            print(sol)
            return
        lam = (1 / n * np.sum(data**k))**(1 / k)
        return lam, k

    def cdf(self, x):
        return 1 - np.exp(-(x / self.lam)**(self.k))

    def p0s_bounds(self, data, p0s='default', bounds='default', for_opt=True):
        if p0s=='default':
            self.p0s = [np.mean(data), 1]
        else:
            self.p0s = p0s
        if bounds=='default':
            self.lower_bounds = [min(data) * 1e-3, min(data) * 1e-3]
            if for_opt:
                self.upper_bounds = [None, None]
            else:
                self.upper_bounds = [np.inf, np.inf]
            self.bounds = (self.lower_bounds, self.upper_bounds)
        else:
            self.bounds = bounds
        if for_opt:
            self.bounds = np.transpose(self.bounds)
        return self.p0s, self.bounds

class Generic:
    def __init__(self):
        self.name = 'Generic'

    def pdf(self, x):
        pass

    def ln_pdf(self, x):
        pass

    def cdf(self, x):
        pass

    def p0s_bounds(self, data, p0s='default', bounds='default', for_opt=True):
        if p0s=='default':
            self.p0s = []
        else:
            self.p0s = p0s
        if bounds=='default':
            self.lower_bounds = []
            self.upper_bounds = []
            self.bounds = (self.lower_bounds, self.upper_bounds)
        else:
            self.bounds = bounds
        if for_opt:
            self.bounds = np.transpose(self.bounds)
        return self.p0s, self.bounds
