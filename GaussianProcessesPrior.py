import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def RBF_kernel(x, l):
    return sf2*np.exp(-1/(2*l**2) * np.abs(x[:,np.newaxis]-x[:,np.newaxis].T)**2)

def plot_samples(xs, fs, l):
    plt.plot(xs, fs, 'gray')
    plt.title(f'l = {l}')
    plt.show()

m = 101 # number of elements to be equally spaced
xs = np.linspace(start=-5, stop=5, num=m, endpoint=True)
mxs = np.zeros(shape=m) # mean vector set to 0

sf2 = 1 # hyperparameter: variance of kernel
l = np.sqrt(2) # hyperparameter: length scale of RBF kernel
Kss = RBF_kernel(xs, l) # covariance matrix

s = 25 # number of samples to draw from the prior
fs = multivariate_normal(mean=mxs, cov=Kss, allow_singular=True).rvs(s).T
plot_samples(xs, fs, l)

l = 0.5 # Different length scale
Kss = RBF_kernel(xs, l)
fs = multivariate_normal(mean=mxs, cov=Kss, allow_singular=True).rvs(s).T
plot_samples(xs, fs, l)


