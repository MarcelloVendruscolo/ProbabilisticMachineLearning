import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import multivariate_normal

# Set path to dataset
path_dataset = 'data/nlindata.mat'

# Use the whole dataset or limit it to a number of data points (default = 5 data points)
limited_dataset = False

# Feature vector: poly_1, poly_2, step_function
input_feature = 'step_function'

# Load data from disk
data = scipy.io.loadmat(path_dataset)

# Full training dataset
x = data["X"]  # inputs
y = data["Y"]  # outputs

if (limited_dataset):
    x = x[0:5]
    y = y[0:5]

beta = float(data["sigma"])**(-2)  # measurement noise precision
N = len(x)  # number of data points

# Define the feature vector
def Phi(a, feature='poly_1'):
    if feature == 'poly_1':
        return np.power(a, range(2)) # rigid feature vector Phi(a) = [1,a]
    elif feature == 'poly_2':
        return np.power(a, range(3)) # more flexible feature vector Phi(a) = [1,a,a**2]
    elif feature == 'step_function':
        return (a > np.linspace(-8, 8, 9).T) # step function Phi(a) = [h(a−8),h(a−6),...,h(a+8)] where h(x) = 1 (if x >= 0) or h(x) = 0 (if x < 0) 
    else:
        raise ValueError()

D = len(Phi(0, input_feature))  # number of features

# Set parameters of the prior on the weights: p(w)=N(mu0,Sigma0)
mu0 = np.zeros((D, 1))
Sigma0 = 10*np.eye(D) / D

# Regression
SN = np.linalg.inv(np.linalg.inv(Sigma0) + beta * Phi(x, input_feature).T @ Phi(x, input_feature))
mN = SN @ (np.linalg.inv(Sigma0) @ mu0 + beta * Phi(x, input_feature).T @ y)

# X data points to evaluate models
xs = np.linspace(-8, 8, num=100)[:, np.newaxis]  # reshape is needed for Phi to work

# Compute the postrior distribution of the function
mpost = Phi(xs, input_feature) @ mN 
vpost = Phi(xs, input_feature) @ SN @ Phi(xs, input_feature).T

# Compute the predictive distribution of the outputs
mpred = mpost
vpred = vpost + beta**(-1)

# Draw samples from the posterior
s = 5
fs = multivariate_normal(mean=mpost.flatten(), cov=vpost, allow_singular=True).rvs(s).T

# Code for the plotting
plt.plot(xs, Phi(xs, input_feature)) # Plot the features
plt.title('features')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.plot(xs, fs, 'gray') # Plot the samples
plt.scatter(x, y, zorder=3)
plt.title('posterior - samples')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Reshape (plt.fill_between requires a one-dimensional array)
xsf = xs.flatten()
mpredf = mpred.flatten()
stdpred = np.sqrt(np.diag(vpred))

plt.plot(xs, mpred, 'black') # Plot credibility regions
plt.fill_between(xsf, mpredf + 3*stdpred, mpredf - 3*stdpred, color='lightgray')
plt.fill_between(xsf, mpredf + 2*stdpred, mpredf - 2*stdpred, color='darkgray')
plt.fill_between(xsf, mpredf + 1*stdpred, mpredf - 1*stdpred, color='gray')
plt.scatter(x, y, zorder=3)
plt.title('predictive distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()