import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def RBF_kernel(x, xp, l):
    return sf2*np.exp(-1/(2*l**2) * np.abs(x[:,np.newaxis]-xp[:,np.newaxis].T)**2)

x = np.array([-4,-3,-1,0,2])
f = np.array([-2,0,1,2,-1])
n = np.size(x)

m = 101 # number of elements to be equally spaced
xs = np.linspace(start=-5, stop=5, num=m, endpoint=True)

l = np.sqrt(2) # hyperparameter: length scale of RBF kernel
sf2 = 1 # hyperparameter: variance of kernel
Kss = RBF_kernel(xs,xs,l) # Covariance matrix
Ks = RBF_kernel(x,xs,l)
K = RBF_kernel(x,x,l)

mu_post = (Ks.T@np.linalg.inv(K))@f
K_post = Kss - Ks.T@np.linalg.inv(K)@Ks

s = 25 # number of samples to draw from the prior
fs = multivariate_normal(mean=mu_post, cov=K_post, allow_singular=True).rvs(s).T

# Plotting the samples
plt.plot(xs, fs, 'gray')
plt.scatter(x,f,zorder = 3)
plt.title('l = sqrt(2), no noise')
plt.show()

# Plotting credibility regions
plt.plot(xs,mu_post,'black')
plt.fill_between(xs,mu_post + 3*np.sqrt(np.diag(K_post)),mu_post - 3*np.sqrt(np.diag(K_post)),color= 'lightgray')
plt.fill_between(xs,mu_post + 2*np.sqrt(np.diag(K_post)),mu_post - 2*np.sqrt(np.diag(K_post)),color= 'darkgray')
plt.fill_between(xs,mu_post + 1*np.sqrt(np.diag(K_post)),mu_post - 1*np.sqrt(np.diag(K_post)),color= 'gray')
plt.scatter(x,f,zorder=3)
plt.title('l = sqrt(2), no noise')
plt.show()

# Include measurement noise
K = RBF_kernel(x,x,l) + 0.1*np.eye(n)
mu_post = (Ks.T@np.linalg.inv(K))@(f)
K_post = Kss - Ks.T@np.linalg.inv(K)@Ks
fs = multivariate_normal(mean=mu_post,cov=K_post,allow_singular=True).rvs(s).T

# Plotting the samples with noise
plt.plot(xs,fs,'gray')
plt.scatter(x,f,zorder=3)
plt.title('l = sqrt(2), with noise')
plt.show()

# Plot credibility regions with noise
plt.plot(xs,mu_post,'black') 
plt.fill_between(xs,mu_post + 3*np.sqrt(np.diag(K_post)),mu_post - 3*np.sqrt(np.diag(K_post)),color='lightgray')
plt.fill_between(xs,mu_post + 2*np.sqrt(np.diag(K_post)),mu_post - 2*np.sqrt(np.diag(K_post)),color='darkgray')
plt.fill_between(xs,mu_post + 1*np.sqrt(np.diag(K_post)),mu_post - 1*np.sqrt(np.diag(K_post)),color='gray')
plt.scatter(x,f,zorder=3)
plt.title('l = sqrt(2), with noise')
plt.show()