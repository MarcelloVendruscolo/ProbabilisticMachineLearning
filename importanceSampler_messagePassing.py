import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

m0 = 0 # Mean of p(x)
s0 = 1 # Variance of p(x)
s = 1 # Variance of p(t|x)
y0 = 1 # Measurement

# Analytical answer for x (derived with message-passing)
px_m = 1/np.sqrt(np.pi)
px_s = 1-1/np.pi

# Analytical answer t (derived with message-passing)
pt_m = 2/np.sqrt(np.pi)
pt_s = 2*(1-2/np.pi)

# Importance sampler
L = 100000 # number of samples
x = np.random.normal(size=L)*np.sqrt(s0)+m0 # draw from p(x)
t = np.random.normal(size=L)*np.sqrt(s)+x # draw from p(t|x)
y = np.sign(t) # proposed function
w = (y==y0)
w = L*w/np.sum(w)

xv = np.linspace(-4,4,1000)
# plot a weighted histogram of x
plt.hist(x, weights=w, bins=150, density=True, label="Importance sampling")
plt.plot(xv, norm.pdf(xv, px_m, np.sqrt(px_s)), label="Moment matching")
plt.xlim((-4,4))
plt.xlabel("x")
plt.legend()
plt.show()

# plot a weighted histogram of t
plt.hist(t, weights=w, bins=150, density=True, label="Importance sampling")
plt.plot(xv, norm.pdf(xv, pt_m, np.sqrt(pt_s)), label="Moment matching")
plt.xlim((-4,4))
plt.xlabel("t")
plt.legend()
plt.show()

# Estimate mean and variance
est_mean = np.sum(x*w)/L
est_var = np.sum(w*(est_mean-x)**2)/L

print("Mean moment-matching:", px_m) # Output: 0.564189583548
print("Mean importance sampling:", est_mean) # Output: 0.559303292236

print("Variance moment-matching", px_s) # Output: 0.6816901138162093
print("Variance importance sampling", est_var) # Output: 0.68086688604