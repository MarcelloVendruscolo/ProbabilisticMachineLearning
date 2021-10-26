import numpy as np
import matplotlib.pyplot as plt

# Target Function
def johnson_dist(x):
  return np.sqrt(2) / np.sqrt(np.pi*(1+(x-1)**2)) * np.exp(-0.5*(3+2*np.arcsinh(x-1))**2)

# Proposal function
m, s2 = -2, 4
def proposal_dist(x):
  return 1/np.sqrt(2*np.pi*s2)*np.exp(-((x-m)**2)/2/s2)

# Importance Sampling:
L = 1000 # number of samples
samples = np.random.normal(loc=0.0, scale=1.0, size=L) * np.sqrt(s2) + m # draw from proposal
weights = johnson_dist(samples)/proposal_dist(samples) # non-normalised weights
weights = weights * L / np.sum(weights) # normalised weights

# Plot a weighted histogram
plt.hist(samples, density=True, weights=weights, bins=50)

# Plot the target distribution and the proposal
x_axis = np.linspace(start=-8, stop=4, num=100)
plt.plot(x_axis, johnson_dist(x_axis), label='Target pdf')
plt.plot(x_axis, proposal_dist(x_axis), label='Proposal pdf')
plt.legend()
plt.show()

for L in [10, 100, 1000, 10000000]:
    samples = np.random.normal(loc=0.0, scale=1.0, size=L) * np.sqrt(s2) + m # draw from proposal 
    weights = johnson_dist(samples)/proposal_dist(samples) # non-normalised weights
    weights = weights * L / np.sum(weights) # normalised weights
    mean = np.sum(weights*samples)/L
    var = np.sum(weights*(mean - samples)**2)/L
    print('Estimates of mean and variance from samples when L = ', L, ' are: ', mean, var)