import numpy as np
import matplotlib.pyplot as plt

L = 300
samples  = np.random.normal(loc=0, scale=1, size=(L,1)) # independent random normal samples (mean = 0, var = 1)

x_axis = np.linspace(start=-4, stop=4, num=100)

plt.hist(samples, bins=10, density=True, label='Samples')
plt.plot(x_axis, 1 / np.sqrt(2*np.pi) * np.exp(-0.5 * x_axis**2))
plt.xlabel('x')
plt.legend()
plt.show()

print('Estimates of mean and variance from samples when L = ', L, ' are: ', np.mean(samples), np.var(samples))

for L in [10, 100, 1000, 10000, 100000]:
    samples = np.random.normal(loc=0, scale=1, size=(L,1))
    print('Estimates of mean and variance from samples when L = ', L, ' are: ', np.mean(samples), np.var(samples))