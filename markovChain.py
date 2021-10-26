import numpy as np
import matplotlib.pyplot as plt

def markov_chain(x):
    return 0.9*x + np.random.normal()*np.sqrt(0.19)

K = 300
x = np.zeros(shape=K)
x[0] = 5
for index in range(K-1):
    x[index+1] = markov_chain(x[index])

x_axis = np.linspace(start=-4, stop=4, num=100)

print('Preparing the plot ...')
plt.hist(x, bins=10, density=True)
plt.plot(x_axis, 1 / np.sqrt(2*np.pi) * np.exp(-0.5 * x_axis**2))
print('Opening the plot ...')
plt.show()
print('Plot closed!')

print('Preparing the plot ...')
plt.plot(x)
print('Opening the plot ...')
plt.show()
print('Plot closed!')

mean = np.mean(x)
var = np.var(x)
print('Estimates of mean and variance from x are: ', mean, var)