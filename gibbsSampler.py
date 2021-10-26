import numpy as np
import matplotlib.pyplot as plt

K = 300
A_vector = np.zeros(shape=K)
B_vector = np.zeros(shape=K)

A_vector[0] = 1 # Initialisation according to the user choice
B_vector[0] = 1 # Initialisation according to the user choice

for index in range(K-1):
    # Sample A given B
    if B_vector[index] == 0:
        A_vector[index+1] = np.random.binomial(n=1, p=0.03/0.84)
    else: 
        A_vector[index+1] = np.random.binomial(n=1, p=0.07/0.16)
    
    # Sample B given A
    if A_vector[index+1] == 0:
        B_vector[index+1] = np.random.binomial(n=1, p=0.1)
    else:
        B_vector[index+1] = np.random.binomial(n=1, p=0.7)

plt.plot(A_vector, label='A')
plt.plot(B_vector,label='B')
plt.legend(loc='upper right')
plt.title('Trace plot')
plt.show()

plt.bar(x=(0,1), height=(sum(1-A_vector)/K,sum(A_vector)/K))
plt.title('Empirical distribution for A: p(A)')
plt.show()

plt.bar(x=(0,1), height=(sum(1-B_vector)/K,sum(B_vector)/K))
plt.title('Empirical distribution for B: p(B)')
plt.show()