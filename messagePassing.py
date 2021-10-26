import numpy as np
from scipy.stats import truncnorm

def mutiplyGauss(m1, s1, m2, s2):
    # Computes the Gaussian distribution N(m,s) being propotional to N(m1,s1) * N(m2,s2)
    # Input: mean (m1, m2) and variance (s1, s2) of first and second Gaussians, respectively
    # Output: m, s mean and variance of the product Gaussian
    s = 1 / (1/s1 + 1/s2)
    m = (m1/s1 + m2/s2) * s
    return m, s

def divideGauss(m1, s1, m2, s2):
    # Computes the Gaussian distribution N(m,s) being propotional to N(m1,s1) / N(m2,s2)
    # Input: mean (m1, m2) and variance (s1, s2) of first and second Gaussians, respectively
    # Output: m, s mean and variance of the quotient Gaussian
    m, s = mutiplyGauss(m1, s1, m2, -s2)
    return m, s

def truncGaussMM(a, b, m0, s0):
    # Computes the mean and variance of a truncated Gaussian distribution
    # Inputs: The interval [a, b] on which the Gaussian (mean mo, var s0) is being truncated
    # Output: m, s mean and variance of the truncated Gaussian
    # scale interval with mean and variance
    a_scaled, b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
    m = truncnorm.mean(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    s = truncnorm.var(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    return m, s

m0 = 0 # The mean of the priorp(x)
s0 = 1 # The variance of the prior p(x)
sv = 1 # The variance of p(t|x)
y0 = 1 # The measurement

# Message mu3 from prior to node x
mu3_m = m0 # mean of message 
mu3_s = s0 # variance of message

# Message mu4 from node x to factor f_xt
mu4_m = mu3_m # mean of message 
mu4_s = mu3_s # variance of message

# Message mu5 from factor f_xt to node t
mu5_m = mu4_m
mu5_s = mu4_s + sv

# Do moment matching of the marginal of t
if y0 == 1:
    a, b = 0, 1000
else:
    a, b = -1000, 0
    
pt_m, pt_s = truncGaussMM(a, b, mu5_m, mu5_s)

# Compute the message from t to f_xt
mu6_m, mu6_s = divideGauss(pt_m, pt_s, mu5_m, mu5_s)

# Compute the message from f_xt to x
mu7_m = mu6_m
mu7_s = mu6_s + sv

# Compute the marginal of x
px_m, px_s = mutiplyGauss(mu3_m, mu3_s, mu7_m, mu7_s)

print(px_m) # Output: 0.564189583548
print(px_s) # Output: 0.681690113816