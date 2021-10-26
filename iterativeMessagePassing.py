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

ax, bx = -1, 1 # The interval [ax,bx] for the uniform prior on x
sv = 1 # The variance of p(t|x)
y0 = 1 # The measurement

# Initialize message mu4 with mean 0 and variance 1 
mu4_m = 0 # mean of message
mu4_s = 1 # variance of message

for j in range(0, 10):

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

    # Do moment matching of the marginal of x
    px_m, px_s = truncGaussMM(ax, bx, mu7_m, mu7_s)

    # Compute the updated message from f_x to x
    mu4_m, mu4_s = divideGauss(px_m, px_s, mu7_m, mu7_s)

print(px_m) # Output: 0.244155540842
print(px_s) # Output: 0.276033234014