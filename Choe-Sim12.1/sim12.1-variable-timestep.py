#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Choe sim12.1, p. 218-219

import numpy as np
import matplotlib.pyplot as plt
import copy

# parameters and initial conditions

#rng = np.random.default_rng(seed=42)
np.random.seed(421)

T = 3.0
N = 8192
X0 = 0.4

t = np.linspace(0, T, N+1)
dt = t[1]-t[0]

# our random increments, Gaussian variables with Var(X)=dt, E(X)=0
dW = np.sqrt(dt)*np.random.randn(N)
W = np.cumsum(np.concatenate([np.zeros(1),dW]))

#this is the exact solution
Exact = W**2 + X0
plt.plot(t, Exact, "k-", label='exact')

#numerical solution
# this routine takes a bunch of normal distributed numbers and combines
# them 2 by 2. This can be done more elegantly but POitRoAE -- Knuth
def reducedW(dW) :
    reddW = np.zeros(dW.size//2)
    for i in range(dW.size//2):
        reddW[i] = dW[2*i] + dW[2*i+1]
    return(reddW)
    
jmax = 5
for j in range(jmax): # we will try for 3 different timesteps
    print("j =",j, "N =", N)
    print("dW = ", dW)
    t = np.linspace(0, T, N+1)
    dt = t[1]-t[0]
    W = np.zeros(N+1)
    X = np.zeros(N+1)
    X[0] = X0
    for i in range(N):
        W[i+1] = W[i] + dW[i]
        X[i+1] = X[i] + 2*W[i]*dW[i] + dt
    plt.plot(t, X, "r-*", label=('Num. dt=%.2e' % (dt)),
            color='%f' %((j+1)/(jmax+2)))
    dW = reducedW(dW)
    N = N//2

## numerical solution
#for j in range(3): # we will try for 3 different timesteps
#    if j>0:
#        dW = reducedW(dW)
#        N = N/2
#    t = np.linspace(0, T, N+1) # a bit ugly, there are N+1 timepoints,
#                               # leading to N intervals.
#    dt = t[1]-t[0]
#    W = np.zeros(N+1)
#    X = np.zeros(N+1)
#    X[0] = X0
#    for i in range(N):
#        W[i+1] = W[i] + dW[i]
#        X[i+1] = X[i] + 2*W[i]*dW[i] + dt
#    plt.plot(t, X, "r-", label=('Num. dt=%.2e' % (dt)))
#
#



plt.ylabel("$X_t$")
plt.xlabel("$t$")
plt.legend()

plt.show()
