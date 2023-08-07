#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Choe sim12.1, p. 218-219

import numpy as np
import matplotlib.pyplot as plt
import copy

#rng = np.random.default_rng(seed=42)
np.random.seed(421)

T = 3.0
N = 2048
X0 = 0.4

#this is the exact solution we calculate it once
Exact = np.zeros(N+1)
Exact[0] = X0
for i in range(N):
    Exact[i+1] = W[i+1]**2 + X0
tExact = copy.deepcopy(t)
plt.plot(t, Exact, "k-", label='exact')

# this routine takes a bunch of normal distributed numbers and combines
# them 2 by 2.
def reducedW(dW) :
    

for j in range(3): # we will try for 3 different timesteps
    t = np.linspace(0, T, N+1) # a bit ugly
    dt = t[1]-t[0]
    if j==0:
        dW = np.sqrt(dt)*np.random.randn(N)
    else:
        dW = reducedW(dW)
    W = np.zeros(N+1)
    X = np.zeros(N+1)
    X[0] = X0
    for i in range(N):
        W[i+1] = W[i] + dW[i]
        X[i+1] = X[i] + 2*W[i]*dW[i] + dt
    plt.plot(t, X, "r-", label=('Num. dt=%.2e' % (dt)))


plt.ylabel("$X_t$")
plt.xlabel("$t$")
plt.legend()

plt.show()
