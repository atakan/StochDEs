#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Choe sim12.1, p. 218-219

# Let's try to make one simulation with one timestep and see error
# statistics

import numpy as np
import matplotlib.pyplot as plt
import copy

# parameters and initial conditions

### fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ax = plt.subplots()

#rng = np.random.default_rng(seed=42)
np.random.seed(421)

T = 3.0
X0 = 0.4
int_sizes = [1024, 2048, 4096]
sigmas = np.zeros_like(int_sizes, dtype=float)
dts = np.zeros_like(int_sizes, dtype=float)
for k, N_int in enumerate(int_sizes):
    t = np.linspace(0, T, N_int+1)
    dt = t[1]-t[0]
    sqrt_dt = np.sqrt(dt)
    Residuals = np.zeros(0) 
    for j in range(1800):
        # our random increments, Gaussian variables with Var(X)=dt, E(X)=0
        dW = sqrt_dt*np.random.randn(N_int)
        W = np.cumsum(np.concatenate([np.zeros(1),dW]))
        #this is the exact solution
        Exact = W**2 + X0
        # numerical solution
        X = np.zeros(N_int+1)
        X[0] = X0
        for i in range(N_int):
            X[i+1] = X[i] + 2*W[i]*dW[i] + dt
        #ax1.plot(t, X, "r-", label=('Num. dt=%.2e' % (dt)))
        Residuals = np.concatenate((Residuals, Exact-X))
    dts[k] = dt
    sigmas[k] = np.std(Residuals)
    print(k, dts[k], sigmas[k])

ax.set_xscale("log")
ax.set_xlabel("$\delta t$")
ax.set_yscale("log")
ax.set_ylabel("$\sigma$ std. dev. of residuals")

ax.plot(dts, sigmas, "-*", label="data")
ax.plot(dts, np.sqrt(dts), "-", label="$\sqrt{\delta t}$")
ax.legend()
plt.show()

### ax1.plot(t, Exact, "k-", label='exact')
### ax1.plot(t, X, "r-", label=('Num. dt=%.2e' % (dt)))
### 
### numbins = 53
### n, bins, patches = ax2.hist(Residuals, bins=numbins, density=True)
### 
### sigma = np.std(Residuals)
### mu = np.mean(Residuals)
### y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
###      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
### ax2.plot(bins, y, '--', label='$\sigma=%.2e \mu=%.2e$' %(sigma, mu))
### 
### ax1.legend()
### ax2.legend()
### 
### plt.show()
