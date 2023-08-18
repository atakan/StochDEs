#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Choe sim12.1, p. 218-219

import numpy as np
import matplotlib.pyplot as plt

# GENERAL PARAMETERS not to be changed from run to run
#rng = np.random.default_rng(seed=42)
np.random.seed(421)
T = 3.0
X0 = 0.4

def single_run(N):
    t = np.linspace(0, T, N+1) # a bit ugly
    dt = t[1]-t[0]
    dW = np.sqrt(dt)*np.random.randn(N)
    W = np.zeros(N+1)
    X = np.zeros(N+1)
    Exact = np.zeros(N+1)
    X[0] = X0
    Exact[0] = X0
    for i in range(N):
        W[i+1] = W[i] + dW[i]
        X[i+1] = X[i] + 2*W[i]*dW[i] + dt
        Exact[i+1] = W[i+1]**2 + X0
    return t, X, Exact

N_runs = 1800
N_interval = 1024
Residuals = np.zeros([Nruns, N_interval])
for irun in range(N_runs):
    t, X, Exact = single_run(N_interval)
    Residuals[irun] = X-Exact
    if irun == N_runs//2:
        plt.plot(t, X, "r-", label='approx')
        plt.plot(t, Exact, "k-", label='exact')

plt.ylabel("$X_t$")
plt.xlabel("$t$")
plt.legend()

plt.show()
