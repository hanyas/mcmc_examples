import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.figsize'] = (10, 5)


def sample_offset(y, x, slope, tau, mu0, tau0):
    N = y.shape[0]
    precision = tau0 + tau * N
    mean = tau0 * mu0 + tau * np.sum(y - slope * x)
    mean /= precision
    return np.random.normal(mean, 1.0 / np.sqrt(precision))


def sample_slope(y, x, offset, tau, mu1, tau1):
    precision = tau1 + tau * np.sum(x * x)
    mean = tau1 * mu1 + tau * np.sum( (y - offset) * x)
    mean /= precision
    return np.random.normal(mean, 1 / np.sqrt(precision))


def sample_noise(y, x, offset, slope, alpha, beta):
    N = y.shape[0]
    alpha_new = alpha + N / 2
    resid = y - offset - slope * x
    beta_new = beta + np.sum(resid * resid) / 2
    return np.random.gamma(alpha_new, 1 / beta_new)


offset_true = -1
slope_true = 2
noise_true = 1

N = 200
x = np.random.uniform(low=0, high=4, size=N)
y = np.random.normal(offset_true + slope_true * x, 1 / np.sqrt(noise_true))

plt.figure()
synth_plot = plt.plot(x, y, "o")
plt.xlabel("x")
plt.ylabel("y")

# specify initial values
init = {"offset": 0, "slope": 0, "tau": 2}

# specify hyper parameters
hypers = {"mu0": 0, "tau0": 1,
          "mu1": 0, "tau1": 1,
          "alpha": 2, "beta": 1}


def gibbs(y, x, iters, init, hypers):
    offset = init["offset"]
    slope = init["slope"]
    tau = init["tau"]

    trace = np.zeros((iters, 3))  # trace to store values of beta_0, beta_1, tau

    for it in range(iters):
        offset = sample_offset(y, x, slope, tau, hypers["mu0"], hypers["tau0"])
        slope = sample_slope(y, x, offset, tau, hypers["mu1"], hypers["tau1"])
        tau = sample_noise(y, x, offset, slope, hypers["alpha"], hypers["beta"])
        trace[it, :] = np.array((offset, slope, tau))

    trace = pd.DataFrame(trace)
    trace.columns = ['offset', 'slope', 'tau']

    return trace


iters = 1000
trace = gibbs(y, x, iters, init, hypers)

traceplot = trace.plot()
traceplot.set_xlabel("Iteration")
traceplot.set_ylabel("Parameter value")

trace_burnt = trace[500:]
hist_plot = trace_burnt.hist(bins=30, layout=(1, 3))

print(trace_burnt.median())
print(trace_burnt.std())
