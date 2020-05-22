import numpy as np
import scipy as sc
import scipy.stats

import matplotlib._color_data as mcd
import matplotlib.pyplot as plt

import seaborn as sns

np.set_printoptions(precision=5)

sns.set_style("white")
sns.set_context("paper")

color_names = ["red", "windows blue", "medium green",
               "dusty purple", "orange", "amber",
               "clay", "pink", "greyish", "light cyan",
               "steel blue", "forest green",
               "pastel purple", "mint", "salmon", "dark brown"]

colors = []
for k in color_names:
    colors.append(mcd.XKCD_COLORS['xkcd:' + k].upper())


def stick_breaking(n_comp, alpha):
    betas = sc.stats.beta(1, alpha).rvs(n_comp)
    weights = np.empty((n_comp,))

    weights[0] = betas[0]
    weights[1:] = betas[1:] * np.cumprod(1.0 - betas[:-1])

    return weights


# sample data
k_true = 4
n = 100
d = 2

hyper_alpha = sc.stats.gamma(9.0, 0.5).rvs()
pi_true = sc.stats.dirichlet(hyper_alpha * np.ones((k_true,)) / k_true).rvs()[0]

hyper_mu = np.zeros((d,))
hyper_sigma = 1.0 * np.eye(d)

mu_true = np.array([[-3.0, 3.0], [-3.0, -3.0], [3.0, 3.0], [3.0, -3.0]])
sigma_true = 1.0 * np.eye(d)

z_true = np.empty((n,), np.int64)
x = np.empty((n, 2))

for i in range(n):
    z_true[i] = np.random.choice(a=len(pi_true), p=pi_true)
    x[i, :] = sc.stats.multivariate_normal(mean=mu_true[z_true[i], :], cov=sigma_true).rvs()

plt.figure()
plt.title('Data')
for m in range(k_true):
    plt.scatter(x[z_true == m, 0], x[z_true == m, 1], color=colors[m])
plt.show()

# DPMM learning

k = n
mu = x.copy()
z = np.random.randint(0, k, n)

inv_sigma = np.linalg.inv(sigma_true)
inv_hyper_sigma = np.linalg.inv(hyper_sigma)

for it in range(100):

    # sample indicators
    for i in range(n):
        pi = np.empty((k,))

        # remove data point from cluster
        z[i] = -1

        m = 0
        while m < k:
            indices = (z == m)
            num = indices.sum()

            # remove cluster
            if num == 0:
                mu = np.delete(mu, m, axis=0)
                pi = np.delete(pi, m)

                k = k - 1

                # rearrange clusters
                z[z > m] = z[z > m] - 1

                continue

            aux = num / (n + hyper_alpha - 1)
            pi[m] = aux * sc.stats.multivariate_normal(mean=mu[m, :], cov=sigma_true).pdf(x[i, :])

            m = m + 1

        aux = hyper_alpha / (n + hyper_alpha - 1)
        new = aux * np.exp(- x[i, :] @ inv_sigma @ x[i, :] / 4.0) / (2.0 * np.sqrt(np.pi))

        pi = np.hstack((pi, new))
        pi = pi / pi.sum()

        z[i] = np.random.choice(a=len(pi), p=pi)

        # add cluster
        if z[i] == k:
            aux = sc.stats.multivariate_normal(mean=0.5 * x[i, :], cov=0.5 * np.eye(2)).rvs()
            mu = np.vstack((mu, aux))
            k = k + 1

    # sample means
    for m in range(k):
        indices = (z == m)
        num = indices.sum()

        x_k = x[indices, :]
        x_bar = np.sum(x_k, axis=0)

        sigma_hat = np.linalg.inv(num * inv_sigma + inv_hyper_sigma)
        mu_hat = sigma_hat @ (inv_sigma @ x_bar + inv_hyper_sigma @ hyper_mu)

        mu[m, :] = np.random.multivariate_normal(mu_hat, sigma_hat)

    # # sample alpha  # a, b = 1.0, 2.0  #
    # tmp = sc.stats.beta(hyper_alpha + 1, n).rvs()  # aux = a + n + 1  # w = aux / (n * (b - np.log(tmp)) + aux)
    # hyper_alpha = w * sc.stats.gamma(a + n, b - np.log(tmp)).rvs() + (1 - w) * sc.stats.gamma(a + n - 1, b - np.log(tmp)).rvs()

plt.figure()
plt.title('DPMM')
for m in range(k):
    plt.scatter(x[z == m, 0], x[z == m, 1], color=colors[m])
plt.show()
