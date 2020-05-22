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

k = 4
n = 100
d = 2

hyper_alpha = sc.stats.gamma(9.0, 0.5).rvs(k)
pi_true = sc.stats.dirichlet(hyper_alpha).rvs()[0]

hyper_mu = np.ones((d,))
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
for m in range(k):
    plt.scatter(x[z_true == m, 0], x[z_true == m, 1], color=colors[m])
plt.show()

pi = sc.stats.dirichlet(hyper_alpha).rvs()[0]
mu = sc.stats.multivariate_normal(mean=hyper_mu, cov=hyper_sigma).rvs(k)
z = np.empty((n,))

for _ in range(1000):
    # sample indicators
    for i in range(n):
        pi_weighted = np.empty((k,))
        for m in range(k):
            pi_weighted[m] = pi[m] * sc.stats.multivariate_normal(mean=mu[m, :], cov=sigma_true).pdf(x[i, :])

        pi_weighted = pi_weighted / pi_weighted.sum()

        z[i] = np.random.choice(a=len(pi_weighted), p=pi_weighted)

    # sample pi
    alpha = np.empty((k,))
    for m in range(k):
        assigned_indices = (z == m)
        alpha[m] = assigned_indices.sum() + hyper_alpha[m]

    pi = sc.stats.dirichlet(alpha).rvs()[0]

    # sample means
    for m in range(k):
        assigned_indices = (z == m)
        num = assigned_indices.sum()

        x_k = x[assigned_indices, :]
        x_bar = np.sum(x_k, axis=0)

        inv_sigma = np.linalg.inv(sigma_true)
        inv_hyper_sigma = np.linalg.inv(hyper_sigma)

        sigma_hat = np.linalg.inv(num * inv_sigma + inv_hyper_sigma)
        mu_hat = sigma_hat @ (inv_sigma @ x_bar + inv_hyper_sigma @ hyper_mu)

        mu[m, :] = np.random.multivariate_normal(mu_hat, sigma_hat)

plt.figure()
plt.title('With Pi')
for m in range(k):
    plt.scatter(x[z == m, 0], x[z == m, 1], color=colors[m])
plt.show()

# mu = sc.stats.multivariate_normal(mean=hyper_mu, cov=hyper_sigma).rvs(k)
# z = np.empty((n,))
#
# for _ in range(25):
#
#     # sample indicators
#     for i in range(n):
#         z_aux = np.delete(z, i)
#         pi_weighted = np.empty((k,))
#         for m in range(k):
#             assigned_indices = (z_aux == m)
#             num = assigned_indices.sum()
#             aux = (num + hyper_alpha[m]) / (n + k * hyper_alpha[m] - 1)
#             pi_weighted[m] = aux * sc.stats.multivariate_normal(mean=mu[m, :], cov=sigma_true).pdf(x[i, :])
#
#         pi_weighted = pi_weighted / pi_weighted.sum()
#
#         z[i] = np.random.choice(a=len(pi_weighted), p=pi_weighted)
#
#     # sample means
#     for m in range(k):
#         assigned_indices = (z == m)
#         num = assigned_indices.sum()
#
#         x_k = x[assigned_indices, :]
#         x_bar = np.sum(x_k, axis=0)
#
#         inv_sigma = np.linalg.inv(sigma_true)
#         inv_hyper_sigma = np.linalg.inv(hyper_sigma)
#
#         sigma_hat = np.linalg.inv(num * inv_sigma + inv_hyper_sigma)
#
#         mu_hat = inv_sigma @ sigma_hat @ x_bar + inv_hyper_sigma @ sigma_hat @ hyper_mu
#
#         mu[m, :] = np.random.multivariate_normal(mu_hat, sigma_hat)
#
# plt.figure()
# plt.title('Without Pi')
# for m in range(k):
#     plt.scatter(x[z == m, 0], x[z == m, 1], color=colors[m])
# plt.show()
