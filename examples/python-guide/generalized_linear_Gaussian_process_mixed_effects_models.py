# -*- coding: utf-8 -*-
"""
Examples of generalized linear Gaussian process and random effects models
for several non-Gaussian likelihoods

@author: Fabio Sigrist
"""

import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')
print("It is recommended that the examples are run in interactive mode")

# Choose likelihood: either "bernoulli_probit" (=default for binary data),
#                     "bernoulli_logit", "poisson", or "gamma"
likelihood = "bernoulli_probit"

# --------------------Grouped random effects model----------------
# Simulate data
n = 5000  # number of samples
m = 500  # number of groups
np.random.seed(1)
# simulate grouped random effects
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
b1 = np.random.normal(size=m)  # simulate random effects
eps = b1[group]
eps = eps - np.mean(eps)
X = np.column_stack((np.ones(n), np.random.uniform(size=n) - 0.5))  # desing matrix / covariate data for fixed effect
beta = np.array([0, 3])  # regression coefficents
f = X.dot(beta)  # fixed effects
# simulate response variable
if likelihood == "bernoulli_probit":
    probs = stats.norm.cdf(f + eps)
    y = np.random.uniform(size=n) < probs
    y = y.astype(np.float64)
elif likelihood == "bernoulli_logit":
    probs = 1 / (1 + np.exp(-(f + eps)))
    y = np.random.uniform(size=n) < probs
    y = y.astype(np.float64)
elif likelihood == "poisson":
    mu = np.exp(f + eps)
    y = stats.poisson.ppf(np.random.uniform(size=n), mu=mu)
elif likelihood == "gamma":
    mu = np.exp(f + eps)
    y = mu * stats.gamma.ppf(np.random.uniform(size=n), a=1)
plt.hist(y, bins=50)  # visualize response variable

# --------------------Train model----------------
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
gp_model.fit(y=y, X=X)  # use params={"trace": True} for monitoring convergence
gp_model.summary()

# --------------------Make predictions----------------
group_test = np.arange(m)
X_test = np.column_stack((np.ones(m), np.zeros(m)))
# Predict latent variable
pred = gp_model.predict(X_pred=X_test, group_data_pred=group_test,
                        predict_var=True, predict_response=False)
print(pred['mu'][0:5])  # Predicted latent mean
print(pred['var'][0:5])  # Predicted latent variance
# Predict response variable
pred_resp = gp_model.predict(X_pred=X_test, group_data_pred=group_test,
                             predict_var=True, predict_response=True)
print(pred_resp['mu'][0:5])  # Predicted response variable (label)
print(pred_resp['var'][0:5])  # # Predicted variance of response

# --------------------Approximate p-values for fixed effects coefficients----------------
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
gp_model.fit(y=y, X=X, params={"std_dev": True})
coefs = gp_model.get_coef()
z_values = coefs[0] / coefs[1]
p_values = 2 * stats.norm.cdf(-np.abs(z_values))
print(p_values)

#--------------------Saving a GPModel and loading it from a file----------------
# Save trained model
gp_model.save_model('gp_model.json')
# Load from file and make predictions again
gp_model_loaded = gpb.GPModel(model_file = 'gp_model.json')
pred_loaded = gp_model_loaded.predict(X_pred=X_test, group_data_pred=group_test,
                                      predict_var=True, predict_response=False)
pred_resp_loaded = gp_model_loaded.predict(X_pred=X_test, group_data_pred=group_test,
                                           predict_var=True, predict_response=True)
# Check equality
print(np.sum(np.abs(pred['mu'] - pred_loaded['mu'])))
print(np.sum(np.abs(pred['var'] - pred_loaded['var'])))
print(np.sum(np.abs(pred_resp['mu'] - pred_resp_loaded['mu'])))
print(np.sum(np.abs(pred_resp['var'] - pred_resp_loaded['var'])))


# --------------------Gaussian process model----------------
# Simulate data
ntrain = 500  # number of samples
np.random.seed(2)
# training and test locations (=features) for Gaussian process
coords_train = np.column_stack((np.random.uniform(size=ntrain), np.random.uniform(size=ntrain)))
# exclude upper right corner
excl = ((coords_train[:, 0] >= 0.7) & (coords_train[:, 1] >= 0.7))
coords_train = coords_train[~excl, :]
ntrain = coords_train.shape[0]
nx = 30  # test data: number of grid points on each axis
coords_test_aux = np.arange(0, 1, 1 / nx)
coords_test_x1, coords_test_x2 = np.meshgrid(coords_test_aux, coords_test_aux)
coords_test = np.column_stack((coords_test_x1.flatten(), coords_test_x2.flatten()))
coords = np.row_stack((coords_train, coords_test))
ntest = nx * nx
n = ntrain + ntest
# Simulate spatial Gaussian process
sigma2_1 = 1  # marginal variance of GP
rho = 0.1  # range parameter
D = np.zeros((n, n))  # distance matrix
for i in range(0, n):
    for j in range(i + 1, n):
        D[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])
        D[j, i] = D[i, j]
Sigma = sigma2_1 * np.exp(-D / rho) + np.diag(np.zeros(n) + 1e-20)
C = np.linalg.cholesky(Sigma)
b1 = np.random.normal(size=n)  # simulate random effects
eps = C.dot(b1)
eps = eps - np.mean(eps)
# simulate response variable
if likelihood == "bernoulli_probit":
    probs = stats.norm.cdf(eps)
    y = np.random.uniform(size=n) < probs
    y = y.astype(np.float64)
elif likelihood == "bernoulli_logit":
    probs = 1 / (1 + np.exp(-eps))
    y = np.random.uniform(size=n) < probs
    y = y.astype(np.float64)
elif likelihood == "poisson":
    mu = np.exp(eps)
    y = stats.poisson.ppf(np.random.uniform(size=n), mu=mu)
elif likelihood == "gamma":
    mu = np.exp(eps)
    y = mu * stats.gamma.ppf(np.random.uniform(size=n), a=1)
# Split into training and test data
y_train = y[0:ntrain]
y_test = y[ntrain:n]
eps_test = eps[ntrain:n]
plt.hist(y_train, bins=50)  # visualize response variable

# Train model
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential", likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()

# Prediction of latent variable
pred = gp_model.predict(gp_coords_pred=coords_test,
                        predict_var=True, predict_response=False)
# Predict response variable (label)
pred_resp = gp_model.predict(gp_coords_pred=coords_test,
                             predict_var=True, predict_response=True)
if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    print("Test error:")
    pred_binary = pred_resp['mu'] > 0.5
    pred_binary = pred_binary.astype(np.float64)
    print(np.mean(pred_binary != y_test))
else:
    print("Test root mean square error:")
    print(np.sqrt(np.mean((pred_resp['mu'] - y_test) ** 2)))

# Visualize predictions and compare to true values
fig, axs = plt.subplots(2, 2)
# data and true GP
eps_test_plot = eps_test.reshape((nx, nx))
CS = axs[0, 0].contourf(coords_test_x1, coords_test_x2, eps_test_plot)
axs[0, 0].plot(coords_train[:, 0], coords_train[:, 1], '+', color="white")
axs[0, 0].set_title("True latent GP and training locations")
# predicted latent mean
pred_mu_plot = pred['mu'].reshape((nx, nx))
CS = axs[0, 1].contourf(coords_test_x1, coords_test_x2, pred_mu_plot)
axs[0, 1].set_title("Predicted latent GP mean")
# prediction uncertainty
pred_var_plot = pred['var'].reshape((nx, nx))
CS = axs[1, 0].contourf(coords_test_x1, coords_test_x2, pred_var_plot)
axs[1, 0].set_title("Predicted latent GP standard deviation")
