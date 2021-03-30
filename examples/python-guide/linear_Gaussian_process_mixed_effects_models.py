# -*- coding: utf-8 -*-
"""
Various examples on how to do inference and prediction for
  (i) grouped (or clustered) random effects models
  (ii) Gaussian process (GP) models
  (iii) models that combine GP and grouped random effects
and on how to save models

@author: Fabio Sigrist
"""

import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
print("It is recommended that the examples are run in interactive mode")

# --------------------Simulate data----------------
n = 1000  # number of samples
# Simulate single level grouped random effects data
m = 100  # number of categories / levels for grouping variable
np.random.seed(1)
# Simulate grouped random effects
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
b = 1 * np.random.normal(size=m)  # simulate random effects
eps = b[group]
# Simulate fixed effects
X = np.column_stack(
    (np.ones(n), np.random.uniform(size=n)))  # design matrix / covariate data for fixed effect
beta = np.array([0, 3])  # regression coefficents
xi = np.sqrt(0.01) * np.random.normal(size=n)  # simulate error term
y = eps + xi + X.dot(beta) # observed data
# Simulate data for two crossed random effects and a random slope
x = np.random.uniform(size=n)  # covariate data for random slope
n_obs_gr = int(n / m)  # number of sampels per group
group_crossed = np.arange(n)  # grouping variable for second random effect
for i in range(m):
    group_crossed[(n_obs_gr * i):(n_obs_gr * (i + 1))] = np.arange(n_obs_gr)
b_crossed = 0.5 * np.random.normal(size=n_obs_gr)  # simulate random effects
b_random_slope = 0.75 * np.random.normal(size=m)
y_crossed_random_slope = b[group] + b_crossed[group_crossed] + x * b_random_slope[group] + xi
# Simulate data for two nested random effects
m_nested = 200  # number of categories / levels for the second nested grouping variable
group_nested = np.arange(n)  # grouping variable for nested lower level random effects
for i in range(m_nested):
    group_nested[int(i * n / m_nested):int((i + 1) * n / m_nested)] = i
b_nested = 1. * np.random.normal(size=m_nested)  # nested lower level random effects
y_nested = b[group] + b_nested[group_nested] + xi  # observed data


# --------------------Grouped random effects model: single-level random effect----------------
# --------------------Training----------------
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
gp_model.fit(y=y, X=X, params={"std_dev": True})
gp_model.summary()
# Use other optimization specifications (gradient descent with Nesterov acceleration)
# and monitor convergence of optimization ("trace": True)
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
gp_model.fit(y=y, X=X, params={"optimizer_cov": "gradient_descent", "lr_cov": 0.1,
                               "std_dev": True, "use_nesterov_acc": True,
                               "maxit": 100, "trace": True})
gp_model.summary()

# --------------------Prediction----------------
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
gp_model.fit(y=y, X=X)

group_test = np.array([1,2,-1])
X_test = np.column_stack((np.ones(len(group_test)), np.random.uniform(size=len(group_test))))
pred = gp_model.predict(group_data_pred=group_test, X_pred=X_test, predict_var = True)
print(pred['mu'])# Predicted mean
print(pred['var'])# Predicted variances

#--------------------Saving a GPModel and loading it from a file----------------
# Train model and make predictions
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
gp_model.fit(y=y, X=X)
group_test = np.array([1,2,-1])
X_test = np.column_stack((np.ones(len(group_test)), np.random.uniform(size=len(group_test))))
pred = gp_model.predict(group_data_pred=group_test, X_pred=X_test, predict_var = True)
# Save model to file
gp_model.save_model('gp_model.json')
# Load from file and make predictions again
gp_model_loaded = gpb.GPModel(model_file = 'gp_model.json')
pred_loaded = gp_model_loaded.predict(group_data_pred=group_test, X_pred=X_test, predict_var = True)
# Check equality
print(pred['mu'] - pred_loaded['mu'])
print(pred['var'] - pred_loaded['var'])

# --------------------Two crossed random effects and a random slope----------------
# Define and train model
group_data = np.column_stack((group, group_crossed))
gp_model = gpb.GPModel(group_data=group_data, group_rand_coef_data=x, ind_effect_group_rand_coef=[1])
gp_model.fit(y=y_crossed_random_slope, params={"std_dev": True})
gp_model.summary()

# --------------------Two nested random effects----------------
# Define and train model
group_data = np.column_stack((group, group_nested))
gp_model = gpb.GPModel(group_data=group_data)
gp_model.fit(y=y_nested, params={"std_dev": True})
gp_model.summary()

# --------------------Using cluster_ids for independent realizations of random effects----------------
# Define and train model
cluster_ids = np.zeros(n)
cluster_ids[int(n/2):n] = 1
gp_model = gpb.GPModel(group_data=group, cluster_ids=cluster_ids)
gp_model.fit(y=y, X=X, params={"std_dev": True})
gp_model.summary()
#Note: gives sames result in this example as when not using cluster_ids (see above)
#   since the random effects of different groups are independent anyway


# --------------------Gaussian process model----------------
#--------------------Simulate data----------------
n = 200  # number of samples
np.random.seed(2)
coords = np.column_stack(
    (np.random.uniform(size=n), np.random.uniform(size=n)))  # locations (=features) for Gaussian process
sigma2_1 = 1 ** 2  # marginal variance of GP
rho = 0.1  # range parameter
sigma2 = 0.5 ** 2  # error variance
D = np.zeros((n, n))  # distance matrix
for i in range(0, n):
    for j in range(i + 1, n):
        D[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])
        D[j, i] = D[i, j]
Sigma = sigma2_1 * np.exp(-D / rho) + np.diag(np.zeros(n) + 1e-20)
C = np.linalg.cholesky(Sigma)
b1 = np.random.normal(size=n)  # simulate random effects
eps = C.dot(b1)
xi = np.sqrt(sigma2) * np.random.normal(size=n)  # simulate error term
y = eps + xi
# Simulate spatially varying coefficient (random coefficient) model data
X_SVC = np.column_stack(
    (np.random.uniform(size=n), np.random.uniform(size=n)))  # covariate data for random coeffients
b2 = np.random.normal(size=n)
b3 = np.random.normal(size=n)
# Note: for simplicity, we assume that all GPs have the same covariance parameters
y_svc = C.dot(b1) + X_SVC[:, 0] * C.dot(b2) + X_SVC[:, 1] * C.dot(b3) + xi

#--------------------Training----------------
gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential")
## Other covariance functions:
# gp_model = gpb.GPModel(gp_coords=coords, cov_function="gaussian")
# gp_model = gpb.GPModel(gp_coords=coords, cov_function="matern", cov_fct_shape=1.5)
# gp_model = gpb.GPModel(gp_coords=coords, cov_function="powered_exponential", cov_fct_shape=1.1)
gp_model.fit(y=y, params={"std_dev": True})
gp_model.summary()

# Other optimization specifications (gradient descent with Nesterov acceleration)
gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential")
gp_model.fit(y=y, params={"optimizer_cov": "gradient_descent", "std_dev": True,
                          "lr_cov": 0.05, "use_nesterov_acc": True})
gp_model.summary()

#--------------------Prediction----------------
np.random.seed(1)
ntest = 5
# prediction locations (=features) for Gaussian process
coords_test = np.column_stack(
    (np.random.uniform(size=ntest), np.random.uniform(size=ntest))) / 10.
pred = gp_model.predict(gp_coords_pred=coords_test, predict_cov_mat=True)
# Predicted (posterior/conditional) mean of GP
print(pred['mu'])
# Predicted (posterior/conditional) covariance matrix of GP
print(pred['cov'])

# --------------------Gaussian process model with Vecchia approximation----------------
gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential",
                       vecchia_approx=True, num_neighbors=30)
gp_model.fit(y=y)
gp_model.summary()

#--------------------Gaussian process model with Wendland covariance function----------------
gp_model = gpb.GPModel(gp_coords=coords, cov_function="wendland",
                       cov_fct_shape=1, cov_fct_taper_range=0.1)
gp_model.fit(y=y)
gp_model.summary()

# --------------------Gaussian process model with random coefficents----------------
# Define and train model
gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential", gp_rand_coef_data=X_SVC)
gp_model.fit(y=y_svc, params={"std_dev": True})
gp_model.summary()
# Note: this is a small sample size for this type of model
#   -> covariance parameters estimates can have high variance


# --------------------Combine Gaussian process with grouped random effects----------------
n = 200  # number of samples
m = 25  # number of categories / levels for grouping variable
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
# incidence matrix relating grouped random effects to samples
Z1 = np.zeros((n, m))
for i in range(m):
    Z1[np.where(group == i), i] = 1
np.random.seed(1)
coords = np.column_stack(
    (np.random.uniform(size=n), np.random.uniform(size=n)))  # locations (=features) for Gaussian process
sigma2_1 = 1 ** 2  # random effect variance
sigma2_2 = 1 ** 2  # marginal variance of GP
rho = 0.1  # range parameter
sigma2 = 0.5 ** 2  # error variance
D = np.zeros((n, n))  # distance matrix
for i in range(0, n):
    for j in range(i + 1, n):
        D[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])
        D[j, i] = D[i, j]
Sigma = sigma2_2 * np.exp(-D / rho) + np.diag(np.zeros(n) + 1e-20)
C = np.linalg.cholesky(Sigma)
b1 = np.random.normal(size=m)  # simulate random effect
b2 = np.random.normal(size=n)
eps = Z1.dot(b1) + C.dot(b2)
xi = np.sqrt(sigma2) * np.random.normal(size=n)  # simulate error term
y = eps + xi

# Define and train model
gp_model = gpb.GPModel(group_data=group, gp_coords=coords, cov_function="exponential")
gp_model.fit(y=y, params={"std_dev": True})
gp_model.summary()
