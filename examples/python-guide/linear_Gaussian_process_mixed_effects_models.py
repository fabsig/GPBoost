# coding: utf-8
# pylint: disable = invalid-name, C0111
import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
print("It is recommended that the examples are run in interactive mode")

"""
This script contains various examples on how to do inference and prediction for
  - grouped (or clustered) random effects models
  - Gaussian process (GP) models
  - models that combine GP and grouped random effects
and on how to save models
"""

# --------------------Grouped random effects model: single-level random effect----------------
# Simulate data
n = 100  # number of samples
m = 25  # number of categories / levels for grouping variable
np.random.seed(1)
# simulate grouped random effects
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
b1 = 1 * np.random.normal(size=m)  # simulate random effects
eps = b1[group]
# simulate fixed effects
X = np.column_stack(
    (np.ones(n), np.random.uniform(size=n)))  # design matrix / covariate data for fixed effect
beta = np.array([0, 3])  # regression coefficents
xi = np.sqrt(0.01) * np.random.normal(size=n)  # simulate error term
y = eps + xi + X.dot(beta) # observed data

# --------------------Training----------------
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
gp_model.fit(y=y, X=X, params={"std_dev": True})
gp_model.summary()
# Other optimization specifications: gradient descent (without Nesterov acceleration)
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
gp_model.fit(y=y, X=X, params={"optimizer_cov": "gradient_descent", "lr_cov": 0.1,
                          "std_dev": True, "use_nesterov_acc": False, "maxit": 100})
gp_model.summary()

# --------------------Prediction----------------
group_test = np.array([1,2,-1])
X_test = np.column_stack((np.ones(len(group_test)), np.random.uniform(size=len(group_test))))
pred = gp_model.predict(group_data_pred=group_test, X_pred=X_test, predict_var = True)
pred['mu']# Predicted mean
pred['var']# Predicted variances

# Evaluate negative log-likelihood
gp_model.neg_log_likelihood(cov_pars=np.array([1, 0.1]), y=y)

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
# NOTE: run the above example first to create the first random effect
# Simulate data
np.random.seed(1)
x = np.random.uniform(size=n)  # covariate data for random slope
n_obs_gr = int(n / m)  # number of sampels per group
group2 = np.arange(n)  # grouping variable for second random effect
for i in range(m):
    group2[(n_obs_gr * i):(n_obs_gr * (i + 1))] = np.arange(n_obs_gr)
b2 = 0.5 * np.random.normal(size=n_obs_gr)  # simulate random effects
b3 = 0.75 * np.random.normal(size=m)
eps2 = b1[group] + b2[group2] + x * b3[group]
y = eps2 + xi  # observed data
# Define and fit model
group_data = np.column_stack((group, group2))
gp_model = gpb.GPModel(group_data=group_data, group_rand_coef_data=x, ind_effect_group_rand_coef=[1])
gp_model.fit(y=y, params={"std_dev": True})
gp_model.summary()


# --------------------Two nested random effects----------------
n = 1000  # number of samples
m1 = 50  # number of categories / levels for the first grouping variable
m2 = 200  # number of categories / levels for the second nested grouping variable
group1 = np.arange(n)  # grouping variable for higher level random effects
for i in range(m1):
    group1[int(i * n / m1):int((i + 1) * n / m1)] = i
group2 = np.arange(n)  # grouping variable for nested lower level random effects
for i in range(m2):
    group2[int(i * n / m2):int((i + 1) * n / m2)] = i
np.random.seed(20)
# simulate random effects
b1 = 1. * np.random.normal(size=m1)  # higher level random effects
b2 = 1. * np.random.normal(size=m2)  # nested lower level random effects
eps = b1[group1] + b2[group2]
xi = 0.5 * np.random.normal(size=n)  # simulate error term
y = eps + xi  # observed data
# Define and fit model
group_data = np.column_stack((group1, group2))
gp_model = gpb.GPModel(group_data=group_data)
gp_model.fit(y=y, params={"std_dev": True})
gp_model.summary()


# --------------------Gaussian process model----------------
# Simulate data
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

# Define and fit model
gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential")
## Other covariance functions:
# gp_model = gpb.GPModel(gp_coords=coords, cov_function="gaussian")
# gp_model = gpb.GPModel(gp_coords=coords, cov_function="matern", cov_fct_shape=1.5)
# gp_model = gpb.GPModel(gp_coords=coords, cov_function="powered_exponential", cov_fct_shape=1.1)
gp_model.fit(y=y, params={"std_dev": True})
gp_model.summary()

# Make predictions
np.random.seed(1)
ntest = 5
# prediction locations (=features) for Gaussian process
coords_test = np.column_stack(
    (np.random.uniform(size=ntest), np.random.uniform(size=ntest))) / 10.
pred = gp_model.predict(gp_coords_pred=coords_test, predict_cov_mat=True)
# Predicted (posterior/conditional) mean of GP
pred['mu']
# Predicted (posterior/conditional) covariance matrix of GP
pred['cov']

# Evaluate negative log-likelihood
gp_model.neg_log_likelihood(cov_pars=np.array([sigma2, sigma2_1, rho]), y=y)

# Other optimization specifications (gradient descent with Nesterov acceleration)
gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential")
gp_model.fit(y=y, params={"optimizer_cov": "gradient_descent", "std_dev": True,
                          "lr_cov": 0.05, "use_nesterov_acc": True})
gp_model.summary()

# --------------------Gaussian process model with Vecchia approximation----------------
gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential",
                       vecchia_approx=True, num_neighbors=30)
gp_model.fit(y=y)
gp_model.summary()

# --------------------Gaussian process model with random coefficents----------------
# Simulate data
n = 500  # number of samples
np.random.seed(1)
coords = np.column_stack(
    (np.random.uniform(size=n), np.random.uniform(size=n)))  # locations (=features) for Gaussian process
sigma2_1 = 1 ** 2  # marginal variance of GP (for simplicity, all GPs have the same parameters)
rho = 0.1  # range parameter
sigma2 = 0.5 ** 2  # error variance
D = np.zeros((n, n))  # distance matrix
for i in range(0, n):
    for j in range(i + 1, n):
        D[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])
        D[j, i] = D[i, j]
Sigma = sigma2_1 * np.exp(-D / rho) + np.diag(np.zeros(n) + 1e-20)
C = np.linalg.cholesky(Sigma)
X_SVC = np.column_stack(
    (np.random.uniform(size=n), np.random.uniform(size=n)))  # covariate data for random coeffients
b1 = np.random.normal(size=n)  # simulate random effect
b2 = np.random.normal(size=n)
b3 = np.random.normal(size=n)
eps = C.dot(b1) + X_SVC[:, 0] * C.dot(b2) + X_SVC[:, 1] * C.dot(b3)
xi = np.sqrt(sigma2) * np.random.normal(size=n)  # simulate error term
y = eps + xi
# Define and fit model (takes a few seconds)
gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential", gp_rand_coef_data=X_SVC)
gp_model.fit(y=y, params={"std_dev": True})
gp_model.summary()


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

# Define and fit model
gp_model = gpb.GPModel(group_data=group, gp_coords=coords, cov_function="exponential")
gp_model.fit(y=y, params={"std_dev": True})
gp_model.summary()
