# coding: utf-8
# pylint: disable = invalid-name, C0111
import gpboost as gpb
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# --------------------Grouped random effects model: single-level random effect----------------
# Simulate data
n = 100  # number of samples
m = 25  # number of categories / levels for grouping variable
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
# incidence matrix relating grouped random effects to samples
Z1 = np.zeros((n, m))
for i in range(m):
    Z1[np.where(group == i), i] = 1
sigma2_1 = 1 ** 2  # random effect variance
sigma2 = 0.5 ** 2  # error variance
np.random.seed(1)
b1 = np.sqrt(sigma2_1) * np.random.normal(size=m)  # simulate random effects
eps = Z1.dot(b1)
xi = np.sqrt(sigma2) * np.random.normal(size=n)  # simulate error term
y = eps + xi  # observed data

# Define and fit model
gp_model = gpb.GPModel(group_data=group)
gp_model.fit(y=y, std_dev=True)
gp_model.summary()

# Make predictions
group_test = np.arange(m)
pred = gp_model.predict(group_data_pred=group_test)
# Compare true and predicted random effects
plt.scatter(b1, pred['mu'])
plt.title("Comparison of true and predicted random effects")
plt.xlabel("truth")
plt.ylabel("predicted")
plt.show()

# Other optimization specifications (gradient descent with Nesterov acceleration)
gp_model = gpb.GPModel(group_data=group)
gp_model.fit(y=y, std_dev=True, params={"optimizer_cov": "gradient_descent", "lr_cov": 0.1,
                                        "use_nesterov_acc": True})
gp_model.summary()

# --------------------Two crossed random effects and a random slope----------------
# NOTE: run the above example first to create the first random effect
# Simulate data
np.random.seed(1)
x = np.random.uniform(size=n)  # covariate data for random slope
n_obs_gr = int(n / m)  # number of sampels per group
group2 = np.arange(n)  # grouping variable for second random effect
for i in range(m):
    group2[(n_obs_gr * i):(n_obs_gr * (i + 1))] = np.arange(n_obs_gr)
# incidence matrix relating grouped random effects to samples
Z2 = np.zeros((n, n_obs_gr))
for i in range(n_obs_gr):
    Z2[np.where(group2 == i), i] = 1
Z3 = np.diag(x).dot(Z1)
sigma2_2 = 0.5 ** 2  # variance of second random effect
sigma2_3 = 0.75 ** 2  # variance of random slope for first random effect
b2 = np.sqrt(sigma2_2) * np.random.normal(size=n_obs_gr)  # simulate random effects
b3 = np.sqrt(sigma2_3) * np.random.normal(size=m)
eps2 = Z1.dot(b1) + Z2.dot(b2) + Z3.dot(b3)
y = eps2 + xi  # observed data
# Define and fit model
group_data = np.column_stack((group, group2))
gp_model = gpb.GPModel(group_data=group_data, group_rand_coef_data=x, ind_effect_group_rand_coef=[1])
gp_model.fit(y=y, std_dev=True)
gp_model.summary()

# --------------------Mixed effects model: random effects and linear fixed effects----------------
# NOTE: run the above example first to create the random effects part
# Simulate data
np.random.seed(1)
X = np.column_stack(
    (np.random.uniform(size=n), np.random.uniform(size=n)))  # desing matrix / covariate data for fixed effect
beta = np.array([3, 3])  # regression coefficents
y = eps2 + xi + X.dot(beta)  # add fixed effect to observed data
# Define and fit model
gp_model = gpb.GPModel(group_data=group_data, group_rand_coef_data=x, ind_effect_group_rand_coef=[1])
gp_model.fit(y=y, X=X, std_dev=True)
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
gp_model.fit(y=y, std_dev=True, params={"optimizer_cov": "gradient_descent",
                                        "lr_cov": 0.1})
gp_model.summary()

# Make predictions
np.random.seed(1)
ntest = 5
# prediction locations (=features) for Gaussian process
coords_test = np.column_stack(
    (np.random.uniform(size=ntest), np.random.uniform(size=ntest))) / 10.
pred = gp_model.predict(gp_coords_pred=coords_test, predict_cov_mat=True)
print("Predicted (posterior/conditional) mean of GP")
pred['mu']
print("Predicted (posterior/conditional) covariance matrix of GP")
pred['cov']

# --------------------Gaussian process model with Vecchia approximation----------------
gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential",
                       vecchia_approx=True, num_neighbors=30)
gp_model.fit(y=y, params={"optimizer_cov": "gradient_descent",
                          "lr_cov": 0.1})
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
gp_model.fit(y=y, std_dev=True, params={"optimizer_cov": "gradient_descent",
                                        "lr_cov": 0.05,
                                        "use_nesterov_acc": True,
                                        "acc_rate_cov": 0.5})
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

# Create Gaussian process model
gp_model = gpb.GPModel(group_data=group, gp_coords=coords, cov_function="exponential")
gp_model.fit(y=y, std_dev=True, params={"optimizer_cov": "gradient_descent",
                                        "lr_cov": 0.05,
                                        "use_nesterov_acc": True,
                                        "acc_rate_cov": 0.5})
gp_model.summary()
