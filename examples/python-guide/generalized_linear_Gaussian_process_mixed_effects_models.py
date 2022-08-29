# -*- coding: utf-8 -*-
"""
Examples on how to do inference and prediction for generalized linear 
mixed effects models with various likelihoods:
    - "gaussian" (=regression)
    - "bernoulli" (=classification)
    - "poisson" and "gamma" (=Poisson and gamma regression)
and various random effects models:
    - grouped (aka clustered) random effects models including random slopes
    - Gaussian process (GP) models
    - combined GP and grouped random effects

Author: Fabio Sigrist
"""

import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
plt.style.use('ggplot')
print("It is recommended that the examples are run in interactive mode")

def simulate_response_variable(lp, rand_eff, likelihood):
    """Function that simulates response variable for various likelihoods"""
    n = len(rand_eff)
    if likelihood == "gaussian":
        xi = 0.25 * np.random.normal(size=n) # error term
        y = lp + rand_eff + xi
    elif likelihood == "bernoulli_probit":
        probs = stats.norm.cdf(lp + rand_eff)
        y = np.random.uniform(size=n) < probs
        y = y.astype(np.float64)
    elif likelihood == "bernoulli_logit":
        probs = 1 / (1 + np.exp(-(lp + rand_eff)))
        y = np.random.uniform(size=n) < probs
        y = y.astype(np.float64)
    elif likelihood == "poisson":
        mu = np.exp(lp + rand_eff)
        y = stats.poisson.ppf(np.random.uniform(size=n), mu=mu)
    elif likelihood == "gamma":
        mu = np.exp(lp + rand_eff)
        y = mu * stats.gamma.ppf(np.random.uniform(size=n), a=1)
    return y

# Choose likelihood: either "gaussian" (=regression), 
#                     "bernoulli_probit", "bernoulli_logit", (=classification)
#                     "poisson", or "gamma"
likelihood = "gaussian"

"""
Grouped random effects
"""
# --------------------Simulate data----------------
# Single-level grouped random effects
n = 1000  # number of samples
m = 100  # number of categories / levels for grouping variable
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
np.random.seed(1)
b = 1 * np.random.normal(size=m)  # simulate random effects
rand_eff = b[group]
rand_eff = rand_eff - np.mean(rand_eff)
# Simulate linear regression fixed effects
X = np.column_stack((np.ones(n), np.random.uniform(size=n) - 0.5)) # design matrix / covariate data for fixed effect
beta = np.array([0, 3]) # regression coefficents
lp = X.dot(beta)
y = simulate_response_variable(lp=lp, rand_eff=rand_eff, likelihood=likelihood)
hst = plt.hist(y, bins=20)  # visualize response variable
plt.show(block=False)
# Crossed grouped random effects and a random slope
x = np.random.uniform(size=n)  # covariate data for random slope
n_obs_gr = int(n / m)  # number of sampels per group
group_crossed = np.arange(n)  # grouping variable for second random effect
for i in range(m):
    group_crossed[(n_obs_gr * i):(n_obs_gr * (i + 1))] = np.arange(n_obs_gr)
b_crossed = 0.5 * np.random.normal(size=n_obs_gr)  # simulate random effects
b_random_slope = 0.75 * np.random.normal(size=m)
rand_eff = b[group] + b_crossed[group_crossed] + x * b_random_slope[group]
rand_eff = rand_eff - np.mean(rand_eff)
y_crossed_random_slope = simulate_response_variable(lp=lp, rand_eff=rand_eff, likelihood=likelihood)
# Nested grouped random effects
group_inner = np.arange(n)  # grouping variable for nested lower level random effects
for i in range(m):
    group_inner[int(i * n / m):int((i + 0.5) * n / m)] = 0
    group_inner[int((i + 0.5) * n / m):int((i + 1) * n / m)] = 1
# Create nested grouping variable 
# Note: you need version 0.7.9 or later to use the function 'get_nested_categories'
group_nested = gpb.get_nested_categories(group, group_inner)
b_nested = 1. * np.random.normal(size=len(np.unique(group_nested)))  # nested lower level random effects
rand_eff = b[group] + b_nested[group_nested]
rand_eff = rand_eff - np.mean(rand_eff)
y_nested = simulate_response_variable(lp=lp, rand_eff=rand_eff, likelihood=likelihood)

# --------------------Training----------------
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
gp_model.fit(y=y, X=X)
gp_model.summary()
# Get coefficients and variance/covariance parameters separately
gp_model.get_coef()
gp_model.get_cov_pars()
# Obtaining standard deviations and p-values for fixed effects coefficients ('std_dev = TRUE')
gp_model.fit(y=y, X=X, params={"std_dev": True})
gp_model.summary()

# Optional arguments for the 'params' argument of the 'fit' function:
# - monitoring convergence: "trace": True
# - calculate standard deviations: "std_dev": True
# - change optimization algorithm options (see below)
# For available optimization options, see
#   https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#optimization-parameters
#gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
#gp_model.fit(y=y, X=X, params={"trace": True, 
#                               "std_dev": True,
#                               "optimizer_cov": "gradient_descent", "lr_cov": 0.1,
#                               "use_nesterov_acc": True, "maxit": 100})

# --------------------Prediction----------------
group_test = np.array([1,2,-1])
X_test = np.column_stack((np.ones(len(group_test)),
                          np.random.uniform(size=len(group_test))))
# Predict latent variable
pred = gp_model.predict(X_pred=X_test, group_data_pred=group_test,
                        predict_var=True, predict_response=False)
print("Predicted latent mean and variance:")
print(pred['mu']) # Predicted latent mean
print(pred['var']) # Predicted latent variance
# Predict response variable (for Gaussian data, latent and response variable predictions are the same)
pred_resp = gp_model.predict(X_pred=X_test, group_data_pred=group_test,
                             predict_var=True, predict_response=True)
print("Predicted response variable mean and variance:")
print(pred_resp['mu']) # Predicted response variable (label)
print(pred_resp['var']) # Predicted variance of response

# --------------------Predict ("estimate") training data random effects----------------
all_training_data_random_effects = gp_model.predict_training_data_random_effects()
# The function 'predict_training_data_random_effects' returns predicted random effects for all data points.
# Unique random effects for every group can be obtained as follows
first_occurences = [np.where(group==i)[0][0] for i in np.unique(group)]
training_data_random_effects = all_training_data_random_effects.iloc[first_occurences]
print(training_data_random_effects[0:5])# Predicted training data random effects
# Compare true and predicted random effects
plt.scatter(b, training_data_random_effects)
plt.title("Comparison of true and predicted random effects")
plt.show(block=False)
# Adding the overall intercept gives the group-wise intercepts
group_wise_intercepts = gp_model.get_coef().iloc[0,0] + training_data_random_effects
# Alternatively, this can also be done as follows
#group_unique = np.unique(group)
#X_zero = np.column_stack((np.zeros(len(group_unique)), np.zeros(len(group_unique))))
#pred_random_effects = gp_model.predict(group_data_pred=group_unique, X_pred=X_zero)
#np.sum(np.abs(training_data_random_effects['Group_1'] - pred_random_effects['mu']))

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

# --------------------Two crossed random effects and a random slope----------------
# Define and train model
group_data = np.column_stack((group, group_crossed))
gp_model = gpb.GPModel(group_data=group_data, group_rand_coef_data=x,
                       ind_effect_group_rand_coef=[1], likelihood=likelihood)
# 'ind_effect_group_rand_coef=[1]' indicates that the random slope is for the first random effect
gp_model.fit(y=y_crossed_random_slope, X=X, params={"std_dev": True})
gp_model.summary()
# Prediction
pred = gp_model.predict(group_data_pred=group_data, group_rand_coef_data_pred=x, X_pred=X)

# Obtain predicted (="estimated") random effects for the training data
all_training_data_random_effects = gp_model.predict_training_data_random_effects()
first_occurences_1 = [np.where(group==i)[0][0] for i in np.unique(group)]
first_occurences_2 = [np.where(group_crossed==i)[0][0] for i in np.unique(group_crossed)]
pred_random_effects = all_training_data_random_effects.iloc[first_occurences_1,0]
pred_random_slopes = all_training_data_random_effects.iloc[first_occurences_1,2]
pred_random_effects_crossed = all_training_data_random_effects.iloc[first_occurences_2,1]
# Compare true and predicted random effects
plt.scatter(b, pred_random_effects, label="Random effects")
plt.scatter(b_random_slope, pred_random_slopes, label="Random slopes")
plt.scatter(b_crossed, pred_random_effects_crossed, label="Crossed random effects")
plt.legend()
plt.title("Comparison of true and predicted random effects")
plt.show(block=False)

# Random slope model in which an intercept random effect is dropped / not included
gp_model = gpb.GPModel(group_data=group_data, group_rand_coef_data=x,
                       ind_effect_group_rand_coef=[1], 
                       drop_intercept_group_rand_effect=[True,False], likelihood=likelihood)
# 'drop_intercept_group_rand_effect=[True,False]' indicates that the first categorical variable 
#   in group_data has no intercept random effect
gp_model.fit(y=y_crossed_random_slope, X=X, params={"std_dev": True})
gp_model.summary()

# --------------------Two nested random effects----------------
# First create nested random effects variable
group_nested = gpb.get_nested_categories(group, group_inner)
group_data = np.column_stack((group, group_nested))
gp_model = gpb.GPModel(group_data=group_data, likelihood=likelihood)
gp_model.fit(y=y_nested, X=X, params={"std_dev": True})
gp_model.summary()

# --------------------Using cluster_ids for independent realizations of random effects----------------
cluster_ids = np.zeros(n)
cluster_ids[int(n/2):n] = 1
gp_model = gpb.GPModel(group_data=group, cluster_ids=cluster_ids, likelihood=likelihood)
gp_model.fit(y=y, X=X, params={"std_dev": True})
gp_model.summary()
#Note: gives sames result in this example as when not using cluster_ids
#   since the random effects of different groups are independent anyway


"""
Gaussian processes
"""
# --------------------Simulate data----------------
ntrain = 600 # number of training samples
np.random.seed(2)
# training and test locations (=features) for Gaussian process
coords_train = np.column_stack((np.random.uniform(size=ntrain), np.random.uniform(size=ntrain)))
# exclude upper right corner
excl = ((coords_train[:, 0] >= 0.6) & (coords_train[:, 1] >= 0.6))
coords_train = coords_train[~excl, :]
ntrain = coords_train.shape[0]
nx = 30  # test data: number of grid points on each axis
coords_test_aux = np.arange(0, 1, 1 / nx)
coords_test_x1, coords_test_x2 = np.meshgrid(coords_test_aux, coords_test_aux)
coords_test = np.column_stack((coords_test_x1.flatten(), coords_test_x2.flatten()))
coords = np.row_stack((coords_train, coords_test))
ntest = nx * nx
n = ntrain + ntest
sigma2_1 = 1  # marginal variance of GP
rho = 0.1  # range parameter
D = np.zeros((n, n))  # distance matrix
for i in range(0, n):
    for j in range(i + 1, n):
        D[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])
        D[j, i] = D[i, j]
Sigma = sigma2_1 * np.exp(-D / rho) + np.diag(np.zeros(n) + 1e-20)
C = np.linalg.cholesky(Sigma)
b = C.dot(np.random.normal(size=n)) # simulate GP
b = b - np.mean(b)
y = simulate_response_variable(lp=0, rand_eff=b, likelihood=likelihood)
# Split into training and test data
y_train = y[0:ntrain]
y_test = y[ntrain:n]
b_train = b[0:ntrain]
b_test = b[ntrain:n]
hst = plt.hist(y_train, bins=50)  # visualize response variable
# Simulate linear regression fixed effects
X = np.column_stack((np.ones(ntrain), np.random.uniform(size=ntrain) - 0.5)) # design matrix / covariate data for fixed effect
beta = np.array([0, 3]) # regression coefficents
lp = X.dot(beta)
y_lin = simulate_response_variable(lp=lp, rand_eff=b_train, likelihood=likelihood)
# Spatially varying coefficient (random coefficient) model
X_SVC = np.column_stack(
    (np.random.uniform(size=ntrain), np.random.uniform(size=ntrain)))  # covariate data for random coefficients
b2 = C[0:ntrain,0:ntrain].dot(np.random.normal(size=ntrain))
b3 = C[0:ntrain,0:ntrain].dot(np.random.normal(size=ntrain))
# Note: for simplicity, we assume that all GPs have the same covariance parameters
rand_eff = b_train + X_SVC[:, 0] * b2 + X_SVC[:, 1] * b3
rand_eff = rand_eff - np.mean(rand_eff)
y_svc = simulate_response_variable(lp=0, rand_eff=rand_eff, likelihood=likelihood)

#--------------------Training----------------
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential",
                       likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()

# Other covariance functions:
# gp_model = gpb.GPModel(gp_coords=coords, cov_function="gaussian", likelihood=likelihood)
# gp_model = gpb.GPModel(gp_coords=coords, cov_function="matern", cov_fct_shape=1.5, likelihood=likelihood)
# gp_model = gpb.GPModel(gp_coords=coords, cov_function="powered_exponential", cov_fct_shape=1.1, likelihood=likelihood)

# Optional arguments for the 'params' argument of the 'fit' function:
# - monitoring convergence: "trace": True
# - obtain standard deviations: "std_dev": True
# - change optimization algorithm options (see below)
# For available optimization options, see
#   https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#optimization-parameters
#gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential", 
#                       likelihood=likelihood)
#gp_model.fit(y=y, X=X, params={"trace": True, 
#                               "std_dev": True,
#                               "optimizer_cov": "gradient_descent", "lr_cov": 0.1,
#                               "use_nesterov_acc": True, "maxit": 100})

#--------------------Prediction----------------
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
fig, axs = plt.subplots(2, 2, figsize=[10,8])
# data and true GP
b_test_plot = b_test.reshape((nx, nx))
CS = axs[0, 0].contourf(coords_test_x1, coords_test_x2, b_test_plot)
axs[0, 0].plot(coords_train[:, 0], coords_train[:, 1], '+', color="white", 
   markersize = 4)
axs[0, 0].set_title("True latent GP and training locations")
# predicted latent mean
pred_mu_plot = pred['mu'].reshape((nx, nx))
CS = axs[0, 1].contourf(coords_test_x1, coords_test_x2, pred_mu_plot)
axs[0, 1].set_title("Predicted latent GP mean")
# prediction uncertainty
pred_var_plot = pred['var'].reshape((nx, nx))
CS = axs[1, 0].contourf(coords_test_x1, coords_test_x2, pred_var_plot)
axs[1, 0].set_title("Predicted latent GP standard deviation")

# Predict latent GP at training data locations (=smoothing)
GP_smooth = gp_model.predict_training_data_random_effects()
# Compare true and predicted random effects
plt.scatter(b_train, GP_smooth)
plt.title("Comparison of true and smoothed GP")
# The above is equivalent to the following
#GP_smooth2 = gp_model.predict(gp_coords_pred=coords_train)
#np.sum(np.abs(GP_smooth['GP'] - GP_smooth2['mu']))

#--------------------Gaussian process model with linear mean function----------------
# Include a liner regression term instead of assuming a zero-mean a.k.a. "universal Kriging"
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential",
                       likelihood=likelihood)
gp_model.fit(y=y_lin, X=X, params={"std_dev": True})
gp_model.summary()

# --------------------Gaussian process model with Vecchia approximation----------------
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential",
                       vecchia_approx=True, num_neighbors=30, likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()

#--------------------Gaussian process model with Wendland covariance function----------------
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="wendland",
                       cov_fct_shape=1, cov_fct_taper_range=0.1, likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()

# --------------------Gaussian process model with random coefficents----------------
# Define and train model
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential", 
                       gp_rand_coef_data=X_SVC, likelihood=likelihood)
gp_model.fit(y=y_svc) # takes some time for non-Gaussian data
pd.set_option('display.max_columns', None)
gp_model.summary()
# Note: this is a small sample size for this type of model
#   -> covariance parameters estimates can have high variance

# Predict latent GP at training data locations (=smoothing)
GP_smooth = gp_model.predict_training_data_random_effects()
# Compare true and predicted random effects
plt.scatter(b_train, GP_smooth['GP'], label="Intercept GP", alpha=0.5)
plt.scatter(b2, GP_smooth['GP_rand_coef_nb_1'], label="1. random coef. GP", alpha=0.5)
plt.scatter(b3, GP_smooth['GP_rand_coef_nb_2'], label="2. random coef. GP", alpha=0.5)
plt.legend()
plt.title("Comparison of true and smoothed GP")

# --------------------Using cluster_ids for independent realizations of GPs----------------
cluster_ids = np.zeros(ntrain)
cluster_ids[int(ntrain/2):ntrain] = 1
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential",
                       cluster_ids=cluster_ids, likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()

# --------------------Evaluate negative log-likelihood----------------
if likelihood == "gaussian":
  cov_pars = [0.1,sigma2_1,rho]
else:
  cov_pars = [sigma2_1,rho]
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential",
                       likelihood=likelihood)
gp_model.neg_log_likelihood(cov_pars=cov_pars, y=y_train)


"""
Combined Gaussian process and grouped random effects
"""
# Simulate data
n = 500  # number of samples
m = 50  # number of categories / levels for grouping variable
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
D = np.zeros((n, n))  # distance matrix
for i in range(0, n):
    for j in range(i + 1, n):
        D[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])
        D[j, i] = D[i, j]
Sigma = sigma2_2 * np.exp(-D / rho) + np.diag(np.zeros(n) + 1e-20)
C = np.linalg.cholesky(Sigma)
b1 = np.random.normal(size=m)  # simulate random effect
b2 = C.dot(np.random.normal(size=n))
rand_eff = Z1.dot(b1) + b2
rand_eff = rand_eff - np.mean(rand_eff)
y_comb = simulate_response_variable(lp=0, rand_eff=rand_eff, likelihood=likelihood)
# Define and train model
gp_model = gpb.GPModel(group_data=group, gp_coords=coords, 
                       cov_function="exponential", likelihood=likelihood)
gp_model.fit(y=y_comb)
gp_model.summary()
