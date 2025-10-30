# -*- coding: utf-8 -*-
"""
Examples on how to do inference and prediction for generalized linear 
mixed effects models with various likelihoods and different random effects models:
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
        xi = 0.1**0.5 * np.random.normal(size=n) # error term, variance = 0.1
        y = lp + rand_eff + xi
    elif likelihood == "binary_probit":
        probs = stats.norm.cdf(lp + rand_eff)
        y = np.random.uniform(size=n) < probs
        y = y.astype(np.float64)
    elif likelihood == "binary_logit":
        probs = 1 / (1 + np.exp(-(lp + rand_eff)))
        y = np.random.uniform(size=n) < probs
        y = y.astype(np.float64)
    elif likelihood == "poisson":
        mu = np.exp(lp + rand_eff)
        y = stats.poisson.ppf(np.random.uniform(size=n), mu=mu)
    elif likelihood == "gamma":
        mu = np.exp(lp + rand_eff)
        y = mu * stats.gamma.ppf(np.random.uniform(size=n), a=1)
    elif likelihood == "negative_binomial":
        mu = np.exp(lp + rand_eff)
        shape = 1.5
        p = shape / (shape + mu)
        y = stats.nbinom.ppf(np.random.uniform(size=n), p=p, n=shape)
    return y

# Choose likelihood: either "gaussian" (=regression), 
#                     "binary_probit", "binary_logit", (=classification)
#                     "poisson", "gamma", or "negative_binomial"
likelihood = "gaussian"

"""
Grouped random effects
"""
# --------------------Simulate data----------------
# Single-level grouped random effects
n = 1000  # number of samples
m = 200  # number of categories / levels for grouping variable
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
np.random.seed(1)
b = 0.25**0.5 * np.random.normal(size=m)  # simulate random effects, variance = 0.25
rand_eff = b[group]
rand_eff = rand_eff - np.mean(rand_eff)
# Simulate linear regression fixed effects
X = np.column_stack((np.ones(n), np.random.uniform(size=n) - 0.5)) # design matrix / covariate data for fixed effect
beta = np.array([0, 2]) # regression coefficents
lp = X.dot(beta)
y = simulate_response_variable(lp=lp, rand_eff=rand_eff, likelihood=likelihood)
hst = plt.hist(y, bins=20)  # visualize response variable
plt.show(block=False)
# Crossed grouped random effects and random slopes
group_crossed = group[np.random.permutation(n)-1] # grouping variable for crossed random effects
b_crossed = 0.25**0.5 * np.random.normal(size=m)  # simulate crossed random effects
b_random_slope = 0.25**0.5 * np.random.normal(size=m)
x = np.random.uniform(size=n)  # covariate data for random slope
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
b_nested = 0.25**0.5 * np.random.normal(size=len(np.unique(group_nested))) # simulate nested random effects
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
# - turning off calculation of standard deviations: "std_dev": False
# - change optimization algorithm options
# - manually set initial values for parameters
# - choose which parameters are estimates
# For more information, see
#   https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#optimization-parameters

gp_model.fit(y=y, X=X, params={"std_dev": False})
gp_model.summary()



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
all_training_data_random_effects = gp_model.predict_training_data_random_effects(predict_var=True)
# The function 'predict_training_data_random_effects' returns predicted random effects for all data points.
# Unique random effects for every group can be obtained as follows
first_occurences = [np.where(group==i)[0][0] for i in np.unique(group)]
training_data_random_effects = all_training_data_random_effects.iloc[first_occurences]
print(training_data_random_effects.head()) # Training data random effects: predictive means and variances
# Compare true and predicted random effects
plt.scatter(b, training_data_random_effects.iloc[:,0])
plt.title("Comparison of true and predicted random effects")
plt.show(block=False)
# Adding the overall intercept gives the group-wise intercepts
group_wise_intercepts = gp_model.get_coef().iloc[0,0] + training_data_random_effects
# The above is equivalent to the following:
# group_unique = np.unique(group)
# x_zero = np.column_stack((np.zeros(len(group_unique)), np.zeros(len(group_unique))))
# pred_random_effects = gp_model.predict(group_data_pred=group_unique, X_pred=x_zero, 
#                                        predict_response=False, predict_var=True)
# print(np.sum(np.abs(training_data_random_effects['Group_1'] - pred_random_effects['mu'])))
# print(np.sum(np.abs(training_data_random_effects['Group_1_var'] - pred_random_effects['var'])))

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

# --------------------Two crossed random effects and random slopes----------------
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

# --------------------Evaluate negative log-likelihood----------------
if likelihood == "gaussian":
  cov_pars = [0.1 ,0.1]
else:
  cov_pars = [0.1]
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
coef = [0, 0.1]
fixed_effects = X.dot(coef)
gp_model.neg_log_likelihood(cov_pars=cov_pars, y=y, fixed_effects=fixed_effects)


"""
Gaussian processes
"""
# --------------------Simulate data----------------
ntrain = 600 # number of training samples
np.random.seed(2)
# training and test locations (=features) for Gaussian process
coords_train = np.column_stack((np.random.uniform(size=ntrain), np.random.uniform(size=ntrain)))
# less data in one area
excl = ((coords_train[:, 0] >= 0.3) & (coords_train[:, 0] <= 0.7) & (coords_train[:, 1] >= 0.3) & 
        (coords_train[:, 1] <= 0.7) & (np.random.uniform(size=ntrain) > 0.1))
coords_train = coords_train[~excl, :]
ntrain = coords_train.shape[0]
nx = 30  # test data: number of grid points on each axis
coords_test_aux = np.arange(0, 1, 1 / nx)
coords_test_x1, coords_test_x2 = np.meshgrid(coords_test_aux, coords_test_aux)
coords_test = np.column_stack((coords_test_x1.flatten(), coords_test_x2.flatten()))
coords = np.vstack((coords_train, coords_test))
ntest = nx * nx
n = ntrain + ntest
sigma2_1 = 0.25  # marginal variance of GP
rho = 0.1  # range parameter
D = np.zeros((n, n))  # distance matrix
for i in range(0, n):
    for j in range(i + 1, n):
        D[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])
        D[j, i] = D[i, j]
D_scaled = 3**0.5 * D / rho
Sigma = sigma2_1 * (1. + D_scaled) * np.exp(-D_scaled) + np.diag(np.zeros(n) + 1e-20) # Matern 1.5 covariance
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
beta = np.array([0, 2]) # regression coefficents
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
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5,
                       likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()
# Optional arguments for the 'params' argument of the 'fit' function:
# - monitoring convergence: "trace": True
# - turning off calculation of standard deviations: "std_dev": False
# - change optimization algorithm options
# - manually set initial values for parameters
# - choose which parameters are estimates
# For more information, see
#   https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#optimization-parameters


#--------------------Prediction----------------
# Prediction of latent variable
pred = gp_model.predict(gp_coords_pred=coords_test,
                        predict_var=True, predict_response=False)
# Predict response variable (label)
pred_resp = gp_model.predict(gp_coords_pred=coords_test,
                             predict_var=True, predict_response=True)
if likelihood in ("binary_probit", "binary_logit"):
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
axs[0, 0].set_title("True GP and training locations")
# predicted latent mean
pred_mu_plot = pred['mu'].reshape((nx, nx))
CS = axs[0, 1].contourf(coords_test_x1, coords_test_x2, pred_mu_plot)
axs[0, 1].set_title("Predictive mean")
# prediction uncertainty
pred_var_plot = pred['var'].reshape((nx, nx))
CS = axs[1, 0].contourf(coords_test_x1, coords_test_x2, pred_var_plot)
axs[1, 0].set_title("Predictive standard deviations")
plt.show(block=False)

# Predict latent GP at training data locations (=smoothing)
GP_smooth = gp_model.predict_training_data_random_effects(predict_var=True)
print(GP_smooth.head()) # Training data random effects: predictive means and variances
# Compare true and predicted random effects
plt.scatter(b_train, GP_smooth.iloc[:,0])
plt.title("Comparison of true and smoothed GP")
plt.show(block=False)
# The above is equivalent to the following:
# GP_smooth2 = gp_model.predict(gp_coords_pred=coords_train, 
#                               predict_response=False, predict_var=True)
# print(np.sum(np.abs(GP_smooth['GP'] - GP_smooth2['mu'])))
# print(np.sum(np.abs(GP_smooth['GP_var'] - GP_smooth2['var'])))

#--------------------Gaussian process model with linear mean function----------------
# Include a liner regression term instead of assuming a zero-mean a.k.a. "universal Kriging"
# Note: you need to include a column of 1's manually for an intercept term
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5,
                       likelihood=likelihood)
gp_model.fit(y=y_lin, X=X, params={"std_dev": True})
gp_model.summary()

#--------------------Gaussian process model anisotropic ARD covariance function----------------
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern_ard",
                       cov_fct_shape = 1.5, likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()

#--------------------Gaussian process model spatio-temporal covariance function----------------
time = np.array(np.tile(np.arange(1, 11), int(600 / 10))[:coords_train.shape[0]]).reshape(-1, 1) # define time 
coords_time_space = np.hstack((time, coords_train)) # the time variable needs to be the first column in the 'gp_coords' argument
gp_model = gpb.GPModel(gp_coords=coords_time_space, cov_function="matern_space_time",
                       cov_fct_shape = 1.5, likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()

# --------------------Gaussian process model with Vecchia approximation----------------
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5,
                       gp_approx="vecchia", num_neighbors=20, likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()
# gp_model.set_prediction_data(num_neighbors_pred=40) # can set number of neigbors for prediction manually
pred_vecchia = gp_model.predict(gp_coords_pred=coords_test, 
                                predict_var=True, predict_response=False)
pred_vecchia = pred_vecchia['mu'].reshape((nx, nx))
plt.contourf(coords_test_x1, coords_test_x2, pred_vecchia)
plt.title("Predicted latent GP mean with Vecchia approxmation")
plt.show(block=False)

# --------------------Gaussian process model with FITC / modified predictive process approximation----------------
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5,
                       gp_approx="fitc", num_ind_points=500, likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()
pred_fitc = gp_model.predict(gp_coords_pred=coords_test, 
                                predict_var=True, predict_response=False)
pred_fitc = pred_fitc['mu'].reshape((nx, nx))
plt.contourf(coords_test_x1, coords_test_x2, pred_fitc)
plt.title("Predicted latent GP mean with FITC approxmation")
plt.show(block=False)

#--------------------Gaussian process model with tapering----------------
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5,
                       gp_approx = "tapering", cov_fct_taper_shape=0., 
                       cov_fct_taper_range=0.5, likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()

# --------------------Gaussian process model with random coefficents----------------
# Define and train model
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5, 
                       gp_rand_coef_data=X_SVC, likelihood=likelihood)
gp_model.fit(y=y_svc) # takes some time for non-Gaussian data
pd.set_option('display.max_columns', None)
gp_model.summary()
# Note: this is a small sample size for this type of model
#   -> covariance parameters estimates can have high variance

# Predict latent GP at training data locations (=smoothing)
GP_smooth = gp_model.predict_training_data_random_effects(predict_var = False) # predict_var = True gives uncertainty for random effect predictions
# Compare true and predicted random effects
plt.scatter(b_train, GP_smooth['GP'], label="Intercept GP", alpha=0.5)
plt.scatter(b2, GP_smooth['GP_rand_coef_nb_1'], label="1. random coef. GP", alpha=0.5)
plt.scatter(b3, GP_smooth['GP_rand_coef_nb_2'], label="2. random coef. GP", alpha=0.5)
plt.legend()
plt.title("Comparison of true and smoothed GP")
plt.show(block=False)

# --------------------Using cluster_ids for independent realizations of GPs----------------
cluster_ids = np.zeros(ntrain)
cluster_ids[int(ntrain/2):ntrain] = 1
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5,
                       cluster_ids=cluster_ids, likelihood=likelihood)
gp_model.fit(y=y_train)
gp_model.summary()

# --------------------Evaluate negative log-likelihood----------------
if likelihood == "gaussian":
  cov_pars = [0.1,sigma2_1,rho]
else:
  cov_pars = [sigma2_1,rho]
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5,
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
sigma2_1 = 0.25  # random effect variance
sigma2_2 = 0.25  # marginal variance of GP
rho = 0.1  # range parameter
D = np.zeros((n, n))  # distance matrix
for i in range(0, n):
    for j in range(i + 1, n):
        D[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])
        D[j, i] = D[i, j]
D_scaled = 3**0.5 * D / rho
Sigma = sigma2_2 * (1. + D_scaled) * np.exp(-D_scaled) + np.diag(np.zeros(n) + 1e-20) # Matern 1.5 covariance
C = np.linalg.cholesky(Sigma)
b1 = sigma2_1**0.5 * np.random.normal(size=m)  # simulate random effect
b2 = C.dot(np.random.normal(size=n))
rand_eff = Z1.dot(b1) + b2
rand_eff = rand_eff - np.mean(rand_eff)
y_comb = simulate_response_variable(lp=0, rand_eff=rand_eff, likelihood=likelihood)
# Define and train model
gp_model = gpb.GPModel(group_data=group, gp_coords=coords, 
                       cov_function="matern", cov_fct_shape=1.5, likelihood=likelihood)
gp_model.fit(y=y_comb)
gp_model.summary()
