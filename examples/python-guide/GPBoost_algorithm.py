# -*- coding: utf-8 -*-
"""
Examples on how to do use the GPBoost and LaGaBoost algorithms 
for various likelihoods:
    - "gaussian" (=regression)
    - "bernoulli" (=classification)
    - "poisson" and "gamma" (=Poisson and gamma regression)
and various random effects models:
    - grouped (aka clustered) random effects models
    - Gaussian process (GP) models

Author: Fabio Sigrist
"""

import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')
print("It is recommended that the examples are run in interactive mode")

def f1d(x):
    """Non-linear fixed effects function for simulation"""
    return 1 / (1 + np.exp(-(x - 0.5) * 10)) - 0.5

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
Combine tree-boosting and grouped random effects model
"""
# --------------------Simulate data----------------
n = 5000  # number of samples
m = 500  # number of groups
np.random.seed(1)
# Simulate grouped random effects
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
b1 = np.sqrt(0.5) * np.random.normal(size=m)  # simulate random effects
rand_eff = b1[group]
rand_eff = rand_eff - np.mean(rand_eff)
# Simulate fixed effects
X = np.random.rand(n, 2)
f = f1d(X[:, 0])
y = simulate_response_variable(lp=f, rand_eff=rand_eff, likelihood=likelihood)
hst = plt.hist(y, bins=20)  # visualize response variable
plt.show()

#--------------------Training----------------
# Define random effects model
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
# The default optimizer for covariance parameters (hyperparameters) is Nesterov-accelerated gradient descent.
# This can be changed to, e.g., Nelder-Mead as follows:
# gp_model.set_optim_params(params={"optimizer_cov": "nelder_mead"})
# Use the option "trace": true to monitor convergence of hyperparameter estimation of the gp_model. E.g.:
# gp_model.set_optim_params(params={"trace": True})

# Create dataset for gpb.train
data_train = gpb.Dataset(X, y)
# Specify boosting parameters as dict
params = {'objective': likelihood, 'learning_rate': 0.01, 'max_depth': 3,
          'verbose': 0, 'monotone_constraints': [1, 0]}
num_boost_round = 250
if likelihood == "gaussian":
    num_boost_round = 35
    params['objective'] = 'regression_l2'
if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    num_boost_round = 500
    params['objective'] = 'binary'
# Note: these parameters are not neccessary optimal for all situations considered here
bst = gpb.train(params=params, train_set=data_train,  gp_model=gp_model,
                num_boost_round=num_boost_round)
gp_model.summary() # Estimated random effects model

# Showing training loss
#gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
#bst = gpb.train(params=params, train_set=data_train,  gp_model=gp_model,
#                num_boost_round=num_boost_round, valid_sets=data_train)

#--------------------Prediction----------------
group_test = np.arange(m)
Xtest = np.zeros((m, 2))
Xtest[:, 0] = np.linspace(0, 1, m)
#1. Predict latent variable (pred_latent=True) and variance
pred = bst.predict(data=Xtest, group_data_pred=group_test, predict_var=True, 
                   pred_latent=True)
# pred_resp['fixed_effect']: predictions for the latent fixed effects / tree ensemble
# pred_resp['random_effect_mean']: mean predictions for the random effects
# pred_resp['random_effect_cov']: predictive (co-)variances (if predict_var=True) of the random effects
# 2. Predict response variable (pred_latent=False)
group_test = -np.ones(m) # only new groups since we are only interested in the fixed effects for visualization
pred_resp = bst.predict(data=Xtest, group_data_pred=group_test, pred_latent=False)
# pred_resp['response_mean']: mean predictions of the response variable 
#   which combines predictions from the tree ensemble and the random effects
# pred_resp['response_var']: predictive variances (if predict_var=True)

# Visualize fitted response variable
fig1, ax1 = plt.subplots()
ax1.plot(Xtest[:, 0], pred_resp['response_mean'], linewidth=2, label="Pred response")
ax1.scatter(X[:, 0], y, linewidth=2, color="black", alpha=0.02)
ax1.set_title("Data and predicted response variable")
ax1.legend()
plt.show()
# Visualize fitted (latent) fixed effects function
fig1, ax1 = plt.subplots()
ax1.plot(Xtest[:, 0], f1d(Xtest[:, 0]), linewidth=2, label="True F")
ax1.plot(Xtest[:, 0], pred['fixed_effect'], linewidth=2, label="Pred F")
if likelihood in ("gaussian", "bernoulli_probit", "bernoulli_logit"):
    ax1.scatter(X[:, 0], y, linewidth=2, color="black", alpha=0.02)
ax1.set_title("Data, true and predicted latent function F")
ax1.legend()
plt.show()
# Compare true and predicted random effects
plt.scatter(b1, pred['random_effect_mean'])
plt.title("Comparison of true and predicted random effects")
plt.xlabel("truth")
plt.ylabel("predicted")
plt.show()

#--------------------Choosing tuning parameters----------------
param_grid = {'learning_rate': [1,0.1,0.01], 'min_data_in_leaf': [1,10,100],
                    'max_depth': [1,3,5,10,-1]}
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
data_train = gpb.Dataset(X, y)
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid,
                                             params=params,
                                             num_try_random=None,
                                             nfold=4,
                                             gp_model=gp_model,
                                             use_gp_model_for_validation=True,
                                             train_set=data_train,
                                             verbose_eval=1,
                                             num_boost_round=1000, 
                                             early_stopping_rounds=10,
                                             seed=1000)
print("Best parameters: " + str(opt_params['best_params']))
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))

#--------------------Cross-validation for determining number of iterations----------------
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
data_train = gpb.Dataset(X, y)
cvbst = gpb.cv(params=params, train_set=data_train,
               gp_model=gp_model, use_gp_model_for_validation=True,
               num_boost_round=1000, early_stopping_rounds=5,
               nfold=4, verbose_eval=True, show_stdv=False, seed=1)
metric_name = list(cvbst.keys())[0]
print("Best number of iterations: " + str(np.argmin(cvbst[metric_name])))

#--------------------Using a validation set for finding number of iterations----------------
# Partition data into training and validation data
np.random.seed(1)
train_ind = np.random.choice(n, int(0.7 * n), replace=False)
test_ind = [i for i in range(n) if i not in train_ind]
data_train = gpb.Dataset(X[train_ind, :], y[train_ind])
data_eval = gpb.Dataset(X[test_ind, :], y[test_ind], reference=data_train)
gp_model = gpb.GPModel(group_data=group[train_ind], likelihood=likelihood)
# Include random effect predictions for validation (=default)
gp_model.set_prediction_data(group_data_pred=group[test_ind])
evals_result = {}  # record eval results for plotting
bst = gpb.train(params=params, train_set=data_train, num_boost_round=1000,
                gp_model=gp_model, valid_sets=data_eval, 
                early_stopping_rounds=10, use_gp_model_for_validation=True,
                evals_result=evals_result)
gpb.plot_metric(evals_result, figsize=(10, 5))# plot validation scores
plt.show()
# Do not include random effect predictions for validation (observe the higher test error)
evals_result = {}  # record eval results for plotting
bst = gpb.train(params=params, train_set=data_train, num_boost_round=1000,
                gp_model=gp_model, valid_sets=data_eval, 
                early_stopping_rounds=10, use_gp_model_for_validation=False,
                evals_result=evals_result)
gpb.plot_metric(evals_result, figsize=(10, 5)) # plot validation scores
plt.show()

#--------------------Do Newton updates for tree leaves (only for Gaussian data)----------------
if likelihood == "gaussian":
    params_N = params.copy()
    params_N['leaves_newton_update'] = True
    evals_result = {}  # record eval results for plotting
    bst = gpb.train(params=params_N, train_set=data_train, num_boost_round=100,
                    gp_model=gp_model, valid_sets=data_eval, early_stopping_rounds=5,
                    use_gp_model_for_validation=True, evals_result=evals_result)
    gpb.plot_metric(evals_result, figsize=(10, 5))# plot validation scores

#--------------------Model interpretation----------------
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
data_train = gpb.Dataset(X, y)
bst = gpb.train(params=params, train_set=data_train,
                gp_model=gp_model, num_boost_round=num_boost_round)
# Split-based feature importances
feature_importances = bst.feature_importance(importance_type='gain')
plt_imp = gpb.plot_importance(bst, importance_type='gain')
# Partial dependence plot
from pdpbox import pdp
import pandas as pd
# Note: for the pdpbox package, the data needs to be a pandas DataFrame
Xpd = pd.DataFrame(X,columns=['variable_1','variable_2'])
pdp_dist = pdp.pdp_isolate(model=bst, dataset=Xpd, model_features=Xpd.columns,
                           feature='variable_1', num_grid_points=50,
                           predict_kwds={"ignore_gp_model": True})
ax = pdp.pdp_plot(pdp_dist, 'variable_1', plot_lines=True, frac_to_plot=0.1)
# Interaction plot
interact = pdp.pdp_interact(model=bst, dataset=Xpd, model_features=Xpd.columns,
                             features=['variable_1','variable_2'],
                             predict_kwds={"ignore_gp_model": True})
pdp.pdp_interact_plot(interact, ['variable_1','variable_2'], x_quantile=True,
                      plot_type='contour', plot_pdp=True)# ignore any error message

# SHAP values and dependence plots (note: you need shap version>=0.36.0)
import shap
shap_values = shap.TreeExplainer(bst).shap_values(X)
shap.summary_plot(shap_values, X)
shap.dependence_plot("Feature 0", shap_values, X)

#--------------------Saving a booster with a gp_model and loading it from a file----------------
# Train model and make prediction
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
data_train = gpb.Dataset(X, y)
bst = gpb.train(params=params, train_set=data_train,
                gp_model=gp_model, num_boost_round=num_boost_round)
group_test = np.array([1,2,-1])
Xtest = np.random.rand(len(group_test), 2)
pred = bst.predict(data=Xtest, group_data_pred=group_test, 
                   predict_var=True, pred_latent=True)
# Save model
bst.save_model('model.json')
# Load from file and make predictions again
bst_loaded = gpb.Booster(model_file = 'model.json')
pred_loaded = bst_loaded.predict(data=Xtest, group_data_pred=group_test, 
                                 predict_var=True, pred_latent=True)
# Check equality
print(pred['fixed_effect'] - pred_loaded['fixed_effect'])
print(pred['random_effect_mean'] - pred_loaded['random_effect_mean'])
print(pred['random_effect_cov'] - pred_loaded['random_effect_cov'])


"""
Combine tree-boosting and Gaussian process model
"""
# --------------------Simulate data----------------
ntrain = 600  # number of samples
np.random.seed(4)
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
# Simulate fixed effects
X_train = np.random.rand(ntrain, 2)
X_test = np.column_stack((np.linspace(0, 1, ntest), np.zeros(ntest)))
X = np.row_stack((X_train, X_test))
f = f1d(X[:, 0])
# Simulate spatial Gaussian process
sigma2_1 = 0.25  # marginal variance of GP
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
y = simulate_response_variable(lp=f, rand_eff=b, likelihood=likelihood)
# Split into training and test data
y_train = y[0:ntrain]
y_test = y[ntrain:n]
b_train = b[0:ntrain]
b_test = b[ntrain:n]
hst = plt.hist(y_train, bins=20)  # visualize response variable
plt.show()

#--------------------Training----------------
# Define Gaussian process model
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential",
                       likelihood=likelihood)
# Create dataset for gpb.train
data_train = gpb.Dataset(X_train, y_train)
# Specify boosting parameters as dict
params = {'learning_rate': 0.1, 'objective': likelihood,
          'verbose': 0, 'monotone_constraints': [1, 0]}
num_boost_round = 25
if likelihood == "gaussian":
    num_boost_round = 10
    params['objective'] = 'regression_l2'
if likelihood == "bernoulli_logit":
    num_boost_round = 50
if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    params['objective'] = 'binary'
bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model,
                num_boost_round=num_boost_round)
gp_model.summary() # Estimated random effects model

#--------------------Prediction----------------
# Predict response variable
pred_resp = bst.predict(data=X_test, gp_coords_pred=coords_test, 
                        pred_latent=False)
# Predict latent variable including variance
pred = bst.predict(data=X_test, gp_coords_pred=coords_test, predict_var=True, 
                   pred_latent=True)

if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    print("Test error:")
    pred_binary = pred_resp['response_mean'] > 0.5
    pred_binary = pred_binary.astype(np.float64)
    print(np.mean(pred_binary != y_test))
else:
    print("Test root mean square error:")
    print(np.sqrt(np.mean((pred_resp['response_mean'] - y_test) ** 2)))
print("Test root mean square error for latent GP:")
print(np.sqrt(np.mean((pred['random_effect_mean'] - b_test) ** 2)))

# Visualize predictions and compare to true values
fig, axs = plt.subplots(2, 2, figsize=[10,8])
# data and true GP
b_test_plot = b_test.reshape((nx, nx))
CS = axs[0, 0].contourf(coords_test_x1, coords_test_x2, b_test_plot)
axs[0, 0].plot(coords_train[:, 0], coords_train[:, 1], '+', color="white", 
   markersize = 4)
axs[0, 0].set_title("True latent GP and training locations")
# predicted latent GP mean
pred_mu_plot = pred['random_effect_mean'].reshape((nx, nx))
CS = axs[0, 1].contourf(coords_test_x1, coords_test_x2, pred_mu_plot)
axs[0, 1].set_title("Predicted latent GP mean")
# prediction uncertainty
pred_var_plot = pred['random_effect_cov'].reshape((nx, nx))
CS = axs[1, 0].contourf(coords_test_x1, coords_test_x2, pred_var_plot)
axs[1, 0].set_title("Predicted latent GP standard deviation")
# latent predictor function F
axs[1, 1].plot(X_test[:, 0], f1d(X_test[:, 0]), linewidth=2, label="True F")
axs[1, 1].plot(X_test[:, 0], pred['fixed_effect'], linewidth=2, label="Pred F")
axs[1, 1].set_title("Predicted and true F")
axs[1, 1].legend()

#--------------------Cross-validation for determining number of iterations----------------
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential",
                       likelihood=likelihood)
cvbst = gpb.cv(params=params, train_set=data_train, gp_model=gp_model,
               use_gp_model_for_validation=True, num_boost_round=200,
               early_stopping_rounds=5, nfold=4, verbose_eval=True,
               show_stdv=False, seed=1)
print("Best number of iterations: " + str(np.argmin(next(iter(cvbst.values())))))
