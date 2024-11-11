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
        xi = np.sqrt(0.05) * np.random.normal(size=n) # error term
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
        shape = 10
        y = mu / shape * stats.gamma.ppf(np.random.uniform(size=n), a=shape)
    elif likelihood == "negative_binomial":
        mu = np.exp(lp + rand_eff)
        shape = 1.5
        p = shape / (shape + mu)
        y = stats.nbinom.ppf(np.random.uniform(size=n), p=p, n=shape)
    return y

# Choose likelihood: either "gaussian" (=regression), 
#                     "bernoulli_probit", "bernoulli_logit", (=classification)
#                     "poisson", "gamma", or "negative_binomial"
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
b1 = np.sqrt(0.25) * np.random.normal(size=m)  # simulate random effects
rand_eff = b1[group]
rand_eff = rand_eff - np.mean(rand_eff)
# Simulate fixed effects
p = 5 # number of predictor variables
X = np.random.rand(n, p)
f = f1d(X[:, 0])
y = simulate_response_variable(lp=f, rand_eff=rand_eff, likelihood=likelihood)
hst = plt.hist(y, bins=20)  # visualize response variable
plt.show(block=False)

#--------------------Training----------------
# Define random effects model
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
# The default optimizer for covariance parameters (hyperparameters) is Nesterov-accelerated gradient descent.
# This can be changed to, e.g., Nelder-Mead as follows:
# gp_model.set_optim_params(params={"optimizer_cov": "nelder_mead"})
# Use the option "trace": true to monitor convergence of hyperparameter estimation of the gp_model. E.g.:
# gp_model.set_optim_params(params={"trace": True})

# Specify boosting parameters
# Note: these parameters are by no means optimal for all data sets but 
#       need to be chosen appropriately, e.g., using 'gpb.grid.search.tune.parameters'
num_boost_round = 250
if likelihood == "gaussian":
    num_boost_round = 50
elif likelihood in ("bernoulli_probit", "bernoulli_logit"):
    num_boost_round = 500
params = {'learning_rate': 0.01, 'max_depth': 3, 
          'num_leaves': 2**10, 'verbose': 0}

# Create dataset for gpb.train
data_train = gpb.Dataset(data=X, label=y)
bst = gpb.train(params=params, train_set=data_train,  gp_model=gp_model,
                num_boost_round=num_boost_round)
gp_model.summary() # Estimated random effects model

# Showing training loss
#gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
#bst = gpb.train(params=params, train_set=data_train,  gp_model=gp_model,
#                num_boost_round=num_boost_round, valid_sets=data_train)

#--------------------Prediction----------------
group_test = np.arange(m) # Predictions for existing groups
group_test_new = -np.ones(m) # Can also do predictions for new/unobserved groups
Xtest = np.zeros((m, p))
Xtest[:, 0] = np.linspace(0, 1, m)
# 1. Predict latent variable (pred_latent=True) and variance
pred = bst.predict(data=Xtest, group_data_pred=group_test, 
                   predict_var=True, pred_latent=True)
# pred['fixed_effect']: predictions for the latent fixed effects / tree ensemble
# pred['random_effect_mean']: mean predictions for the random effects
# pred['random_effect_cov']: predictive (co-)variances (if predict_var=True) of the random effects
# 2. Predict response variable (pred_latent=False)
pred_resp = bst.predict(data=Xtest, group_data_pred=group_test_new,
                        predict_var=True, pred_latent=False)
# pred_resp['response_mean']: mean predictions of the response variable 
#   which combines predictions from the tree ensemble and the random effects
# pred_resp['response_var']: predictive (co-)variances (if predict_var=True)

# Visualize fitted response variable
fig1, ax1 = plt.subplots()
ax1.plot(Xtest[:, 0], pred_resp['response_mean'], linewidth=2, label="Pred response")
ax1.scatter(X[:, 0], y, linewidth=2, color="black", alpha=0.02)
ax1.set_title("Data and predicted response variable")
ax1.legend()
plt.show(block=False)
# Visualize fitted (latent) fixed effects function
fig1, ax1 = plt.subplots()
ax1.plot(Xtest[:, 0], f1d(Xtest[:, 0]), linewidth=2, label="True F")
ax1.plot(Xtest[:, 0], pred['fixed_effect'], linewidth=2, label="Pred F")
ax1.set_title("Tue and predicted latent function F")
ax1.legend()
plt.show(block=False)
# Compare true and predicted random effects
plt.scatter(b1, pred['random_effect_mean'])
plt.title("Comparison of true and predicted random effects")
plt.xlabel("truth")
plt.ylabel("predicted")
plt.show(block=False)

#--------------------Choosing tuning parameters using the TPESampler from optuna----------------
# Define search space
# Note: if the best combination found below is close to the bounday for a paramter, you might want to extend the corresponding range
search_space = { 'learning_rate': [0.001, 10],
                'min_data_in_leaf': [1, 1000],
                'max_depth': [-1,-1], # -1 means no depth limit as we tune 'num_leaves'. Can also additionaly tune 'max_depth', e.g., 'max_depth': [-1,10]
                'num_leaves': [2, 1024],
                'lambda_l2': [0, 100],
                'max_bin': [63, np.min([10000,n])],
                'line_search_step_length': [True, False] }
# Define metric
metric = "mse"
if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    metric = "binary_logloss"
# Note: can also use metric = "test_neg_log_likelihood". For more options, see https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
# Run parameter optimization using the TPE algorithm and 4-fold CV 
opt_params = gpb.tune_pars_TPE_algorithm_optuna(X=X, y=y, search_space=search_space, 
                                                nfold=4, gp_model=gp_model, metric=metric, tpe_seed=1,
                                                max_num_boost_round=1000, n_trials=100, early_stopping_rounds=20)
print("Best parameters: " + str(opt_params['best_params']))
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))

# Alternatively and faster: using manually defined validation data instead of cross-validation
np.random.seed(10)
permute_aux = np.random.permutation(n)
train_tune_idx = permute_aux[0:int(0.8 * n)] # use 20% of the data as validation data
valid_tune_idx = permute_aux[int(0.8 * n):n]
folds = [(train_tune_idx, valid_tune_idx)]
opt_params = gpb.tune_pars_TPE_algorithm_optuna(X=X, y=y, search_space=search_space, 
                                                folds=folds, gp_model=gp_model, metric=metric, tpe_seed=1,
                                                max_num_boost_round=1000, n_trials=100, early_stopping_rounds=20)

#--------------------Choosing tuning parameters using random grid search----------------
# Define parameter search grid
# Note: if the best combination found below is close to the bounday for a paramter, you might want to extend the corresponding range
param_grid = { 'learning_rate': [0.001, 0.01, 0.1, 1, 10], 
              'min_data_in_leaf': [1, 10, 100, 1000],
              'max_depth': [-1], # -1 means no depth limit as we tune 'num_leaves'. Can also additionaly tune 'max_depth', e.g., 'max_depth': [-1, 1, 2, 3, 5, 10]
              'num_leaves': 2**np.arange(1,10),
              'lambda_l2': [0, 1, 10, 100],
              'max_bin': [250, 500, 1000, np.min([10000,n])],
              'line_search_step_length': [True, False]}
other_params = {'verbose': 0} # avoid trace information when training models
# Define metric
metric = "mse"
if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    metric = "binary_logloss"
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
data_train = gpb.Dataset(data=X, label=y)
# Run parameter optimization using random grid search and 4-fold CV
# Note: deterministic grid search can be done by setting 'num_try_random=None'
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid, params=other_params,
                                             num_try_random=100, nfold=4, seed=1000,
                                             train_set=data_train, gp_model=gp_model,
                                             use_gp_model_for_validation=True, verbose_eval=1,
                                             num_boost_round=1000, early_stopping_rounds=20,
                                             metric=metric)                            
print("Best parameters: " + str(opt_params['best_params']))
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))

# Alternatively and faster: using manually defined validation data instead of cross-validation
np.random.seed(10)
permute_aux = np.random.permutation(n)
train_tune_idx = permute_aux[0:int(0.8 * n)] # use 20% of the data as validation data
valid_tune_idx = permute_aux[int(0.8 * n):n]
folds = [(train_tune_idx, valid_tune_idx)]
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid, params=other_params,
                                             num_try_random=100, folds=folds, seed=1000,
                                             train_set=data_train, gp_model=gp_model,
                                             use_gp_model_for_validation=True, verbose_eval=1,
                                             num_boost_round=1000, early_stopping_rounds=20,
                                             metric=metric)
  
#--------------------Cross-validation for determining number of iterations----------------
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
data_train = gpb.Dataset(data=X, label=y)
cvbst = gpb.cv(params=params, train_set=data_train,
               gp_model=gp_model, use_gp_model_for_validation=True,
               num_boost_round=1000, early_stopping_rounds=20,
               nfold=4, verbose_eval=True, show_stdv=False, seed=1)
metric_name = list(cvbst.keys())[0]
print("Best number of iterations: " + str(np.argmin(cvbst[metric_name]) + 1))

#--------------------Using a validation set for finding number of iterations----------------
# Partition data into training and validation data
np.random.seed(1)
train_ind = np.random.choice(n, int(0.8 * n), replace=False)
test_ind = [i for i in range(n) if i not in train_ind]
data_train = gpb.Dataset(X[train_ind, :], y[train_ind])
data_eval = gpb.Dataset(X[test_ind, :], y[test_ind], reference=data_train)
gp_model = gpb.GPModel(group_data=group[train_ind], likelihood=likelihood)
gp_model.set_prediction_data(group_data_pred=group[test_ind])
evals_result = {}  # record eval results for plotting
bst = gpb.train(params=params, train_set=data_train, num_boost_round=1000,
                gp_model=gp_model, valid_sets=data_eval, 
                early_stopping_rounds=20, use_gp_model_for_validation=True,
                evals_result=evals_result)
gpb.plot_metric(evals_result, figsize=(10, 5))# plot validation scores
plt.show(block=False)

#--------------------Do Newton updates for tree leaves (only for Gaussian data)----------------
if likelihood == "gaussian":
    params_newton = params.copy()
    params_newton['leaves_newton_update'] = True
    params_newton['learning_rate'] = 0.1
    evals_result = {}  # record eval results for plotting
    bst = gpb.train(params=params_newton, train_set=data_train, num_boost_round=1000,
                    gp_model=gp_model, valid_sets=data_eval, early_stopping_rounds=5,
                    use_gp_model_for_validation=True, evals_result=evals_result)
    gpb.plot_metric(evals_result, figsize=(10, 5))# plot validation scores

#--------------------Model interpretation----------------
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
data_train = gpb.Dataset(data=X, label=y)
bst = gpb.train(params=params, train_set=data_train,
                gp_model=gp_model, num_boost_round=num_boost_round)
# Split-based feature importances
feature_importances = bst.feature_importance(importance_type='gain')
plt_imp = gpb.plot_importance(bst, importance_type='gain')
# Partial dependence plot
from pdpbox import pdp
# note: pdpbox can also be run with newer versions of matplotlib. In case 
#       problems occurr during installation, try "pip install pdpbox --no-dependencies"
import pandas as pd
# Note: for the pdpbox package, the data needs to be a pandas DataFrame
Xpd = pd.DataFrame(X, columns=['variable_' + str(i) for i in range(p)])
pdp_dist = pdp.PDPIsolate(model=bst, df=Xpd.copy(), model_features=Xpd.columns, # need to copy() since PDPIsolate modifies the df
                           feature='variable_0', feature_name='variable_0', 
                           n_classes=0, num_grid_points=50,
                           predict_kwds={"ignore_gp_model": True})
fig, axes = pdp_dist.plot(engine='matplotlib', plot_lines=True, frac_to_plot=0.1)
# Interaction plot
interact = pdp.PDPInteract(model=bst, df=Xpd.copy(), model_features=Xpd.columns,
                             features=['variable_0','variable_1'],
                             feature_names=['variable_0','variable_1'],
                             n_classes=0, predict_kwds={"ignore_gp_model": True})
fig, axes = interact.plot(engine='matplotlib', plot_type='contour')
"""
# Note: the above code is for pdpbox version 0.3.0 or latter, for earlier versions use:
# pdp_dist = pdp.pdp_isolate(model=bst, dataset=Xpd, model_features=Xpd.columns,
#                            feature='variable_0', num_grid_points=50,
#                            predict_kwds={"ignore_gp_model": True})
# ax = pdp.pdp_plot(pdp_dist, 'variable_0', plot_lines=True, frac_to_plot=0.1)
# interact = pdp.pdp_interact(model=bst, dataset=Xpd, model_features=Xpd.columns,
#                              features=['variable_0','variable_1'],
#                              predict_kwds={"ignore_gp_model": True})
# pdp.pdp_interact_plot(interact, ['variable_0','variable_1'], x_quantile=True,
#                       plot_type='contour', plot_pdp=True) # Ignore the error message 'got an unexpected keyword argument 'contour_label_fontsize'' in 'pdp_interact_plot'
"""
# SHAP values and dependence plots (note: shap version>=0.36.0 is required)
import shap
shap_values = shap.TreeExplainer(bst).shap_values(X)
shap.summary_plot(shap_values, X)
shap.dependence_plot("Feature 0", shap_values, X)
# SHAP interaction values
shap_interaction_values = shap.TreeExplainer(bst).shap_interaction_values(shap_values)
shap.summary_plot(shap_interaction_values, X)
shap.dependence_plot(("Feature 0", "Feature 1"), shap_interaction_values, X, display_features=X)

#--------------------Saving a booster with a gp_model and loading it from a file----------------
# Train model and make prediction
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
data_train = gpb.Dataset(data=X, label=y)
bst = gpb.train(params=params, train_set=data_train,
                gp_model=gp_model, num_boost_round=num_boost_round)
group_test = np.array([1,2,-1])
Xtest = np.random.rand(len(group_test), p)
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

# Note: can also convert to string and load from string
# model_str = bst.model_to_string()
# bst_loaded = gpb.Booster(model_str = model_str)


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
D_scaled = 3**0.5 * D / rho
Sigma = sigma2_1 * (1. + D_scaled) * np.exp(-D_scaled) + np.diag(np.zeros(n) + 1e-20) # Matern 1.5 covariance
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
plt.show(block=False)

# Specify boosting parameters as dict
params = {'learning_rate': 0.1, 'max_depth': 3, 'verbose': 0}
num_boost_round = 10
if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    num_boost_round = 50

#--------------------Training----------------
# Define Gaussian process model
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5,
                       likelihood=likelihood)
# Create dataset for gpb.train
data_train = gpb.Dataset(X_train, y_train)
bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model,
                num_boost_round=num_boost_round)
gp_model.summary() # Estimated random effects model

#--------------------Prediction----------------
# 1. Predict response variable (pred_latent=False)
pred_resp = bst.predict(data=X_test, gp_coords_pred=coords_test, 
                        predict_var=True, pred_latent=False)
# pred_resp['response_mean']: mean predictions of the response variable 
#   which combines predictions from the tree ensemble and the Gaussian process
# pred_resp['response_var']: predictive (co-)variances (if predict_var=True)
# 2. Predict latent variables (pred_latent=True)
pred = bst.predict(data=X_test, gp_coords_pred=coords_test, 
                   predict_var=True, pred_latent=True)
# pred['fixed_effect']: predictions for the latent fixed effects / tree ensemble
# pred['random_effect_mean']: mean predictions for the random effects
# pred['random_effect_cov']: predictive (co-)variances (if predict_var=True) of the (latent) Gaussian process
# 3. Can also calculate predictive covariances
pred_cov = bst.predict(data=X_test[0:3,], gp_coords_pred=coords_test[0:3,],
                       predict_cov_mat=True, pred_latent=True)
# pred_cov['random_effect_cov']: predictive covariances of the (latent) Gaussian process
if likelihood == "gaussian":
    # Predictive covariances for the response variable are currently only supported for Gaussian likelihoods
    pred_resp_cov = bst.predict(data=X_test[0:3,], gp_coords_pred=coords_test[0:3,],
                                predict_cov_mat=True, pred_latent=False)
    # pred_resp_cov['response_var']: predictive covariances of the response variable

# Evaluate predictions
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

#--------------------Choosing tuning parameters----------------
"""
Choosing tuning parameters carefully is important.
See the above demo code for grouped random effects on how this can be done.
You just have to replace the gp_model. E.g.,    
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5, likelihood=likelihood)
"""

#--------------------Model interpretation----------------
"""
See the above demo code for grouped random effects on how this can be done.
"""

