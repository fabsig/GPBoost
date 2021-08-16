# -*- coding: utf-8 -*-
"""
Examples on how to use the GPBoost algorithm for combining tree-boosting
with random effects and Gaussian process models

@author: Fabio Sigrist
"""

import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
def f1d(x):
    """Non-linear function for simulation"""
    return (1.7 * (1 / (1 + np.exp(-(x - 0.5) * 20)) + 0.75 * x))
print("It is recommended that the examples are run in interactive mode")

# --------------------Combine tree-boosting and grouped random effects model----------------
# --------------------Simulate data----------------
n = 5000  # number of samples
m = 500  # number of groups
np.random.seed(1)
# Simulate grouped random effects
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
b1 = np.random.normal(size=m)  # simulate random effects
eps = b1[group]
# Simulate fixed effects
X = np.random.rand(n, 2)
f = f1d(X[:, 0])
xi = np.sqrt(0.01) * np.random.normal(size=n)  # simulate error term
y = f + eps + xi  # observed data

#--------------------Training----------------
# Define GPModel
gp_model = gpb.GPModel(group_data=group)
# The default optimizer for covariance parameters (hyperparameters) is Nesterov-accelerated gradient descent.
# This can be changed to, e.g., Nelder-Mead as follows:
# gp_model.set_optim_params(params={"optimizer_cov": "nelder_mead"})
# Use the option "trace": true to monitor convergence of hyperparameter estimation of the gp_model. E.g.:
# gp_model.set_optim_params(params={"trace": True})
# Create dataset for gpb.train
data_train = gpb.Dataset(X, y)
# Specify boosting parameters as dict
params = { 'objective': 'regression_l2',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_data_in_leaf': 5,
            'verbose': 0 }
# Train GPBoost model
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=15)
# Estimated random effects model
gp_model.summary()

# Showing training loss
gp_model = gpb.GPModel(group_data=group)
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=15,
                valid_sets=data_train)

#--------------------Prediction----------------
group_test = np.arange(m)
Xtest = np.zeros((m, 2))
Xtest[:, 0] = np.linspace(0, 1, m)
pred = bst.predict(data=Xtest, group_data_pred=group_test)
# pred['fixed_effect'] contains the predictions for the fixed effects / tree ensemble
# pred['random_effect_mean'] contains the mean predictions for the latent random effects
# pred['random_effect_cov'] contains the predictive (co-)variances (if predict_var=True) of the random effects
# To obtain a unique prediction which combines fixed and random effects,
#   sum the two components up: pred_combined = pred['fixed_effect'] + pred['random_effect_mean']

# Compare true and predicted random effects
plt.figure("Comparison of true and predicted random effects")
plt.scatter(b1, pred['random_effect_mean'])
plt.title("Comparison of true and predicted random effects")
plt.xlabel("truth")
plt.ylabel("predicted")
plt.show()
# Fixed effect
plt.figure("Comparison of true and fitted fixed effect")
plt.scatter(Xtest[:, 0], pred['fixed_effect'], linewidth=2, color="b", label="fit")
x = np.linspace(0, 1, 200, endpoint=True)
plt.plot(x, f1d(x), linewidth=2, color="r", label="true")
plt.title("Comparison of true and fitted fixed effect")
plt.legend()
plt.show()

#--------------------Using a validation set----------------
np.random.seed(1)
train_ind = np.random.choice(n, int(0.9 * n), replace=False)
test_ind = [i for i in range(n) if i not in train_ind]
data_train = gpb.Dataset(X[train_ind, :], y[train_ind])
data_eval = gpb.Dataset(X[test_ind, :], y[test_ind], reference=data_train)
gp_model = gpb.GPModel(group_data=group[train_ind])

# Include random effect predictions for validation (=default)
gp_model.set_prediction_data(group_data_pred=group[test_ind])
evals_result = {}  # record eval results for plotting
bst = gpb.train(params=params,
                train_set=data_train,
                num_boost_round=100,
                gp_model=gp_model,
                valid_sets=data_eval,
                early_stopping_rounds=5,
                use_gp_model_for_validation=True,
                evals_result=evals_result)
# plot validation scores
gpb.plot_metric(evals_result, figsize=(10, 5))
plt.show()

# Do not include random effect predictions for validation (observe the higher test error)
evals_result = {}  # record eval results for plotting
bst = gpb.train(params=params,
                train_set=data_train,
                num_boost_round=100,
                gp_model=gp_model,
                valid_sets=data_eval,
                early_stopping_rounds=5,
                use_gp_model_for_validation=False,
                evals_result=evals_result)
# plot validation scores
gpb.plot_metric(evals_result, figsize=(10, 5))
plt.show()

#--------------------Cross-validation for determining number of iterations----------------
gp_model = gpb.GPModel(group_data=group)
data_train = gpb.Dataset(X, y)
cvbst = gpb.cv(params=params, train_set=data_train,
               gp_model=gp_model, use_gp_model_for_validation=True,
               num_boost_round=100, early_stopping_rounds=5,
               nfold=2, verbose_eval=True, show_stdv=False, seed=1)
print("Best number of iterations: " + str(np.argmin(cvbst['l2-mean'])))

#--------------------Model interpretation----------------
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
data_train = gpb.Dataset(X, y)
bst = gpb.train(params=params, train_set=data_train,
                gp_model=gp_model, num_boost_round=15)
# Calculate and plot feature importances
feature_importances = bst.feature_importance(importance_type='gain')
plt = gpb.plot_importance(bst, importance_type='gain')
# SHAP values and dependence plots
# Note: you need shap version>=0.36.0
import shap
shap_values = shap.TreeExplainer(bst).shap_values(X)
shap.summary_plot(shap_values, X)
shap.dependence_plot("Feature 0", shap_values, X)

#--------------------Saving a booster with a gp_model and loading it from a file----------------
# Train model and make prediction
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
data_train = gpb.Dataset(X, y)
bst = gpb.train(params=params, train_set=data_train,
                gp_model=gp_model, num_boost_round=15)
group_test = np.array([1,2,-1])
Xtest = np.random.rand(len(group_test), 2)
pred = bst.predict(data=Xtest, group_data_pred=group_test, predict_var=True)
# Save model
bst.save_model('model.json')
# Load from file and make predictions again
bst_loaded = gpb.Booster(model_file = 'model.json')
pred_loaded = bst_loaded.predict(data=Xtest, group_data_pred=group_test, predict_var=True)
# Check equality
print(pred['fixed_effect'] - pred_loaded['fixed_effect'])
print(pred['random_effect_mean'] - pred_loaded['random_effect_mean'])
print(pred['random_effect_cov'] - pred_loaded['random_effect_cov'])

#--------------------Do Newton updates for tree leaves----------------
print("Training with Newton updates for tree leaves")
params = { 'objective': 'regression_l2',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_data_in_leaf': 5,
            'verbose': 0,
            'leaves_newton_update': True }
evals_result = {}  # record eval results for plotting
gp_model.set_prediction_data(group_data_pred=group[test_ind])
bst = gpb.train(params=params,
                train_set=data_train,
                num_boost_round=100,
                gp_model=gp_model,
                valid_sets=data_eval,
                early_stopping_rounds=5,
                use_gp_model_for_validation=True,
                evals_result=evals_result)
# plot validation scores
gpb.plot_metric(evals_result, figsize=(10, 5))


# --------------------Combine tree-boosting and Gaussian process model----------------
# Simulate data
n = 200  # number of samples
np.random.seed(1)
X = np.random.rand(n, 2)
F = f1d(X[:, 0])
# Simulate Gaussian process
coords = np.column_stack(
    (np.random.uniform(size=n), np.random.uniform(size=n)))  # locations (=features) for Gaussian process
sigma2_1 = 1 ** 2  # marginal variance of GP
rho = 0.1  # range parameter
sigma2 = 0.1 ** 2  # error variance
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
y = F + eps + xi  # observed data

# define GPModel
gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential")
# The default optimizer for covariance parameters (hyperparameters) is Fisher scoring.
# This can be changed as follows:
# gp_model.set_optim_params(params={"optimizer_cov": "gradient_descent", "lr_cov": 0.05,
#                                   "use_nesterov_acc": True, "acc_rate_cov": 0.5})
# Use the option "trace": true to monitor convergence of hyperparameter estimation of the gp_model. E.g.:
# gp_model.set_optim_params(params={"trace": True})
# create dataset for gpb.train
data_train = gpb.Dataset(X, y)
# specify your configurations as a dict
params = { 'objective': 'regression_l2',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_data_in_leaf': 5,
            'verbose': 0 }

# train
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=8)
print("Estimated random effects model")
gp_model.summary()

# Make predictions
np.random.seed(1)
ntest = 5
Xtest = np.random.rand(ntest, 2)
# prediction locations (=features) for Gaussian process
coords_test = np.column_stack(
    (np.random.uniform(size=ntest), np.random.uniform(size=ntest))) / 10.
pred = bst.predict(data=Xtest, gp_coords_pred=coords_test, predict_cov_mat=True)
# Predicted fixed effect from tree ensemble
pred['fixed_effect']
# Predicted (posterior) mean of GP
pred['random_effect_mean']
# Predicted (posterior) covariance matrix of GP
pred['random_effect_cov']

