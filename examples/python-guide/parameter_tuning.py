# -*- coding: utf-8 -*-
"""
Examples on how to do parameter tuning for the GPBoost algorithm

@author: Fabio Sigrist
"""

import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')
def f1d(x):
    """Non-linear function for simulation"""
    return 1 / (1 + np.exp(-(x - 0.5) * 10)) - 0.5
print("It is recommended that the examples are run in interactive mode")

# --------------------Simulate data grouped random effects data----------------
n = 1000  # number of samples
m = 100  # number of groups
np.random.seed(1000)
# simulate grouped random effects
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
b1 = np.sqrt(0.5) * np.random.normal(size=m)  # simulate random effects
eps = b1[group]
eps = eps - np.mean(eps)
# simulate fixed effects
X = np.random.rand(n, 2)
f = f1d(X[:, 0])
# simulate response variable
probs = stats.norm.cdf(f + eps)
y = np.random.uniform(size=n) < probs
y = y.astype(np.float64)

# --------------------Parameter tuning using cross-validation: deterministic and random grid search----------------
# Create random effects model and Dataset
gp_model = gpb.GPModel(group_data=group, likelihood = "bernoulli_probit")
data_train = gpb.Dataset(X, y)
# Other parameters not contained in the grid of tuning parameters
params = { 'objective': 'binary', 'verbose': 0, 'num_leaves': 2**10, 'max_bin': 255 }

# Small grid and deterministic search
param_grid_small = {'learning_rate': [1, 0.1,0.01], 'min_data_in_leaf': [20,100],
                    'max_depth': [5,10]}
# Note: it is advisable to also try larger learning rates (e.g., 1 or larger) since when using 
#         gradient boosting the scale of the gradient can depend on the loss function and the data.
#         This is in contrast to Newton boosting, where learning rates smaller than 0.1 are typically used
#         since the natural gradient is not scale dependent.
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid_small,
                                             params=params,
                                             num_try_random=None,
                                             nfold=4,
                                             gp_model=gp_model,
                                             use_gp_model_for_validation=True,
                                             train_set=data_train,
                                             verbose_eval=1,
                                             num_boost_round=1000, 
                                             early_stopping_rounds=10,
                                             seed=1000,
                                             metrics='binary_logloss')
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))
print("Best parameters: " + str(opt_params['best_params']))
# I obtained the following best parameters:
#***** New best score (0.6053142918517325) found for the following parameter combination:
#{'learning_rate': 0.1, 'min_data_in_leaf': 100, 'max_depth': 5, 'num_boost_round': 44}

# larger grid and random search
param_grid_large = {'learning_rate': [5,1,0.5,0.1,0.05,0.01], 'min_data_in_leaf': [5,10,20,50,100,200],
                    'max_depth': [1,3,5,10,20], 'max_bin': [255,500,1000,2000]}
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid_large,
                                             params=params,
                                             num_try_random=10,
                                             nfold=4,
                                             gp_model=gp_model,
                                             use_gp_model_for_validation=True,
                                             train_set=data_train,
                                             verbose_eval=1,
                                             num_boost_round=1000, 
                                             early_stopping_rounds=10,
                                             seed=1000,
                                             metrics='binary_logloss')
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))
print("Best parameters: " + str(opt_params['best_params']))

# Using another metric (AUC) instead of the log-loss
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid_small,
                                             params=params,
                                             num_try_random=5,
                                             nfold=4,
                                             gp_model=gp_model,
                                             use_gp_model_for_validation=True,
                                             train_set=data_train,
                                             verbose_eval=1,
                                             num_boost_round=1000, 
                                             early_stopping_rounds=10,
                                             seed=100,
                                             metrics='auc')
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))
print("Best parameters: " + str(opt_params['best_params']))
# Note: it is coincidence that the AUC and the log-loss have similar values on this data

# --------------------Parameter tuning using a validation set----------------
# Define training and validation data by setting indices of 'folds'
permut = np.random.RandomState(10).choice(a=n, size=n, replace=False)
train_idx = permut[0:int(n/2)]
valid_idx = permut[int(n/2):n]
folds = [(train_idx, valid_idx)]
# Parameter tuning using validation data
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid_small,
                                             params=params,
                                             folds=folds,
                                             gp_model=gp_model,
                                             use_gp_model_for_validation=True,
                                             train_set=data_train,
                                             verbose_eval=1,
                                             num_boost_round=1000, 
                                             early_stopping_rounds=10,
                                             seed=1000,
                                             metrics='binary_logloss')
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))
print("Best parameters: " + str(opt_params['best_params']))

