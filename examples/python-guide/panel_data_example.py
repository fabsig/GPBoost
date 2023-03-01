# -*- coding: utf-8 -*-
"""
Example on how to use the GPBoost algorithm for panel data

@author: Fabio Sigrist
"""

import gpboost as gpb
from statsmodels.datasets import grunfeld
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Load data
data = grunfeld.load_pandas().data
 # Visualize response variable
plt.hist(data['invest'], bins=50)
plt.title("Histogram of response variable")

"""
Boosting with two crossed firm and year grouped random effects
"""
# Define random effects model (assuming firm and year random effects) 
gp_model = gpb.GPModel(group_data=data[['firm', 'year']])
# Create dataset for gpb.train
data_train = gpb.Dataset(data=data[['value', 'capital']], label=data['invest'])
# Specify boosting parameters as dict
# Note: no attempt has been done to optimaly choose tuning parameters
params = { 'objective': 'regression_l2',
            'learning_rate': 1,
            'max_depth': 6,
            'min_data_in_leaf': 1,
            'verbose': 0 }
# Train GPBoost model
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=1800)
# Estimated random effects model (variances of random effects)
gp_model.summary()

# Cross-validation for determining number of boosting iterations
gp_model = gpb.GPModel(group_data=data[['firm', 'year']])
data_train = gpb.Dataset(data=data[['value', 'capital']], label=data['invest'])
cvbst = gpb.cv(params=params, train_set=data_train,
               gp_model=gp_model, use_gp_model_for_validation=True,
               num_boost_round=5000, early_stopping_rounds=5,
               nfold=2, verbose_eval=True, show_stdv=False, seed=1)
print("Best number of iterations: " + str(np.argmin(cvbst['l2-mean'])))


"""
Linear mixed effecst model with two crossed firm and year grouped random effects
"""
lin_gp_model = gpb.GPModel(group_data=data[['firm', 'year']])
# Add interecept for linear model
X = data[['value', 'capital']]
X['intercept'] = 1
lin_gp_model.fit(y=data['invest'], X=X, params={"std_dev": True})
lin_gp_model.summary()


"""
Boosting with grouped firm random effects and one "global" AR(1) year random effect
     I.e., all firms share the same temporal AR(1) effect.
"""
gp_model_ar1 = gpb.GPModel(group_data=data['firm'], gp_coords=data['year'], cov_function="exponential")
data_train = gpb.Dataset(data=data[['value', 'capital']], label=data['invest'])
# Train GPBoost model (takes a few seconds)
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model_ar1,
                num_boost_round=1800)
# Estimated random effects model (variances of random effects and range parameters)
gp_model_ar1.summary()
cov_pars = gp_model_ar1.get_cov_pars()
phi_hat = np.exp(-1 / cov_pars['GP_range'][0])
sigma2_hat = cov_pars['GP_var'][0] * (1. - phi_hat ** 2)
print("Estimated innovation variance and AR(1) coefficient of year effect:")
print([sigma2_hat ,phi_hat])


"""
Boosting with grouped firm random effects and separate AR(1) year random effects per firm
     I.e., every firms has its own temporal AR(1) effect. This can be done using
     the 'cluster_ids' parameter.
"""
gp_model_ar1 = gpb.GPModel(group_data=data['firm'], gp_coords=data['year'], 
                           cluster_ids=data['firm'], cov_function="exponential")
# Need to use the more robust option gradient_descent instead of fisher_scoring in this example
gp_model_ar1.set_optim_params(params={"optimizer_cov": "gradient_descent"})
data_train = gpb.Dataset(data=data[['value', 'capital']], label=data['invest'])
# Train GPBoost model (takes a few seconds)
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model_ar1,
                num_boost_round=1800)
# Estimated random effects model (variances of random effects and range parameters)
gp_model_ar1.summary()
cov_pars = gp_model_ar1.get_cov_pars()
phi_hat = np.exp(-1 / cov_pars['GP_range'][0])
sigma2_hat = cov_pars['GP_var'][0] * (1. - phi_hat ** 2)
print("Estimated innovation variance and AR(1) coefficient of year effect:")
print([sigma2_hat ,phi_hat])

