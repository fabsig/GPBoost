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

# Define random effects model (assuming firm and year random effects) 
gp_model = gpb.GPModel(group_data=data[['firm', 'year']])
# Create dataset for gpb.train
data_train = gpb.Dataset(data=data[['value', 'capital']], label=data['invest'])
# Specify boosting parameters as dict
# Note: no attempt has been done in appropriately selecting tuning parameters
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
# Estimated random effects model
gp_model.summary()

# Cross-validation for determining number of iterations
gp_model = gpb.GPModel(group_data=data[['firm', 'year']])
data_train = gpb.Dataset(data=data[['value', 'capital']], label=data['invest'])
cvbst = gpb.cv(params=params, train_set=data_train,
               gp_model=gp_model, use_gp_model_for_validation=True,
               num_boost_round=5000, early_stopping_rounds=5,
               nfold=2, verbose_eval=True, show_stdv=False, seed=1)
print("Best number of iterations: " + str(np.argmin(cvbst['l2-mean'])))

# Linear mixed effecst models (with firm and year random effects)
lin_gp_model = gpb.GPModel(group_data=data[['firm', 'year']])
# Add interecept for linear model
X = data[['value', 'capital']]
X['intercept'] = 1
lin_gp_model.fit(y=data['invest'], X=X, params={"std_dev": True})
lin_gp_model.summary()
