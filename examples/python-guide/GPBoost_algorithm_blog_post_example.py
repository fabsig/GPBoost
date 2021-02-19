# -*- coding: utf-8 -*-
"""
Demo code used in this blog post: 
https://towardsdatascience.com/tree-boosted-mixed-effects-models-4df610b624cb

@author: Fabio Sigrist
"""

import gpboost as gpb
import numpy as np
import sklearn.datasets as datasets
import time
import pandas as pd
print("It is recommended that the examples are run in interactive mode")

# --------------------Simulate data----------------
ntrain = 5000 # number of samples for training
n = 2 * ntrain # combined number of training and test data
m = 500  # number of categories / levels for grouping variable
sigma2_1 = 1  # random effect variance
sigma2 = 1 ** 2  # error variance
# Simulate non-linear mean function
np.random.seed(1)
X, F = datasets.make_friedman3(n_samples=n)
X = pd.DataFrame(X,columns=['variable_1','variable_2','variable_3','variable_4'])
F = F * 10**0.5 # with this choice, the fixed-effects regression function has the same variance as the random effects
# Simulate random effects
group_train = np.arange(ntrain)  # grouping variable
for i in range(m):
    group_train[int(i * ntrain / m):int((i + 1) * ntrain / m)] = i
group_test = np.arange(ntrain) # grouping variable for test data. Some existing and some new groups
m_test = 2 * m
for i in range(m_test):
    group_test[int(i * ntrain / m_test):int((i + 1) * ntrain / m_test)] = i
group = np.concatenate((group_train,group_test))
b = np.sqrt(sigma2_1) * np.random.normal(size=m_test)  # simulate random effects
Zb = b[group]
# Put everything together
xi = np.sqrt(sigma2) * np.random.normal(size=n)  # simulate error term
y = F + Zb + xi  # observed data
# split train and test data
y_train = y[0:ntrain]
y_test = y[ntrain:n]
X_train = X.iloc[0:ntrain,]
X_test = X.iloc[ntrain:n,]

# --------------------Learning and prediction----------------
# Define and train GPModel
gp_model = gpb.GPModel(group_data=group_train)
# create dataset for gpb.train function
data_train = gpb.Dataset(X_train, y_train)
# specify tree-boosting parameters as a dict
params = { 'objective': 'regression_l2', 'learning_rate': 0.1,
    'max_depth': 6, 'min_data_in_leaf': 5, 'verbose': 0 }
# train model
bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model, num_boost_round=32)
gp_model.summary() # estimated covariance parameters
# Covariance parameters in the following order:
# ['Error_term', 'Group_1']
# [0.9183072 1.013057 ]

# Make predictions
pred = bst.predict(data=X_test, group_data_pred=group_test)
y_pred = pred['fixed_effect'] + pred['random_effect_mean'] # sum predictions of fixed effect and random effect
np.sqrt(np.mean((y_test - y_pred) ** 2)) # root mean square error (RMSE) on test data. Approx. = 1.25

# Parameter tuning using cross-validation (only number of boosting iterations)
gp_model = gpb.GPModel(group_data=group_train)
cvbst = gpb.cv(params=params, train_set=data_train,
               gp_model=gp_model, use_gp_model_for_validation=False,
               num_boost_round=100, early_stopping_rounds=5,
               nfold=4, verbose_eval=True, show_stdv=False, seed=1)
best_iter = np.argmin(cvbst['l2-mean'])
print("Best number of iterations: " + str(best_iter))
# Best number of iterations: 32

# --------------------Model interpretation----------------
# Plotting feature importances
gpb.plot_importance(bst)

# Partial dependence plots
from pdpbox import pdp
# Single variable plots (takes a few seconds to compute)
pdp_dist = pdp.pdp_isolate(model=bst, dataset=X_train, model_features=X_train.columns,
                           feature='variable_2', num_grid_points=50)
pdp.pdp_plot(pdp_dist, 'variable_2', plot_lines=True)
# Two variable interaction plot
inter_rf = pdp.pdp_interact(model=bst, dataset=X_train, model_features=X_train.columns,
                             features=['variable_1','variable_2'])
pdp.pdp_interact_plot(inter_rf, ['variable_1','variable_2'], x_quantile=True,
                      plot_type='contour', plot_pdp=True)# ignore any error message

# SHAP values and dependence plots
# Note: you need shap version>=0.36.0
import shap
shap_values = shap.TreeExplainer(bst).shap_values(X_test)
shap.summary_plot(shap_values, X_test)
shap.dependence_plot("variable_2", shap_values, X_test)


# --------------------Comparison to alternative approaches----------------
results = pd.DataFrame(columns = ["RMSE","Time"],
                       index = ["GPBoost", "Linear_ME","Boosting_Ign","Boosting_Cat","MERF"])
# 1. GPBoost
gp_model = gpb.GPModel(group_data=group_train)
start_time = time.time() # measure time
bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model, num_boost_round=best_iter)
results.loc["GPBoost","Time"] = time.time() - start_time
pred = bst.predict(data=X_test, group_data_pred=group_test)
y_pred = pred['fixed_effect'] + pred['random_effect_mean'] # sum predictions of fixed effect and random effect
results.loc["GPBoost","RMSE"] = np.sqrt(np.mean((y_test - y_pred) ** 2))

# 2. Linear mixed effects model ('Linear_ME')
gp_model = gpb.GPModel(group_data=group_train)
X_train_linear = np.column_stack((np.ones(ntrain),X_train))
X_test_linear = np.column_stack((np.ones(ntrain),X_test))
start_time = time.time() # measure time
gp_model.fit(y=y_train, X=X_train_linear) # add a column of 1's for intercept
results.loc["Linear_ME","Time"] = time.time() - start_time
y_pred = gp_model.predict(group_data_pred=group_test, X_pred=X_test_linear)
F_pred = X_test_linear.dot(gp_model.get_coef())
results.loc["Linear_ME","RMSE"] = np.sqrt(np.mean((y_test - y_pred['mu']) ** 2))

# 3. Gradient tree-boosting ignoring the grouping variable ('Boosting_Ign')
cvbst = gpb.cv(params=params, train_set=data_train,
               num_boost_round=100, early_stopping_rounds=5,
               nfold=4, verbose_eval=True, show_stdv=False, seed=1)
best_iter = np.argmin(cvbst['l2-mean'])
print("Best number of iterations: " + str(best_iter))
# Best number of iterations: 19
start_time = time.time() # measure time
bst = gpb.train(params=params, train_set=data_train, num_boost_round=best_iter)
results.loc["Boosting_Ign","Time"] = time.time() - start_time
y_pred = bst.predict(data=X_test)
results.loc["Boosting_Ign","RMSE"] = np.sqrt(np.mean((y_test - y_pred) ** 2))

# 4. Gradient tree-boosting including the grouping variable as a categorical variable ('Boosting_Cat')
X_train_cat = np.column_stack((group_train,X_train))
X_test_cat = np.column_stack((group_test,X_test))
data_train_cat = gpb.Dataset(X_train_cat, y_train, categorical_feature=[0])
cvbst = gpb.cv(params=params, train_set=data_train_cat,
               num_boost_round=1000, early_stopping_rounds=5,
               nfold=4, verbose_eval=True, show_stdv=False, seed=1)
best_iter = np.argmin(cvbst['l2-mean'])
print("Best number of iterations: " + str(best_iter))
# Best number of iterations: 49
start_time = time.time() # measure time
bst = gpb.train(params=params, train_set=data_train_cat, num_boost_round=best_iter)
results.loc["Boosting_Cat","Time"] = time.time() - start_time
y_pred = bst.predict(data=X_test_cat)
results.loc["Boosting_Cat","RMSE"] = np.sqrt(np.mean((y_test - y_pred) ** 2))

# 5. Mixed-effects random forest ('MERF')
from merf import MERF
rf_params={'max_depth': 6, 'n_estimators': 300}
merf_model = MERF(max_iterations=100, rf_params=rf_params)
print("Warning: the following takes a lot of time")
start_time = time.time() # measure time
merf_model.fit(pd.DataFrame(X_train), np.ones(shape=(ntrain,1)), pd.Series(group_train), y_train)
results.loc["MERF","Time"] = time.time() - start_time
y_pred = merf_model.predict(pd.DataFrame(X_test), np.ones(shape=(ntrain,1)), pd.Series(group_test))
results.loc["MERF","RMSE"] = np.sqrt(np.mean((y_test - y_pred) ** 2))

print(results.apply(pd.to_numeric).round(3))
