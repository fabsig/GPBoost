# coding: utf-8
# pylint: disable = invalid-name, C0111
import numpy as np
import gpboost as gpb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

print('Simulating data...')
# Simulate and create your dataset

def f1d(x):
    """Non-linear function for simulation"""
    return(1.7*(1/(1+np.exp(-(x-0.5)*20))+0.75*x))
def sim_data(n):
    """Function that simulates data. Two covariates of which only one has an effect"""
    X = np.random.rand(n,2)
    # mean function plus noise
    y = f1d(X[:,0]) + np.random.normal(scale=0.1, size=n)
    return([X,y])

# Simulate data
n = 1000
data = sim_data(2 * n)
Xtrain = data[0][0:n,:]
ytrain = data[1][0:n]
Xtest = data[0][n:(2*n),:]
ytest = data[1][n:(2*n)]


print('Starting training...')
# train
bst = gpb.GPBoostRegressor(max_depth=6,
                        learning_rate=0.1,
                        n_estimators=100)
bst.fit(Xtrain, ytrain,
        eval_set=[(Xtest, ytest)],
        eval_metric='l1',
        early_stopping_rounds=5)

print('Starting predicting...')
# predict
y_pred = bst.predict(Xtest, num_iteration=bst.best_iteration_)
# eval
print('The rmse of prediction is:', mean_squared_error(ytest, y_pred) ** 0.5)

# feature importances
print('Feature importances:', list(bst.feature_importances_))


# self-defined eval metric
# f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# l4 loss
def l4_loss(y_true, y_pred):
    return 'l4_loss', np.mean((y_pred - y_true) ** 4), False

print('Starting training with custom eval function...')
# train
bst.fit(Xtrain, ytrain,
        eval_set=[(Xtest, ytest)],
        eval_metric=l4_loss,
        early_stopping_rounds=5)


# other scikit-learn modules
bstcv = gpb.GPBoostRegressor()

# max_depth
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 50],
    'max_depth': [1, 5, 10]
}

bst = GridSearchCV(bstcv, param_grid, cv=3)
bst.fit(Xtrain, ytrain)

print('Best parameters found by grid search are:', bst.best_params_)
