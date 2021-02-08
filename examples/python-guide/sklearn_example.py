# coding: utf-8
# pylint: disable = invalid-name, C0111
import numpy as np
import gpboost as gpb
import random

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

print('Simulating data...')

# Simulate data
n = 5000  # number of samples
m = 500  # number of groups
# Simulate grouped random effects
np.random.seed(1)
# simulate grouped random effects
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
b1 = np.random.normal(size=m)  # simulate random effects
eps = b1[group]
# simulate fixed effects
def f1d(x):
    """Non-linear function for simulation"""
    return (1.7 * (1 / (1 + np.exp(-(x - 0.5) * 20)) + 0.75 * x))
X = np.random.rand(n, 2)
f = f1d(X[:, 0])
xi = np.sqrt(0.01) * np.random.normal(size=n)  # simulate error term
y = f + eps + xi  # observed data

print('Starting training...')
# define GPModel
gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
# train
bst = gpb.GPBoostRegressor(max_depth=6,
                           learning_rate=0.05,
                           min_data_in_leaf=5,
                           n_estimators=15)
bst.fit(X, y, gp_model=gp_model)
print("Estimated random effects model")
gp_model.summary()

print('Starting predicting...')
# predict
group_test = np.arange(m)
Xtest = np.zeros((m, 2))
Xtest[:, 0] = np.linspace(0, 1, m)
pred = bst.predict(X=Xtest, group_data_pred=group_test)
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

# feature importances
print('Feature importances:', list(bst.feature_importances_))

# Using validation set
print('Using validation set...')
# split into training an test data
train_ind = random.sample(range(n), int(n / 2))
test_ind = [x for x in range(n) if (x not in train_ind)]
X_train = X[train_ind, :]
y_train = y[train_ind]
group_train = group[train_ind]
X_test = X[test_ind, :]
y_test = y[test_ind]
group_test = group[test_ind]
# train
gp_model = gpb.GPModel(group_data=group_train, likelihood="gaussian")
gp_model.set_prediction_data(group_data_pred=group_test)
bst = gpb.GPBoostRegressor(max_depth=6,
                           learning_rate=0.05,
                           min_data_in_leaf=5,
                           n_estimators=100)
bst.fit(X_train, y_train, gp_model=gp_model,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)
