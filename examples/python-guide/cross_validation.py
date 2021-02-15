# coding: utf-8
# pylint: disable = invalid-name, C0111
import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
print("It is recommended that the examples are run in interactive mode")
print("This script shows how cross-validation can be done for finding the number of iterations")

#--------------------Cross validation for tree-boosting without GP or random effects----------------
print('Simulating data...')
# Simulate and create your dataset
def f1d(x):
    """Non-linear function for simulation"""
    return (1.7 * (1 / (1 + np.exp(-(x - 0.5) * 20)) + 0.75 * x))
x = np.linspace(0, 1, 200, endpoint=True)
plt.plot(x, f1d(x), linewidth=2, color="r")
plt.title("Mean function")
plt.show()
def sim_data(n):
    """Function that simulates data. Two covariates of which only one has an effect"""
    X = np.random.rand(n, 2)
    # mean function plus noise
    y = f1d(X[:, 0]) + np.random.normal(scale=0.1, size=n)
    return ([X, y])

# Simulate data
n = 1000
data = sim_data(2 * n)
# create dataset for gpb.train
data_train = gpb.Dataset(data[0][0:n, :], data[1][0:n])

# specify your configurations as a dict
params = { 'objective': 'regression_l2',
            'metric': {'l2', 'l1'},
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_data_in_leaf': 5,
            'verbose': 0 }

# do cross-validation
cvbst = gpb.cv(params=params, train_set=data_train,
               num_boost_round=100, early_stopping_rounds=5,
               nfold=2, verbose_eval=True, show_stdv=False, seed=1)
print("Best number of iterations: " + str(np.argmin(cvbst['l2-mean'])))


# --------------------Combine tree-boosting and grouped random effects model----------------
# Simulate data
def f1d(x):
    """Non-linear function for simulation"""
    return (1.7 * (1 / (1 + np.exp(-(x - 0.5) * 20)) + 0.75 * x))
x = np.linspace(0, 1, 200, endpoint=True)
plt.figure("Mean function")
plt.plot(x, f1d(x), linewidth=2, color="r")
plt.title("Mean function")
plt.show()
n = 1000  # number of samples
np.random.seed(1)
X = np.random.rand(n, 2)
F = f1d(X[:, 0])
# Simulate grouped random effects
m = 25  # number of categories / levels for grouping variable
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
# incidence matrix relating grouped random effects to samples
Z1 = np.zeros((n, m))
for i in range(m):
    Z1[np.where(group == i), i] = 1
sigma2_1 = 1 ** 2  # random effect variance
sigma2 = 0.1 ** 2  # error variance
b1 = np.sqrt(sigma2_1) * np.random.normal(size=m)  # simulate random effects
eps = Z1.dot(b1)
xi = np.sqrt(sigma2) * np.random.normal(size=n)  # simulate error term
y = F + eps + xi  # observed data

# define GPModel
gp_model = gpb.GPModel(group_data=group)
gp_model.set_optim_params(params={"optimizer_cov": "fisher_scoring"})
# create dataset for gpb.train
data_train = gpb.Dataset(X, y)
# specify your configurations as a dict
params = { 'objective': 'regression_l2',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_data_in_leaf': 5,
            'verbose': 0 }

# do cross-validation
gp_model = gpb.GPModel(group_data=group)
cvbst = gpb.cv(params=params, train_set=data_train,
               gp_model=gp_model, use_gp_model_for_validation=True,
               num_boost_round=100, early_stopping_rounds=5,
               nfold=2, verbose_eval=True, show_stdv=False, seed=1)
print("Best number of iterations: " + str(np.argmin(cvbst['l2-mean'])))

# Do not include random effect predictions for validation (observe the higher test error)
cvbst = gpb.cv(params=params, train_set=data_train,
               gp_model=gp_model, use_gp_model_for_validation=False,
               num_boost_round=100, early_stopping_rounds=5,
               nfold=2, verbose_eval=True, show_stdv=False, seed=1)
print("Best number of iterations: " + str(np.argmin(cvbst['l2-mean'])))

