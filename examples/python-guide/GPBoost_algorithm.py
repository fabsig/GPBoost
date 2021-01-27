# coding: utf-8
# pylint: disable = invalid-name, C0111
import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
print("It is recommended that the examples are run in interactive mode")

# --------------------Combine tree-boosting and grouped random effects model----------------
print('Simulating data...')

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
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_data_in_leaf': 5,
            'verbose': 1 }

print('Training GPBoost model...')
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=16)
print("Estimated random effects model")
gp_model.summary()

print('Starting predicting...')
# predict
group_test = np.arange(m)
Xtest = np.zeros((m, 2))
Xtest[:, 0] = np.linspace(0, 1, m)
pred = bst.predict(data=Xtest, group_data_pred=group_test)
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
plt.plot(x, f1d(x), linewidth=2, color="r", label="true")
plt.title("Comparison of true and fitted fixed effect")
plt.legend()
plt.show()

# Showing training loss
print('Showing training loss...')
gp_model = gpb.GPModel(group_data=group)
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=16,
                valid_sets=data_train)

# Using validation set
print('Using validation set...')
np.random.seed(1)
train_ind = np.random.choice(n, int(0.9 * n), replace=False)
test_ind = [i for i in range(n) if i not in train_ind]
data_train = gpb.Dataset(X[train_ind, :], y[train_ind])
data_eval = gpb.Dataset(X[test_ind, :], y[test_ind], reference=data_train)
gp_model = gpb.GPModel(group_data=group[train_ind])

# Do not include random effect predictions for validation
print("Training with validation data and use_gp_model_for_validation = False")
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

# Include random effect predictions for validation (observe the lower test error)
print("Training with validation data and use_gp_model_for_validation = True")
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

# Do Newton updates for tree leaves
print("Training with Newton updates for tree leaves")
params = {
    'objective': 'regression_l2',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_data_in_leaf': 5,
    'verbose': 0,
    'leaves_newton_update': True
}
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

# --------------------Combine tree-boosting and Gaussian process model----------------
print('Simulating data...')

# Simulate data
def f1d(x):
    """Non-linear function for simulation"""
    return (1.7 * (1 / (1 + np.exp(-(x - 0.5) * 20)) + 0.75 * x))

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
params = {
    'objective': 'regression_l2',
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_data_in_leaf': 5,
    'verbose': 0
}

print('Starting training...')
# train
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=8)
print("Estimated random effects model")
gp_model.summary()

print('Starting predicting...')
# Make predictions
np.random.seed(1)
ntest = 5
Xtest = np.random.rand(ntest, 2)
# prediction locations (=features) for Gaussian process
coords_test = np.column_stack(
    (np.random.uniform(size=ntest), np.random.uniform(size=ntest))) / 10.
pred = bst.predict(data=Xtest, gp_coords_pred=coords_test, predict_cov_mat=True)
print("Predicted fixed effect from tree ensemble")
pred['fixed_effect']
print("Predicted (posterior) mean of GP")
pred['random_effect_mean']
print("Predicted (posterior) covariance matrix of GP")
pred['random_effect_cov']

