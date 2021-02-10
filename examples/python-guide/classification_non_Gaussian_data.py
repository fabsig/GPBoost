# coding: utf-8
# pylint: disable = invalid-name, C0111
import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use('ggplot')
print("It is recommended that the examples are run in interactive mode")

"""
Examples of generalized linear Gaussian process and random effects models
for several non-Gaussian likelihoods
"""

# Choose likelihood: either "bernoulli_probit" (=default for binary data), "bernoulli_logit",
#                      "poisson", or "gamma"
likelihood = "bernoulli_probit"

# Non-linear function for simulation
def f1d(x):
    return 1 / (1 + np.exp(-(x - 0.5) * 20)) - 0.5

# Parameters for gpboost in examples below
# Note: the tuning parameters are by no means optimal for all situations considered here
params = {'learning_rate': 0.1, 'min_data_in_leaf': 20, 'objective': likelihood,
          'verbose': 0, 'monotone_constraints': [1, 0]}
num_boost_round = 25
if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    params['objective'] = 'binary'

# --------------------Grouped random effects model----------------
# Simulate data
n = 5000  # number of samples
m = 500  # number of groups
np.random.seed(1)
# simulate grouped random effects
group = np.arange(n)  # grouping variable
for i in range(m):
    group[int(i * n / m):int((i + 1) * n / m)] = i
b1 = 0.5 * np.random.normal(size=m)  # simulate random effects
eps = b1[group]
eps = eps - np.mean(eps)
# simulate fixed effects
X = np.random.rand(n, 2)
f = f1d(X[:, 0])
# simulate response variable
if likelihood == "bernoulli_probit":
    probs = stats.norm.cdf(f + eps)
    y = np.random.uniform(size=n) < probs
    y = y.astype(np.float64)
elif likelihood == "bernoulli_logit":
    probs = 1 / (1 + np.exp(-(f + eps)))
    y = np.random.uniform(size=n) < probs
    y = y.astype(np.float64)
elif likelihood == "poisson":
    mu = np.exp(f + eps)
    y = stats.poisson.ppf(np.random.uniform(size=n), mu=mu)
elif likelihood == "gamma":
    mu = np.exp(f + eps)
    y = mu * stats.gamma.ppf(np.random.uniform(size=n), a=1)
fig1, ax1 = plt.subplots()
ax1.hist(y, bins=50)  # visualize response variable

# create dataset for gpb.train
data_train = gpb.Dataset(X, y)
# Train model
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
# Use the option "trace": true to monitor convergence of hyperparameter estimation of the gp_model. E.g.:
# gp_model.set_optim_params(params={"trace": True})
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=num_boost_round)
gp_model.summary()  # Trained random effects model

# Make predictions
nplot = 200  # number of predictions
X_test_plot = np.column_stack((np.linspace(0, 1, nplot), np.zeros(nplot)))
group_data_pred = -np.ones(nplot)
# Predict response variable
pred_resp = bst.predict(data=X_test_plot, group_data_pred=group_data_pred, raw_score=False)
# Predict latent variable including variance
pred = bst.predict(data=X_test_plot, group_data_pred=group_data_pred,
                   predict_var=True, raw_score=True)

# Visualize predictions
fig1, ax1 = plt.subplots()
ax1.plot(X_test_plot[:, 0], f1d(X_test_plot[:, 0]), linewidth=2, label="True F")
ax1.plot(X_test_plot[:, 0], pred['fixed_effect'], linewidth=2, label="Pred F")
if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    ax1.scatter(X[:, 0], y, linewidth=2, color="black", alpha=0.02)
ax1.set_title("Data, true and predicted latent function F")
ax1.legend()

fig1, ax1 = plt.subplots()
ax1.plot(X_test_plot[:, 0], pred_resp['response_mean'], linewidth=2, label="Pred response")
ax1.scatter(X[:, 0], y, linewidth=2, color="black", alpha=0.02)
ax1.set_title("Data and predicted response variable")
ax1.legend()

# Cross-validation for finding number of iterations
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
cvbst = gpb.cv(params=params, train_set=data_train,
               gp_model=gp_model, use_gp_model_for_validation=True,
               num_boost_round=200, early_stopping_rounds=5,
               nfold=4, verbose_eval=True, show_stdv=False, seed=1)
print("Best number of iterations: " + str(np.argmin(next(iter(cvbst.values())))))

# Showing training loss
print('Showing training loss...')
gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=5,
                valid_sets=data_train)


# --------------------Gaussian process model----------------
# Simulate data
ntrain = 500  # number of samples
np.random.seed(4)
# training and test locations (=features) for Gaussian process
coords_train = np.column_stack((np.random.uniform(size=ntrain), np.random.uniform(size=ntrain)))
# exclude upper right corner
excl = ((coords_train[:, 0] >= 0.7) & (coords_train[:, 1] >= 0.7))
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
Sigma = sigma2_1 * np.exp(-D / rho) + np.diag(np.zeros(n) + 1e-20)
C = np.linalg.cholesky(Sigma)
b1 = np.random.normal(size=n)  # simulate random effects
eps = C.dot(b1)
eps = eps - np.mean(eps)
# simulate response variable
if likelihood == "bernoulli_probit":
    probs = stats.norm.cdf(f + eps)
    y = np.random.uniform(size=n) < probs
    y = y.astype(np.float64)
elif likelihood == "bernoulli_logit":
    probs = 1 / (1 + np.exp(-(f + eps)))
    y = np.random.uniform(size=n) < probs
    y = y.astype(np.float64)
elif likelihood == "poisson":
    mu = np.exp(f + eps)
    y = stats.poisson.ppf(np.random.uniform(size=n), mu=mu)
elif likelihood == "gamma":
    mu = np.exp(f + eps)
    y = mu * stats.gamma.ppf(np.random.uniform(size=n), a=1)
# Split into training and test data
y_train = y[0:ntrain]
data_train = gpb.Dataset(X_train, y_train)  # create dataset for gpb.train
y_test = y[ntrain:n]
eps_test = eps[ntrain:n]
plt.hist(y_train, bins=50)  # visualize response variable

# Train model
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential", likelihood=likelihood)
# Use the option "trace": true to monitor convergence of hyperparameter estimation of the gp_model. E.g.:
# gp_model.set_optim_params(params={"trace": True})
print("Train GPBoost model with GP model (takes a few seconds)")
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=num_boost_round)
gp_model.summary()  # Trained GP model

# Predict response variable
pred_resp = bst.predict(data=X_test, gp_coords_pred=coords_test, raw_score=False)
# Predict latent variable including variance
pred = bst.predict(data=X_test, gp_coords_pred=coords_test, predict_var=True, raw_score=True)

if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    print("Test error:")
    pred_binary = pred_resp['response_mean'] > 0.5
    pred_binary = pred_binary.astype(np.float64)
    print(np.mean(pred_binary != y_test))
else:
    print("Test root mean square error:")
    print(np.sqrt(np.mean((pred_resp['response_mean'] - y_test) ** 2)))
print("Test root mean square error for latent GP:")
print(np.sqrt(np.mean((pred['random_effect_mean'] - eps_test) ** 2)))

# Visualize predictions and compare to true values
fig, axs = plt.subplots(2, 2)
# data and true GP
eps_test_plot = eps_test.reshape((nx, nx))
CS = axs[0, 0].contourf(coords_test_x1, coords_test_x2, eps_test_plot)
axs[0, 0].plot(coords_train[:, 0], coords_train[:, 1], '+', color="white")
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

# Cross-validation for finding number of iterations (takes a few seconds)
gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential", likelihood=likelihood)
cvbst = gpb.cv(params=params, train_set=data_train,
               gp_model=gp_model, use_gp_model_for_validation=True,
               num_boost_round=200, early_stopping_rounds=5,
               nfold=4, verbose_eval=True, show_stdv=False, seed=1)
print("Best number of iterations: " + str(np.argmin(next(iter(cvbst.values())))))
