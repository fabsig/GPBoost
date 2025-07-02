# -*- coding: utf-8 -*-
"""
## Examples on how to use GPBoost for 
##  - the Grabit model of Sigrist and Hirnschall (2019)
##  - mean-scale regression / heteroscedastic regression with a Gaussian likelihood

@author: Fabio Sigrist
"""

import sklearn.datasets as datasets
import numpy as np
import gpboost as gpb
import matplotlib.pyplot as plt

"""
Example 1
"""
# simulate data
np.random.seed(1)
n = 10000
X, lp = datasets.make_friedman3(n_samples=n)
X_test, lp_test = datasets.make_friedman3(n_samples=n)
lp = lp*5+0.2
lp_test = lp_test*5+0.2
y = np.random.normal(loc=lp,scale=1)
y_test = np.random.normal(loc=lp_test,scale=1)
# apply censoring
yu = 8
yl = 5
y[y>=yu] = yu
y[y<=yl] = yl
# censoring fractions
print(np.sum(y==yu) / n)
print(np.sum(y==yl) / n)

# train model and make predictions
params = {'objective': 'tobit', 'verbose': 0, 'yl': yl, 'yu': yu, 'sigma': 1.}
dtrain = gpb.Dataset(X, y)
bst = gpb.train(params=params, train_set=dtrain, num_boost_round=100)
y_pred = bst.predict(X_test)
# mean square error (approx. 1.1 for n=10'000)
print("Test error of Grabit: " + str(((y_pred-y_test)**2).mean()))
# compare to standard least squares gradient boosting (approx. 1.8 for n=10'000)
params = {'objective': 'regression_l2', 'verbose': 0}
bst = gpb.train(params=params, train_set=dtrain, num_boost_round=100)
y_pred_ls = bst.predict(X_test)
print("Test error of standard least squares gradient boosting: " + str(((y_pred_ls-y_test)**2).mean()))

# measure time
import time
start = time.time()
bst = gpb.train(params=params, train_set=dtrain, num_boost_round=100)
end = time.time()
print(end - start)
# approx. 0.1 sec for n='10'000 on a standard laptop


"""
Example 2: 2-d non-linear function
"""

def nonlin_fct(x1,x2):
    r=x1**2+x2**2
    r=np.pi*2*1*(r**0.75)
    f=2*np.cos(r)
    return(f)
def plot_2d_fct(x1, x2, y, *,
                    zlim=None, elev=30, azim=-60,
                    title="3-D surface", filename=None):
    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")
    surf_kw = dict(rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor='k')
    if zlim is not None:
        ax.set_zlim(zlim)
        surf_kw["vmin"], surf_kw["vmax"] = zlim
    surf = ax.plot_surface(x1, x2, y, **surf_kw)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("")
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.7, aspect=15)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, .95])
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
    else:
        plt.show(block=True)
    
##True function
nx = 100
x = np.arange(-1+1/nx,1,2/nx)
x1, x2 = np.meshgrid(x, x)
yt = nonlin_fct(x1,x2)
zlim = (-1.75,1.75)
plot_2d_fct(x1,x2,yt.reshape((100,-1)),title="True F",zlim=zlim)
        
# simulate data
n = 10000
np.random.seed(10)  
X = np.random.rand(n,2)
X = (X-0.5)*2
y = nonlin_fct(X[:,0],X[:,1])+np.random.normal(scale=1, size=n)
# apply xensoring
yc = y.copy()
yl = np.percentile(y,q=33.33)
yu = np.percentile(y,q=66.66)
yc[y>=yu] = yu
yc[y<=yl] = yl

# train Grabit model and make predictions
params = {'objective': 'tobit', 'verbose': 0, 'yl': yl, 'yu': yu, 'sigma': 1.,
          'learning_rate': 0.1, 'max_depth': 3}
dtrain = gpb.Dataset(X, yc)
bst = gpb.train(params=params, train_set=dtrain, num_boost_round=100)
X_pred = np.transpose(np.array([x1.flatten(),x2.flatten()]))
y_pred = bst.predict(X_pred)
plot_2d_fct(x1,x2,y_pred.reshape((100,-1)),title="Grabit",zlim=zlim)
# compare to standard least squares gradient boosting
params = {'objective': 'regression_l2', 'verbose': 0, 'yl': yl, 'yu': yu, 'sigma': 1.,
          'learning_rate': 0.1, 'max_depth': 3}
dtrain = gpb.Dataset(X, yc)
bst = gpb.train(params=params, train_set=dtrain, num_boost_round=100)
X_pred = np.transpose(np.array([x1.flatten(),x2.flatten()]))
y_pred = bst.predict(X_pred)
plot_2d_fct(x1,x2,y_pred.reshape((100,-1)),title="L2 Boosting",zlim=zlim)


####################
## Heteroscedastic mean-scale regression
####################

def f1d(x: np.ndarray) -> np.ndarray:
    """Non-linear fixed-effect function used in the original R code."""
    return 1 / (1 + np.exp(-(x - 0.5) * 10)) - 0.5

n = 1000
p = 5 # number of predictor variables
# Simulate data
np.random.seed(10)
X = np.random.rand(n, p)
x1 = X[:, 0]
f_mean  = f1d(x1)
f_stdev = np.exp(x1 * 3 - 4)
y = f_mean + f_stdev * np.random.randn(n)

dtrain = gpb.Dataset(data=X, label=y)

## Parameter tuning
param_grid = {
    "learning_rate":   [0.001, 0.01, 0.1, 1],
    "min_data_in_leaf":[1, 10, 100, 1000],
    "max_depth":       [-1],               # no limit; we tune 'num_leaves' instead
    "num_leaves":      [2 ** i for i in range(1, 11)],
    "lambda_l2":       [0, 1, 10, 100],
    "max_bin":         [250, 500, 1000, min(n, 10_000)]
}

metric = "crps_gaussian"
other_params = {'verbose': 0, 'objective': "mean_scale_regression"}
np.random.seed(1)
opt_params = gpb.grid_search_tune_parameters(param_grid = param_grid, train_set = dtrain,
    num_try_random = 100, nfold = 5, num_boost_round = 1000, early_stopping_rounds = 20,
    verbose_eval = 1, metric = metric, seed = 4, params = other_params)
print("Best parameters found:\n", opt_params)

# Train model
params = {"learning_rate": 0.1, "max_depth": -1, "num_leaves": 8,
    "max_bin": 250, "lambda_l2": 100, "min_data_in_leaf": 1, 'verbose': 0, 
    'objective': "mean_scale_regression" }
bst = gpb.train(params = params, train_set = dtrain, num_boost_round = 100)

# Make predictions
npred = 100
X_test       = np.zeros((npred, p))
X_test[:, 0] = np.linspace(0, 1, npred)
y_pred = bst.predict(X_test) 
pred_mean = y_pred["pred_mean"]
pred_sd = np.sqrt(y_pred["pred_var"])

# Plot data and predictions
plt.scatter(x1, y, s=10, alpha=0.4, label="data")
plt.plot(X_test[:, 0], pred_mean, lw=3, color="red", label="pred. mean")
plt.plot(X_test[:, 0], pred_mean + 2*pred_sd, lw=2, ls="--", color="red", label="mean ± 2 sd")
plt.plot(X_test[:, 0], pred_mean - 2*pred_sd, lw=2, ls="--", color="red")
plt.xlabel("X[:, 0]")
plt.ylabel("y / prediction")
plt.legend()
plt.show()
