# -*- coding: utf-8 -*-
"""
Examples on how to use GPBoost for the Grabit model of Sigrist and Hirnschall (2019)

@author: Fabio Sigrist
"""

import sklearn.datasets as datasets
import numpy as np
import gpboost as gpb

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nonlin_fct(x1,x2):
    r=x1**2+x2**2
    r=np.pi*2*1*(r**0.75)
    f=2*np.cos(r)
    return(f)
def plot_2d_fct(x1,x2,y,title="2d function",elev=45,azim=120,zlim=None,filename=None):
    fig = plt.figure(figsize=(8, 7))
    ax = Axes3D(fig)
    if zlim is not None:
        ax.set_zlim3d(zlim)
        surf = ax.plot_surface(x1, x2, y, rstride=1, cstride=1,
                       cmap=plt.cm.BuPu, edgecolor='k',vmax=zlim[1])
    else:
        surf = ax.plot_surface(x1, x2, y, rstride=1, cstride=1,
                   cmap=plt.cm.BuPu, edgecolor='k')
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel('')
    #  pretty init view
    ax.view_init(elev=elev, azim=azim)
    plt.colorbar(surf)
    plt.suptitle(title)
    plt.subplots_adjust(top=0.9)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename,dpi=200)
    
##True function
nx = 100
x = np.arange(-1+1/nx,1,2/nx)
x1, x2 = np.meshgrid(x, x)
yt = nonlin_fct(x1,x2)
zlim = (-1.75,1.75)
plot_2d_fct(x1,x2,yt,title="True F",zlim=zlim)
        
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

