# coding: utf-8
# pylint: disable = invalid-name, C0111
import gpboost as gpb
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
print("It is recommended that the examples are run in interactive mode")

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
Xtrain = data[0][0:n, :]
ytrain = data[1][0:n]
Xtest = data[0][n:(2 * n), :]
ytest = data[1][n:(2 * n)]

# create dataset for gpb.train
data_train = gpb.Dataset(Xtrain, ytrain)
data_eval = gpb.Dataset(Xtest, ytest, reference=data_train)

# specify your configurations as a dict
params = {
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_data_in_leaf': 5,
    'verbose': 0
}

print('Starting training...')
# train
evals_result = {}  # record eval results for plotting
bst = gpb.train(params=params,
                train_set=data_train,
                num_boost_round=100,
                valid_sets=data_eval,
                early_stopping_rounds=5,
                evals_result = evals_result)

# plot validation scores
gpb.plot_metric(evals_result, metric='l1', figsize=(10, 5))
plt.show()

print('Saving model...')
# save model to file
bst.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = bst.predict(Xtest, num_iteration=bst.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(ytest, y_pred) ** 0.5)

# compare fit to truth
Xtest_plot = np.zeros((200, 2))
Xtest_plot[:, 0] = np.linspace(0, 1, 200, endpoint=True)
y_pred_plot = bst.predict(Xtest_plot, num_iteration=bst.best_iteration)
plt.plot(Xtest_plot[:, 0], y_pred_plot, linewidth=2, color="b", label="fit")
plt.plot(x, f1d(x), linewidth=2, color="r", label="true")
plt.title("Comparison of true and fitted value")
plt.legend()
plt.show()

print('Starting training with custom evaluation function...')
# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
# quantile loss
def quantile_loss(preds, train_data):
    alpha = 0.95
    labels = train_data.get_label()
    diff = labels - preds
    dummy = diff < 0
    return 'quantile_loss', np.mean((alpha - dummy) * diff), False

params = {
    'objective': 'regression_l2',
    'metric': 'quantile_loss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_data_in_leaf': 5,
    'verbose': 0
}

bst = gpb.train(params=params,
                train_set=data_train,
                num_boost_round=100,
                feval=quantile_loss,
                valid_sets=data_eval,
                early_stopping_rounds=5)


# Several custom evaluation functions
# l4 loss
def l4_loss(preds, train_data):
    labels = train_data.get_label()
    return 'l4_loss', np.mean((preds - labels) ** 4), False


bst = gpb.train(params=params,
                train_set=data_train,
                num_boost_round=100,
                feval=lambda preds, train_data: [quantile_loss(preds, train_data),
                                                 l4_loss(preds, train_data)],
                valid_sets=data_eval,
                early_stopping_rounds=5)

# feature importances
bst = gpb.train(params=params,
                train_set=data_train,
                num_boost_round=20)

print('Feature importances:', list(bst.feature_importance()))

print('Plotting feature importances...')
ax = gpb.plot_importance(bst, max_num_features=10)
plt.show()

print('Plotting split value histogram...')
ax = gpb.plot_split_value_histogram(bst, feature='Column_0', bins='auto')
plt.show()

print('Plotting 10th tree...')  # one tree use categorical feature to split
ax = gpb.plot_tree(bst, tree_index=10, figsize=(15, 15), show_info=['split_gain'])
plt.show()

print('Starting training Boosting with Nesterov accelaration...')
# Boosting with Nesterov acceleration
params = {
    'objective': 'regression_l2',
    'metric': {'l2'},
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_data_in_leaf': 5,
    'verbose': 1,
    'use_nesterov_acc': True
}
bst = gpb.train(params=params,
                train_set=data_train,
                num_boost_round=100,
                valid_sets=data_eval,
                early_stopping_rounds=5)
# predict
y_pred = bst.predict(Xtest, num_iteration=bst.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(ytest, y_pred) ** 0.5)
