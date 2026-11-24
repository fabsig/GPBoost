# coding: utf-8
"""Tests of the functions for choosing tuning parameters with optuna (TPE) and SMAC."""
import logging
import os
import subprocess
import sys

import numpy as np
import pytest
from sklearn.model_selection import KFold

import gpboost as gpb
from gpboost.engine import _parse_tuning_search_space, _tuning_deterministic_params


def _simulate_data(n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    group = np.repeat(np.arange(20), n // 20)
    b = rng.normal(scale=0.5, size=20)
    y = np.sin(4 * X[:, 0]) + X[:, 1] + b[group] + rng.normal(scale=0.3, size=n)
    return X, y, group


def _folds(n):
    idx = np.arange(n)
    return [(idx[idx % 2 == 0], idx[idx % 2 == 1]), (idx[idx % 2 == 1], idx[idx % 2 == 0])]


def _cv_best_score(params, X, y, folds, metric, gp_model=None, higher_better=False):
    # The same parameters for the construction of the histograms as in the tuning functions
    params = dict(params, verbose=-1, force_col_wise=True, deterministic=True)
    cvbst = gpb.cv(params=params, train_set=gpb.Dataset(X, y), gp_model=gp_model, num_boost_round=50,
                   early_stopping_rounds=5, folds=folds, metric=metric, verbose_eval=False, show_stdv=False)
    scores = cvbst[list(cvbst.keys())[0]]
    return np.max(scores) if higher_better else np.min(scores)


def _simulate_spatial_data(n=200, seed=2):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(size=(n, 2))
    X = rng.uniform(size=(n, 2))
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    b = np.linalg.cholesky(np.exp(-dist / 0.2) + 1e-10 * np.eye(n)) @ rng.normal(size=n)
    y = np.sin(4 * X[:, 0]) + X[:, 1] + b + rng.normal(scale=0.3, size=n)
    return X, y, coords


def _spatial_gp_model(coords, cov_function='matern', cov_fct_shape=2.5):
    gp_model = gpb.GPModel(gp_coords=coords, cov_function=cov_function, cov_fct_shape=cov_fct_shape,
                           gp_approx='vecchia', num_neighbors=10, vecchia_ordering='random', seed=1)
    gp_model.set_prediction_data(vecchia_pred_type='order_obs_first_cond_obs_only', num_neighbors_pred=10)
    return gp_model


def _cv_score_of_best_params(best_params, best_iter, X, y, coords, folds):
    """Score of 'cv()' after 'best_iter' iterations with the parameters of the booster and the GP model chosen by a
    tuning function."""
    params = dict(best_params, verbose=-1, force_col_wise=True, deterministic=True)
    gp_model_params = {param: params.pop(param) for param in ['cov_fct_shape', 'cov_function'] if param in params}
    cvbst = gpb.cv(params=params, train_set=gpb.Dataset(X, y), gp_model=_spatial_gp_model(coords, **gp_model_params),
                   num_boost_round=best_iter, folds=folds, metric='test_neg_log_likelihood', verbose_eval=False)
    return cvbst[list(cvbst.keys())[0]][-1]


def test_parse_tuning_search_space():
    parsed = _parse_tuning_search_space({'learning_rate': [0.001, 1], 'lambda_l2': [0, 100], 'reg_alpha': [1, 10],
                                         'num_leaves': [2, 1024], 'max_depth': [-1, 10],
                                         'feature_fraction': [0.5, 1], 'line_search_step_length': [True, False]})
    assert parsed['learning_rate'] == {'kind': 'float', 'lower': 0.001, 'upper': 1, 'log': True}
    assert parsed['lambda_l2'] == {'kind': 'float', 'lower': 1e-8, 'upper': 100, 'log': True}
    assert parsed['reg_alpha'] == {'kind': 'float', 'lower': 1, 'upper': 10, 'log': True}
    assert parsed['num_leaves'] == {'kind': 'int', 'lower': 2, 'upper': 1024, 'log': True}
    assert parsed['max_depth'] == {'kind': 'int', 'lower': -1, 'upper': 10, 'log': False}
    assert parsed['feature_fraction'] == {'kind': 'float', 'lower': 0.5, 'upper': 1, 'log': False}
    assert parsed['line_search_step_length'] == {'kind': 'categorical', 'values': [True, False]}
    with pytest.raises(ValueError, match="lower bound >= 0"):
        _parse_tuning_search_space({'lambda_l1': [-1, 10]})
    with pytest.raises(ValueError, match="upper bound > 0"):
        _parse_tuning_search_space({'lambda_l2': [0, 0]})
    with pytest.raises(ValueError, match="positive bounds"):
        _parse_tuning_search_space({'learning_rate': [0, 1]})
    with pytest.raises(ValueError, match="Unknown parameter"):
        _parse_tuning_search_space({'not_a_parameter': [0, 1]})
    with pytest.raises(ValueError, match="length 2"):
        _parse_tuning_search_space({'learning_rate': [0.1, 0.5, 1]})
    # Parameters on a logarithmic scale for which 0 is a valid value
    parsed = _parse_tuning_search_space({'min_gain_to_split': [0, 1], 'min_sum_hessian_in_leaf': [0, 10],
                                         'min_data_in_leaf': [0, 50], 'min_child_samples': [5, 50]})
    assert parsed['min_gain_to_split'] == {'kind': 'float', 'lower': 1e-8, 'upper': 1, 'log': True}
    assert parsed['min_sum_hessian_in_leaf'] == {'kind': 'float', 'lower': 1e-8, 'upper': 10, 'log': True}
    assert parsed['min_data_in_leaf'] == {'kind': 'int', 'lower': 1, 'upper': 50, 'log': True}
    assert parsed['min_child_samples'] == {'kind': 'int', 'lower': 5, 'upper': 50, 'log': True}
    with pytest.raises(ValueError, match="lower bound >= 0"):
        _parse_tuning_search_space({'min_data_in_leaf': [-1, 50]})
    with pytest.raises(ValueError, match="positive bounds"):
        _parse_tuning_search_space({'num_leaves': [0, 1024]})
    with pytest.raises(ValueError, match="not larger than the upper bound"):
        _parse_tuning_search_space({'feature_fraction': [1, 0.5]})


def test_parse_tuning_search_space_values():
    # Sets of numbers are sorted, and numpy numbers are converted to Python numbers
    parsed = _parse_tuning_search_space({'cov_fct_shape': {'values': [2.5, 0.5, np.float64(1.5)]},
                                         'cov_function': {'values': ['matern', 'gaussian']},
                                         'line_search_step_length': {'values': [True, False]},
                                         'max_depth': {'values': [5, -1]}, 'learning_rate': {'values': [0.1]},
                                         'cov_fct_order': [1, 2]})
    assert parsed['cov_fct_shape'] == {'kind': 'ordinal', 'values': [0.5, 1.5, 2.5]}
    assert type(parsed['cov_fct_shape']['values'][1]) is float
    assert parsed['cov_function'] == {'kind': 'categorical', 'values': ['matern', 'gaussian']}
    assert parsed['line_search_step_length'] == {'kind': 'categorical', 'values': [True, False]}
    assert parsed['max_depth'] == {'kind': 'ordinal', 'values': [-1, 5]}
    assert parsed['learning_rate'] == {'kind': 'ordinal', 'values': [0.1]}
    assert parsed['cov_fct_order'] == {'kind': 'int', 'lower': 1, 'upper': 2, 'log': False}
    # A list of two numbers is a range
    assert _parse_tuning_search_space({'cov_fct_shape': [0.5, 2.5]})['cov_fct_shape'] == \
        {'kind': 'float', 'lower': 0.5, 'upper': 2.5, 'log': False}
    with pytest.raises(ValueError, match=r"must be a list \[lower, upper\] or a dictionary \{'values'"):
        _parse_tuning_search_space({'cov_fct_shape': {'value': [0.5, 1.5]}})
    with pytest.raises(ValueError, match="at least one value"):
        _parse_tuning_search_space({'cov_fct_shape': {'values': []}})
    with pytest.raises(ValueError, match="must not contain a value twice"):
        _parse_tuning_search_space({'cov_fct_shape': {'values': [0.5, 1.5, 0.5]}})
    with pytest.raises(ValueError, match=r"search_space\['cov_function'\] must be a set of values"):
        _parse_tuning_search_space({'cov_function': ['matern', 'gaussian']})
    with pytest.raises(ValueError, match="Unknown parameter 'num_neighbors'"):
        _parse_tuning_search_space({'num_neighbors': {'values': [10, 20]}})
    # Parameters of the GP model require a GP model
    X, y, _ = _simulate_data()
    with pytest.raises(ValueError, match="'cov_fct_shape' is a parameter of the GP model and can only be tuned with"):
        gpb.tune_pars_TPE_algorithm_optuna(search_space={'cov_fct_shape': {'values': [0.5, 1.5]}}, n_trials=1, X=X,
                                           y=y)
    with pytest.raises(ValueError, match="'cov_function' is a parameter of the GP model and can only be tuned with"):
        gpb.grid_search_tune_parameters(param_grid={'cov_function': ['matern', 'gaussian']},
                                        train_set=gpb.Dataset(X, y))


def test_tuning_deterministic_params():
    assert _tuning_deterministic_params({}) == {'force_col_wise': True, 'deterministic': True}
    # Parameters given by the user are not changed
    assert _tuning_deterministic_params({'force_row_wise': True}) == {'force_row_wise': True, 'deterministic': True}
    assert _tuning_deterministic_params({'force_col_wise': False, 'deterministic': False}) == \
        {'force_col_wise': False, 'deterministic': False}
    assert _tuning_deterministic_params({}, {'force_row_wise': [True, False]}) == {'deterministic': True}


def test_tune_pars_TPE_algorithm_optuna():
    X, y, _ = _simulate_data()
    search_space = {'learning_rate': [0.01, 1], 'lambda_l2': [0, 100], 'min_data_in_leaf': [1, 50]}
    folds = _folds(len(y))
    opt_params = gpb.tune_pars_TPE_algorithm_optuna(search_space=search_space, n_trials=6, X=X, y=y,
                                                    max_num_boost_round=50, early_stopping_rounds=5,
                                                    folds=folds, metric='mse', params={'verbose': -1},
                                                    tpe_seed=1, verbose_eval=0)
    best_params = opt_params['best_params']
    assert 0.01 <= best_params['learning_rate'] <= 1
    assert 1e-8 <= best_params['lambda_l2'] <= 100
    assert 1 <= best_params['min_data_in_leaf'] <= 50
    assert opt_params['best_iter'] >= 1
    best_score = _cv_best_score(best_params, X, y, folds, 'mse')
    assert abs(opt_params['best_score'][0] - best_score) < 1e-10
    # Folds given as a generator are used in every trial, and so are scikit-learn splitters
    opt_params_gen = gpb.tune_pars_TPE_algorithm_optuna(search_space=search_space, n_trials=6, X=X, y=y,
                                                        max_num_boost_round=50, early_stopping_rounds=5,
                                                        folds=(fold for fold in folds), metric='mse',
                                                        params={'verbose': -1}, tpe_seed=1, verbose_eval=0)
    assert opt_params_gen['best_params'] == best_params
    opt_params_kfold = gpb.tune_pars_TPE_algorithm_optuna(search_space=search_space, n_trials=2, X=X, y=y,
                                                          max_num_boost_round=50, early_stopping_rounds=5,
                                                          folds=KFold(n_splits=2, shuffle=True, random_state=1),
                                                          metric='mse', params={'verbose': -1}, tpe_seed=1,
                                                          verbose_eval=0)
    assert opt_params_kfold['best_iter'] >= 1


def test_tune_pars_SMAC_gpboost(capsys):
    pytest.importorskip("smac")
    X, y, group = _simulate_data()
    folds = _folds(len(y))
    search_space = {'learning_rate': [0.01, 1], 'lambda_l2': [0, 100], 'min_data_in_leaf': [1, 50],
                    'max_depth': [-1, -1], 'line_search_step_length': [True, False]}
    root_logger = logging.getLogger()
    root_handlers, root_level = list(root_logger.handlers), root_logger.level
    gp_model = gpb.GPModel(group_data=group, likelihood='gaussian')
    opt_params = gpb.tune_pars_SMAC(search_space=search_space, n_trials=8, X=X, y=y, gp_model=gp_model,
                                    max_num_boost_round=50, early_stopping_rounds=5, folds=folds, metric='mse',
                                    params={'verbose': -1}, smac_seed=1, verbose_eval=0)
    assert capsys.readouterr().out == ''
    assert root_logger.handlers == root_handlers and root_logger.level == root_level
    best_params = opt_params['best_params']
    assert set(best_params) == set(search_space)
    assert best_params['max_depth'] == -1
    assert 0.01 <= best_params['learning_rate'] <= 1
    assert 1e-8 <= best_params['lambda_l2'] <= 100
    assert 1 <= best_params['min_data_in_leaf'] <= 50 and isinstance(best_params['min_data_in_leaf'], int)
    assert isinstance(best_params['line_search_step_length'], bool)
    assert opt_params['best_iter'] >= 1
    # The order in which OpenMP adds up the partial sums of the threads can change the last digits
    gp_model = gpb.GPModel(group_data=group, likelihood='gaussian')
    best_score = _cv_best_score(best_params, X, y, folds, 'mse', gp_model=gp_model)
    assert abs(opt_params['best_score'] - best_score) < 1e-8


def test_tune_pars_SMAC_gp_model_params():
    pytest.importorskip("smac")
    X, y, coords = _simulate_spatial_data()
    folds = _folds(len(y))
    # The smoothness of 'gp_model' is not in the search space, so that the scores show whether the GP models of the
    # folds use the smoothness of the trials
    gp_model = _spatial_gp_model(coords, cov_fct_shape=2.5)
    args = dict(n_trials=6, X=X, y=y, gp_model=gp_model, max_num_boost_round=50, early_stopping_rounds=5,
                folds=folds, metric='test_neg_log_likelihood', params={'verbose': -1}, smac_seed=1, verbose_eval=0)
    opt_params = gpb.tune_pars_SMAC(search_space={'learning_rate': [0.01, 1], 'min_data_in_leaf': [1, 50],
                                                  'cov_fct_shape': {'values': [0.5, 1.5]}}, **args)
    best_params = opt_params['best_params']
    assert best_params['cov_fct_shape'] in [0.5, 1.5]
    assert gp_model.cov_fct_shape == 2.5
    best_score = _cv_score_of_best_params(best_params, opt_params['best_iter'], X, y, coords, folds)
    assert abs(opt_params['best_score'] - best_score) < 1e-8
    # A parameter of the GP model with the single value of 'gp_model' is held fixed and gives the same trials as
    # without it
    search_space = {'learning_rate': [0.01, 1], 'min_data_in_leaf': [1, 50]}
    opt_params = gpb.tune_pars_SMAC(search_space=search_space, **args)
    opt_params_fixed = gpb.tune_pars_SMAC(search_space=dict(search_space, cov_fct_shape={'values': [2.5]}), **args)
    assert opt_params_fixed['best_params'] == dict(opt_params['best_params'], cov_fct_shape=2.5)
    assert opt_params_fixed['best_iter'] == opt_params['best_iter']
    assert abs(opt_params_fixed['best_score'] - opt_params['best_score']) < 1e-8


def test_tuning_gp_model_params():
    X, y, coords = _simulate_spatial_data()
    folds = _folds(len(y))
    gp_model = _spatial_gp_model(coords, cov_fct_shape=2.5)
    opt_params = gpb.tune_pars_TPE_algorithm_optuna(search_space={'learning_rate': [0.05, 0.5],
                                                                  'cov_fct_shape': {'values': [0.5, 1.5]}},
                                                    n_trials=3, X=X, y=y, gp_model=gp_model, max_num_boost_round=50,
                                                    early_stopping_rounds=5, folds=folds,
                                                    metric='test_neg_log_likelihood', params={'verbose': -1},
                                                    tpe_seed=1, verbose_eval=0)
    best_params = opt_params['best_params']
    assert best_params['cov_fct_shape'] in [0.5, 1.5]
    assert gp_model.cov_fct_shape == 2.5
    best_score = _cv_score_of_best_params(best_params, opt_params['best_iter'], X, y, coords, folds)
    assert abs(opt_params['best_score'][0] - best_score) < 1e-8
    # Grid search, also over the covariance function
    opt_params = gpb.grid_search_tune_parameters(param_grid={'cov_fct_shape': [0.5, 1.5],
                                                             'cov_function': ['matern', 'exponential']},
                                                 train_set=gpb.Dataset(X, y), gp_model=gp_model,
                                                 params={'learning_rate': 0.1, 'verbose': -1}, num_boost_round=50,
                                                 early_stopping_rounds=5, folds=folds,
                                                 metric='test_neg_log_likelihood', verbose_eval=0,
                                                 return_all_combinations=True)
    assert gp_model.cov_fct_shape == 2.5 and gp_model.cov_function == 'matern'
    all_combinations = opt_params['all_combinations'].values()
    assert len(all_combinations) == 4
    for combination in all_combinations:
        score = _cv_score_of_best_params(dict(combination['params'], learning_rate=0.1),
                                         combination['num_boost_round'], X, y, coords, folds)
        assert abs(combination['score'] - score) < 1e-8
    # The exponential covariance function does not depend on the smoothness
    scores_exponential = [combination['score'] for combination in all_combinations
                          if combination['params']['cov_function'] == 'exponential']
    assert abs(scores_exponential[0] - scores_exponential[1]) < 1e-8


def test_tuning_cov_function_number_of_cov_pars():
    # Covariance functions with more and with fewer covariance parameters than the one of 'gp_model'
    X, y, coords = _simulate_spatial_data()
    folds = _folds(len(y))

    def grid_search(gp_model, cov_functions):
        return gpb.grid_search_tune_parameters(param_grid={'cov_function': cov_functions},
                                               train_set=gpb.Dataset(X, y), gp_model=gp_model,
                                               params={'learning_rate': 0.1, 'verbose': -1}, num_boost_round=50,
                                               early_stopping_rounds=5, folds=folds,
                                               metric='test_neg_log_likelihood', verbose_eval=0,
                                               return_all_combinations=True)

    for cov_function, cov_functions in (('matern', ['matern', 'matern_ard']), ('matern_ard', ['matern'])):
        gp_model = _spatial_gp_model(coords, cov_function=cov_function)
        num_cov_pars = gp_model.num_cov_pars
        opt_params = grid_search(gp_model, cov_functions)
        assert gp_model.cov_function == cov_function and gp_model.num_cov_pars == num_cov_pars
        for combination in opt_params['all_combinations'].values():
            score = _cv_score_of_best_params(dict(combination['params'], learning_rate=0.1),
                                             combination['num_boost_round'], X, y, coords, folds)
            assert abs(combination['score'] - score) < 1e-8
    # A mask that estimates all covariance parameters is the same as no mask
    scores = [combination['score'] for combination in
              grid_search(_spatial_gp_model(coords), ['matern', 'matern_ard'])['all_combinations'].values()]
    gp_model = _spatial_gp_model(coords)
    gp_model.set_optim_params(params={'estimate_cov_par_index': [1, 1, 1]})
    scores_mask = [combination['score'] for combination in
                   grid_search(gp_model, ['matern', 'matern_ard'])['all_combinations'].values()]
    assert np.max(np.abs(np.array(scores_mask) - np.array(scores))) < 1e-8


def test_tuning_ar1_multifidelity_cov_function():
    rng = np.random.default_rng(215)
    n = 120
    fidelity = np.tile([0, 1], n // 2)
    coords = np.column_stack([rng.uniform(size=(n, 3)), fidelity])
    X = rng.uniform(size=(n, 2))
    y = 5 * fidelity + np.sin(4 * X[:, 0]) + rng.normal(scale=0.4, size=n)
    indices = rng.permutation(n)
    folds = [(indices[:80], indices[80:])]
    params = {'learning_rate': 0.2, 'min_data_in_leaf': 5, 'verbose': -1, 'deterministic': True,
              'force_col_wise': True}

    def grid_search(cov_functions, fidelity_specific_mean, train_set=None):
        if train_set is None:
            train_set = gpb.Dataset(X, y)
        gp_model = gpb.GPModel(gp_coords=coords, cov_function=cov_functions[0],
                               fidelity_specific_mean=fidelity_specific_mean)
        return gpb.grid_search_tune_parameters(param_grid={'cov_function': cov_functions},
                                               train_set=train_set, gp_model=gp_model, params=params,
                                               num_boost_round=10, folds=folds, metric='test_neg_log_likelihood',
                                               verbose_eval=0, return_all_combinations=True)

    # With a fidelity-specific mean, the trees have the fidelity indicator as an additional feature, and covariance
    # functions with and without it cannot be compared in one search. Without it, they can
    for cov_functions, fidelity_specific_mean in ((['ar1_mf_matern', 'ar1_mf_exponential'], True),
                                                  (['ar1_mf_matern', 'matern_ard'], False)):
        opt_params = grid_search(cov_functions, fidelity_specific_mean)
        for combination in opt_params['all_combinations'].values():
            gp_model = gpb.GPModel(gp_coords=coords, cov_function=combination['params']['cov_function'],
                                   fidelity_specific_mean=fidelity_specific_mean)
            cvbst = gpb.cv(params=params, train_set=gpb.Dataset(X, y), gp_model=gp_model, num_boost_round=10,
                           folds=folds, metric='test_neg_log_likelihood', verbose_eval=False)
            assert abs(combination['score'] - np.min(cvbst['test_neg_log_likelihood-mean'])) < 1e-8
    with pytest.raises(ValueError, match="all values of 'cov_function' must then be AR1 multifidelity"):
        grid_search(['ar1_mf_matern', 'matern_ard'], True)
    # A search with a fidelity-specific mean adds the fidelity indicator to the Dataset as a feature, and a later
    #   search without it must not use this Dataset
    train_set = gpb.Dataset(X, y)
    grid_search(['ar1_mf_matern'], True, train_set=train_set)
    with pytest.raises(ValueError, match="Use a new Dataset for a model without a fidelity-specific mean"):
        grid_search(['matern_ard'], True, train_set=train_set)
    with pytest.raises(ValueError, match="'ar1_mf_matern' is an AR1 multifidelity covariance function and can only"):
        gpb.grid_search_tune_parameters(param_grid={'cov_function': ['ar1_mf_matern']}, train_set=gpb.Dataset(X, y),
                                        gp_model=gpb.GPModel(gp_coords=coords, cov_function='matern_ard'))


def test_tuning_gp_model_params_errors():
    X, y, coords = _simulate_spatial_data()
    with pytest.raises(ValueError, match="can only be tuned with a 'gp_model' that has a Gaussian process"):
        gpb.grid_search_tune_parameters(param_grid={'cov_function': ['matern', 'exponential']},
                                        train_set=gpb.Dataset(X, y),
                                        gp_model=gpb.GPModel(group_data=np.arange(len(y)) % 10))
    # The number and the meaning of the covariance parameters depend on 'cov_function' and 'cov_fct_order'
    gp_model = _spatial_gp_model(coords)
    gp_model.set_optim_params(params={'init_cov_pars': [0.1, 1., 0.2]})
    with pytest.raises(ValueError, match="cannot be tuned with initial values 'init_cov_pars'"):
        gpb.tune_pars_TPE_algorithm_optuna(search_space={'cov_function': {'values': ['matern', 'exponential']}},
                                           n_trials=1, X=X, y=y, gp_model=gp_model)
    gp_model = _spatial_gp_model(coords)
    gp_model.set_optim_params(params={'estimate_cov_par_index': [1, 0, 1]})
    with pytest.raises(ValueError, match="or with covariance parameters that are not estimated"):
        gpb.grid_search_tune_parameters(param_grid={'cov_fct_order': [1, 2]}, train_set=gpb.Dataset(X, y),
                                        gp_model=gp_model)
    with pytest.raises(ValueError, match="must be names of covariance functions"):
        gpb.tune_pars_TPE_algorithm_optuna(search_space={'cov_function': {'values': [1, 2]}}, n_trials=1, X=X, y=y,
                                           gp_model=_spatial_gp_model(coords))


# The last digits of a cross-validation score can differ between processes, since the threads add up their
# partial sums in the order in which they finish (observed on Linux with GCC), and the proposals of SMAC then
# differ after some trials. The script therefore replaces the cross-validation by a fixed function of the
# parameters, so that only the proposals of SMAC are compared
_SMAC_TRIALS_SCRIPT = """
import numpy as np
import gpboost as gpb
import gpboost.engine
def _score_of_parameters(params, gp_model_params, **kwargs):
    score = ((np.log10(params['learning_rate']) + 1) ** 2 + ((params['min_data_in_leaf'] - 20) / 30) ** 2
             + np.log10(params['lambda_l2']) / 10 + 0.1 * params['line_search_step_length']
             + 0.1 * abs(gp_model_params['cov_fct_shape'] - 1.5))
    return score, 1, False
gpboost.engine._tuning_cv_score = _score_of_parameters
rng = np.random.default_rng(1)
X = rng.uniform(size=(200, 2))
y = np.sin(4 * X[:, 0]) + X[:, 1] + rng.normal(scale=0.3, size=200)
idx = np.arange(200)
folds = [(idx[idx % 2 == 0], idx[idx % 2 == 1]), (idx[idx % 2 == 1], idx[idx % 2 == 0])]
gp_model = gpb.GPModel(gp_coords=rng.uniform(size=(200, 2)), cov_function='matern', cov_fct_shape=1.5)
gpb.tune_pars_SMAC(search_space={'learning_rate': [0.01, 1], 'min_data_in_leaf': [1, 50], 'lambda_l2': [0, 100],
                                 'line_search_step_length': [True, False],
                                 'cov_fct_shape': {'values': [0.5, 1.5, 2.5]}},
                   n_trials=16, X=X, y=y, gp_model=gp_model, max_num_boost_round=20, early_stopping_rounds=5,
                   folds=folds, metric='mse', params={'verbose': -1}, smac_seed=1, verbose_eval=1)
"""


def test_tune_pars_SMAC_reproducible():
    pytest.importorskip("smac")
    from ConfigSpace import Configuration
    configuration_hash = Configuration.__hash__
    X, y, _ = _simulate_data()
    gpb.tune_pars_SMAC(search_space={'learning_rate': [0.01, 1]}, n_trials=2, X=X, y=y, max_num_boost_round=10,
                       folds=_folds(len(y)), metric='mse', params={'verbose': -1}, verbose_eval=0)
    assert Configuration.__hash__ is configuration_hash
    # The proposals of SMAC do not depend on the hash randomization of Python
    trials = []
    for hash_seed in ["1", "2"]:
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=os.pathsep.join(sys.path))
        output = subprocess.run([sys.executable, "-c", _SMAC_TRIALS_SCRIPT], env=env, capture_output=True,
                                text=True, check=True).stdout
        trials.append([line for line in output.splitlines() if line.startswith("Trial")])
    assert len(trials[0]) == 16
    assert trials[0] == trials[1]


def test_tune_pars_SMAC_higher_better_and_failed_trials(capsys):
    pytest.importorskip("smac")
    X, y, _ = _simulate_data()
    y_binary = (y > np.median(y)).astype(float)
    folds = _folds(len(y))
    # bagging_fraction > 1 is invalid. With this seed, the first trial of the initial design fails before any valid
    # score is known, and the second one is valid
    search_space = {'learning_rate': [0.01, 1], 'bagging_fraction': [0.5, 1.5]}
    params = {'objective': 'binary', 'bagging_freq': 1, 'verbose': -1}
    opt_params = gpb.tune_pars_SMAC(search_space=search_space, n_trials=10, X=X, y=y_binary,
                                    max_num_boost_round=50, early_stopping_rounds=5, folds=folds, metric='auc',
                                    params=params, smac_seed=3, verbose_eval=1)
    output = [line for line in capsys.readouterr().out.splitlines()
              if line.startswith(("Trial", "An error occurred", "*****"))]
    assert output[0].startswith("Trial 1 of 10") and output[1].startswith("An error occurred")
    assert output[2].startswith("Trial 2 of 10") and output[3].startswith("***** New best test score")
    assert any(line.startswith("Trial 10 of 10") for line in output)
    best_params = opt_params['best_params']
    assert best_params['bagging_fraction'] <= 1
    best_score = _cv_best_score(dict(params, **best_params), X, y_binary, folds, 'auc', higher_better=True)
    assert abs(opt_params['best_score'] - best_score) < 1e-10
    # No valid trial at all
    with pytest.raises(ValueError, match="Did not find any valid parameter combination"):
        gpb.tune_pars_SMAC(search_space={'learning_rate': [0.01, 1], 'bagging_fraction': [1.5, 3]}, n_trials=3,
                           X=X, y=y_binary, max_num_boost_round=10, folds=folds, metric='auc', params=params,
                           verbose_eval=0)


def _binary_data():
    X, y, _ = _simulate_data()
    return X, (y > np.median(y)).astype(float), _folds(len(y))


def _agreement(pred, data):
    return "agreement", float(np.mean((pred > 0.5) == data.get_label())), True


_TUNING_ARGS_BINARY = dict(search_space={'learning_rate': [0.05, 0.5]}, max_num_boost_round=50,
                           early_stopping_rounds=5, verbose_eval=0)


def test_tuning_metric_direction():
    # Whether higher scores are better is taken from the first metric of the cross-validation. AUC given in 'params'
    # was minimized, and so was a custom metric with 'is_higher_better' = True when the built-in metric was disabled
    X, y_binary, folds = _binary_data()
    params = {'objective': 'binary', 'verbose': -1}
    args = dict(_TUNING_ARGS_BINARY, n_trials=2, X=X, y=y_binary, folds=folds, tpe_seed=1)
    opt_params_metric = gpb.tune_pars_TPE_algorithm_optuna(**args, params=params, metric='auc')
    opt_params_in_params = gpb.tune_pars_TPE_algorithm_optuna(**args, params=dict(params, metric='auc'))
    assert opt_params_in_params == opt_params_metric
    best_score = _cv_best_score(dict(params, **opt_params_metric['best_params']), X, y_binary, folds, 'auc',
                                higher_better=True)
    assert abs(opt_params_metric['best_score'][0] - best_score) < 1e-10
    opt_params_feval = gpb.tune_pars_TPE_algorithm_optuna(**args, params=params, metric='None', feval=_agreement)
    cvbst = gpb.cv(params=dict(params, force_col_wise=True, deterministic=True, **opt_params_feval['best_params']),
                   train_set=gpb.Dataset(X, y_binary), num_boost_round=50, early_stopping_rounds=5, folds=folds,
                   metric='None', feval=_agreement, verbose_eval=False)
    assert abs(opt_params_feval['best_score'][0] - np.max(cvbst['agreement-mean'])) < 1e-10
    assert opt_params_feval['best_iter'] == int(np.argmax(cvbst['agreement-mean'])) + 1
    # Grid search
    grid_args = dict(param_grid={'learning_rate': [0.1]}, train_set=gpb.Dataset(X, y_binary), num_boost_round=50,
                     early_stopping_rounds=5, folds=folds, verbose_eval=0)
    opt_params_grid = gpb.grid_search_tune_parameters(**grid_args, params=dict(params, metric='auc'))
    best_score = _cv_best_score(dict(params, learning_rate=0.1), X, y_binary, folds, 'auc', higher_better=True)
    assert abs(opt_params_grid['best_score'] - best_score) < 1e-10
    assert opt_params_grid == gpb.grid_search_tune_parameters(**grid_args, params=params, metric='auc')


def test_tune_pars_SMAC_metric_direction():
    pytest.importorskip("smac")
    X, y_binary, folds = _binary_data()
    params = {'objective': 'binary', 'verbose': -1}
    args = dict(_TUNING_ARGS_BINARY, n_trials=1, X=X, y=y_binary, folds=folds, smac_seed=7)
    opt_params_metric = gpb.tune_pars_SMAC(**args, params=params, metric='auc')
    opt_params_in_params = gpb.tune_pars_SMAC(**args, params=dict(params, metric='auc'))
    assert opt_params_in_params == opt_params_metric
    best_score = _cv_best_score(dict(params, **opt_params_metric['best_params']), X, y_binary, folds, 'auc',
                                higher_better=True)
    assert abs(opt_params_metric['best_score'] - best_score) < 1e-10


def test_grid_search_return_all_combinations():
    # 'best_iter' was the number of iterations of the last parameter combination when all combinations were returned
    X, y, _ = _simulate_data()
    opt_params = gpb.grid_search_tune_parameters(param_grid={'learning_rate': [0.3, 0.01]}, train_set=gpb.Dataset(X, y),
                                                 num_boost_round=30, folds=_folds(len(y)), params={'verbose': -1},
                                                 metric='mse', verbose_eval=0, return_all_combinations=True)
    all_combinations = opt_params['all_combinations']
    assert all_combinations[0]['num_boost_round'] != all_combinations[1]['num_boost_round']
    best = min(all_combinations.values(), key=lambda combination: combination['score'])
    assert opt_params['best_params'] == best['params'] and opt_params['best_score'] == best['score']
    assert opt_params['best_iter'] == best['num_boost_round']


def test_tune_pars_SMAC_finite_search_space(capsys):
    pytest.importorskip("smac")
    # SMAC stops when it has evaluated the two parameter combinations of the search space. This raised an error
    X, y_binary, folds = _binary_data()
    params = {'objective': 'binary', 'verbose': -1}
    opt_params = gpb.tune_pars_SMAC(search_space={'max_depth': [1, 2]}, n_trials=4, X=X, y=y_binary,
                                    max_num_boost_round=50, early_stopping_rounds=5, folds=folds, params=params,
                                    metric='auc', verbose_eval=1)
    output = capsys.readouterr().out
    assert "Trial 2 of 4" in output and "Trial 3 of 4" not in output
    assert "SMAC has found no further parameter combination" in output
    scores = {max_depth: _cv_best_score(dict(params, max_depth=max_depth), X, y_binary, folds, 'auc',
                                        higher_better=True) for max_depth in [1, 2]}
    assert opt_params['best_params']['max_depth'] == max(scores, key=scores.get)
    assert abs(opt_params['best_score'] - max(scores.values())) < 1e-10


class _ShuffledKFold(KFold):
    """KFold with shuffling whose random state changes with every call of 'split()'."""

    def __init__(self):
        super().__init__(n_splits=2, shuffle=True, random_state=np.random.RandomState(4))
        self.num_calls = 0

    def split(self, *args, **kwargs):
        self.num_calls += 1
        return super().split(*args, **kwargs)


_TUNING_ARGS_MSE = dict(search_space={'learning_rate': [0.01, 1]}, n_trials=3, max_num_boost_round=50,
                        early_stopping_rounds=5, metric='mse', params={'verbose': -1}, verbose_eval=0)


def test_tuning_splitter_folds():
    # A splitter is called once, so that every trial uses the same folds. It was called in every trial, and a
    # shuffled splitter with a 'RandomState' instance gave other folds each time
    X, y, _ = _simulate_data()
    folds_first_call = list(_ShuffledKFold().split(np.zeros(len(y)), y))
    splitter = _ShuffledKFold()
    opt_params = gpb.tune_pars_TPE_algorithm_optuna(**_TUNING_ARGS_MSE, X=X, y=y, folds=splitter, tpe_seed=1)
    assert splitter.num_calls == 1
    best_score = _cv_best_score(opt_params['best_params'], X, y, folds_first_call, 'mse')
    assert abs(opt_params['best_score'][0] - best_score) < 1e-10
    splitter = _ShuffledKFold()
    opt_params = gpb.grid_search_tune_parameters(param_grid={'learning_rate': [0.01, 0.1, 1]},
                                                 train_set=gpb.Dataset(X, y), num_boost_round=50,
                                                 early_stopping_rounds=5, folds=splitter, metric='mse',
                                                 params={'verbose': -1}, verbose_eval=0)
    assert splitter.num_calls == 1
    best_score = _cv_best_score(opt_params['best_params'], X, y, folds_first_call, 'mse')
    assert abs(opt_params['best_score'] - best_score) < 1e-10


def test_tune_pars_SMAC_splitter_folds():
    pytest.importorskip("smac")
    X, y, _ = _simulate_data()
    folds_first_call = list(_ShuffledKFold().split(np.zeros(len(y)), y))
    splitter = _ShuffledKFold()
    opt_params = gpb.tune_pars_SMAC(**_TUNING_ARGS_MSE, X=X, y=y, folds=splitter, smac_seed=1)
    assert splitter.num_calls == 1
    best_score = _cv_best_score(opt_params['best_params'], X, y, folds_first_call, 'mse')
    assert abs(opt_params['best_score'] - best_score) < 1e-10
