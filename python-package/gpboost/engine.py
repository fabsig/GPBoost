# coding: utf-8
"""
Library with training routines of GPBoost.

Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
Modified work Copyright (c) 2020 - 2024 Fabio Sigrist. All rights reserved.
Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
"""
import collections
import copy
from operator import attrgetter

import numpy as np
import optuna

from . import callback
from .basic import Booster, Dataset, GPBoostError, _ConfigAliases, _InnerPredictor, _log_warning, GPModel, _format_check_1D_data, is_numeric, is_1d_list, _get_bad_pandas_dtypes, _get_bad_pandas_dtypes_int
from .compat import SKLEARN_INSTALLED, _GPBoostGroupKFold, _GPBoostStratifiedKFold, pd_Series, is_dtype_sparse, pd_DataFrame


def train(params, train_set, num_boost_round=100,
          gp_model=None, line_search_step_length=False,
          use_gp_model_for_validation=True, train_gp_model_cov_pars=True,
          valid_sets=None, valid_names=None,
          fobj=None, feval=None, init_model=None,
          feature_name='auto', categorical_feature='auto',
          early_stopping_rounds=None, evals_result=None,
          verbose_eval=True, learning_rates=None,
          keep_training_booster=False, callbacks=None):
    """Training function.

    Parameters
    ----------
    params : dict
        Parameters for training.
        See https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#tuning-parameters--hyperparameters-for-the-tree-boosting-part
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    gp_model : GPModel or None, optional (default=None)
        GPModel object for the GPBoost algorithm
    line_search_step_length : bool, optional (default=False)
        If True, a line search is done to find the optimal step length for every boosting update
        (see, e.g., Friedman 2001). This is then multiplied by the 'learning_rate'.
        Applies only to the GPBoost algorithm
    use_gp_model_for_validation : bool, optional (default=True)
        If True, the 'gp_model' (Gaussian process and/or random effects) is also used (in addition to the tree model)
        for calculating predictions on the validation data. If False, the 'gp_model' (random effects part) is ignored
        for making predictions and only the tree ensemble is used for making predictions for calculating the validation / test error.
    train_gp_model_cov_pars : bool, optional (default=True)
        If True, the covariance parameters of the 'gp_model' (Gaussian process and/or random effects) are estimated
        in every boosting iterations, otherwise the 'gp_model' parameters are not estimated. In the latter case, you
        need to either estimate them beforehand or provide values via the 'init_cov_pars' parameter when creating
        the 'gp_model'
    valid_sets : list of Datasets or None, optional (default=None)
        List of data to be evaluated on during training.
    valid_names : list of strings or None, optional (default=None)
        Names of ``valid_sets``.
    fobj : callable or None, optional (default=None)
        Customized objective function. Only for independent boosting.
        The GPBoost algorithm currently does not support this.
        Should accept two parameters: preds, train_data,
        and return (grad, hess).

            preds : list or numpy 1-D array
                The predicted values.
            train_data : Dataset
                The training dataset.
            grad : list or numpy 1-D array
                The value of the first order derivative (gradient) for each sample point.
            hess : list or numpy 1-D array
                The value of the second order derivative (Hessian) for each sample point.

        For binary task, the preds is margin.
        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
        and you should group grad and hess in this way as well.
    feval : callable, list of callable functions or None, optional (default=None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, train_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : list or numpy 1-D array
                The predicted values.
            train_data : Dataset
                The training dataset.
            eval_name : string
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        For binary task, the preds is probability of positive class (or margin in case of specified ``fobj``).
        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].
        To ignore the default metric corresponding to the used objective,
        set the ``metric`` parameter to the string ``"None"`` in ``params``.
    init_model : string, Booster or None, optional (default=None)
        Filename of GPBoost model or Booster instance used for continue training.
    feature_name : list of strings or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of strings or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
    early_stopping_rounds : int or None, optional (default=None)
        Activates early stopping. The model will train until the validation score stops improving.
        Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
        to continue training.
        Requires at least one validation data and one metric.
        If there's more than one, will check all of them. But the training data is ignored anyway.
        To check only the first metric, set the ``first_metric_only`` parameter to ``True`` in ``params``.
        The index of iteration that has the best performance will be saved in the ``best_iteration`` field
        if early stopping logic is enabled by setting ``early_stopping_rounds``.
    evals_result: dict or None, optional (default=None)
        This dictionary used to store all evaluation results of all the items in ``valid_sets``.
        .. rubric:: Example

        With a ``valid_sets`` = [valid_set, train_set],
        ``valid_names`` = ['eval', 'train']
        and a ``params`` = {'metric': 'logloss'}
        returns {'train': {'logloss': ['0.48253', '0.35953', ...]},
        'eval': {'logloss': ['0.480385', '0.357756', ...]}}.

    verbose_eval : bool or int, optional (default=True)
        Requires at least one validation data.
        If True, the eval metric on the valid set is printed at each boosting stage.
        If int, the eval metric on the valid set is printed at every ``verbose_eval`` boosting stage.
        The last boosting stage or the boosting stage found by using ``early_stopping_rounds`` is also printed.

        .. rubric:: Example

        With ``verbose_eval`` = 4 and at least one item in ``valid_sets``,
        an evaluation metric is printed every 4 (instead of 1) boosting stages.

    learning_rates : list, callable or None, optional (default=None)
        List of learning rates for each boosting round
        or a customized function that calculates ``learning_rate``
        in terms of current number of round (e.g. yields learning rate decay).
    keep_training_booster : bool, optional (default=False)
        Whether the returned Booster will be used to keep training.
        If False, the returned value will be converted into _InnerPredictor before returning.
        When your model is very large and cause the memory error,
        you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.
        You can still use _InnerPredictor as ``init_model`` for future continue training.
    callbacks : list of callables or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.

    Returns
    -------
    booster : Booster
        The trained Booster model.

    Example
    -------
    >>> gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    >>> data_train = gpb.Dataset(X, y)
    >>> params = {'learning_rate': 0.01, 'max_depth': 3, 'num_leaves': 2**10, 'verbose': 0}
    >>> bst = gpb.train(params=params, train_set=data_train,  gp_model=gp_model,
    >>>                 num_boost_round=100)

    :Authors:
        Authors of the LightGBM Python package
        Fabio Sigrist
    """
    # create predictor first
    params = copy.deepcopy(params)
    if gp_model is not None:
        if not isinstance(gp_model, GPModel):
            raise TypeError('gp_model should be GPModel instance, met {}'
                            .format(type(gp_model).__name__))
    if fobj is not None:
        for obj_alias in _ConfigAliases.get("objective"):
            params.pop(obj_alias, None)
        params['objective'] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            num_boost_round = params.pop(alias)
            _log_warning("Found `{}` in params. Will use it instead of argument".format(alias))
    params["num_iterations"] = num_boost_round
    for alias in _ConfigAliases.get("early_stopping_round"):
        if alias in params:
            early_stopping_rounds = params.pop(alias)
            _log_warning("Found `{}` in params. Will use it instead of argument".format(alias))
    params["early_stopping_round"] = early_stopping_rounds
    first_metric_only = params.get('first_metric_only', False)

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    if isinstance(init_model, str):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    else:
        predictor = None
    init_iteration = predictor.num_total_iteration if predictor is not None else 0
    # check dataset
    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")

    train_set._update_params(params) \
        ._set_predictor(predictor) \
        .set_feature_name(feature_name) \
        .set_categorical_feature(categorical_feature)

    is_valid_contain_train = False
    train_data_name = "training"
    reduced_valid_sets = []
    name_valid_sets = []
    if valid_sets is not None:
        if isinstance(valid_sets, Dataset):
            valid_sets = [valid_sets]
        if isinstance(valid_names, str):
            valid_names = [valid_names]
        for i, valid_data in enumerate(valid_sets):
            # reduce cost for prediction training data
            if valid_data is train_set:
                is_valid_contain_train = True
                if valid_names is not None:
                    train_data_name = valid_names[i]
                continue
            if not isinstance(valid_data, Dataset):
                raise TypeError("Training only accepts Dataset object")
            reduced_valid_sets.append(valid_data._update_params(params).set_reference(train_set))
            if valid_names is not None and len(valid_names) > i:
                name_valid_sets.append(valid_names[i])
            else:
                name_valid_sets.append('valid_' + str(i))

    if gp_model is not None:
        # some checks
        if gp_model.has_covariates:
            raise ValueError("The 'gp_model' cannot have covariates 'X' (a linear predictor) in the GPBoost algorithm.")
        if use_gp_model_for_validation and not not valid_sets and not not feval:
            raise ValueError("use_gp_model_for_validation=True is currently "
                             "not supported for custom validation functions. If you need this feature, contact the "
                             "developer of this package or open a GitHub issue.")
        if use_gp_model_for_validation and len(reduced_valid_sets) > 1:
            raise ValueError("Can use only one validation set when use_gp_model_for_validation = True")
        if not is_valid_contain_train and use_gp_model_for_validation and len(
                reduced_valid_sets) > 0 and not gp_model.prediction_data_is_set:
            raise ValueError("Prediction data for 'gp_model' has not been set. "
                             "This needs to be set prior to trainig when having a validation set and 'use_gp_model_for_validation=True'. "
                             "Either call 'gp_model.set_prediction_data(...)' first or use 'use_gp_model_for_validation=False'.")
        # update gp_model related parameters
        params['use_gp_model_for_validation'] = use_gp_model_for_validation
        params['train_gp_model_cov_pars'] = train_gp_model_cov_pars
        params['line_search_step_length'] = line_search_step_length
        # Set the default metric to the (approximate marginal) negative log-likelihood if only the training loss should be calculated
        if is_valid_contain_train and len(reduced_valid_sets) == 0 and params.get('metric') is None:
            if gp_model._get_likelihood_name() == "gaussian":
                params['metric'] = "neg_log_likelihood"
            else:
                params['metric'] = "approx_neg_marginal_log_likelihood"

    # process callbacks
    if callbacks is None:
        callbacks = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks = set(callbacks)

    # Most of legacy advanced options becomes callbacks
    if verbose_eval is True:
        callbacks.add(callback.print_evaluation())
    elif isinstance(verbose_eval, int):
        callbacks.add(callback.print_evaluation(verbose_eval))

    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.add(callback.early_stopping(early_stopping_rounds, first_metric_only, verbose=bool(verbose_eval)))

    if learning_rates is not None:
        callbacks.add(callback.reset_parameter(learning_rate=learning_rates))

    if evals_result is not None:
        callbacks.add(callback.record_evaluation(evals_result))

    callbacks_before_iter = {cb for cb in callbacks if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter('order'))

    # construct booster
    try:
        booster = Booster(params=params, train_set=train_set, gp_model=gp_model)
        if is_valid_contain_train:
            booster.set_train_data_name(train_data_name)
        for valid_set, name_valid_set in zip(reduced_valid_sets, name_valid_sets):
            booster.add_valid(valid_set, name_valid_set)
    finally:
        train_set._reverse_update_params()
        for valid_set in reduced_valid_sets:
            valid_set._reverse_update_params()
    booster.best_iteration = 0

    # start training
    for i in range(init_iteration, init_iteration + num_boost_round):
        for cb in callbacks_before_iter:
            cb(callback.CallbackEnv(model=booster,
                                    params=params,
                                    iteration=i,
                                    begin_iteration=init_iteration,
                                    end_iteration=init_iteration + num_boost_round,
                                    evaluation_result_list=None))

        booster.update(fobj=fobj)

        evaluation_result_list = []
        # check evaluation result.
        if valid_sets is not None:
            if is_valid_contain_train:
                evaluation_result_list.extend(booster.eval_train(feval))
            evaluation_result_list.extend(booster.eval_valid(feval))
        try:
            for cb in callbacks_after_iter:
                cb(callback.CallbackEnv(model=booster,
                                        params=params,
                                        iteration=i,
                                        begin_iteration=init_iteration,
                                        end_iteration=init_iteration + num_boost_round,
                                        evaluation_result_list=evaluation_result_list))
        except callback.EarlyStopException as earlyStopException:
            booster.best_iteration = earlyStopException.best_iteration + 1
            evaluation_result_list = earlyStopException.best_score
            break

    booster.best_score = collections.defaultdict(collections.OrderedDict)
    for dataset_name, eval_name, score, _ in evaluation_result_list:
        booster.best_score[dataset_name][eval_name] = score
    if not keep_training_booster and gp_model is None:
        booster.model_from_string(booster.model_to_string(), False).free_dataset()

    return booster


class CVBooster:
    """CVBooster in GPBoost.

    Auxiliary data structure to hold and redirect all boosters of ``cv`` function.
    This class has the same methods as Booster class.
    All method calls are actually performed for underlying Boosters and then all returned results are returned in a list.

    Attributes
    ----------
    boosters : list of Booster
        The list of underlying fitted models.
    best_iteration : int
        The best iteration of fitted model.
    """

    def __init__(self):
        """Initialize the CVBooster.

        Generally, no need to instantiate manually.
        """
        self.boosters = []
        self.best_iteration = -1

    def _append(self, booster):
        """Add a booster to CVBooster."""
        self.boosters.append(booster)

    def __getattr__(self, name):
        """Redirect methods call of CVBooster."""

        def handler_function(*args, **kwargs):
            """Call methods with each booster, and concatenate their results."""
            ret = []
            for booster in self.boosters:
                ret.append(getattr(booster, name)(*args, **kwargs))
            return ret

        return handler_function


def _make_n_folds(full_data, folds, nfold, params, seed, gp_model=None, use_gp_model_for_validation=True,
                  fpreproc=None, stratified=False, shuffle=True, eval_train_metric=False):
    """Make a n-fold list of Booster from random indices."""
    full_data = full_data.construct()
    num_data = full_data.num_data()
    if folds is not None:
        if not hasattr(folds, '__iter__') and not hasattr(folds, 'split'):
            raise AttributeError("folds should be a generator or iterator of (train_idx, test_idx) tuples "
                                 "or scikit-learn splitter object with split method")
        if hasattr(folds, 'split'):
            group_info = full_data.get_group()
            if group_info is not None:
                group_info = np.array(group_info, dtype=np.int32, copy=False)
                flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            else:
                flatted_group = np.zeros(num_data, dtype=np.int32)
            folds = folds.split(X=np.zeros(num_data), y=full_data.get_label(), groups=flatted_group)
    else:
        if any(params.get(obj_alias, "") in {"lambdarank", "rank_xendcg", "xendcg",
                                             "xe_ndcg", "xe_ndcg_mart", "xendcg_mart"}
               for obj_alias in _ConfigAliases.get("objective")):
            if not SKLEARN_INSTALLED:
                raise GPBoostError('scikit-learn is required for ranking cv')
            # ranking task, split according to groups
            group_info = np.array(full_data.get_group(), dtype=np.int32, copy=False)
            flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            group_kfold = _GPBoostGroupKFold(n_splits=nfold)
            folds = group_kfold.split(X=np.zeros(num_data), groups=flatted_group)
        elif stratified:
            if not SKLEARN_INSTALLED:
                raise GPBoostError('scikit-learn is required for stratified cv')
            skf = _GPBoostStratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
            folds = skf.split(X=np.zeros(num_data), y=full_data.get_label())
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(num_data)
            else:
                randidx = np.arange(num_data)
            kstep = int(num_data / nfold)
            test_id = [randidx[i: i + kstep] for i in range(0, num_data, kstep)]
            train_id = [np.concatenate([test_id[i] for i in range(nfold) if k != i]) for k in range(nfold)]
            folds = zip(train_id, test_id)

    ret = CVBooster()
    for train_idx, test_idx in folds:
        train_set = full_data.subset(sorted(train_idx))
        if full_data.free_raw_data:
            valid_set = full_data.subset(sorted(test_idx))
        else:
            valid_set = full_data.subset(sorted(test_idx), reference=train_set)
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            train_set, valid_set, tparam = fpreproc(train_set, valid_set, params.copy())
        else:
            tparam = params
        if gp_model is not None:
            train_idx = sorted(train_idx)
            test_idx = sorted(test_idx)

            group_data = None
            group_data_pred = None
            group_rand_coef_data = None
            group_rand_coef_data_pred = None
            gp_coords = None
            gp_coords_pred = None
            gp_rand_coef_data = None
            gp_rand_coef_data_pred = None
            cluster_ids = None
            cluster_ids_pred = None
            if gp_model.group_data is not None:
                group_data = gp_model.group_data[train_idx]
                group_data_pred = gp_model.group_data[test_idx]
                if gp_model.group_rand_coef_data is not None:
                    group_rand_coef_data = gp_model.group_rand_coef_data[train_idx]
                    group_rand_coef_data_pred = gp_model.group_rand_coef_data[test_idx]
            if gp_model.gp_coords is not None:
                gp_coords = gp_model.gp_coords[train_idx]
                gp_coords_pred = gp_model.gp_coords[test_idx]
                if gp_model.gp_rand_coef_data is not None:
                    gp_rand_coef_data = gp_model.gp_rand_coef_data[train_idx]
                    gp_rand_coef_data_pred = gp_model.gp_rand_coef_data[test_idx]
            if gp_model.cluster_ids is not None:
                cluster_ids = gp_model.cluster_ids[train_idx]
                cluster_ids_pred = gp_model.cluster_ids[test_idx]
            gp_model_train = GPModel(group_data=group_data,
                                     group_rand_coef_data=group_rand_coef_data,
                                     ind_effect_group_rand_coef=gp_model.ind_effect_group_rand_coef,
                                     drop_intercept_group_rand_effect=gp_model.drop_intercept_group_rand_effect,
                                     gp_coords=gp_coords,
                                     gp_rand_coef_data=gp_rand_coef_data,
                                     cov_function=gp_model.cov_function,
                                     cov_fct_shape=gp_model.cov_fct_shape,
                                     gp_approx=gp_model.gp_approx,
                                     cov_fct_taper_range=gp_model.cov_fct_taper_range,
                                     cov_fct_taper_shape=gp_model.cov_fct_taper_shape,
                                     num_neighbors=gp_model.num_neighbors,
                                     vecchia_ordering=gp_model.vecchia_ordering,
                                     num_ind_points=gp_model.num_ind_points,
                                     matrix_inversion_method=gp_model.matrix_inversion_method,
                                     seed=gp_model.seed,
                                     cluster_ids=cluster_ids,
                                     likelihood=gp_model._get_likelihood_name(),
                                     free_raw_data=True)
            if use_gp_model_for_validation:
                gp_model_train.set_prediction_data(group_data_pred=group_data_pred,
                                                   group_rand_coef_data_pred=group_rand_coef_data_pred,
                                                   gp_coords_pred=gp_coords_pred,
                                                   gp_rand_coef_data_pred=gp_rand_coef_data_pred,
                                                   cluster_ids_pred=cluster_ids_pred,
                                                   vecchia_pred_type=gp_model.vecchia_pred_type,
                                                   num_neighbors_pred=gp_model.num_neighbors_pred,
                                                   cg_delta_conv_pred=gp_model.cg_delta_conv_pred,
                                                   nsim_var_pred=gp_model.nsim_var_pred)
            cvbooster = Booster(params=tparam, train_set=train_set, gp_model=gp_model_train)
            gp_model._set_likelihood(
                gp_model_train._get_likelihood_name())  # potentially change likelihood in case this was done in the booster to reflect implied changes in the default optimizer for different likelihoods
            gp_model_train.set_optim_params(params=gp_model._get_optim_params())
        else:  # no gp_model
            cvbooster = Booster(tparam, train_set)
        if eval_train_metric:
            cvbooster.add_valid(train_set, 'train')
        cvbooster.add_valid(valid_set, 'valid')
        ret._append(cvbooster)
    return ret


def _agg_cv_result(raw_results, eval_train_metric=False):
    """Aggregate cross-validation results."""
    cvmap = collections.OrderedDict()
    metric_type = {}
    for one_result in raw_results:
        for one_line in one_result:
            if eval_train_metric:
                key = "{} {}".format(one_line[0], one_line[1])
            else:
                key = one_line[1]
            metric_type[key] = one_line[3]
            cvmap.setdefault(key, [])
            cvmap[key].append(one_line[2])
    return [('cv_agg', k, np.mean(v), metric_type[k], np.std(v)) for k, v in cvmap.items()]


def cv(params, train_set, num_boost_round=100,
       gp_model=None, line_search_step_length=False,
       use_gp_model_for_validation=True, fit_GP_cov_pars_OOS=False, train_gp_model_cov_pars=True,
       folds=None, nfold=5, stratified=False, shuffle=True,
       metric=None, fobj=None, feval=None, init_model=None,
       feature_name='auto', categorical_feature='auto',
       early_stopping_rounds=None, fpreproc=None,
       verbose_eval=None, show_stdv=False, seed=0,
       callbacks=None, eval_train_metric=False,
       return_cvbooster=False, metrics=None):
    """Perform cross-validation for choosing number of boosting iterations.

    Parameters
    ----------
    params : dict
        Parameters for Booster.
        See https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#tuning-parameters--hyperparameters-for-the-tree-boosting-part
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    gp_model : GPModel or None, optional (default=None)
        GPModel object for the GPBoost algorithm
    line_search_step_length : bool, optional (default=False)
        If True, a line search is done to find the optimal step length for every boosting update
        (see, e.g., Friedman 2001). This is then multiplied by the 'learning_rate'.
        Applies only to the GPBoost algorithm
    use_gp_model_for_validation : bool, optional (default=True)
        If True, the 'gp_model' (Gaussian process and/or random effects) is also used (in addition to the tree model)
        for calculating predictions on the validation data. If False, the 'gp_model' (random effects part) is ignored
        for making predictions and only the tree ensemble is used for making predictions for calculating the validation / test error.
    fit_GP_cov_pars_OOS : bool, optional (default=False)
        If TRUE, the covariance parameters of the 'gp_model' model are estimated using the out-of-sample (OOS) predictions
        on the validation data using the optimal number of iterations (after performing the CV).
        This corresponds to the GPBoostOOS algorithm.
    train_gp_model_cov_pars : bool, optional (default=True)
        If True, the covariance parameters of the 'gp_model' (Gaussian process and/or random effects) are estimated
        in every boosting iterations, otherwise the 'gp_model' parameters are not estimated. In the latter case, you
        need to either estimate them beforehand or provide values via the 'init_cov_pars' parameter when creating
        the 'gp_model'
    folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
        If generator or iterator, it should yield the train and test indices for each fold.
        If object, it should be one of the scikit-learn splitter classes
        (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        and have ``split`` method.
        This argument has highest priority over other data split arguments.
    nfold : int, optional (default=5)
        Number of folds in CV.
    stratified : bool, optional (default=False)
        Whether to perform stratified sampling.
    shuffle : bool, optional (default=True)
        Whether to shuffle before splitting data.
    metric : string, list of strings or None, optional (default=None)
        Evaluation metric to be monitored when doing CV and parameter tuning.
        If not None, the metric in ``params`` will be overridden.
        Non-exhaustive list of supported metrics: "test_neg_log_likelihood", "mse", "rmse", "mae",
        "auc", "average_precision", "binary_logloss", "binary_error"
        See https://gpboost.readthedocs.io/en/latest/Parameters.html#metric-parameters
        for a complete list of valid metrics.
    fobj : callable or None, optional (default=None)
        Customized objective function. Only for independent boosting.
        The GPBoost algorithm currently does not support this.
        Should accept two parameters: preds, train_data,
        and return (grad, hess).

            preds : list or numpy 1-D array
                The predicted values.
            train_data : Dataset
                The training dataset.
            grad : list or numpy 1-D array
                The value of the first order derivative (gradient) for each sample point.
            hess : list or numpy 1-D array
                The value of the second order derivative (Hessian) for each sample point.

        For binary task, the preds is margin.
        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
        and you should group grad and hess in this way as well.
    feval : callable, list of callable functions or None, optional (default=None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, train_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : list or numpy 1-D array
                The predicted values.
            train_data : Dataset
                The training dataset.
            eval_name : string
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        For binary task, the preds is probability of positive class (or margin in case of specified ``fobj``).
        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].
        To ignore the default metric corresponding to the used objective,
        set ``metric`` to the string ``"None"``.
    init_model : string, Booster or None, optional (default=None)
        Filename of GPBoost model or Booster instance used for continue training.
    feature_name : list of strings or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of strings or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
    early_stopping_rounds : int or None, optional (default=None)
        Activates early stopping. The ``metric`` needs to improve at least every 
        ``early_stopping_rounds`` round(s) to continue.
    fpreproc : callable or None, optional (default=None)
        Preprocessing function that takes (dtrain, dtest, params)
        and returns transformed versions of those.
    verbose_eval : bool, int, or None, optional (default=None)
        Whether to display the progress.
        If None, progress will be displayed when np.ndarray is returned.
        If True, progress will be displayed at every boosting stage.
        If int, progress will be displayed at every given ``verbose_eval`` boosting stage.
    show_stdv : bool, optional (default=False)
        Whether to display the standard deviation in progress.
        Results are not affected by this parameter, and always contain std.
    seed : int, optional (default=0)
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callables or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    eval_train_metric : bool, optional (default=False)
        Whether to display the train metric in progress.
        The score of the metric is calculated again after each training step, so there is some impact on performance.
    return_cvbooster : bool, optional (default=False)
        Whether to return Booster models trained on each fold through ``CVBooster``.
    metrics : string, list of strings or None, discontinued (default=None)
        This is discontinued. Use the renamed equivalent argument 'metric' instead

    Returns
    -------
    eval_hist : dict
        Evaluation history.
        The dictionary has the following format:
        {'metric1-mean': [values], 'metric1-stdv': [values],
        'metric2-mean': [values], 'metric2-stdv': [values],
        ...}.
        If ``return_cvbooster=True``, also returns trained boosters via ``cvbooster`` key.

    Example
    -------
    >>> gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    >>> data_train = gpb.Dataset(X, y)
    >>> params = {'learning_rate': 0.01, 'max_depth': 3, 'num_leaves': 2**10, 'verbose': 0}
    >>> cvbst = gpb.cv(params=params, train_set=data_train,
    >>>                gp_model=gp_model, use_gp_model_for_validation=True,
    >>>                num_boost_round=1000, early_stopping_rounds=5,
    >>>                nfold=4, verbose_eval=True, show_stdv=False, seed=1)

    :Authors:
        Authors of the LightGBM Python package
        Fabio Sigrist
    """
    if metrics is not None:
        raise GPBoostError("The argument 'metrics' is discontinued. "
                           "Use the renamed equivalent argument 'metric' instead")
    if fit_GP_cov_pars_OOS:
        raise ValueError("The GPBoostOOS algorithm (fit_GP_cov_pars_OOS=True) is not yet implemented in Python.")
    if not isinstance(train_set, Dataset):
        raise TypeError("cv only accepts Dataset objects as train_set")
    if train_set.free_raw_data:
        _log_warning('For true out-of-sample (cross-) validation, it is recommended to set free_raw_data = False '
                     'when constructing the Dataset')

    params = copy.deepcopy(params)
    if fobj is not None:
        for obj_alias in _ConfigAliases.get("objective"):
            params.pop(obj_alias, None)
        params['objective'] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            _log_warning("Found `{}` in params. Will use it instead of argument".format(alias))
            num_boost_round = params.pop(alias)
    params["num_iterations"] = num_boost_round
    for alias in _ConfigAliases.get("early_stopping_round"):
        if alias in params:
            _log_warning("Found `{}` in params. Will use it instead of argument".format(alias))
            early_stopping_rounds = params.pop(alias)
    params["early_stopping_round"] = early_stopping_rounds
    first_metric_only = params.get('first_metric_only', False)

    if gp_model is not None:
        # some checks
        if use_gp_model_for_validation and not not feval:
            raise ValueError("use_gp_model_for_validation=True is currently "
                             "not supported for custom validation functions. If you need this feature, contact the "
                             "developer of this package or open a GitHub issue.")
        if stratified:
            raise ValueError("stratified=True is not supported when a gp_model is provided")
        # update gp_model related parameters
        params['use_gp_model_for_validation'] = use_gp_model_for_validation
        params['train_gp_model_cov_pars'] = train_gp_model_cov_pars
        params['line_search_step_length'] = line_search_step_length

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    if isinstance(init_model, str):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    else:
        predictor = None

    if metric is not None:
        for metric_alias in _ConfigAliases.get("metric"):
            params.pop(metric_alias, None)
        params['metric'] = metric

    train_set._update_params(params) \
        ._set_predictor(predictor) \
        .set_feature_name(feature_name) \
        .set_categorical_feature(categorical_feature)

    results = collections.defaultdict(list)
    cvfolds = _make_n_folds(train_set, folds=folds, nfold=nfold,
                            params=params, seed=seed, gp_model=gp_model,
                            use_gp_model_for_validation=use_gp_model_for_validation,
                            fpreproc=fpreproc, stratified=stratified, shuffle=shuffle,
                            eval_train_metric=eval_train_metric)

    # setup callbacks
    if callbacks is None:
        callbacks = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks = set(callbacks)
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.add(callback.early_stopping(early_stopping_rounds, first_metric_only, verbose=False))
    if verbose_eval is True:
        callbacks.add(callback.print_evaluation(show_stdv=show_stdv))
    elif isinstance(verbose_eval, int):
        callbacks.add(callback.print_evaluation(verbose_eval, show_stdv=show_stdv))

    callbacks_before_iter = {cb for cb in callbacks if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter('order'))

    for i in range(num_boost_round):
        for cb in callbacks_before_iter:
            cb(callback.CallbackEnv(model=cvfolds,
                                    params=params,
                                    iteration=i,
                                    begin_iteration=0,
                                    end_iteration=num_boost_round,
                                    evaluation_result_list=None))
        try:
            cvfolds.update(fobj=fobj)
        except Exception as e:
            print('Error in boosting iteration ' + str(i))
            break
        res = _agg_cv_result(cvfolds.eval_valid(feval), eval_train_metric)
        for _, key, mean, _, std in res:
            results[key + '-mean'].append(mean)
            results[key + '-stdv'].append(std)
        try:
            for cb in callbacks_after_iter:
                cb(callback.CallbackEnv(model=cvfolds,
                                        params=params,
                                        iteration=i,
                                        begin_iteration=0,
                                        end_iteration=num_boost_round,
                                        evaluation_result_list=res))
        except callback.EarlyStopException as earlyStopException:
            cvfolds.best_iteration = earlyStopException.best_iteration + 1
            for k in results:
                results[k] = results[k][:cvfolds.best_iteration]
            break

    if return_cvbooster:
        results['cvbooster'] = cvfolds

    return dict(results)


def _get_grid_size(param_grid):
    """Determine total number of parameter combinations on a grid

    Parameters
    ----------
    param_grid : dict
        Parameter grid

    Returns
    -------
    grid_size : int
        Parameter grid size

    :Authors:
        Fabio Sigrist
    """
    grid_size = 1
    for param in param_grid:
        grid_size = grid_size * len(param_grid[param])
    return (grid_size)


def _get_param_combination(param_comb_number, param_grid):
    """Select parameter combination from a grid of parameters

    Parameters
    ----------
    param_comb_number : int
        Index number of parameter combination on parameter grid that should be returned (counting starts at 0).
    param_grid : dict
        Parameter grid

    Returns
    -------
    param_comb : dict
        Parameter combination

    :Authors:
        Fabio Sigrist
    """
    param_comb = {}
    nk = param_comb_number
    for param in param_grid:
        ind_p = int(nk % len(param_grid[param]))
        param_comb[param] = param_grid[param][ind_p]
        nk = (nk - ind_p) / len(param_grid[param])
    return (param_comb)


def grid_search_tune_parameters(param_grid, train_set, params=None, num_try_random=None,
                                num_boost_round=100, gp_model=None,
                                line_search_step_length=False,
                                use_gp_model_for_validation=True, train_gp_model_cov_pars=True,
                                folds=None, nfold=5, stratified=False, shuffle=True,
                                metric=None, fobj=None, feval=None, init_model=None,
                                feature_name='auto', categorical_feature='auto',
                                early_stopping_rounds=None, fpreproc=None,
                                verbose_eval=1, seed=0, callbacks=None, metrics=None,
                                return_all_combinations=False):
    """Function for choosing tuning parameters from a grid in a determinstic or random way using cross validation or validation data sets.

    Parameters
    ----------
    param_grid : dict
        Candidate parameters defining the grid over which a search is done.
        See https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#tuning-parameters--hyperparameters-for-the-tree-boosting-part
    train_set : Dataset
        Data to be trained on.
    params : dict, optional (default=None)
        Other parameters not included in param_grid.
    num_try_random : int, optional (default=None)
        Number of random trial on parameter grid. If none, a deterministic search is done
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    gp_model : GPModel or None, optional (default=None)
        GPModel object for the GPBoost algorithm
    line_search_step_length : bool, optional (default=False)
        If True, a line search is done to find the optimal step length for every boosting update
        (see, e.g., Friedman 2001). This is then multiplied by the 'learning_rate'.
        Applies only to the GPBoost algorithm
    use_gp_model_for_validation : bool, optional (default=True)
        If True, the 'gp_model' (Gaussian process and/or random effects) is also used (in addition to the tree model)
        for calculating predictions on the validation data. If False, the 'gp_model' (random effects part) is ignored
        for making predictions and only the tree ensemble is used for making predictions for calculating the validation / test error.
    train_gp_model_cov_pars : bool, optional (default=True)
        If True, the covariance parameters of the 'gp_model' (Gaussian process and/or random effects) are estimated
        in every boosting iterations, otherwise the 'gp_model' parameters are not estimated. In the latter case, you
        need to either estimate them beforehand or provide values via the 'init_cov_pars' parameter when creating
        the 'gp_model'
    folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
        If generator or iterator, it should yield the train and test indices for each fold.
        If object, it should be one of the scikit-learn splitter classes
        (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        and have ``split`` method.
        This argument has highest priority over other data split arguments.
    nfold : int, optional (default=5)
        Number of folds in CV.
    stratified : bool, optional (default=False)
        Whether to perform stratified sampling.
    shuffle : bool, optional (default=True)
        Whether to shuffle before splitting data.
    metric : string, list of strings or None, optional (default=None)
        Evaluation metric to be monitored when doing parameter tuning.
        If not None, the metric in ``params`` will be overridden.
        Non-exhaustive list of supported metrics: "test_neg_log_likelihood", "mse", "rmse", "mae",
        "auc", "average_precision", "binary_logloss", "binary_error"
        See https://gpboost.readthedocs.io/en/latest/Parameters.html#metric-parameters
        for a complete list of valid metrics.
    fobj : callable or None, optional (default=None)
        Customized objective function. Only for independent boosting.
        The GPBoost algorithm currently does not support this.
        Should accept two parameters: preds, train_data,
        and return (grad, hess).

            preds : list or numpy 1-D array
                The predicted values.
            train_data : Dataset
                The training dataset.
            grad : list or numpy 1-D array
                The value of the first order derivative (gradient) for each sample point.
            hess : list or numpy 1-D array
                The value of the second order derivative (Hessian) for each sample point.

        For binary task, the preds is margin.
        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
        and you should group grad and hess in this way as well.

    feval : callable, list of callable functions or None, optional (default=None)
        Customized evaluation function.
        If more than one evaluation function is provided, only the first evaluation function will be used to choose tuning parameters
        Each evaluation function should accept two parameters: preds, train_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : list or numpy 1-D array
                The predicted values.
            train_data : Dataset
                The training dataset.
            eval_name : string
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        For binary task, the preds is probability of positive class (or margin in case of specified ``fobj``).
        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].
        To ignore the default metric corresponding to the used objective,
        set ``metric`` to the string ``"None"``.
    init_model : string, Booster or None, optional (default=None)
        Filename of GPBoost model or Booster instance used for continue training.
    feature_name : list of strings or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of strings or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
    early_stopping_rounds : int or None, optional (default=None)
        Activates early stopping. The ``metric`` needs to improve at least every 
        ``early_stopping_rounds`` round(s) to continue.
    fpreproc : callable or None, optional (default=None)
        Preprocessing function that takes (dtrain, dtest, params)
        and returns transformed versions of those.
    verbose_eval : int or None, optional (default=1)
        Whether to display information on the progress of tuning parameter choice.
        If None or 0, verbose is of.
        If = 1, summary progress information is displayed for every parameter combination.
        If >= 2, detailed progress is displayed at every boosting stage for every parameter combination.
    seed : int, optional (default=0)
        Seed used to generate folds and random grid search (passed to numpy.random.seed).
    callbacks : list of callables or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    metrics : string, list of strings or None, discontinued (default=None)
        This is discontinued. Use the renamed equivalent argument 'metric' instead
    return_all_combinations : bool, optional (default=False)
        If True, all tried parameter combinations are returned

    Returns
    -------
    return : dict
        Dictionary with the best parameter combination and score
        The dictionary has the following format:
        {'best_params': best_params, 'best_num_boost_round': best_num_boost_round, 'best_score': best_score}
        If return_all_combinations is True, then the dictionary contains an additional entry 'all_combinations'

    Example
    -------
    >>> param_grid = {'learning_rate': [1,0.1,0.01], 
    >>>   'min_data_in_leaf': [10,100,1000],
    >>>   'max_depth': [1,2,3,5,10],
    >>>   'lambda_l2': [0,1,10]}
    >>> other_params = {'num_leaves': 2**10, 'verbose': 0}
    >>> # Note: here we try different values for 'max_depth' and thus set 'num_leaves' to a large value.
    >>> #       An alternative strategy is to impose no limit on 'max_depth',  
    >>> #       and try different values for 'num_leaves' as follows:
    >>> # param_grid = {'learning_rate': [1,0.1,0.01], 
    >>> #               'min_data_in_leaf': [10,100,1000],
    >>> #               'num_leaves': 2**np.arange(1,11),
    >>> #               'lambda_l2': [0,1,10]}
    >>> # other_params = {'max_depth': -1, 'verbose': 0}
    >>> gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    >>> data_train = gpb.Dataset(X, y)
    >>> opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid, params=other_params,
    >>>                                              num_try_random=None, nfold=4,
    >>>                                              train_set=data_train, gp_model=gp_model,
    >>>                                              use_gp_model_for_validation=True, verbose_eval=1,
    >>>                                              num_boost_round=1000, early_stopping_rounds=10,
    >>>                                              seed=1000)
    >>> print("Best parameters: " + str(opt_params['best_params']))
    >>> print("Best number of iterations: " + str(opt_params['best_iter']))
    >>> print("Best score: " + str(opt_params['best_score']))
    >>> 
    >>> # Using manually defined validation data instead of cross-validation
    >>> permute_aux = np.random.permutation(n)
    >>> train_tune_idx = permute_aux[0:int(0.8 * n)]
    >>> valid_tune_idx = permute_aux[int(0.8 * n):n]
    >>> folds = [(train_tune_idx, valid_tune_idx)]
    >>> opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid, params=other_params,
    >>>                                              num_try_random=None, folds=folds,
    >>>                                              train_set=data_train, gp_model=gp_model,
    >>>                                              use_gp_model_for_validation=True, verbose_eval=1,
    >>>                                              num_boost_round=1000, early_stopping_rounds=10,
    >>>                                              seed=1000)

    :Authors:
        Fabio Sigrist
    """
    if metrics is not None:
        raise GPBoostError("The argument 'metrics' is discontinued. "
                           "Use the renamed equivalent argument 'metric' instead")
    # Check correct format
    if not isinstance(param_grid, dict):
        raise ValueError("param_grid needs to be a dict")
    if verbose_eval is None:
        verbose_eval = 0
    else:
        if not isinstance(verbose_eval, int):
            raise ValueError("verbose_eval needs to be int")
    if params is None:
        params = {}
    else:
        params = copy.deepcopy(params)
    param_grid = copy.deepcopy(param_grid)
    for param in param_grid:
        if is_numeric(param_grid[param]):
            param_grid[param] = [param_grid[param]]
        param_grid[param] = _format_check_1D_data(param_grid[param],
                                                  data_name=param, check_data_type=False,
                                                  check_must_be_int=False, convert_to_type=None)
    higher_better = False
    if metric is not None:
        if isinstance(metric, str):
            metric = [metric]
        if metric[0].startswith(('auc', 'ndcg@', 'map@', 'average_precision')):
            higher_better = True
    elif feval is not None:
        if callable(feval):
            feval = [feval]
        PH1, PH2, higher_better = feval[0](np.array([0]), Dataset(np.array([0]), np.array([0])))
    # Determine combinations of parameter values that should be tried out
    grid_size = _get_grid_size(param_grid)
    if num_try_random is not None:
        if num_try_random > grid_size:
            raise ValueError("num_try_random is larger than the number of all possible combinations of parameters in param_grid ")
        try_param_combs = np.random.RandomState(seed).choice(a=grid_size, size=num_try_random, replace=False)
        if verbose_eval >= 1:
            print("Starting random grid search with " + str(num_try_random) + " trials out of " + str(
                grid_size) + " parameter combinations ")
    else:
        try_param_combs = range(grid_size)
        if verbose_eval >= 1:
            print("Starting deterministic grid search with " + str(grid_size) + " parameter combinations ")
    if verbose_eval < 2:
        verbose_eval_cv = False
    else:
        verbose_eval_cv = True
    if return_all_combinations:
        all_combinations = {}
    best_score = 1e99
    current_score = 1e99
    if higher_better:
        best_score = -1e99
        current_score = -1e99
    best_params = {}
    best_num_boost_round = num_boost_round
    counter_num_comb = 1
    if 'max_bin' in param_grid:
        if train_set.handle is not None:
            raise ValueError("'train_set' cannot be constructed already when 'max_bin' is in 'param_grid' ")
        else:
            train_set_not_constructed = copy.deepcopy(train_set)

    for param_comb_number in try_param_combs:
        param_comb = _get_param_combination(param_comb_number=param_comb_number, param_grid=param_grid)
        for param in param_comb:
            params[param] = param_comb[param]
        if verbose_eval >= 1:
            print("Trying parameter combination " + str(counter_num_comb) +
                  " of " + str(len(try_param_combs)) + ": " + str(param_comb))
        if 'max_bin' in param_grid:
            train_set = copy.deepcopy(train_set_not_constructed)
        current_score_is_better = False
        try:
            cvbst = cv(params=params, train_set=train_set, num_boost_round=num_boost_round, gp_model=gp_model,
                       line_search_step_length=line_search_step_length,
                       use_gp_model_for_validation=use_gp_model_for_validation,
                       train_gp_model_cov_pars=train_gp_model_cov_pars,
                       folds=folds, nfold=nfold, stratified=stratified, shuffle=shuffle,
                       metric=metric, fobj=fobj, feval=feval, init_model=init_model,
                       feature_name=feature_name, categorical_feature=categorical_feature,
                       early_stopping_rounds=early_stopping_rounds, fpreproc=fpreproc,
                       verbose_eval=verbose_eval_cv, seed=seed, callbacks=callbacks,
                       eval_train_metric=False, return_cvbooster=False)
            if higher_better:
                current_score = np.max(cvbst[next(iter(cvbst))])
                if current_score > best_score:
                    current_score_is_better = True
            else:
                current_score = np.min(cvbst[next(iter(cvbst))])
                if current_score < best_score:
                    current_score_is_better = True
        except Exception as err: # Note: this is typically not called anymore since gpv.cv() now already contains a tryCatch statement
            if verbose_eval < 1:
                print("Error for parameter combination " + str(counter_num_comb) +
                      " of " + str(len(try_param_combs)) + ": " + str(param_comb))
        if current_score_is_better:
            best_score = current_score
            best_params = param_comb
            if higher_better:
                best_num_boost_round = np.argmax(cvbst[next(iter(cvbst))]) + 1
            else:
                best_num_boost_round = np.argmin(cvbst[next(iter(cvbst))]) + 1
            if verbose_eval >= 1:
                metric_name = list(cvbst.keys())[0]
                metric_name = metric_name.split('-mean', 1)[0]
                print("***** New best test score ("+metric_name+" = " + str(best_score) +
                      ") found for the following parameter combination:")
                best_params_print = copy.deepcopy(best_params)
                best_params_print['num_boost_round'] = best_num_boost_round
                print(best_params_print)
        counter_num_comb = counter_num_comb + 1
        if return_all_combinations:
            if higher_better:
                best_num_boost_round = np.argmax(cvbst[next(iter(cvbst))]) + 1
            else:
                best_num_boost_round = np.argmin(cvbst[next(iter(cvbst))]) + 1
            all_combinations[param_comb_number] = {'params': param_comb, 'num_boost_round': best_num_boost_round,
                                                   'score': current_score}
    if return_all_combinations:
        return {'best_params': best_params, 'best_iter': best_num_boost_round, 'best_score': best_score,
                    'all_combinations': all_combinations}
    else:
        return {'best_params': best_params, 'best_iter': best_num_boost_round, 'best_score': best_score}

def tune_pars_TPE_algorithm_optuna(X, y, param_grid, n_trials,
                                max_num_boost_round=1000, early_stopping_rounds=None,
                                metric=None, gp_model=None, folds=None, nfold=5,
                                cv_seed=0, tpe_seed=0,
                                params=None, verbose_train=0, verbose_eval=1,
                                use_gp_model_for_validation=True,
                                train_gp_model_cov_pars=True, feval=None,
                                categorical_feature='auto'):
    """Function for choosing tuning parameters using the TPE (Tree-structured Parzen Estimator) algorithm implemented in optuna

    Parameters
    ----------
    X : string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse or list of numpy arrays
        Predictor variables data for creating a gpb.Dataset.
        If string, it represents the path to txt file.
    y : list, numpy 1-D array, pandas Series / one-column DataFrame or None, optional (default=None)
        Response variable / label data.
    param_grid : dict
        The range for every parameter over which a search is done.
        The format for every entry of the dict must be 
        'parameter_name': [lower, upper].
        See https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#tuning-parameters--hyperparameters-for-the-tree-boosting-part
    n_trials: int
        The number of trials for the TPESampler.
    max_num_boost_round : int, optional (default=1000)
        Maximal number of boosting iterations.
    early_stopping_rounds : int or None, optional (default=None)
        Activates early stopping. The ``metric`` needs to improve at least every 
        ``early_stopping_rounds`` round(s) to continue.
    metric : string, list of strings or None, optional (default=None)
        Evaluation metric to be monitored when doing parameter tuning.
        If not None, the metric in ``params`` will be overridden.
        Non-exhaustive list of supported metrics: "test_neg_log_likelihood", "mse", "rmse", "mae",
        "auc", "average_precision", "binary_logloss", "binary_error"
        See https://gpboost.readthedocs.io/en/latest/Parameters.html#metric-parameters
        for a complete list of valid metrics.
    gp_model : GPModel or None, optional (default=None)
        GPModel object for the GPBoost algorithm
    folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
        If generator or iterator, it should yield the train and test indices for each fold.
        If object, it should be one of the scikit-learn splitter classes
        (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        and have ``split`` method.
        This argument has highest priority over other data split arguments.
    nfold : int, optional (default=5)
        Number of folds in CV.
    cv_seed : int, optional (default=0)
        Seed used to generate folds if CV is used (passed to numpy.random.seed).
    tpe_seed : int, optional (default=0)
        Seed for TPESampler of optuna    
    params : dict, optional (default=None)
        Other parameters not included in param_grid.
    verbose_train: int, optional (default=0)
        Controls the level of verbosity of the tree-boosting part during estimation
        < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
    verbose_eval : int or None, optional (default=1)
        Whether to display information on the progress of tuning parameter choice.
        If None or 0, verbose is of.
        If = 1, summary progress information is displayed for every trial of a parameter combination.
        If >= 2, detailed progress is displayed at every boosting stage for every parameter combination.
    use_gp_model_for_validation : bool, optional (default=True)
        If True, the 'gp_model' (Gaussian process and/or random effects) is also used (in addition to the tree model)
        for calculating predictions on the validation data. If False, the 'gp_model' (random effects part) is ignored
        for making predictions and only the tree ensemble is used for making predictions for calculating the validation / test error.
    train_gp_model_cov_pars : bool, optional (default=True)
        If True, the covariance parameters of the 'gp_model' (Gaussian process and/or random effects) are estimated
        in every boosting iterations, otherwise the 'gp_model' parameters are not estimated. In the latter case, you
        need to either estimate them beforehand or provide values via the 'init_cov_pars' parameter when creating
        the 'gp_model'
    feval : callable, list of callable functions or None, optional (default=None)
        Customized evaluation function.
        If more than one evaluation function is provided, only the first evaluation function will be used to choose tuning parameters
        Each evaluation function should accept two parameters: preds, train_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : list or numpy 1-D array
                The predicted values.
            train_data : Dataset
                The training dataset.
            eval_name : string
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        For binary task, the preds is probability of positive class (or margin in case of specified ``fobj``).
        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].
        To ignore the default metric corresponding to the used objective,
        set ``metric`` to the string ``"None"``.
    categorical_feature : list of strings or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
        
    Returns
    -------
    return : dict
        Dictionary with the best parameter combination and score
        The dictionary has the following format:
        {'best_params': best_params, 'best_num_boost_round': best_num_boost_round, 'best_score': best_score}

    Example
    -------
    >>> param_grid= { 'learning_rate': [0.001, 10],
    >>>   'min_data_in_leaf': [1, 1000],
    >>>   'max_depth': [-1, 10],
    >>>   'num_leaves': [2, 1024],
    >>>   'lambda_l2': [0, 100],
    >>>   'max_bin': [63, np.min([10000,n])],
    >>>   'line_search_step_length': [True, False] }
    >>> metric = "mse"
    >>> if likelihood in ("bernoulli_probit", "bernoulli_logit"):
    >>>   metric = "binary_logloss"
    >>> # Can also use metric = "test_neg_log_likelihood"
    >>> # For more options, see https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
    >>> gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
    >>> # Run parameter optimization using 4-fold CV
    >>> opt_params = tune_pars_TPE_algorithm_optuna(X=X, y=y, param_grid=param_grid, 
    >>>                                             nfold=4, gp_model=gp_model, metric=metric, tpe_seed=1,
    >>>                                             max_num_boost_round=1000, n_trials=100, early_stopping_rounds=20)
    >>> print("Best parameters: " + str(opt_params['best_params']))
    >>> print("Best number of iterations: " + str(opt_params['best_iter']))
    >>> print("Best score: " + str(opt_params['best_score']))
    >>> 
    >>> # Alternatively and faster: using manually defined validation data instead of cross-validation
    >>> permute_aux = np.random.permutation(n)
    >>> train_tune_idx = permute_aux[0:int(0.8 * n)] # use 20% of the data as validation data
    >>> valid_tune_idx = permute_aux[int(0.8 * n):n]
    >>> folds = [(train_tune_idx, valid_tune_idx)]
    >>> opt_params = tune_pars_TPE_algorithm_optuna(X=X, y=y, param_grid=param_grid, 
    >>>                                             folds=folds, gp_model=gp_model, metric=metric, tpe_seed=1,
    >>>                                             max_num_boost_round=1000, n_trials=100, early_stopping_rounds=20)

    :Authors:
        Fabio Sigrist
    """
    if not isinstance(param_grid, dict):
        raise ValueError("param_grid must be a dictionary")
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    
    if params is None:
        params = {}
    else:
        params = copy.deepcopy(params)
    param_grid = copy.deepcopy(param_grid)
    metric_higher_better = False
    if metric is not None:
        if isinstance(metric, str):
            metric = [metric]
        if metric[0].startswith(('auc', 'ndcg@', 'map@', 'average_precision')):
            metric_higher_better = True
    elif feval is not None:
        if callable(feval):
            feval = [feval]
        PH1, PH2, metric_higher_better = feval[0](np.array([0]), Dataset(np.array([0]), np.array([0])))
    best_score = -1e99 if metric_higher_better else 1e99
    best_iter = -1
    verbose_eval_cv = verbose_eval >= 2

    def objective_opt(trial):
        nonlocal best_score, best_iter
        """Objective function for tuning parameter search with Optuna."""
        # Parse parameters
        params_loc = {}
        for param in param_grid:
            if param in ['learning_rate', 'shrinkage_rate',
                         'min_gain_to_split', 'min_split_gain',
                         'min_sum_hessian_in_leaf', 'min_sum_hessian_per_leaf', 'min_sum_hessian', 'min_hessian', 'min_child_weight']:
                params_loc[param] = trial.suggest_float(param, param_grid[param][0], param_grid[param][1], log=True)
            elif param in ['lambda_l2', 'reg_lambda', 'lambda',
                           'lambda_l1', 'reg_alpha',
                           'bagging_fraction', 'sub_row', 'subsample', 'bagging',
                           'feature_fraction', 'sub_feature', 'colsample_bytree',
                           'cat_l2',
                           'cat_smooth']:
                params_loc[param] = trial.suggest_float(param, param_grid[param][0], param_grid[param][1], log=False)
            elif param in ['num_leaves', 'num_leaf', 'max_leaves', 'max_leaf',
                           'max_depth',
                           'min_data_in_leaf', 'min_data_per_leaf', 'min_data', 'min_child_samples',
                           'max_bin']:
                params_loc[param] = trial.suggest_int(param, param_grid[param][0], param_grid[param][1])
            elif param in ['line_search_step_length']:
                params_loc[param] = trial.suggest_categorical(param, [param_grid[param][0], param_grid[param][1]])
            else: 
                raise ValueError(f"Unknown parameter '{param}'")
        params_loc.update({'verbose': verbose_train})
        params_loc.update(params)

        # Train the model
        data_bst = Dataset(data=X, label=y)
        cvbst = cv(params=params_loc, train_set=data_bst, gp_model=gp_model, 
                   use_gp_model_for_validation=use_gp_model_for_validation,  
                   train_gp_model_cov_pars=train_gp_model_cov_pars,
                   num_boost_round=max_num_boost_round, 
                   early_stopping_rounds=early_stopping_rounds,
                   folds=folds, nfold=nfold, verbose_eval=verbose_eval_cv, show_stdv=False, 
                   seed=cv_seed, metric=metric, feval=feval,
                   categorical_feature=categorical_feature)
        metric_name = list(cvbst.keys())[0]
        best_score_trial = np.min(cvbst[metric_name])
        best_iter_trial = np.argmin(cvbst[metric_name]) + 1
        if metric_higher_better:
            best_score_trial = np.max(cvbst[metric_name])
            best_iter_trial = np.argmax(cvbst[metric_name]) + 1

        # Save the best number of iterations
        found_better_combination = False
        if metric_higher_better:
            if best_score_trial > best_score:
                found_better_combination = True
        else:
            if best_score_trial < best_score:
                found_better_combination = True
        if found_better_combination:
            best_score = best_score_trial
            best_iter = best_iter_trial

        return best_score_trial
    
    direction = 'maximize' if metric_higher_better else 'minimize'
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=tpe_seed))
    study.optimize(objective_opt, n_trials=n_trials)
    return {'best_params': study.best_trial.params, 'best_iter': best_iter, 'best_score': study.best_trial.values}
