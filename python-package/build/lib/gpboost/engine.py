# coding: utf-8
# pylint: disable = invalid-name, W0105
"""Library with training routines of GPBoost."""
from __future__ import absolute_import

import collections
import copy
import warnings
from operator import attrgetter

import numpy as np

from . import callback
from .basic import Booster, Dataset, GPBoostError, _ConfigAliases, _InnerPredictor, GPModel
from .compat import (SKLEARN_INSTALLED, _GPBoostGroupKFold, _GPBoostStratifiedKFold,
                     string_type, integer_types, range_, zip_)


def train(params, train_set, num_boost_round=100,
          gp_model=None, train_gp_model_cov_pars=True,
          valid_sets=None, valid_names=None,
          fobj=None, feval=None, init_model=None,
          feature_name='auto', categorical_feature='auto',
          early_stopping_rounds=None, evals_result=None,
          use_gp_model_for_validation=False,
          verbose_eval=True, learning_rates=None,
          keep_training_booster=True, callbacks=None):
    """Perform the training with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for training, see Parameters.rst for more information.
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    gp_model : GPModel or None, optional (default=None)
        GPModel object for Gaussian process boosting. Can currently only be used for objective = "regression"
    train_gp_model_cov_pars : bool, optional (default=True)
        If True, the covariance parameters of the Gaussian process are estimated in every boosting iterations,
        otherwise the GPModel parameters are not estimated. In the latter case, you need to either esimate them
        beforehand or provide the values via the 'init_cov_pars' parameter when creating the GPModel
    valid_sets : list of Datasets or None, optional (default=None)
        List of data to be evaluated on during training.
    valid_names : list of strings or None, optional (default=None)
        Names of ``valid_sets``.
    fobj : callable or None, optional (default=None)
        Customized objective function.
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

        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
        and you should group grad and hess in this way as well.

    feval : callable or None, optional (default=None)
        Customized evaluation function.
        Should accept two parameters: preds, train_data,
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

    use_gp_model_for_validation : bool, optional (default=False)
        If True, the Gaussian process is also used (in addition to the tree model) for calculating predictions on
        the validation data
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
    keep_training_booster : bool, optional (default=True)
        Whether the returned Booster will be used to keep training.
        If False, the returned value will be converted into _InnerPredictor before returning.
        You can still use _InnerPredictor as ``init_model`` for future continue training.
    callbacks : list of callables or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.

    Returns
    -------
    booster : Booster
        The trained Booster model.
    """
    # create predictor first
    params = copy.deepcopy(params)
    if gp_model is not None:
        if not isinstance(gp_model, GPModel):
            raise TypeError('gp_model should be GPModel instance, met {}'
                            .format(type(gp_model).__name__))
        params['has_gp_model'] = True
    if fobj is not None:
        for obj_alias in _ConfigAliases.get("objective"):
            params.pop(obj_alias, None)
        params['objective'] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            num_boost_round = params.pop(alias)
            warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
            break
    for alias in _ConfigAliases.get("early_stopping_round"):
        if alias in params:
            early_stopping_rounds = params.pop(alias)
            warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
            break
    first_metric_only = params.pop('first_metric_only', False)

    params['use_gp_model_for_validation'] = use_gp_model_for_validation
    params['train_gp_model_cov_pars'] = train_gp_model_cov_pars

    if use_gp_model_for_validation:
        if gp_model is None:
            raise ValueError("gp_model missing but is should be used for validation")
        if not gp_model.prediction_data_is_set:
            raise ValueError("Prediction data for gp_model has not been set. Call gp_model.set_prediction_data() first")
        if not isinstance(valid_sets, Dataset):
            if len(valid_sets) > 1:
                raise ValueError("Can use only one validation set when use_gp_model_for_validation = True")
        if feval is not None:
            raise ValueError("Using the Gaussian process for making predictions for the validation data is currently "
                             "not supported for custom validation functions. If you need this feature, contact the "
                             "developer of this package.")

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    if isinstance(init_model, string_type):
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
        if isinstance(valid_names, string_type):
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
    elif isinstance(verbose_eval, integer_types):
        callbacks.add(callback.print_evaluation(verbose_eval))

    if early_stopping_rounds is not None:
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
        for valid_set, name_valid_set in zip_(reduced_valid_sets, name_valid_sets):
            booster.add_valid(valid_set, name_valid_set)
    finally:
        train_set._reverse_update_params()
        for valid_set in reduced_valid_sets:
            valid_set._reverse_update_params()
    booster.best_iteration = 0

    # start training
    for i in range_(init_iteration, init_iteration + num_boost_round):
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
    if not keep_training_booster:
        booster.model_from_string(booster.model_to_string(), False).free_dataset()
    return booster


class _CVBooster(object):
    """Auxiliary data struct to hold all boosters of CV."""

    def __init__(self):
        self.boosters = []
        self.best_iteration = -1

    def append(self, booster):
        """Add a booster to _CVBooster."""
        self.boosters.append(booster)

    def __getattr__(self, name):
        """Redirect methods call of _CVBooster."""
        def handler_function(*args, **kwargs):
            """Call methods with each booster, and concatenate their results."""
            ret = []
            for booster in self.boosters:
                ret.append(getattr(booster, name)(*args, **kwargs))
            return ret
        return handler_function


def _make_n_folds(full_data, folds, nfold, params, seed, gp_model=None, use_gp_model_for_validation=False,
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
                flatted_group = np.repeat(range_(len(group_info)), repeats=group_info)
            else:
                flatted_group = np.zeros(num_data, dtype=np.int32)
            folds = folds.split(X=np.zeros(num_data), y=full_data.get_label(), groups=flatted_group)
    else:
        if any(params.get(obj_alias, "") == "lambdarank" for obj_alias in _ConfigAliases.get("objective")):
            if not SKLEARN_INSTALLED:
                raise GPBoostError('Scikit-learn is required for lambdarank cv.')
            # lambdarank task, split according to groups
            group_info = np.array(full_data.get_group(), dtype=np.int32, copy=False)
            flatted_group = np.repeat(range_(len(group_info)), repeats=group_info)
            group_kfold = _GPBoostGroupKFold(n_splits=nfold)
            folds = group_kfold.split(X=np.zeros(num_data), groups=flatted_group)
        elif stratified:
            if not SKLEARN_INSTALLED:
                raise GPBoostError('Scikit-learn is required for stratified cv.')
            skf = _GPBoostStratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
            folds = skf.split(X=np.zeros(num_data), y=full_data.get_label())
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(num_data)
            else:
                randidx = np.arange(num_data)
            kstep = int(num_data / nfold)
            test_id = [randidx[i: i + kstep] for i in range_(0, num_data, kstep)]
            train_id = [np.concatenate([test_id[i] for i in range_(nfold) if k != i]) for k in range_(nfold)]
            folds = zip_(train_id, test_id)

    ret = _CVBooster()
    for train_idx, test_idx in folds:
        train_set = full_data.subset(sorted(train_idx))
        valid_set = full_data.subset(sorted(test_idx))
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
                group_data = gp_model.group_data[train_idx, :]
                group_data_pred = gp_model.group_data[test_idx, :]
                if gp_model.group_rand_coef_data is not None:
                    group_rand_coef_data = gp_model.group_rand_coef_data[train_idx, :]
                    group_rand_coef_data_pred = gp_model.group_rand_coef_data[test_idx, :]
            if gp_model.gp_coords is not None:
                gp_coords = gp_model.gp_coords[train_idx, :]
                gp_coords_pred = gp_model.gp_coords[test_idx, :]
                if gp_model.gp_rand_coef_data is not None:
                    gp_rand_coef_data = gp_model.gp_rand_coef_data[train_idx, :]
                    gp_rand_coef_data_pred = gp_model.gp_rand_coef_data[test_idx, :]
            if gp_model.cluster_ids is not None:
                cluster_ids = gp_model.cluster_ids[train_idx, :]
                cluster_ids_pred = gp_model.cluster_ids[test_idx, :]
            vecchia_approx = gp_model.vecchia_approx
            num_neighbors = gp_model.num_neighbors
            vecchia_ordering = gp_model.vecchia_ordering
            vecchia_pred_type = gp_model.vecchia_pred_type
            num_neighbors_pred = gp_model.num_neighbors_pred
            cov_function = gp_model.cov_function
            ind_effect_group_rand_coef = gp_model.ind_effect_group_rand_coef
            gp_model_train = GPModel(group_data=group_data,
                                         group_rand_coef_data=group_rand_coef_data,
                                         ind_effect_group_rand_coef=ind_effect_group_rand_coef,
                                         gp_coords=gp_coords,
                                         gp_rand_coef_data=gp_rand_coef_data,
                                         cov_function=cov_function,
                                         vecchia_approx=vecchia_approx,
                                         num_neighbors=num_neighbors,
                                         vecchia_ordering=vecchia_ordering,
                                         vecchia_pred_type=vecchia_pred_type,
                                         num_neighbors_pred=num_neighbors_pred,
                                         cluster_ids=cluster_ids,
                                         free_raw_data=True)
            gp_model_train.set_optim_params(params=gp_model.params)
            if use_gp_model_for_validation:
                gp_model_train.set_prediction_data(group_data_pred=group_data_pred,
                                                   group_rand_coef_data_pred=group_rand_coef_data_pred,
                                                   gp_coords_pred=gp_coords_pred,
                                                   gp_rand_coef_data_pred=gp_rand_coef_data_pred,
                                                   cluster_ids_pred=cluster_ids_pred)
        else:
            gp_model_train = None
        cvbooster = Booster(params=tparam, train_set=train_set, gp_model=gp_model_train)
        if eval_train_metric:
            cvbooster.add_valid(train_set, 'train')
        cvbooster.add_valid(valid_set, 'valid')
        ret.append(cvbooster)
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
       gp_model=None, train_gp_model_cov_pars=True,
       use_gp_model_for_validation=False,
       folds=None, nfold=5, stratified=False, shuffle=True,
       metrics=None, fobj=None, feval=None, init_model=None,
       feature_name='auto', categorical_feature='auto',
       early_stopping_rounds=None, fpreproc=None,
       verbose_eval=True, show_stdv=False, seed=0,
       callbacks=None, eval_train_metric=False):
    """Perform the cross-validation with given paramaters.

    Parameters
    ----------
    params : dict
        Parameters for Booster.
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    gp_model : GPModel or None, optional (default=None)
        GPModel object for Gaussian process boosting. Can currently only be used for objective = "regression"
    train_gp_model_cov_pars : bool, optional (default=True)
        If True, the covariance parameters of the Gaussian process are estimated in every boosting iterations,
        otherwise the GPModel parameters are not estimated. In the latter case, you need to either esimate them
        beforehand or provide the values via the 'init_cov_pars' parameter when creating the GPModel
    use_gp_model_for_validation : bool, optional (default=False)
        If True, the Gaussian process is also used (in addition to the tree model) for calculating predictions on
        the validation data
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
    metrics : string, list of strings or None, optional (default=None)
        Evaluation metrics to be monitored while CV.
        If not None, the metric in ``params`` will be overridden.
    fobj : callable or None, optional (default=None)
        Customized objective function.
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

        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
        and you should group grad and hess in this way as well.

    feval : callable or None, optional (default=None)
        Customized evaluation function.
        Should accept two parameters: preds, train_data,
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

        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].
        To ignore the default metric corresponding to the used objective,
        set ``metrics`` to the string ``"None"``.
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
        Activates early stopping.
        CV score needs to improve at least every ``early_stopping_rounds`` round(s)
        to continue.
        Requires at least one metric. If there's more than one, will check all of them.
        To check only the first metric, set the ``first_metric_only`` parameter to ``True`` in ``params``.
        Last entry in evaluation history is the one from the best iteration.
    fpreproc : callable or None, optional (default=None)
        Preprocessing function that takes (dtrain, dtest, params)
        and returns transformed versions of those.
    verbose_eval : bool, int, or None, optional (default=None)
        Whether to display the progress.
        If None, progress will be displayed when np.ndarray is returned.
        If True, progress will be displayed at every boosting stage.
        If int, progress will be displayed at every given ``verbose_eval`` boosting stage.
    show_stdv : bool, optional (default=True)
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

    Returns
    -------
    eval_hist : dict
        Evaluation history.
        The dictionary has the following format:
        {'metric1-mean': [values], 'metric1-stdv': [values],
        'metric2-mean': [values], 'metric2-stdv': [values],
        ...}.
    """
    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")

    params = copy.deepcopy(params)
    if fobj is not None:
        for obj_alias in _ConfigAliases.get("objective"):
            params.pop(obj_alias, None)
        params['objective'] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
            num_boost_round = params.pop(alias)
            break
    for alias in _ConfigAliases.get("early_stopping_round"):
        if alias in params:
            warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
            early_stopping_rounds = params.pop(alias)
            break
    first_metric_only = params.pop('first_metric_only', False)

    params['use_gp_model_for_validation'] = use_gp_model_for_validation
    params['train_gp_model_cov_pars'] = train_gp_model_cov_pars

    if use_gp_model_for_validation:
        if gp_model is None:
            raise ValueError("gp_model missing but is should be used for validation")
        if feval is not None:
            raise ValueError("Using the Gaussian process for making predictions for the validation data is currently "
                             "not supported for custom validation functions. If you need this feature, contact the "
                             "developer of this package.")
    if gp_model is None and stratified:
        raise ValueError("stratified=True is not supported when a GPModel is provided")

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    if isinstance(init_model, string_type):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    else:
        predictor = None
    train_set._update_params(params) \
             ._set_predictor(predictor) \
             .set_feature_name(feature_name) \
             .set_categorical_feature(categorical_feature)

    if metrics is not None:
        for metric_alias in _ConfigAliases.get("metric"):
            params.pop(metric_alias, None)
        params['metric'] = metrics

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
    if early_stopping_rounds is not None:
        callbacks.add(callback.early_stopping(early_stopping_rounds, first_metric_only, verbose=False))
    if verbose_eval is True:
        callbacks.add(callback.print_evaluation(show_stdv=show_stdv))
    elif isinstance(verbose_eval, integer_types):
        callbacks.add(callback.print_evaluation(verbose_eval, show_stdv=show_stdv))

    callbacks_before_iter = {cb for cb in callbacks if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter('order'))

    for i in range_(num_boost_round):
        for cb in callbacks_before_iter:
            cb(callback.CallbackEnv(model=cvfolds,
                                    params=params,
                                    iteration=i,
                                    begin_iteration=0,
                                    end_iteration=num_boost_round,
                                    evaluation_result_list=None))
        cvfolds.update(fobj=fobj)
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

    return dict(results)
