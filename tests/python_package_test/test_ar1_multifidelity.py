# coding: utf-8
"""Tests for the AR1 multifidelity covariance in the GPBoost algorithm.

With an 'ar1_mf_<base>' covariance and the default fidelity_specific_mean=True the fidelity
indicator is added to the boosting features, so that the two fidelities can have different
means. Training and cross-validation have to agree on that feature matrix, and the feature
names of the Dataset have to follow the added column.
"""
import numpy as np
import pytest

import gpboost as gpb

_COV_PARS = np.array([0.08, 1.1, 0.25, 0.5, 0.12, -0.6])
_PARAMS = {"learning_rate": 0.1, "max_depth": 2, "min_data_in_leaf": 4,
           "objective": "regression_l2", "metric": "l2", "verbose": 0}


def _sim_multifidelity(seed=1):
    x_low = np.linspace(0.02, 0.98, 18)
    x_high = np.linspace(0.04, 0.96, 14) + 0.001
    coords = np.vstack((np.column_stack((x_low, np.zeros(len(x_low)))),
                        np.column_stack((x_high, np.ones(len(x_high))))))
    features = np.column_stack((coords[:, 0], np.sin(4. * coords[:, 0])))
    # The two fidelities differ by a constant, which only the fidelity feature can represent
    y = np.sin(6. * coords[:, 0]) + 5. * coords[:, 1] + \
        0.1 * np.random.default_rng(seed).normal(size=coords.shape[0])
    return coords, features, y


def _gp_model(coords, fidelity_specific_mean=True):
    gp_model = gpb.GPModel(gp_coords=coords, cov_function="ar1_mf_exponential", likelihood="gaussian",
                           fidelity_specific_mean=fidelity_specific_mean)
    gp_model.set_optim_params(params={"init_cov_pars": _COV_PARS,
                                      "init_coef_aux_pars_from_iid_model": False})
    return gp_model


def _folds(n):
    return [(np.arange(0, n, 2), np.arange(1, n, 2)), (np.arange(1, n, 2), np.arange(0, n, 2))]


@pytest.mark.parametrize("feature_name", ["auto", ["x", "nonlinear"]])
def test_train_adds_the_fidelity_feature(feature_name):
    coords, features, y = _sim_multifidelity()
    train_set = gpb.Dataset(features, label=y, feature_name=feature_name, free_raw_data=False)
    booster = gpb.train(params=_PARAMS, train_set=train_set, gp_model=_gp_model(coords),
                        num_boost_round=3, train_gp_model_cov_pars=False)
    assert train_set.data.shape[1] == 3
    assert len(booster.feature_name()) == 3
    if feature_name != "auto":
        assert booster.feature_name() == feature_name + ["AR1_MF_fidelity"]


@pytest.mark.parametrize("feature_name", ["auto", ["x", "nonlinear"]])
def test_cv_uses_the_same_features_as_train(feature_name):
    coords, features, y = _sim_multifidelity()

    def run(fidelity_specific_mean):
        train_set = gpb.Dataset(features, label=y, feature_name=feature_name, free_raw_data=False)
        result = gpb.cv(params=_PARAMS, train_set=train_set,
                        gp_model=_gp_model(coords, fidelity_specific_mean),
                        num_boost_round=25, folds=_folds(len(y)), train_gp_model_cov_pars=False,
                        use_gp_model_for_validation=False, verbose_eval=False)
        return train_set, min(result["l2-mean"])

    with_feature, error_with = run(True)
    without_feature, error_without = run(False)
    assert with_feature.data.shape[1] == 3
    assert without_feature.data.shape[1] == 2
    if feature_name != "auto":
        assert list(with_feature.feature_name) == feature_name + ["AR1_MF_fidelity"]
        assert list(without_feature.feature_name) == feature_name
    assert error_with < 0.5 * error_without


def test_dataset_with_the_fidelity_feature_requires_a_fidelity_specific_mean():
    # The trees of a model without a fidelity-specific mean would use the fidelity indicator that 'train()' and 'cv()'
    # have added to a Dataset as a feature
    coords, features, y = _sim_multifidelity()
    train_set = gpb.Dataset(features, label=y, free_raw_data=False)
    gpb.train(params=_PARAMS, train_set=train_set, gp_model=_gp_model(coords), num_boost_round=1,
              train_gp_model_cov_pars=False)
    gpb.cv(params=_PARAMS, train_set=train_set, gp_model=_gp_model(coords), num_boost_round=1,
           folds=_folds(len(y)), train_gp_model_cov_pars=False, use_gp_model_for_validation=False,
           verbose_eval=False)
    for gp_model in (_gp_model(coords, fidelity_specific_mean=False), None):
        with pytest.raises(ValueError, match="Use a new Dataset for a model without a fidelity-specific mean"):
            gpb.train(params=_PARAMS, train_set=train_set, gp_model=gp_model, num_boost_round=1,
                      train_gp_model_cov_pars=False)
        with pytest.raises(ValueError, match="Use a new Dataset for a model without a fidelity-specific mean"):
            gpb.cv(params=_PARAMS, train_set=train_set, gp_model=gp_model, num_boost_round=1,
                   folds=_folds(len(y)), train_gp_model_cov_pars=False, use_gp_model_for_validation=False,
                   verbose_eval=False)
    with pytest.raises(ValueError, match="Use a new Dataset for a model without a fidelity-specific mean"):
        gpb.train(params=_PARAMS, train_set=gpb.Dataset(features, label=y), valid_sets=[train_set],
                  num_boost_round=1)


@pytest.mark.parametrize("free_raw_data", [False, True])
def test_subsets_and_binary_files_of_a_dataset_with_the_fidelity_feature(free_raw_data, tmp_path):
    coords, features, y = _sim_multifidelity()
    train_set = gpb.Dataset(features, label=y, free_raw_data=free_raw_data)
    gpb.train(params=_PARAMS, train_set=train_set, gp_model=_gp_model(coords), num_boost_round=1,
              train_gp_model_cov_pars=False)
    # A subset has the fidelity indicator as a feature too
    rows = np.arange(24)
    with pytest.raises(ValueError, match="Use a new Dataset for a model without a fidelity-specific mean"):
        gpb.train(params=_PARAMS, train_set=train_set.subset(rows),
                  gp_model=_gp_model(coords[rows], fidelity_specific_mean=False), num_boost_round=1,
                  train_gp_model_cov_pars=False)
    # The indices can be any iterable
    booster = gpb.train(params=_PARAMS, train_set=train_set.subset(iter(rows)), gp_model=_gp_model(coords[rows]),
                        num_boost_round=3, train_gp_model_cov_pars=False)
    assert booster.num_feature() == 3
    # Predictions of the GP require the raw data
    if not free_raw_data:
        pred = booster.predict(data=features[24:], gp_coords_pred=coords[24:])
        assert np.all(np.isfinite(pred["response_mean"]))
    # A Dataset loaded from a binary file would not know that its last feature is the fidelity indicator
    with pytest.raises(ValueError, match="cannot be saved to a binary file"):
        train_set.save_binary(str(tmp_path / "train.bin"))


def test_subsets_without_raw_data_created_before_the_fidelity_feature_is_added():
    # Such a subset gets the features that its reference has when it is constructed
    coords, features, y = _sim_multifidelity()
    train_set = gpb.Dataset(features, label=y, free_raw_data=True)
    rows = np.arange(24)
    subset = train_set.subset(rows)
    nested_subset = subset.subset(np.arange(12))
    gpb.train(params=_PARAMS, train_set=train_set, gp_model=_gp_model(coords), num_boost_round=1,
              train_gp_model_cov_pars=False)
    for dataset in (subset, nested_subset):
        with pytest.raises(ValueError, match="Use a new Dataset for a model without a fidelity-specific mean"):
            gpb.train(params=_PARAMS, train_set=dataset, num_boost_round=1)
    booster = gpb.train(params=_PARAMS, train_set=subset, gp_model=_gp_model(coords[rows]), num_boost_round=3,
                        train_gp_model_cov_pars=False)
    assert booster.num_feature() == 3
