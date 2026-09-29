# coding: utf-8
"""API tests for the 'hurst' and 'hurst_ard' covariance functions of order 1 and 2 ('cov_fct_order').

The negative log-likelihoods are compared with a direct calculation in NumPy, and the order has to survive
saving and loading as well as the reconstruction of the model in cross-validation.
"""
import numpy as np
import pytest

import gpboost as gpb


def _hurst_cov(coords, sigma2, H, order, ranges=None):
    x = np.array(coords, dtype=float).reshape(len(coords), -1)
    if ranges is not None:
        x[:, 1:] = x[:, 1:] / np.asarray(ranges)
    u = np.sum(x ** 2, axis=1)
    u_xy = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    if order == 1:
        return sigma2 / 2 * (np.add.outer(u ** H, u ** H) - u_xy ** H)
    return sigma2 / (2 * (2 * H - 1)) * (u_xy ** H - np.add.outer(u ** H, u ** H) +
                                        2 * H * (x @ x.T) * np.add.outer(u ** (H - 1), u ** (H - 1)))


def _gauss_nll(y, Sigma):
    return 0.5 * y @ np.linalg.solve(Sigma, y) + 0.5 * np.linalg.slogdet(Sigma)[1] + 0.5 * len(y) * np.log(2 * np.pi)


def _sim_time(n=80, seed=1):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(size=n))
    y = 1 + 2 * t + np.sin(6 * t) + rng.normal(scale=0.1, size=n)
    return t, y


@pytest.mark.parametrize("order,H", [(1, 0.3), (2, 1.3)])
def test_neg_log_likelihood_matches_direct_calculation(order, H):
    rng = np.random.default_rng(2)
    coords = rng.uniform(size=(60, 2))
    y = rng.normal(size=60)
    gp_model = gpb.GPModel(gp_coords=coords, cov_function="hurst", cov_fct_order=order)
    nll = gp_model.neg_log_likelihood(cov_pars=np.array([0.1, 1.2, H]), y=y)
    Sigma = _hurst_cov(coords, 1.2, H, order) + 0.1 * np.eye(60)
    np.testing.assert_allclose(nll, _gauss_nll(y, Sigma), rtol=1e-10)
    gp_model = gpb.GPModel(gp_coords=coords, cov_function="hurst_ard", cov_fct_order=order)
    nll = gp_model.neg_log_likelihood(cov_pars=np.array([0.1, 1.2, H, 0.6]), y=y)
    Sigma = _hurst_cov(coords, 1.2, H, order, ranges=[0.6]) + 0.1 * np.eye(60)
    np.testing.assert_allclose(nll, _gauss_nll(y, Sigma), rtol=1e-10)


def test_continuous_time_rw2_with_fixed_H():
    t, y = _sim_time()
    X = np.column_stack((np.ones(len(t)), t))
    gp_model = gpb.GPModel(gp_coords=t, cov_function="hurst", cov_fct_order=2)
    gp_model.fit(y=y, X=X, params={"init_cov_pars": np.array([0.1, 1., 1.5]),
                                   "estimate_cov_par_index": np.array([1, 1, 0])})
    cov_pars = np.asarray(gp_model.get_cov_pars(std_err=True), dtype=float)
    assert cov_pars[0, 2] == 1.5
    # H is not estimated and has no standard error
    assert np.isnan(cov_pars[1, 2])
    assert np.all(np.isfinite(cov_pars[1, :2]))


def test_order_is_saved_and_loaded(tmp_path):
    t, y = _sim_time()
    gp_model = gpb.GPModel(gp_coords=t, cov_function="hurst", cov_fct_order=2)
    gp_model.fit(y=y)
    t_pred = np.array([0.25, 1.2])
    pred = gp_model.predict(gp_coords_pred=t_pred, predict_var=True)
    fname = str(tmp_path / "gp_model_hurst.json")
    gp_model.save_model(fname)
    loaded = gpb.GPModel(model_file=fname)
    assert loaded.cov_fct_order == 2
    pred_loaded = loaded.predict(gp_coords_pred=t_pred, predict_var=True)
    np.testing.assert_allclose(pred["mu"], pred_loaded["mu"], rtol=1e-10)
    np.testing.assert_allclose(pred["var"], pred_loaded["var"], rtol=1e-10)
    # a model saved without 'cov_fct_order' (i.e., with an older version) has order 1
    gp_model = gpb.GPModel(gp_coords=t, cov_function="hurst")
    gp_model.fit(y=y)
    model_dict = gp_model.model_to_dict(include_response_data=True)
    del model_dict["cov_fct_order"]
    loaded = gpb.GPModel(model_dict=model_dict)
    assert loaded.cov_fct_order == 1
    np.testing.assert_allclose(gp_model.predict(gp_coords_pred=t_pred)["mu"], loaded.predict(gp_coords_pred=t_pred)["mu"], rtol=1e-10)


def test_invalid_orders_and_hurst_exponents():
    t, y = _sim_time()
    for kwargs in [dict(cov_function="hurst", cov_fct_order=3), dict(cov_function="hurst", cov_fct_order=0),
                   dict(cov_function="hurst", cov_fct_order=1.5), dict(cov_function="hurst", cov_fct_order=True),
                   dict(cov_function="matern", cov_fct_order=2)]:
        with pytest.raises(Exception):
            gpb.GPModel(gp_coords=t, **kwargs)
    gp_model = gpb.GPModel(gp_coords=t, cov_function="hurst", cov_fct_order=2)
    for H in [0.5, 2.]:
        with pytest.raises(Exception):
            gp_model.neg_log_likelihood(cov_pars=np.array([0.1, 1., H]), y=y)


def test_order_is_used_in_cross_validation():
    t, y = _sim_time()
    data_train = gpb.Dataset(np.column_stack((np.ones(len(t)), t)), y)
    folds = [(np.where(np.arange(len(t)) % 2 == i)[0], np.where(np.arange(len(t)) % 2 != i)[0]) for i in range(2)]
    results = []
    for order in [1, 2]:
        gp_model = gpb.GPModel(gp_coords=t, cov_function="hurst", cov_fct_order=order)
        cvbst = gpb.cv(params={"learning_rate": 0.1, "max_depth": 2, "verbose": -1}, train_set=data_train, gp_model=gp_model,
                       num_boost_round=3, folds=folds, metric="l2", return_cvbooster=False)
        results.append(cvbst["l2-mean"])
    assert not np.allclose(results[0], results[1])
