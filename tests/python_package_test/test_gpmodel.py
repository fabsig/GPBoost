# coding: utf-8
"""API tests for GPModel.

These deliberately check the API contract - shapes, keys, types, round trips,
error handling - and only loose statistical recovery of the simulated variance
parameters. They are not a port of the R test suite and they carry no expected
values computed on a reference platform, because such values do not reproduce
bit-wise across compilers and would turn every rebuild into a golden update.

The model classes covered are chosen so that the sparse Cholesky paths are
exercised as well: crossed grouped random effects, tapering, FITC and Vecchia.
"""
import numpy as np
import pytest

import gpboost as gpb


def _sim_grouped(n=500, n_groups=50, sigma2_re=0.5, sigma2_err=0.25, seed=1):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), n // n_groups)
    b = rng.normal(scale=np.sqrt(sigma2_re), size=n_groups)
    eps = rng.normal(scale=np.sqrt(sigma2_err), size=len(group))
    return group, b[group] + eps


def _sim_coords(n=200, seed=2):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(size=(n, 2))
    # a smooth field, so that a spatial model has something to find
    b = np.sin(6 * coords[:, 0]) + np.cos(5 * coords[:, 1])
    y = b + rng.normal(scale=0.3, size=n)
    return coords, y


# ---------------------------------------------------------------- basic API

def test_grouped_random_effects_fit_predict():
    group, y = _sim_grouped()
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    gp_model.fit(y=y)

    cov_pars = gp_model.get_cov_pars()
    assert np.asarray(cov_pars).size >= 2          # error variance and group variance
    assert np.all(np.asarray(cov_pars, dtype=float) > 0)
    assert np.isfinite(gp_model.get_current_neg_log_likelihood())

    group_test = np.array([0, 1, 2])
    pred = gp_model.predict(group_data_pred=group_test, predict_var=True)
    assert set(["mu", "var", "cov"]).issubset(pred.keys())
    assert pred["mu"].shape == (3,)
    assert pred["var"].shape == (3,)
    assert np.all(pred["var"] > 0)
    assert np.all(np.isfinite(pred["mu"]))


def test_recovers_simulated_variances_roughly():
    # loose on purpose: this guards against a broken factorization, not against
    # small numerical differences between compilers
    group, y = _sim_grouped(n=2000, n_groups=200, sigma2_re=0.8, sigma2_err=0.2, seed=7)
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    gp_model.fit(y=y)
    pars = np.asarray(gp_model.get_cov_pars(), dtype=float).ravel()
    err_var, re_var = pars[0], pars[1]
    assert 0.1 < err_var < 0.4
    assert 0.4 < re_var < 1.4


def test_linear_fixed_effects():
    group, y = _sim_grouped()
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(len(y)), rng.normal(size=len(y))])
    y = y + X[:, 1] * 2.0
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    gp_model.fit(y=y, X=X)

    coef = np.asarray(gp_model.get_coef(), dtype=float).ravel()
    assert coef.size >= 2
    assert abs(coef[1] - 2.0) < 0.3                # slope is well identified


def test_neg_log_likelihood_responds_to_cov_pars():
    group, y = _sim_grouped()
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    nll1 = gp_model.neg_log_likelihood(cov_pars=np.array([0.25, 0.5]), y=y)
    nll2 = gp_model.neg_log_likelihood(cov_pars=np.array([1.0, 2.0]), y=y)
    assert np.isfinite(nll1) and np.isfinite(nll2)
    assert nll1 != nll2

    # the fitted optimum must not be worse than an arbitrary starting point
    gp_model.fit(y=y)
    assert gp_model.get_current_neg_log_likelihood() <= max(nll1, nll2) + 1e-6


# --------------------------------------------------- sparse Cholesky paths

def test_two_crossed_grouped_random_effects():
    # no Gaussian process, so this is a sparse Cholesky
    rng = np.random.default_rng(11)
    n, m = 2000, 100
    g1 = rng.integers(0, m, n)
    g2 = rng.integers(0, m, n)
    y = (rng.normal(scale=0.7, size=m)[g1] + rng.normal(scale=0.7, size=m)[g2]
         + rng.normal(scale=0.5, size=n))
    group = np.column_stack([g1, g2])

    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    gp_model.fit(y=y)
    pars = np.asarray(gp_model.get_cov_pars(), dtype=float).ravel()
    assert pars.size >= 3                          # error variance and two group variances
    assert np.all(pars > 0)

    pred = gp_model.predict(group_data_pred=np.column_stack([[0, 1], [0, 1]]),
                            predict_var=True)
    assert pred["mu"].shape == (2,)
    assert np.all(pred["var"] > 0)


@pytest.mark.parametrize("gp_approx,kwargs", [
    ("none", {}),
    ("vecchia", {"num_neighbors": 15}),
    ("tapering", {"cov_fct_taper_shape": 0, "cov_fct_taper_range": 0.3}),
    ("fitc", {"num_ind_points": 50}),
])
def test_gp_approximations(gp_approx, kwargs):
    coords, y = _sim_coords()
    gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential",
                           likelihood="gaussian", gp_approx=gp_approx, **kwargs)
    gp_model.fit(y=y, params={"maxit": 20})
    assert np.isfinite(gp_model.get_current_neg_log_likelihood())
    assert np.all(np.asarray(gp_model.get_cov_pars(), dtype=float) > 0)

    coords_test = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.2]])
    pred = gp_model.predict(gp_coords_pred=coords_test, predict_var=True)
    assert pred["mu"].shape == (3,)
    assert np.all(pred["var"] > 0)


@pytest.mark.parametrize("likelihood", ["bernoulli_probit", "poisson", "gamma"])
def test_non_gaussian_likelihoods(likelihood):
    # these run the Laplace approximation, i.e. the mode finding
    rng = np.random.default_rng(23)
    group, lin_pred = _sim_grouped(n=600, n_groups=60, seed=5)
    if likelihood == "bernoulli_probit":
        y = (rng.uniform(size=len(lin_pred)) < 0.5 + 0.2 * np.sign(lin_pred)).astype(float)
    elif likelihood == "poisson":
        y = rng.poisson(np.exp(lin_pred - 0.5)).astype(float)
    else:
        y = rng.gamma(shape=2.0, scale=np.exp(lin_pred) / 2.0)

    gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
    gp_model.fit(y=y, params={"maxit": 20})
    assert np.isfinite(gp_model.get_current_neg_log_likelihood())

    pred = gp_model.predict(group_data_pred=np.array([0, 1]), predict_var=True,
                            predict_response=True)
    assert pred["mu"].shape == (2,)
    assert np.all(np.isfinite(pred["mu"]))


# ------------------------------------------------------------- predictions

def test_predict_cov_mat_is_consistent_with_var():
    coords, y = _sim_coords()
    gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential", likelihood="gaussian")
    gp_model.fit(y=y, params={"maxit": 20})
    coords_test = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.2]])

    pred_cov = gp_model.predict(gp_coords_pred=coords_test, predict_cov_mat=True)
    pred_var = gp_model.predict(gp_coords_pred=coords_test, predict_var=True)

    cov = np.asarray(pred_cov["cov"])
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov, cov.T, atol=1e-8)                 # symmetric
    np.testing.assert_allclose(np.diag(cov), pred_var["var"], rtol=1e-6)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-8)                    # positive semi definite


def test_predict_training_data_random_effects():
    group, y = _sim_grouped()
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    gp_model.fit(y=y)
    re = gp_model.predict_training_data_random_effects()
    arr = np.asarray(re, dtype=float)
    assert arr.shape[0] == len(y)
    assert np.all(np.isfinite(arr))


# ------------------------------------------------------ persistence and misc

def test_save_and_load_model_round_trip(tmp_path):
    group, y = _sim_grouped()
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    gp_model.fit(y=y)
    group_test = np.array([0, 1, 2])
    pred_before = gp_model.predict(group_data_pred=group_test, predict_var=True)

    fname = str(tmp_path / "gp_model.json")
    gp_model.save_model(fname)
    loaded = gpb.GPModel(model_file=fname)
    pred_after = loaded.predict(group_data_pred=group_test, predict_var=True)

    np.testing.assert_allclose(pred_before["mu"], pred_after["mu"], rtol=1e-10)
    np.testing.assert_allclose(pred_before["var"], pred_after["var"], rtol=1e-10)


def test_model_to_dict_round_trip():
    group, y = _sim_grouped()
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    gp_model.fit(y=y)
    d = gp_model.model_to_dict(include_response_data=True)
    assert isinstance(d, dict) and len(d) > 0

    loaded = gpb.GPModel(model_dict=d)
    group_test = np.array([0, 1])
    np.testing.assert_allclose(
        gp_model.predict(group_data_pred=group_test)["mu"],
        loaded.predict(group_data_pred=group_test)["mu"], rtol=1e-10)


def test_set_optim_params_limits_iterations():
    group, y = _sim_grouped()
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    gp_model.set_optim_params(params={"maxit": 1, "trace": False})
    gp_model.fit(y=y)
    assert np.isfinite(gp_model.get_current_neg_log_likelihood())


def test_summary_runs():
    group, y = _sim_grouped()
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    gp_model.fit(y=y)
    gp_model.summary()          # must not raise


def test_invalid_input_raises():
    group, y = _sim_grouped()
    with pytest.raises(Exception):
        gpb.GPModel(likelihood="gaussian")                      # no random effect at all
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian")
    with pytest.raises(Exception):
        gp_model.fit(y=y[:-1])                                  # y of the wrong length
