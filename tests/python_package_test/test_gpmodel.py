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




@pytest.mark.parametrize("gp_approx,kwargs,contains_nugget,deterministic", [
    ("none", {}, False, True),
    ("fitc", {"num_ind_points": 8, "ind_points_selection": "random"}, False, True),
    ("full_scale_tapering", {"num_ind_points": 8, "ind_points_selection": "random",
                             "cov_fct_taper_range": 1e6, "cov_fct_taper_shape": 2}, False, False),
    ("vecchia", {"num_neighbors": 20, "vecchia_ordering": "none"}, True, True),
    ("full_scale_vecchia", {"num_ind_points": 8, "ind_points_selection": "random",
                            "num_neighbors": 20, "vecchia_ordering": "none"}, True, True),
])
def test_predict_for_a_cluster_without_observed_data(gp_approx, kwargs, contains_nugget, deterministic):
    """A cluster in 'cluster_ids_pred' that does not occur in 'cluster_ids' has no observed data.

    The predictive distribution of the latent GP is then its prior, i.e. the mean is zero and the
    variance is the marginal variance. The Vecchia-based approximations return the variance of the
    observable process, which additionally contains the nugget effect. Predicting for such a cluster
    must not change the model, i.e. the predictions for the observed clusters stay the same. The
    predictive variances of 'full_scale_tapering' are not deterministic (repeating the same prediction
    for the same model gives slightly different variances), hence the flag 'deterministic'.
    """
    coords, y = _sim_coords(n=100)
    cluster_ids = np.repeat([0, 1], 50)
    rng = np.random.default_rng(7)
    coords_pred = rng.uniform(size=(30, 2))
    cluster_ids_pred = np.tile([0, 1, 2], 10)      # cluster 2 has not been observed
    is_new = cluster_ids_pred == 2
    cov_pars = np.array([0.1, 1.3, 0.2])           # error variance, marginal variance, range

    gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential", likelihood="gaussian",
                           gp_approx=gp_approx, cluster_ids=cluster_ids, **kwargs)
    pred = gp_model.predict(y=y, cov_pars=cov_pars, gp_coords_pred=coords_pred,
                            cluster_ids_pred=cluster_ids_pred, predict_var=True,
                            predict_response=False)
    expected_var = cov_pars[1] + cov_pars[0] if contains_nugget else cov_pars[1]
    np.testing.assert_allclose(pred["mu"][is_new], 0.0, atol=1e-10)
    np.testing.assert_allclose(pred["var"][is_new], expected_var, atol=1e-3)

    pred_obs = gp_model.predict(y=y, cov_pars=cov_pars, gp_coords_pred=coords_pred[~is_new],
                                cluster_ids_pred=cluster_ids_pred[~is_new], predict_var=True,
                                predict_response=False)
    np.testing.assert_allclose(pred["mu"][~is_new], pred_obs["mu"], atol=1e-10)
    if deterministic:
        np.testing.assert_allclose(pred["var"][~is_new], pred_obs["var"], atol=1e-10)

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


def _sim_several_fe_predictors(n=200, n_groups=20, seed=11):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), n // n_groups)
    X = np.column_stack((np.ones(n), rng.uniform(size=n)))
    b = rng.normal(scale=np.sqrt(0.5), size=n_groups)
    mean = X @ np.array([0.3, 0.7]) + b[group]
    # second fixed effects predictor: the log-variance / log-shape
    log_scale = X @ np.array([-0.5, 1.2])
    y = {"gaussian_heteroscedastic": mean + rng.normal(size=n) * np.exp(0.5 * log_scale),
         "gamma_varying_shape": rng.gamma(shape=np.exp(log_scale), scale=np.exp(mean - log_scale))}
    return group, X, y


@pytest.mark.parametrize("likelihood,num_sets_fe", [("gamma_varying_shape", 2),
                                                    ("gaussian_heteroscedastic", 2)])
def test_save_and_load_several_fixed_effects_predictors(tmp_path, likelihood, num_sets_fe):
    # A model is loaded by passing the saved coefficients as 'init_coef' to a pseudo call to
    # 'fit' (with maxit = 0), which is why both are tested here together
    group, X, ys = _sim_several_fe_predictors()
    y = ys[likelihood]
    num_coef = X.shape[1] * num_sets_fe
    gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
    gp_model.fit(y=y, X=X, params={"maxit": 20, "optimizer_cov": "lbfgs", "optimizer_coef": "lbfgs",
                                   "init_coef_aux_pars_from_iid_model": False, "trace": False})
    coef = np.asarray(gp_model.get_coef(format_pandas=False), dtype=float).ravel()
    assert coef.shape == (num_coef,)
    assert np.all(np.isfinite(coef))
    group_test = np.array([0, 1, 999])  # the last group is not in the training data
    X_test = np.column_stack((np.ones(3), np.array([0.1, 0.4, 0.8])))
    pred = gp_model.predict(group_data_pred=group_test, X_pred=X_test,
                            predict_var=True, predict_response=True)

    fname = str(tmp_path / "gp_model.json")
    gp_model.save_model(fname)
    loaded = gpb.GPModel(model_file=fname)
    coef_loaded = np.asarray(loaded.get_coef(format_pandas=False), dtype=float).ravel()
    np.testing.assert_allclose(coef_loaded, coef, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(loaded.get_cov_pars(format_pandas=False), dtype=float).ravel(),
                               np.asarray(gp_model.get_cov_pars(format_pandas=False), dtype=float).ravel(),
                               rtol=1e-10)
    assert loaded.get_current_neg_log_likelihood() == pytest.approx(
        gp_model.get_current_neg_log_likelihood(), rel=1e-10)
    pred_loaded = loaded.predict(group_data_pred=group_test, X_pred=X_test,
                                 predict_var=True, predict_response=True)
    np.testing.assert_allclose(pred_loaded["mu"], pred["mu"], rtol=1e-10)
    np.testing.assert_allclose(pred_loaded["var"], pred["var"], rtol=1e-10)

    from_dict = gpb.GPModel(model_dict=gp_model.model_to_dict(include_response_data=True))
    np.testing.assert_allclose(np.asarray(from_dict.get_coef(format_pandas=False), dtype=float).ravel(),
                               coef, rtol=1e-10)


@pytest.mark.parametrize("likelihood,num_sets_fe", [("gamma_varying_shape", 2),
                                                    ("gaussian_heteroscedastic", 2)])
def test_init_coef_covers_all_fixed_effects_predictors(likelihood, num_sets_fe):
    # 'init_coef' holds the coefficients of all fixed effects predictors, so with maxit = 0 the
    # fitted coefficients are exactly the provided initial values
    group, X, ys = _sim_several_fe_predictors()
    init_coef = np.tile([0.1, -0.2], num_sets_fe)
    gp_model = gpb.GPModel(group_data=group, likelihood=likelihood)
    gp_model.fit(y=ys[likelihood], X=X,
                 params={"maxit": 0, "init_coef": init_coef,
                         "init_coef_aux_pars_from_iid_model": False, "trace": False})
    np.testing.assert_allclose(np.asarray(gp_model.get_coef(format_pandas=False), dtype=float).ravel(),
                               init_coef, rtol=1e-10)


def test_init_coef_before_covariate_data_is_known():
    # The number of covariates is derived from the length of 'init_coef' and the number of
    # fixed effects predictors
    group, X, ys = _sim_several_fe_predictors()
    init_coef = np.array([0.1, -0.2, 0.3, -0.4])
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian_heteroscedastic")
    gp_model.set_optim_params(params={"init_coef": init_coef, "trace": False})
    assert gp_model.num_covariates == 2
    gp_model.fit(y=ys["gaussian_heteroscedastic"], X=X, params={"maxit": 0})
    np.testing.assert_allclose(np.asarray(gp_model.get_coef(format_pandas=False), dtype=float).ravel(),
                               init_coef, rtol=1e-10)
    with pytest.raises(ValueError, match="init_coef"):
        gpb.GPModel(group_data=group, likelihood="gaussian_heteroscedastic").set_optim_params(
            params={"init_coef": np.array([0.1, -0.2, 0.3]), "trace": False})


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


def _sim_crossed(n=2000, m=100, seed=11):
    rng = np.random.default_rng(seed)
    g1 = rng.integers(0, m, n)
    g2 = rng.integers(0, m, n)
    y = (rng.normal(scale=0.7, size=m)[g1] + rng.normal(scale=0.7, size=m)[g2]
         + rng.normal(scale=0.5, size=n))
    return np.column_stack([g1, g2]), y


def _fit_iterative(group, y, **cg_params):
    params = {"cg_preconditioner_type": "ssor", "num_rand_vec_trace": 100,
              "seed_rand_vec_trace": 1, "trace": False}
    params.update(cg_params)
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian",
                           matrix_inversion_method="iterative")
    gp_model.fit(y=y, params=params)
    return gp_model


def test_cg_default_stopping_rule_is_unchanged():
    # if none of the new options are given, the behaviour must be the historic one, so passing
    # the documented defaults explicitly must give identical results
    group, y = _sim_crossed()
    default = _fit_iterative(group, y)
    explicit = _fit_iterative(group, y, cg_convergence_criterion="absolute",
                              cg_multi_rhs_convergence="average")
    np.testing.assert_allclose(np.asarray(default.get_cov_pars(), dtype=float).ravel(),
                               np.asarray(explicit.get_cov_pars(), dtype=float).ravel(), rtol=1e-12)
    assert default.get_current_neg_log_likelihood() == pytest.approx(
        explicit.get_current_neg_log_likelihood(), rel=1e-12)


@pytest.mark.parametrize("multi_rhs", ["average", "max", "per_rhs"])
def test_cg_relative_stopping_rule(multi_rhs):
    # ||r||_2 <= max(cg_abs_tol, cg_rel_tol * ||b||_2) with a tight tolerance must reproduce
    # the fit of a tight absolute rule, whichever way it is aggregated over the right-hand sides
    group, y = _sim_crossed()
    reference = _fit_iterative(group, y, cg_delta_conv=1e-6)
    relative = _fit_iterative(group, y, cg_convergence_criterion="relative", cg_rel_tol=1e-8,
                              cg_abs_tol=1e-8, cg_multi_rhs_convergence=multi_rhs)
    np.testing.assert_allclose(np.asarray(relative.get_cov_pars(), dtype=float).ravel(),
                               np.asarray(reference.get_cov_pars(), dtype=float).ravel(),
                               rtol=0.1, atol=0.05)
    assert np.isfinite(relative.get_current_neg_log_likelihood())


def test_cg_relative_rule_handles_a_tiny_absolute_floor():
    # a right-hand side that is zero or very small must not produce NaN
    group, y = _sim_crossed()
    gp_model = _fit_iterative(group, y, cg_convergence_criterion="relative", cg_rel_tol=1e-6,
                              cg_abs_tol=1e-30)
    assert np.all(np.isfinite(np.asarray(gp_model.get_cov_pars(), dtype=float)))
    assert np.isfinite(gp_model.get_current_neg_log_likelihood())


def test_cg_prediction_stopping_rule():
    group, y = _sim_crossed()
    gp_model = _fit_iterative(group, y, cg_convergence_criterion="relative", cg_rel_tol=1e-8,
                              cg_abs_tol=1e-8)
    group_pred = np.column_stack([[0, 1, 999], [1, 0, 999]])
    gp_model.set_prediction_data(cg_convergence_criterion_pred="relative",
                                 cg_rel_tol_pred=1e-10, cg_abs_tol_pred=1e-10)
    relative = gp_model.predict(group_data_pred=group_pred, predict_var=True)
    gp_model.set_prediction_data(cg_convergence_criterion_pred="absolute", cg_delta_conv_pred=1e-8)
    absolute = gp_model.predict(group_data_pred=group_pred, predict_var=True)
    np.testing.assert_allclose(relative["mu"], absolute["mu"], rtol=0, atol=1e-3)
    np.testing.assert_allclose(relative["var"], absolute["var"], rtol=0, atol=1e-2)


def test_invalid_cg_stopping_rule_options_raise():
    group, y = _sim_crossed(n=200, m=20)
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian",
                           matrix_inversion_method="iterative")
    for bad in ({"cg_convergence_criterion": "reltive"},
                {"cg_multi_rhs_convergence": "maximum"},
                {"cg_rel_tol": -1.},
                {"cg_abs_tol": 0.}):
        with pytest.raises(Exception):
            gpb.GPModel(group_data=group, likelihood="gaussian",
                        matrix_inversion_method="iterative").fit(y=y, params=dict(bad, trace=False))
    with pytest.raises(ValueError):
        gp_model.set_prediction_data(cg_convergence_criterion_pred="reltive")

def test_cg_tolerance_above_every_probe_norm_still_fits():
    # the stochastic Lanczos quadrature averages over all probe vectors, so a probe must not be
    # dropped from that average just because the zero vector satisfies the solver tolerance
    group, y = _sim_crossed()
    gp_model = _fit_iterative(group, y, cg_convergence_criterion="relative", cg_rel_tol=1e-8,
                              cg_abs_tol=1e6)
    assert np.all(np.isfinite(np.asarray(gp_model.get_cov_pars(), dtype=float)))
    assert np.isfinite(gp_model.get_current_neg_log_likelihood())


def test_cg_prediction_options_are_inherited_independently():
    # setting one prediction option explicitly must not stop the others from following their
    # parameter estimation counterpart
    group, y = _sim_crossed()
    gp_model = gpb.GPModel(group_data=group, likelihood="gaussian",
                           matrix_inversion_method="iterative")
    gp_model.set_prediction_data(cg_rel_tol_pred=1e-6)
    gp_model.fit(y=y, params={"cg_preconditioner_type": "ssor", "num_rand_vec_trace": 100,
                              "seed_rand_vec_trace": 1, "trace": False,
                              "cg_convergence_criterion": "relative", "cg_rel_tol": 1e-3})
    group_pred = np.column_stack([[0, 1, 999], [1, 0, 999]])
    pred = gp_model.predict(group_data_pred=group_pred, predict_var=True)
    assert np.all(np.isfinite(pred["mu"])) and np.all(pred["var"] > 0)


def test_non_finite_cg_tolerances_raise():
    group, y = _sim_crossed(n=200, m=20)
    for bad in ({"cg_rel_tol": float("inf")}, {"cg_abs_tol": float("inf")},
                {"cg_delta_conv": float("inf")}):
        with pytest.raises(Exception):
            gpb.GPModel(group_data=group, likelihood="gaussian",
                        matrix_inversion_method="iterative").fit(y=y, params=dict(bad, trace=False))

def test_cg_relative_rule_has_its_own_default_tolerances():
    # cg_rel_tol and cg_abs_tol default to 1e-6 and 1e-8 and are independent of cg_delta_conv, so
    # changing cg_delta_conv must leave a "relative" fit untouched
    group, y = _sim_crossed()
    default_rel = _fit_iterative(group, y, cg_convergence_criterion="relative")
    other_delta = _fit_iterative(group, y, cg_convergence_criterion="relative", cg_delta_conv=1e-1)
    np.testing.assert_allclose(np.asarray(default_rel.get_cov_pars(), dtype=float).ravel(),
                               np.asarray(other_delta.get_cov_pars(), dtype=float).ravel(), rtol=1e-12)
    # passing the documented defaults explicitly changes nothing
    explicit = _fit_iterative(group, y, cg_convergence_criterion="relative", cg_rel_tol=1e-6,
                              cg_abs_tol=1e-8)
    np.testing.assert_allclose(np.asarray(default_rel.get_cov_pars(), dtype=float).ravel(),
                               np.asarray(explicit.get_cov_pars(), dtype=float).ravel(), rtol=1e-12)

# no estimation routine may read any of the settings that configure predictions
_PRED_OPTS = [{"cg_rel_tol_pred": 1e-12},
              {"cg_abs_tol_pred": 1e-14},
              {"cg_convergence_criterion_pred": "absolute"},
              {"cg_delta_conv_pred": 1e-10}]


@pytest.mark.parametrize("pred_opts", _PRED_OPTS)
@pytest.mark.parametrize("likelihood", ["gaussian", "poisson"])
def test_cg_prediction_settings_do_not_change_the_fit(pred_opts, likelihood):
    # the estimation routines must not read the stopping rule configured for predictions. The
    # poisson case is the one that reaches the grouped-RE Laplace gradient, which was reading it
    group, y = _sim_crossed()
    if likelihood == "poisson":
        y = np.random.default_rng(3).poisson(np.exp(y / 2)).astype(float)
    fit_params = {"cg_preconditioner_type": "ssor", "num_rand_vec_trace": 100,
                  "seed_rand_vec_trace": 1, "trace": False,
                  "cg_convergence_criterion": "relative", "cg_rel_tol": 1e-6}
    if likelihood == "poisson":
        fit_params.update({"optimizer_cov": "gradient_descent", "lr_cov": 0.1, "maxit": 5})

    def fit(opts=None):
        gp_model = gpb.GPModel(group_data=group, likelihood=likelihood,
                               matrix_inversion_method="iterative")
        if opts is not None:
            gp_model.set_prediction_data(**opts)
        gp_model.fit(y=y, params=fit_params)
        return np.asarray(gp_model.get_cov_pars(), dtype=float).ravel()

    reference = fit()
    assert np.all(np.isfinite(reference))
    np.testing.assert_allclose(fit(pred_opts), reference, rtol=1e-12)
