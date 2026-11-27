# coding: utf-8
"""Tests of the BLAS library that is used for dense matrix calculations."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

import gpboost as gpb
from gpboost.basic import GPBoostError


def restore_blas(blas_before):
    """Restore whether OpenBLAS is used as it was before a test."""
    if blas_before == "builtin":
        gpb.set_blas("builtin")
    else:
        gpb.set_blas("auto")


def test_blas_can_be_queried_and_switched_off():
    blas_before = gpb.get_blas()
    try:
        assert isinstance(blas_before, str)
        gpb.set_blas("builtin")
        assert gpb.get_blas() == "builtin"
        gpb.set_blas("auto")
        assert gpb.get_blas() == "builtin" or gpb.get_blas().startswith("OpenBLAS")
        with pytest.raises(GPBoostError):
            gpb.set_blas("openblas")
        with pytest.raises(ValueError):
            gpb.set_blas(1)
    finally:
        restore_blas(blas_before)


def test_openblas_and_builtin_code_agree_for_an_exact_gaussian_process():
    # Large enough that OpenBLAS uses several threads for the Cholesky factorization and the inverse, see
    # include/GPBoost/blas_backend.h
    n = 900
    n_pred = 50
    rng = np.random.default_rng(1)
    coords_all = rng.uniform(size=(n + n_pred, 2))
    cov_pars = np.array([0.3, 1., 0.1])  # error variance, marginal variance, range
    Sigma_all = cov_pars[1] * np.exp(-cdist(coords_all, coords_all) / cov_pars[2])
    Sigma = Sigma_all[:n, :n] + cov_pars[0] * np.eye(n)
    L = np.linalg.cholesky(Sigma)
    y = L @ rng.normal(size=n)
    # Negative log-likelihood and predictive distribution calculated with numpy
    L_inv_y = np.linalg.solve(L, y)
    nll_exact = np.sum(np.log(np.diag(L))) + 0.5 * np.sum(L_inv_y ** 2) + n / 2 * np.log(2 * np.pi)
    L_inv_Sigma_cross_t = np.linalg.solve(L, Sigma_all[:n, n:])
    pred_mean_exact = L_inv_Sigma_cross_t.T @ L_inv_y
    pred_cov_exact = (Sigma_all[n:, n:] + cov_pars[0] * np.eye(n_pred) -
                      L_inv_Sigma_cross_t.T @ L_inv_Sigma_cross_t)
    blas_before = gpb.get_blas()
    results = {}
    try:
        for blas in ["auto", "builtin"]:
            gpb.set_blas(blas)
            gp_model = gpb.GPModel(gp_coords=coords_all[:n], cov_function="exponential")
            nll = gp_model.neg_log_likelihood(cov_pars=cov_pars, y=y)
            assert abs(nll - nll_exact) / abs(nll_exact) < 1e-10
            pred = gp_model.predict(y=y, gp_coords_pred=coords_all[n:], cov_pars=cov_pars,
                                    predict_cov_mat=True, predict_response=True)
            assert np.max(np.abs(pred['mu'] - pred_mean_exact)) < 1e-10
            assert np.max(np.abs(pred['cov'] - pred_cov_exact)) < 1e-10
            # The gradient and the standard errors use the inverse of the covariance matrix
            gp_model.fit(y=y)
            results[blas] = gp_model.get_cov_pars(std_err=True).values
    finally:
        restore_blas(blas_before)
    # The estimates differ at most by the accuracy of the optimization
    assert np.max(np.abs(results["auto"] - results["builtin"])) < 1e-5
