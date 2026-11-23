context("Tweedie likelihood")

# Non-convex / stochastic optimization: a different compiler or standard library (e.g. clang + libc++
# on Linux, used by the sanitizer containers of R-hub and CRAN) can converge to a different stationary
# point with practically the same likelihood.
# The tight tolerances below therefore only hold on the reference platform on which the expected
# values were calculated. 'relax_tolerance*()' of helper-tolerances.R relaxes them elsewhere and
# reports once per test run which of the two is in force.

if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

sim_rand_unif <- function(n, init_c=0.1){
  mod_lcg <- 2^32
  sim <- rep(NA, n)
  sim[1] <- floor(init_c * mod_lcg)
  for(i in 2:n) sim[i] <- (22695477 * sim[i-1] + 1) %% mod_lcg
  sim / mod_lcg
}
sim_tweedie <- function(mu, phi, p, init_count, init_gamma){
  lambda <- mu^(2-p) / (phi * (2-p))
  counts <- qpois(sim_rand_unif(length(mu), init_count), lambda=lambda)
  ans <- numeric(length(mu))
  ind <- counts > 0
  ans[ind] <- qgamma(sim_rand_unif(sum(ind), init_gamma), shape=counts[ind] * (2-p) / (p-1), scale=phi * (p-1) * mu[ind]^(p-1))
  ans
}

test_that("Tweedie likelihood covers grouped, crossed, Vecchia, and combined models", {
  n <- 120
  x <- 2 * sim_rand_unif(n, 0.17) - 1
  coords <- cbind(sim_rand_unif(n, 0.31), sim_rand_unif(n, 0.67))
  group1 <- rep(seq_len(20), each=6)
  group2 <- rep(seq_len(12), times=10)
  b1 <- 0.45 * qnorm(sim_rand_unif(20, 0.73))
  b2 <- 0.30 * qnorm(sim_rand_unif(12, 0.29))
  gp_cov <- 0.25 * exp(-as.matrix(dist(coords)) / 0.25) + diag(1e-10, n)
  gp_effect <- drop(t(chol(gp_cov)) %*% qnorm(sim_rand_unif(n, 0.83)))
  phi <- 0.7
  p <- 1.55
  params_chol <- list(maxit=30, delta_rel_conv=1e-5, init_coef_aux_pars_from_iid_model=FALSE)
  params_iter <- list(maxit=20, delta_rel_conv=1e-4, cg_preconditioner_type="ssor", num_rand_vec_trace=50, cg_max_num_it=500,
                      cg_max_num_it_tridiag=500, cg_delta_conv=1e-7, init_coef_aux_pars_from_iid_model=FALSE)
  params_vecchia_iter <- params_iter
  params_vecchia_iter$cg_preconditioner_type <- "vadu"
  params_vecchia_iter$num_rand_vec_trace <- 200
  tolerance_cholesky <- 1e-4
  tolerance_iterative <- 1e-3
  # The covariance-parameter likelihood of the Vecchia GP models is very flat (a marginal-variance /
  # range ridge), so the fitted covariance parameters and derived predictions differ noticeably across
  # compilers (e.g. gcc vs MSVC) even though the log-likelihood agrees. Use a loose tolerance for those
  # parameter and prediction checks while keeping the negative log-likelihood checks tight.
  tolerance_vecchia <- 0.1

  eta_group <- 0.25 + 0.55 * x + b1[group1]
  y_group <- sim_tweedie(exp(eta_group), phi, p, 0.41, 0.91)
  out <- capture.output(fit_group <- fitGPModel(group_data=group1, y=y_group, X=cbind(1, x), likelihood="tweedie", params=params_chol))
  # An estimated power inside the supported range gives no warning
  expect_false(any(grepl("Tweedie power", out)))
  expect_equal(unname(fit_group$get_aux_pars()), c(0.6787341, 1.5850101), tolerance=tolerance_cholesky)
  expect_equal(unname(fit_group$get_coef()), c(0.2322353, 0.5857995), tolerance=tolerance_cholesky)
  expect_equal(unname(fit_group$get_cov_pars()), 0.1604697, tolerance=tolerance_cholesky)
  expect_equal(fit_group$get_current_neg_log_likelihood(), 162.8851995, tolerance=tolerance_cholesky)
  evaluated_nll <- fit_group$neg_log_likelihood(unname(fit_group$get_cov_pars()), y_group, fixed_effects=drop(cbind(1, x) %*% fit_group$get_coef()),
                                                aux_pars=unname(fit_group$get_aux_pars()))
  expect_equal(evaluated_nll, 162.8851995, tolerance=tolerance_cholesky)
  pred_group <- predict(fit_group, group_data_pred=group1[1:4], X_pred=cbind(1, x[1:4]), predict_response=TRUE, predict_var=TRUE)
  expect_equal(unname(pred_group$mu), c(1.1225127, 1.0200591, 1.0724432, 0.9222752), tolerance=tolerance_cholesky)
  expect_equal(unname(pred_group$var), c(0.9167285, 0.7850720, 0.8513994, 0.6669177), tolerance=tolerance_cholesky)

  eta_crossed <- 0.15 + b1[group1] + b2[group2]
  y_crossed <- sim_tweedie(exp(eta_crossed), phi, p, 0.47, 0.87)
  expected_crossed <- list(
    cholesky=list(aux=c(0.7761666, 1.55), cov=c(0.3481216, 0.0704993), nll=155.8381249,
                  mu=c(1.4236499, 1.8128322, 1.4891734), var=c(1.6474327, 2.4168528, 1.7691571)),
    iterative=list(aux=c(0.7782680, 1.55), cov=c(0.3455210, 0.0659401), nll=155.9376561,
                   mu=c(1.4239169, 1.7984077, 1.4871377), var=c(1.6486800, 2.3873606, 1.7661078)))
  for (method in c("cholesky", "iterative")) {
    fit_crossed <- fitGPModel(group_data=cbind(group1, group2), y=y_crossed, likelihood="tweedie_fixed_p", likelihood_additional_param=p,
                              matrix_inversion_method=method, params=if (method == "cholesky") params_chol else params_iter)
    aux_crossed <- unname(fit_crossed$get_aux_pars())
    expected <- expected_crossed[[method]]
    tolerance <- if (method == "cholesky") tolerance_cholesky else tolerance_iterative
    expect_equal(aux_crossed, expected$aux, tolerance=tolerance)
    expect_identical(aux_crossed[2], p)
    expect_equal(unname(fit_crossed$get_cov_pars()), expected$cov, tolerance=tolerance)
    expect_equal(fit_crossed$get_current_neg_log_likelihood(), expected$nll, tolerance=tolerance)
    expect_equal(fit_crossed$neg_log_likelihood(unname(fit_crossed$get_cov_pars()), y_crossed, aux_pars=aux_crossed), expected$nll, tolerance=tolerance)
    pred_crossed <- predict(fit_crossed, group_data_pred=cbind(group1[1:3], group2[1:3]), predict_response=TRUE, predict_var=TRUE)
    expect_equal(unname(pred_crossed$mu), expected$mu, tolerance=tolerance)
    expect_equal(unname(pred_crossed$var), expected$var, tolerance=tolerance)
  }

  eta_gp <- 0.2 + 0.4 * x + gp_effect
  y_gp <- sim_tweedie(exp(eta_gp), phi, p, 0.53, 0.79)
  expected_vecchia <- list(
    cholesky=list(aux=c(0.7142933, 1.55), coef=c(0.3208155, 0.3062940), cov=c(0.0484851, 0.1169696), nll=170.5114178,
                  mu=c(1.0549200, 1.1673261, 1.0618704), var=c(0.8290707, 0.9698431, 0.8370135)),
    iterative=list(aux=c(0.7104205, 1.55), coef=c(0.3180712, 0.2937038), cov=c(0.05309573, 0.1086405), nll=170.5147,
                   mu=c(1.057045, 1.192346, 1.070532), var=c(0.8331259, 1.002747, 0.8476099)))
  for (method in c("cholesky", "iterative")) {
    fit_vecchia <- fitGPModel(gp_coords=coords, gp_approx="vecchia", num_neighbors=15, matrix_inversion_method=method, y=y_gp, X=cbind(1, x),
                              likelihood="tweedie_fixed_p", likelihood_additional_param=p, params=if (method == "cholesky") params_chol else params_vecchia_iter)
    aux_vecchia <- unname(fit_vecchia$get_aux_pars())
    expected <- expected_vecchia[[method]]
    tolerance <- if (method == "cholesky") tolerance_cholesky else tolerance_iterative
    expect_equal(aux_vecchia, expected$aux, tolerance=tolerance_vecchia)
    expect_identical(aux_vecchia[2], p)
    expect_equal(unname(fit_vecchia$get_coef()), expected$coef, tolerance=tolerance_vecchia)
    expect_equal(unname(fit_vecchia$get_cov_pars()), expected$cov, tolerance=tolerance_vecchia)
    expect_equal(fit_vecchia$get_current_neg_log_likelihood(), expected$nll, tolerance=relax_tolerance(tolerance))
    evaluated_nll <- fit_vecchia$neg_log_likelihood(unname(fit_vecchia$get_cov_pars()), y_gp, fixed_effects=drop(cbind(1, x) %*% fit_vecchia$get_coef()), aux_pars=aux_vecchia)
    expect_equal(evaluated_nll, expected$nll, tolerance=relax_tolerance(tolerance))
    pred_vecchia <- predict(fit_vecchia, gp_coords_pred=coords[1:3, ], X_pred=cbind(1, x[1:3]), predict_response=TRUE, predict_var=TRUE)
    expect_equal(unname(pred_vecchia$mu), expected$mu, tolerance=tolerance_vecchia)
    expect_equal(unname(pred_vecchia$var), expected$var, tolerance=tolerance_vecchia)
  }

  eta_combined <- 0.1 + 0.35 * x + b1[group1] + gp_effect
  y_combined <- sim_tweedie(exp(eta_combined), phi, p, 0.59, 0.71)
  fit_combined <- fitGPModel(group_data=group1, gp_coords=coords, gp_approx="none", y=y_combined, X=cbind(1, x), likelihood="tweedie_fixed_p",
                             likelihood_additional_param=p, params=params_chol)
  aux_combined <- unname(fit_combined$get_aux_pars())
  expect_equal(aux_combined, c(0.7052811, 1.55), tolerance=tolerance_cholesky)
  expect_identical(aux_combined[2], p)
  expect_equal(unname(fit_combined$get_coef()), c(-0.0668874, 0.5604593), tolerance=tolerance_cholesky)
  expect_equal(unname(fit_combined$get_cov_pars()), c(0.2181935, 0.1529311, 0.2724234), tolerance=tolerance_cholesky)
  expect_equal(fit_combined$get_current_neg_log_likelihood(), 155.0430917, tolerance=tolerance_cholesky)
  evaluated_nll <- fit_combined$neg_log_likelihood(unname(fit_combined$get_cov_pars()), y_combined,
                                                   fixed_effects=drop(cbind(1, x) %*% fit_combined$get_coef()), aux_pars=aux_combined)
  expect_equal(evaluated_nll, 155.0430917, tolerance=tolerance_cholesky)
  pred_combined <- predict(fit_combined, group_data_pred=group1[1:3], gp_coords_pred=coords[1:3, ], X_pred=cbind(1, x[1:3]), predict_response=TRUE, predict_var=TRUE)
  expect_equal(unname(pred_combined$mu), c(1.2644083, 1.1316402, 0.9778109), tolerance=tolerance_cholesky)
  expect_equal(unname(pred_combined$var), c(1.2368039, 1.0198414, 0.8369473), tolerance=tolerance_cholesky)

  # Vecchia GP with multiple observations at the same locations (the use_random_effects_indices_of_data_ path), Cholesky and iterative
  nu_rep <- 40L
  coords_rep_u <- cbind(sim_rand_unif(nu_rep, 0.19), sim_rand_unif(nu_rep, 0.53))
  rep_idx <- rep(seq_len(nu_rep), length.out=n)
  coords_rep <- coords_rep_u[rep_idx, ]
  gp_cov_rep <- 0.25 * exp(-as.matrix(dist(coords_rep_u)) / 0.25) + diag(1e-10, nu_rep)
  gp_eff_rep <- drop(t(chol(gp_cov_rep)) %*% qnorm(sim_rand_unif(nu_rep, 0.61)))[rep_idx]
  eta_rep <- 0.2 + 0.4 * x + gp_eff_rep
  y_rep <- sim_tweedie(exp(eta_rep), phi, p, 0.37, 0.83)
  expected_rep <- list(
    cholesky=list(aux=c(0.6281294, 1.55), coef=c(0.1889439, 0.5301197), cov=c(0.1435919, 0.055215), nll=160.5661,
                  mu=c(1.2088353, 1.3688655, 0.7609807), var=c(0.9736622, 1.1826878, 0.4815947)),
    iterative=list(aux=c(0.6300628, 1.55), coef=c(0.1883982, 0.538425), cov=c(0.1510919, 0.05802348), nll=160.5544,
                   mu=c(1.213593, 1.380825, 0.7578053), var=c(0.9832903, 1.21219, 0.4863258)))
  for (method in c("cholesky", "iterative")) {
    params <- if (method == "cholesky") params_chol else params_vecchia_iter
    fit_rep <- fitGPModel(gp_coords=coords_rep, gp_approx="vecchia", num_neighbors=15, matrix_inversion_method=method, y=y_rep, X=cbind(1, x),
                          likelihood="tweedie_fixed_p", likelihood_additional_param=p, params=params)
    reference <- expected_rep[[method]]
    tolerance <- if (method == "cholesky") tolerance_cholesky else tolerance_iterative
    expect_equal(unname(fit_rep$get_aux_pars()), reference$aux, tolerance=tolerance_vecchia)
    expect_identical(unname(fit_rep$get_aux_pars())[2], p)
    expect_equal(unname(fit_rep$get_coef()), reference$coef, tolerance=tolerance_vecchia)
    expect_equal(unname(fit_rep$get_cov_pars()), reference$cov, tolerance=tolerance_vecchia)
    expect_equal(fit_rep$get_current_neg_log_likelihood(), reference$nll, tolerance=tolerance)
    evaluated_nll <- fit_rep$neg_log_likelihood(unname(fit_rep$get_cov_pars()), y_rep, fixed_effects=drop(cbind(1, x) %*% fit_rep$get_coef()), aux_pars=unname(fit_rep$get_aux_pars()))
    expect_equal(evaluated_nll, reference$nll, tolerance=tolerance)
    pred_rep <- predict(fit_rep, gp_coords_pred=coords_rep_u[1:3, ], X_pred=cbind(1, x[1:3]), predict_response=TRUE, predict_var=TRUE)
    expect_equal(unname(pred_rep$mu), reference$mu, tolerance=tolerance_vecchia)
    expect_equal(unname(pred_rep$var), reference$var, tolerance=tolerance_vecchia)
  }
})

test_that("Tweedie response validation and fixed-power interface are explicit", {
  expect_error(GPModel(num_data=3, likelihood="tweedie_fixed_p"), "No value was provided for 'likelihood_additional_param'", fixed=TRUE)
  expect_error(GPModel(num_data=3, likelihood="tweedie_fixed_p", likelihood_additional_param=1), "only the compound")
  expect_error(fitGPModel(group_data=1:3, y=c(0, 0, 0), likelihood="tweedie_fixed_p", likelihood_additional_param=1.5), "only zeros")
  expect_error(fitGPModel(group_data=1:3, y=c(0, -1, 2), likelihood="tweedie_fixed_p", likelihood_additional_param=1.5), "finite and nonnegative")
  # The supported range of the power is (1.001, 1.999)
  expect_error(GPModel(num_data=3, likelihood="tweedie_fixed_p", likelihood_additional_param=1.0005), "too close to 1")
  expect_error(GPModel(num_data=3, likelihood="tweedie_fixed_p", likelihood_additional_param=1.9995), "too close to 2")
})

test_that("Tweedie log-likelihood for a power close to one or two", {
  # The expected values are the negative sums of the log-densities in 50-digit arithmetic for the double values of p,
  # y = (0, 0.5, 1, 2), mu = (1, 0.5, 1, 2), and phi = 1. The tiny random effect variance makes the Laplace
  # approximation equal to the likelihood
  y <- c(0, 0.5, 1, 2)
  mu <- c(1, 0.5, 1, 2)
  gp_model <- GPModel(group_data=seq_along(y), likelihood="tweedie")
  nll_at <- function(p) gp_model$neg_log_likelihood(cov_pars=1e-20, y=y, fixed_effects=log(mu), aux_pars=c(1, p))
  expect_lt(abs(nll_at(1.005) - 36.571417724918119), 1e-7)
  expect_lt(abs(nll_at(1.995) - 203.00117959943266), 1e-7)
})

test_that("Tweedie likelihood of a concentrated density and maximum likelihood estimate of the dispersion", {
  # For a small dispersion and a power close to one, the normalizer and the location-dependent part of the canonical form of
  # the log-density are both of the order of y^(2-p) / ((p-1) phi) = 1e12 and cancel, and so do their derivatives wrt
  # log(phi). The expected values are calculated in 30-digit arithmetic for the double values of y. The tiny random effect
  # variance makes the Laplace approximation equal to the likelihood
  n <- 100
  p <- 1.00101
  y <- 1 + sqrt(1e-9) * qnorm(sim_rand_unif(n, 0.43))
  gp_model <- GPModel(group_data=seq_len(n), likelihood="tweedie_fixed_p", likelihood_additional_param=p)
  nll <- gp_model$neg_log_likelihood(cov_pars=1e-20, y=y, fixed_effects=rep(0, n), aux_pars=c(1e-9, p))
  expect_lt(abs(nll + 895.181202767865), 1e-6)
  # The estimate of the dispersion solves the score equation
  capture.output(fit <- fitGPModel(group_data=seq_len(n), y=y, likelihood="tweedie_fixed_p", likelihood_additional_param=p,
                                   params=list(init_cov_pars=1e-20, estimate_cov_par_index=0, init_aux_pars=c(2e-9, p),
                                               delta_rel_conv=1e-12, maxit=1000)), file="NUL")
  expect_lt(abs(fit$get_aux_pars()[1] / 9.8176542262e-10 - 1), 1e-4)
})

test_that("an estimated Tweedie power at a bound of the supported range is valid for a new model", {
  # Where the likelihood is flat in the transformed power theta = log((p - 1.001) / (1.999 - p)), an optimizer can move theta
  # so far that p rounds to a bound or that exp(theta) underflows or overflows. Nelder-Mead does so here, started at the
  # representable power closest to a bound: at the lower bound for Poisson-like data and at the upper bound for data without
  # zeros. p rounds to the bound for the first two data sets, and exp(theta) underflows or overflows for the last two
  group <- rep(1:4, length.out=20)
  counts <- rep(0:4, length.out=20)
  z <- qnorm(seq(0.05, 0.95, length.out=20))
  p_lower <- 1.001 + 2^-52
  p_upper <- 1.999 - 2^-52
  cases <- list(list(y=counts, p_init=p_lower), list(y=exp(z), p_init=p_upper),
                list(y=0.5 * counts, p_init=p_lower), list(y=exp(0.5 * z), p_init=p_upper))
  for (case in cases) {
    capture.output(fit <- fitGPModel(group_data=group, y=case$y, likelihood="tweedie",
                                     params=list(optimizer_cov="nelder_mead", init_aux_pars=c(1, case$p_init),
                                                 init_coef_aux_pars_from_iid_model=FALSE)), file="NUL")
    p_hat <- fit$get_aux_pars()[2]
    expect_gt(p_hat, 1.001)
    expect_lt(p_hat, 1.999)
    filename <- tempfile(fileext=".json")
    saveGPModel(fit, filename)
    capture.output(fit_loaded <- loadGPModel(filename), file="NUL")
    unlink(filename)
    expect_equal(fit_loaded$get_aux_pars(), fit$get_aux_pars())
  }
})

test_that("Tweedie power close to a bound of the supported range: warnings and estimation", {
  group <- rep(1:50, each=10)
  x <- 2 * sim_rand_unif(500, 0.23) - 1
  b <- 0.5 * qnorm(sim_rand_unif(50, 0.37))
  offset <- 0.3 + 0.5 * x
  eta <- offset + b[group]
  # For Poisson data, the estimated power is at the lower bound. A fit without covariates warns and suggests 'poisson'
  y_pois <- qpois(sim_rand_unif(500, 0.59), lambda=exp(eta))
  out <- capture.output(fit <- fitGPModel(group_data=group, y=y_pois, offset=offset, likelihood="tweedie"))
  expect_lt(fit$get_aux_pars()[2], 1.002)
  expect_true(any(grepl("close to the lower bound", out)))
  expect_true(any(grepl("likelihood = 'poisson'", out, fixed=TRUE)))
  # For data with a power close to 2, a fit with covariates warns and suggests 'gamma' and 'hurdle_gamma', or, for a
  # varying dispersion, 'gamma_varying_shape' and 'hurdle_regression_gamma_varying_shape'
  y_near_gamma <- sim_tweedie(exp(eta), 1, 1.999, 0.13, 0.17)
  out <- capture.output(fit <- fitGPModel(group_data=group, y=y_near_gamma, X=cbind(1, x), likelihood="tweedie"))
  expect_gt(fit$get_aux_pars()[2], 1.998)
  expect_true(any(grepl("close to the upper bound", out)))
  expect_true(any(grepl("likelihood = 'gamma' (or 'hurdle_gamma' if y contains zeros)", out, fixed=TRUE)))
  out <- capture.output(fit <- fitGPModel(group_data=group, y=y_near_gamma, X=cbind(1, x), likelihood="tweedie_varying_dispersion"))
  expect_true(any(grepl("likelihood = 'gamma_varying_shape' (or 'hurdle_regression_gamma_varying_shape' if y contains zeros)", out, fixed=TRUE)))
  expect_true(any(grepl("likelihood = 'tweedie_varying_dispersion_fixed_p'", out, fixed=TRUE)))
  # For these data, the power of the model without random effects, which gives the initial values, is close to the upper
  # bound, while the maximum of the likelihood is not. The estimate equals the one of a tightly converged fit that starts
  # in the middle of the range
  y <- sim_tweedie(exp(eta), 1, 1.95, 0.53, 0.57)
  fit <- fitGPModel(group_data=group, y=y, X=cbind(1, x), likelihood="tweedie")
  fit_ref <- fitGPModel(group_data=group, y=y, X=cbind(1, x), likelihood="tweedie",
                        params=list(init_aux_pars=c(1, 1.5), delta_rel_conv=1e-12, maxit=5000))
  expect_lt(fit_ref$get_aux_pars()[2], 1.95)
  expect_lt(abs(fit$get_aux_pars()[2] - fit_ref$get_aux_pars()[2]), 0.01)
  expect_lt(fit$get_current_neg_log_likelihood() - fit_ref$get_current_neg_log_likelihood(), 0.01)
})

}

