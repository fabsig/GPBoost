context("Tweedie likelihood")

test_that("Tweedie likelihood covers grouped, crossed, Vecchia, and combined models", {
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
  tolerance_cholesky <- 1e-4
  tolerance_iterative <- 1e-3

  eta_group <- 0.25 + 0.55 * x + b1[group1]
  y_group <- sim_tweedie(exp(eta_group), phi, p, 0.41, 0.91)
  fit_group <- fitGPModel(group_data=group1, y=y_group, X=cbind(1, x), likelihood="tweedie", params=params_chol)
  expect_equal(unname(fit_group$get_aux_pars()), c(0.6784574, 1.5862270), tolerance=tolerance_cholesky)
  expect_equal(unname(fit_group$get_coef()), c(0.2322927, 0.5900659), tolerance=tolerance_cholesky)
  expect_equal(unname(fit_group$get_cov_pars()), 0.1609974, tolerance=tolerance_cholesky)
  expect_equal(fit_group$get_current_neg_log_likelihood(), 162.8843124, tolerance=tolerance_cholesky)
  evaluated_nll <- fit_group$neg_log_likelihood(unname(fit_group$get_cov_pars()), y_group, fixed_effects=drop(cbind(1, x) %*% fit_group$get_coef()),
                                                aux_pars=unname(fit_group$get_aux_pars()))
  expect_equal(evaluated_nll, 162.8843124, tolerance=tolerance_cholesky)
  pred_group <- predict(fit_group, group_data_pred=group1[1:4], X_pred=cbind(1, x[1:4]), predict_response=TRUE, predict_var=TRUE)
  expect_equal(unname(pred_group$mu), c(1.1208706, 1.0178572, 1.0705185, 0.9196092), tolerance=tolerance_cholesky)
  expect_equal(unname(pred_group$var), c(0.9145121, 0.7822045, 0.8488394, 0.6636168), tolerance=tolerance_cholesky)

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
    iterative=list(aux=c(0.7190565, 1.55), coef=c(0.2896672, 0.2941669), cov=c(0.0780043, 0.1481743), nll=170.6336208,
                   mu=c(1.0302059, 1.1814947, 1.0385672), var=c(0.8187500, 1.0088491, 0.8314185)))
  for (method in c("cholesky", "iterative")) {
    fit_vecchia <- fitGPModel(gp_coords=coords, gp_approx="vecchia", num_neighbors=15, matrix_inversion_method=method, y=y_gp, X=cbind(1, x),
                              likelihood="tweedie_fixed_p", likelihood_additional_param=p, params=if (method == "cholesky") params_chol else params_vecchia_iter)
    aux_vecchia <- unname(fit_vecchia$get_aux_pars())
    expected <- expected_vecchia[[method]]
    tolerance <- if (method == "cholesky") tolerance_cholesky else tolerance_iterative
    expect_equal(aux_vecchia, expected$aux, tolerance=tolerance)
    expect_identical(aux_vecchia[2], p)
    expect_equal(unname(fit_vecchia$get_coef()), expected$coef, tolerance=tolerance)
    expect_equal(unname(fit_vecchia$get_cov_pars()), expected$cov, tolerance=tolerance)
    expect_equal(fit_vecchia$get_current_neg_log_likelihood(), expected$nll, tolerance=tolerance)
    evaluated_nll <- fit_vecchia$neg_log_likelihood(unname(fit_vecchia$get_cov_pars()), y_gp, fixed_effects=drop(cbind(1, x) %*% fit_vecchia$get_coef()), aux_pars=aux_vecchia)
    expect_equal(evaluated_nll, expected$nll, tolerance=tolerance)
    pred_vecchia <- predict(fit_vecchia, gp_coords_pred=coords[1:3, ], X_pred=cbind(1, x[1:3]), predict_response=TRUE, predict_var=TRUE)
    expect_equal(unname(pred_vecchia$mu), expected$mu, tolerance=tolerance)
    expect_equal(unname(pred_vecchia$var), expected$var, tolerance=tolerance*5)
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
    iterative=list(aux=c(0.6237624, 1.55), coef=c(0.1960408, 0.5337103), cov=c(0.1472277, 0.0660624), nll=160.5628,
                   mu=c(1.2209169, 1.3826488, 0.7664825), var=c(0.9818941, 1.2062249, 0.4878628)))
  for (method in c("cholesky", "iterative")) {
    params <- if (method == "cholesky") params_chol else params_vecchia_iter
    fit_rep <- fitGPModel(gp_coords=coords_rep, gp_approx="vecchia", num_neighbors=15, matrix_inversion_method=method, y=y_rep, X=cbind(1, x),
                          likelihood="tweedie_fixed_p", likelihood_additional_param=p, params=params)
    reference <- expected_rep[[method]]
    tolerance <- if (method == "cholesky") tolerance_cholesky else tolerance_iterative
    expect_equal(unname(fit_rep$get_aux_pars()), reference$aux, tolerance=tolerance)
    expect_identical(unname(fit_rep$get_aux_pars())[2], p)
    expect_equal(unname(fit_rep$get_coef()), reference$coef, tolerance=tolerance)
    expect_equal(unname(fit_rep$get_cov_pars()), reference$cov, tolerance=tolerance)
    expect_equal(fit_rep$get_current_neg_log_likelihood(), reference$nll, tolerance=tolerance)
    evaluated_nll <- fit_rep$neg_log_likelihood(unname(fit_rep$get_cov_pars()), y_rep, fixed_effects=drop(cbind(1, x) %*% fit_rep$get_coef()), aux_pars=unname(fit_rep$get_aux_pars()))
    expect_equal(evaluated_nll, reference$nll, tolerance=tolerance)
    pred_rep <- predict(fit_rep, gp_coords_pred=coords_rep_u[1:3, ], X_pred=cbind(1, x[1:3]), predict_response=TRUE, predict_var=TRUE)
    expect_equal(unname(pred_rep$mu), reference$mu, tolerance=tolerance)
    expect_equal(unname(pred_rep$var), reference$var, tolerance=tolerance*5)
  }
})

test_that("Tweedie response validation and fixed-power interface are explicit", {
  expect_error(GPModel(num_data=3, likelihood="tweedie_fixed_p"), "No value was provided for 'likelihood_additional_param'", fixed=TRUE)
  expect_error(GPModel(num_data=3, likelihood="tweedie_fixed_p", likelihood_additional_param=1), "only the compound")
  expect_error(fitGPModel(group_data=1:3, y=c(0, 0, 0), likelihood="tweedie_fixed_p", likelihood_additional_param=1.5), "only zeros")
  expect_error(fitGPModel(group_data=1:3, y=c(0, -1, 2), likelihood="tweedie_fixed_p", likelihood_additional_param=1.5), "finite and nonnegative")
})

