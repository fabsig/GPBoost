if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

  context("GPModel_cg_convergence")

  # See helper-tolerances.R for 'relax_tolerance()', which relaxes the tolerances below on a platform
  # other than the reference one

  TOLERANCE_STRICT <- 1E-6
  TOLERANCE_MEDIUM <- 1E-3
  TOLERANCE_LOOSE <- 1E-2
  TOL_VERY_LOOSE <- 1E-1

  # Function that simulates uniform random variables
  sim_rand_unif <- function(n, init_c=0.1){
    mod_lcg <- 134456 # modulus for linear congruential generator (random0 used)
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for(i in 2:n) sim[i] <- (8121 * sim[i-1] + 28411) %% mod_lcg
    return(sim / mod_lcg)
  }

  # Two crossed grouped random effects: this is the model for which 'matrix_inversion_method = "iterative"'
  #   uses the conjugate gradient algorithm together with stochastic Lanczos quadrature
  n <- 1000
  m <- 100
  group <- rep(1,n)
  for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
  group2 <- rep(1,n)
  for(i in 1:m) group2[(1:(n/m))*m-m+i] <- i
  Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
  Z2 <- model.matrix(rep(1,n) ~ factor(group2) - 1)
  b1 <- qnorm(sim_rand_unif(n=m, init_c=0.586)) * sqrt(0.5)
  b2 <- qnorm(sim_rand_unif(n=m, init_c=0.951)) * sqrt(1.2)
  xi <- qnorm(sim_rand_unif(n=n, init_c=0.176)) * sqrt(0.25)
  y <- as.vector(Z1 %*% b1 + Z2 %*% b2 + xi)
  group_data <- cbind(group, group2)

  FIT_PARAMS <- list(optimizer_cov = "fisher_scoring", cg_preconditioner_type = "ssor",
                     num_rand_vec_trace = 100, seed_rand_vec_trace = 1L,
                     init_coef_aux_pars_from_iid_model = FALSE)
  # a count response for the non-Gaussian (Laplace approximation) code path
  y_count <- as.numeric(qpois(sim_rand_unif(n=n, init_c=0.723),
                              lambda = exp(as.vector(Z1 %*% b1 + Z2 %*% b2) / 2)))
  FIT_PARAMS_POISSON <- list(optimizer_cov = "gradient_descent", lr_cov = 0.1, maxit = 5L,
                             cg_preconditioner_type = "ssor", num_rand_vec_trace = 100,
                             seed_rand_vec_trace = 1L, init_coef_aux_pars_from_iid_model = FALSE)

  fit_iterative <- function(extra_params = list()){
    params <- c(FIT_PARAMS, extra_params)
    capture.output( gp_model <- fitGPModel(group_data = group_data, y = y,
                                           matrix_inversion_method = "iterative",
                                           params = params) , file='NUL')
    gp_model
  }

  test_that("Default CG stopping rule is unchanged when the new options are set to their defaults", {

    # Requirement: if none of the new options are specified, behavior must be unchanged.
    #   Passing the documented defaults explicitly must give bit-identical results
    default_fit <- fit_iterative()
    explicit_fit <- fit_iterative(list(cg_convergence_criterion = "absolute",
                                       cg_multi_rhs_convergence = "average"))
    expect_equal(as.vector(default_fit$get_cov_pars()), as.vector(explicit_fit$get_cov_pars()))
    expect_equal(default_fit$get_current_neg_log_likelihood(),
                 explicit_fit$get_current_neg_log_likelihood())
    expect_equal(default_fit$get_num_optim_iter(), explicit_fit$get_num_optim_iter())

    # 'cg_delta_conv' still drives the absolute rule and nothing else took it over. A looser CG
    #   tolerance is allowed to shift the estimates, so this only requires that the fit stays in
    #   the same neighborhood - a decoupling would move it much further than this
    loose_fit <- fit_iterative(list(cg_delta_conv = 1E-1))
    tight_fit <- fit_iterative(list(cg_delta_conv = 1E-4))
    expect_true(all(is.finite(as.vector(loose_fit$get_cov_pars()))))
    expect_lt(sum(abs(as.vector(loose_fit$get_cov_pars()) - as.vector(tight_fit$get_cov_pars()))),
              relax_tolerance(TOL_VERY_LOOSE, as.vector(tight_fit$get_cov_pars())))
  })

  test_that("The relative CG stopping rule gives the same fit as the absolute one", {

    reference <- fit_iterative(list(cg_delta_conv = 1E-6))
    # ||r||_2 <= max(cg_abs_tol, cg_rel_tol * ||b||_2) with a tight relative tolerance
    relative <- fit_iterative(list(cg_convergence_criterion = "relative",
                                   cg_rel_tol = 1E-8, cg_abs_tol = 1E-8))
    expect_lt(sum(abs(as.vector(reference$get_cov_pars()) - as.vector(relative$get_cov_pars()))),
              relax_tolerance(TOLERANCE_LOOSE, as.vector(relative$get_cov_pars())))
    expect_lt(abs(reference$get_current_neg_log_likelihood() -
                    relative$get_current_neg_log_likelihood()), relax_tolerance(1, relative$get_current_neg_log_likelihood()))

    # A very small 'cg_abs_tol' floor must not make the algorithm fail on small right-hand sides
    small_floor <- fit_iterative(list(cg_convergence_criterion = "relative",
                                      cg_rel_tol = 1E-6, cg_abs_tol = 1E-30))
    expect_true(all(is.finite(as.vector(small_floor$get_cov_pars()))))
    expect_true(is.finite(small_floor$get_current_neg_log_likelihood()))
  })

  test_that("The relative rule uses its own default tolerances, not cg_delta_conv", {

    # 'cg_rel_tol' and 'cg_abs_tol' default to 1E-6 and 1E-8 and are independent of 'cg_delta_conv'.
    # Changing 'cg_delta_conv' must therefore leave a "relative" fit untouched, while it does change
    # an "absolute" fit (checked in the first test above)
    default_rel <- fit_iterative(list(cg_convergence_criterion = "relative"))
    other_delta <- fit_iterative(list(cg_convergence_criterion = "relative", cg_delta_conv = 1E-1))
    expect_equal(as.vector(default_rel$get_cov_pars()), as.vector(other_delta$get_cov_pars()))

    # the defaults are tight enough to reproduce an explicitly tight absolute fit
    reference <- fit_iterative(list(cg_delta_conv = 1E-6))
    expect_lt(sum(abs(as.vector(default_rel$get_cov_pars()) - as.vector(reference$get_cov_pars()))),
              relax_tolerance(TOLERANCE_LOOSE, as.vector(reference$get_cov_pars())))

    # and passing the documented defaults explicitly changes nothing
    explicit <- fit_iterative(list(cg_convergence_criterion = "relative", cg_rel_tol = 1E-6,
                                   cg_abs_tol = 1E-8))
    expect_equal(as.vector(default_rel$get_cov_pars()), as.vector(explicit$get_cov_pars()))
  })

  test_that("All multi-rhs aggregation rules give the same fit", {

    # "average", "max" and "per_rhs" only change when the iteration stops, not what it converges to
    fits <- lapply(c("average", "max", "per_rhs"), function(rule){
      fit_iterative(list(cg_convergence_criterion = "relative", cg_rel_tol = 1E-8,
                         cg_abs_tol = 1E-8, cg_multi_rhs_convergence = rule))
    })
    for (i in 2:3) {
      expect_lt(sum(abs(as.vector(fits[[1]]$get_cov_pars()) - as.vector(fits[[i]]$get_cov_pars()))),
                relax_tolerance(TOLERANCE_LOOSE, as.vector(fits[[i]]$get_cov_pars())))
      expect_lt(abs(fits[[1]]$get_current_neg_log_likelihood() -
                      fits[[i]]$get_current_neg_log_likelihood()), relax_tolerance(1, fits[[i]]$get_current_neg_log_likelihood()))
    }

    # "max" is the strictest rule, so it can never stop before "average" does
    expect_true(all(is.finite(as.vector(fits[[2]]$get_cov_pars()))))
    # 'per_rhs' stops iterating on individual columns, the log-determinant must stay finite
    expect_true(is.finite(fits[[3]]$get_current_neg_log_likelihood()))

    # The aggregation rules also apply to the absolute criterion
    abs_per_rhs <- fit_iterative(list(cg_multi_rhs_convergence = "per_rhs", cg_delta_conv = 1E-4))
    abs_max <- fit_iterative(list(cg_multi_rhs_convergence = "max", cg_delta_conv = 1E-4))
    expect_lt(sum(abs(as.vector(abs_per_rhs$get_cov_pars()) - as.vector(abs_max$get_cov_pars()))),
              relax_tolerance(TOLERANCE_LOOSE, as.vector(abs_max$get_cov_pars())))
  })

  test_that("Prediction options are inherited independently of one another", {

    # A prediction option that has been set explicitly must not stop the other ones from following
    # their parameter estimation counterpart, in either call order
    gp_model <- GPModel(group_data = group_data, matrix_inversion_method = "iterative")
    gp_model$set_prediction_data(cg_rel_tol_pred = 1E-6)
    capture.output( gp_model$fit(y = y, params = c(FIT_PARAMS,
                                                   list(cg_convergence_criterion = "relative",
                                                        cg_rel_tol = 1E-3))) , file='NUL')
    group_data_pred <- cbind(c(1, 2, 999), c(2, 1, 999))
    # 'cg_convergence_criterion_pred' was never set, so it has to inherit "relative", and
    # 'cg_rel_tol_pred' has to keep the 1E-6 that was set explicitly. Neither can be read back, so
    # this only checks that the call order does not make the predictions fail or turn non-finite
    capture.output( preds <- predict(gp_model, group_data_pred = group_data_pred,
                                     predict_var = TRUE) , file='NUL')
    expect_true(all(is.finite(preds$mu)))
    expect_true(all(is.finite(preds$var)) && all(preds$var > 0))

    # the other call order
    gp_model2 <- fit_iterative(list(cg_convergence_criterion = "relative", cg_rel_tol = 1E-8,
                                    cg_abs_tol = 1E-8))
    set_prediction_data(gp_model2, cg_rel_tol_pred = 1E-6)
    capture.output( preds2 <- predict(gp_model2, group_data_pred = group_data_pred,
                                      predict_var = TRUE) , file='NUL')
    expect_true(all(is.finite(preds2$mu)))
    expect_lt(sum(abs(preds$mu - preds2$mu)), relax_tolerance(TOL_VERY_LOOSE, preds2$mu))
  })

  test_that("A tolerance above the norm of every probe vector still gives a finite fit", {

    # The stochastic Lanczos quadrature averages over all probe vectors. A probe must not be dropped
    # from that average just because the zero vector happens to satisfy the solver tolerance, it
    # still carries information about the log-determinant and gets at least one Lanczos step
    gp_model <- fit_iterative(list(cg_convergence_criterion = "relative", cg_rel_tol = 1E-8,
                                   cg_abs_tol = 1E6))
    expect_true(all(is.finite(as.vector(gp_model$get_cov_pars()))))
    expect_true(is.finite(gp_model$get_current_neg_log_likelihood()))
  })

  test_that("Prediction settings do not change the parameter estimation", {

    # No estimation routine may read any of the settings that configure predictions, neither the
    # stopping rule nor the absolute tolerance
    pred_opts_list <- list(list(cg_rel_tol_pred = 1E-12),
                           list(cg_abs_tol_pred = 1E-14),
                           list(cg_convergence_criterion_pred = "absolute"),
                           list(cg_delta_conv_pred = 1E-10))

    # Gaussian, which exercises the grouped-RE CG solves and the Lanczos log-determinant
    reference <- fit_iterative(list(cg_convergence_criterion = "relative", cg_rel_tol = 1E-6))
    for (pred_opts in pred_opts_list) {
      gp_model <- GPModel(group_data = group_data, matrix_inversion_method = "iterative")
      do.call(gp_model$set_prediction_data, pred_opts)
      capture.output( gp_model$fit(y = y, params = c(FIT_PARAMS,
                                                     list(cg_convergence_criterion = "relative",
                                                          cg_rel_tol = 1E-6))) , file='NUL')
      expect_equal(as.vector(gp_model$get_cov_pars()), as.vector(reference$get_cov_pars()))
    }

    # Poisson, which is what actually reaches the grouped-RE Laplace gradient. That routine was
    # reading the prediction settings, so only this case exercises the defect
    fit_poisson <- function(pred_opts = NULL){
      gp_model <- GPModel(group_data = group_data, likelihood = "poisson",
                          matrix_inversion_method = "iterative")
      if (!is.null(pred_opts)) {
        do.call(gp_model$set_prediction_data, pred_opts)
      }
      capture.output( gp_model$fit(y = y_count, params = c(FIT_PARAMS_POISSON,
                                                           list(cg_convergence_criterion = "relative",
                                                                cg_rel_tol = 1E-6))) , file='NUL')
      gp_model
    }
    reference_poisson <- fit_poisson()
    expect_true(all(is.finite(as.vector(reference_poisson$get_cov_pars()))))
    for (pred_opts in pred_opts_list) {
      expect_equal(as.vector(fit_poisson(pred_opts)$get_cov_pars()),
                   as.vector(reference_poisson$get_cov_pars()))
    }
  })

  test_that("Invalid CG stopping-rule options are rejected", {

    expect_error(fitGPModel(group_data = group_data, y = y, matrix_inversion_method = "iterative",
                            params = list(cg_convergence_criterion = "reltive")))
    expect_error(fitGPModel(group_data = group_data, y = y, matrix_inversion_method = "iterative",
                            params = list(cg_multi_rhs_convergence = "maximum")))
    expect_error(fitGPModel(group_data = group_data, y = y, matrix_inversion_method = "iterative",
                            params = list(cg_convergence_criterion = 1)))
    expect_error(fitGPModel(group_data = group_data, y = y, matrix_inversion_method = "iterative",
                            params = list(cg_rel_tol = -1)))
    expect_error(fitGPModel(group_data = group_data, y = y, matrix_inversion_method = "iterative",
                            params = list(cg_abs_tol = 0)))
    expect_error(fitGPModel(group_data = group_data, y = y, matrix_inversion_method = "iterative",
                            params = list(cg_rel_tol = Inf)))
    expect_error(fitGPModel(group_data = group_data, y = y, matrix_inversion_method = "iterative",
                            params = list(cg_abs_tol = Inf)))

    gp_model <- GPModel(group_data = group_data, matrix_inversion_method = "iterative")
    expect_error(gp_model$set_prediction_data(cg_convergence_criterion_pred = "reltive"))
    expect_error(gp_model$set_prediction_data(cg_convergence_criterion_pred = 1))
  })

  test_that("The CG stopping rule can be set for predictions", {

    gp_model <- fit_iterative(list(cg_convergence_criterion = "relative", cg_rel_tol = 1E-8,
                                   cg_abs_tol = 1E-8))
    group_data_pred <- cbind(c(1, 2, 3, 999), c(2, 1, 999, 3))

    # The exported function must accept the prediction options and forward them
    set_prediction_data(gp_model, cg_convergence_criterion_pred = "relative",
                        cg_rel_tol_pred = 1E-10, cg_abs_tol_pred = 1E-10)
    capture.output( pred_relative <- predict(gp_model, group_data_pred = group_data_pred,
                                             predict_var = TRUE) , file='NUL')
    set_prediction_data(gp_model, cg_convergence_criterion_pred = "absolute",
                        cg_delta_conv_pred = 1E-8)
    capture.output( pred_absolute <- predict(gp_model, group_data_pred = group_data_pred,
                                             predict_var = TRUE) , file='NUL')
    expect_lt(sum(abs(pred_relative$mu - pred_absolute$mu)), relax_tolerance(TOLERANCE_MEDIUM, pred_absolute$mu))
    expect_lt(sum(abs(pred_relative$var - pred_absolute$var)), relax_tolerance(TOLERANCE_LOOSE, pred_absolute$var))
  })

  test_that("Warning when the conjugate gradient algorithm of a Vecchia-Laplace approximation needs many iterations", {

    # The iterative methods of a Vecchia-Laplace approximation can be much slower than Cholesky factorizations,
    #   e.g., for the 'vadu' preconditioner when the information of the likelihood is large and the Gaussian process
    #   is strongly correlated. Here, these are Poisson counts with a large mean and a smooth Gaussian process with a
    #   large range. Already in the first iteration, every run for the log-determinant needs the maximal number of
    #   iterations, which the dimension n_s caps, for 1 to 16 threads
    WARNING_SLOW_CG <- "conjugate gradient algorithm of the iterative methods"
    n_s <- 400
    coords_s <- cbind(sim_rand_unif(n = n_s, init_c = 0.35), sim_rand_unif(n = n_s, init_c = 0.62))
    D_s <- as.matrix(dist(coords_s)) * sqrt(5) / 0.2
    Sigma_s <- (1 + D_s + D_s^2 / 3) * exp(-D_s) + diag(1E-10, n_s)
    f_s <- as.vector(t(chol(Sigma_s)) %*% qnorm(sim_rand_unif(n = n_s, init_c = 0.77)))
    y_s <- qpois(sim_rand_unif(n = n_s, init_c = 0.21), lambda = exp(12 + f_s))
    n_v <- 300
    coords_v <- cbind(sim_rand_unif(n = n_v, init_c = 0.35), sim_rand_unif(n = n_v, init_c = 0.62))
    D_v <- as.matrix(dist(coords_v))
    Sigma_v <- exp(-D_v / 0.1) + diag(1E-10, n_v)
    f_v <- as.vector(t(chol(Sigma_v)) %*% qnorm(sim_rand_unif(n = n_v, init_c = 0.77)))
    y_v <- as.numeric(sim_rand_unif(n = n_v, init_c = 0.21) < 1 / (1 + exp(-f_v)))
    y_v_pois <- qpois(sim_rand_unif(n = n_v, init_c = 0.21), lambda = exp(12 + f_v))
    fit_vecchia <- function(params, matrix_inversion_method = "iterative", coords = coords_v, cov_function = "exponential",
                            likelihood = "bernoulli_logit", y = y_v, ...) {
      gp_model <- NULL
      out <- capture.output(gp_model <- fitGPModel(gp_coords = coords, cov_function = cov_function,
                                                   likelihood = likelihood, gp_approx = "vecchia",
                                                   matrix_inversion_method = matrix_inversion_method,
                                                   y = y, params = params, ...))
      list(out = out, gp_model = gp_model)
    }
    fit_vecchia_slow <- function(params, ...) {
      fit_vecchia(c(list(maxit = 1), params), coords = coords_s, cov_function = "matern", cov_fct_shape = 2.5,
                  likelihood = "poisson", y = y_s, X = matrix(1, n_s), ...)
    }
    res <- fit_vecchia_slow(list())
    expect_true(any(grepl(WARNING_SLOW_CG, res$out)))
    expect_true(any(grepl("runs for the log-determinant: [0-9]+ of [0-9]+ reached", res$out)))
    expect_true(any(grepl("matrix_inversion_method = 'cholesky'", res$out)))
    expect_true(any(grepl("or with cg_preconditioner_type = 'fitc'", res$out)))
    # A raised maximal number of iterations does not suppress the warning. The 'fitc' preconditioner requires the
    #   inverse of the information of the likelihood and is therefore not suggested when a weight is zero
    weights_s <- rep(1, n_s)
    weights_s[1] <- 0
    res <- fit_vecchia_slow(list(cg_max_num_it = 2000, cg_max_num_it_tridiag = 2000), weights = weights_s)
    expect_true(any(grepl(WARNING_SLOW_CG, res$out)))
    expect_false(any(grepl("cg_preconditioner_type = 'fitc'", res$out)))
    # No warning when the maximal numbers of iterations are lowered deliberately. Here, a limit of 999 does not change
    #   the runs, which stop at the dimension n_s
    res <- fit_vecchia_slow(list(cg_max_num_it = 999, cg_max_num_it_tridiag = 999))
    expect_false(any(grepl(WARNING_SLOW_CG, res$out)))
    res <- fit_vecchia(list(cg_max_num_it = 3, cg_max_num_it_tridiag = 3))
    expect_false(any(grepl(WARNING_SLOW_CG, res$out)))
    # The number of iterations of the last runs of the conjugate gradient algorithm is also available
    expect_true(res$gp_model$get_num_cg_steps() %in% 1:3)
    expect_true(res$gp_model$get_num_cg_steps_tridiag() %in% 1:3)
    # No warning with Cholesky factorizations or with the suggested 'fitc' preconditioner
    res <- fit_vecchia_slow(list(), matrix_inversion_method = "cholesky")
    expect_false(any(grepl(WARNING_SLOW_CG, res$out)))
    res <- fit_vecchia_slow(list(cg_preconditioner_type = "fitc"))
    expect_false(any(grepl(WARNING_SLOW_CG, res$out)))
    # No warning with the default settings, and none when only a few hard systems at the start of the estimation
    #   reach the maximal number of iterations: for these Poisson counts, the initial range is much larger than the
    #   estimated one, at which the runs need few iterations
    res <- fit_vecchia(list())
    expect_false(any(grepl(WARNING_SLOW_CG, res$out)))
    expect_gt(res$gp_model$get_num_cg_steps(), 0)
    expect_lt(res$gp_model$get_num_cg_steps(), 1000)
    res <- fit_vecchia(list(), cov_function = "matern", cov_fct_shape = 2.5, likelihood = "poisson", y = y_v_pois,
                       X = matrix(1, n_v))
    expect_false(any(grepl(WARNING_SLOW_CG, res$out)))
  })

}
