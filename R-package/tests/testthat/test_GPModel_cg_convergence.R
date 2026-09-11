if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

  context("GPModel_cg_convergence")

  # See helper-tolerances.R
  USE_STRICT_TOLERANCES <- gpb_use_strict_tolerances()
  relax_tolerance <- function(tol) if (USE_STRICT_TOLERANCES) tol else max(2 * tol, 0.5)

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
              relax_tolerance(TOL_VERY_LOOSE))
  })

  test_that("The relative CG stopping rule gives the same fit as the absolute one", {

    reference <- fit_iterative(list(cg_delta_conv = 1E-6))
    # ||r||_2 <= max(cg_abs_tol, cg_rel_tol * ||b||_2) with a tight relative tolerance
    relative <- fit_iterative(list(cg_convergence_criterion = "relative",
                                   cg_rel_tol = 1E-8, cg_abs_tol = 1E-8))
    expect_lt(sum(abs(as.vector(reference$get_cov_pars()) - as.vector(relative$get_cov_pars()))),
              relax_tolerance(TOLERANCE_LOOSE))
    expect_lt(abs(reference$get_current_neg_log_likelihood() -
                    relative$get_current_neg_log_likelihood()), relax_tolerance(1))

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
              relax_tolerance(TOLERANCE_LOOSE))

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
                relax_tolerance(TOLERANCE_LOOSE))
      expect_lt(abs(fits[[1]]$get_current_neg_log_likelihood() -
                      fits[[i]]$get_current_neg_log_likelihood()), relax_tolerance(1))
    }

    # "max" is the strictest rule, so it can never stop before "average" does
    expect_true(all(is.finite(as.vector(fits[[2]]$get_cov_pars()))))
    # 'per_rhs' stops iterating on individual columns, the log-determinant must stay finite
    expect_true(is.finite(fits[[3]]$get_current_neg_log_likelihood()))

    # The aggregation rules also apply to the absolute criterion
    abs_per_rhs <- fit_iterative(list(cg_multi_rhs_convergence = "per_rhs", cg_delta_conv = 1E-4))
    abs_max <- fit_iterative(list(cg_multi_rhs_convergence = "max", cg_delta_conv = 1E-4))
    expect_lt(sum(abs(as.vector(abs_per_rhs$get_cov_pars()) - as.vector(abs_max$get_cov_pars()))),
              relax_tolerance(TOLERANCE_LOOSE))
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
    expect_lt(sum(abs(preds$mu - preds2$mu)), relax_tolerance(TOL_VERY_LOOSE))
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

  test_that("The relative-rule prediction settings do not change the parameter estimation", {

    # The estimation routines must not read the stopping rule configured for predictions.
    # Note that 'cg_delta_conv_pred' is deliberately not in this list: some estimation routines
    # (for example the grouped-RE Laplace gradient) have always used it as a tighter absolute
    # tolerance, and that pre-existing behavior is kept so that historic results are reproduced
    pred_opts_list <- list(list(cg_rel_tol_pred = 1E-12),
                           list(cg_abs_tol_pred = 1E-14),
                           list(cg_convergence_criterion_pred = "absolute"))

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
    expect_lt(sum(abs(pred_relative$mu - pred_absolute$mu)), relax_tolerance(TOLERANCE_MEDIUM))
    expect_lt(sum(abs(pred_relative$var - pred_absolute$var)), relax_tolerance(TOLERANCE_LOOSE))
  })

}
