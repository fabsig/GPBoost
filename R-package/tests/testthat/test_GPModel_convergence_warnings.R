if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

  context("GPModel_convergence_warnings")

  # The severity convention of the convergence reporting of a 'GPModel':
  #   Debug   internal optimizer iterations, the behavior of an individual line search, a line search that
  #           reached 'max_linesearch', and an unsuccessful last line search when the restarts have
  #           established that nothing more can be gained (convergence status 0)
  #   Warning the final non-convergence of the optimizer: the maximal number of iterations was reached
  #           (convergence status 1) or the line search of an lbfgs optimizer was unsuccessful and it
  #           could not be established that nothing more can be gained (convergence status 2)
  #
  # The messages are written to R's output stream by Rprintf() and not by warning(), so they are
  # captured with capture.output() and not with expect_warning()

  WARNING_MAX_ITER <- "did not converge after the maximum number of iterations"
  WARNING_LINE_SEARCH <- "has terminated since its line search has not been successful"

  has_warning <- function(out, pattern) {
    any(grepl("[Warning]", out, fixed = TRUE) & grepl(pattern, out, fixed = TRUE))
  }

  convergence_status <- function(gp_model) {
    gp_model$.__enclos_env__$private$get_convergence_status()
  }

  # Whatever the optimizer did, the reported severity has to correspond to the convergence status
  expect_reporting_matches_status <- function(out, gp_model) {
    status <- convergence_status(gp_model)
    expect_equal(has_warning(out, WARNING_MAX_ITER), status == 1L)
    expect_equal(has_warning(out, WARNING_LINE_SEARCH), status == 2L)
  }

  # Function that simulates uniform random variables
  sim_rand_unif <- function(n, init_c=0.1){
    mod_lcg <- 134456 # modulus for linear congruential generator (random0 used)
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for(i in 2:n) sim[i] <- (8121 * sim[i-1] + 28411) %% mod_lcg
    return(sim / mod_lcg)
  }

  # Create data: a grouped random effects model
  n <- 500
  m <- 50
  group <- rep(1,n)
  for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
  b <- qnorm(sim_rand_unif(n=m, init_c=0.546))
  xi <- sqrt(0.5) * qnorm(sim_rand_unif(n=n, init_c=0.1))
  X <- cbind(rep(1,n), sin((1:n-n/2)^2*2*pi/n))
  beta <- c(2,2)
  y <- b[group] + xi + as.vector(X %*% beta)

  test_that("reaching the maximal number of iterations gives a warning", {
    out <- capture.output({
      gp_model <- fitGPModel(group_data = group, y = y, X = X, likelihood = "gaussian",
                             params = list(optimizer_cov = "gradient_descent", lr_cov = 1E-6,
                                           use_nesterov_acc = FALSE, maxit = 2))
    })
    expect_equal(convergence_status(gp_model), 1L)
    expect_true(has_warning(out, WARNING_MAX_ITER))
    expect_false(has_warning(out, WARNING_LINE_SEARCH))
    # The estimates are still returned
    expect_equal(length(gp_model$get_coef()), 2L)
  })

  test_that("a converged estimation gives no warning", {
    out <- capture.output({
      gp_model <- fitGPModel(group_data = group, y = y, X = X, likelihood = "gaussian",
                             params = list(optimizer_cov = "gradient_descent", lr_cov = 0.1,
                                           use_nesterov_acc = TRUE, maxit = 1000))
    })
    expect_equal(convergence_status(gp_model), 0L)
    expect_false(has_warning(out, WARNING_MAX_ITER))
    expect_false(has_warning(out, WARNING_LINE_SEARCH))
  })

  test_that("lbfgs reports a warning only for an unresolved line search failure", {
    # A line search that reaches 'max_linesearch' but accepts a usable point, and an unsuccessful last
    # line search after which the restarts no longer improved the objective function, both leave the
    # convergence status at 0 and must not produce a warning. Which of the cases occurs depends on the
    # build, so the invariant and not a fixed outcome is tested here.
    # A 'delta_rel_conv' that is practically zero makes the optimizer run until its line search fails
    # instead of until the relative change is small, which is what reaches the convergence status 2
    for (optimizer in c("lbfgs", "lbfgs_linesearch_nocedal_wright")) {
      for (max_num_restarts in c(0, 2)) {
        for (delta_rel_conv in c(1E-6, 1E-30)) {
          out <- capture.output({
            gp_model <- fitGPModel(group_data = group, y = y, X = X, likelihood = "gaussian",
                                   params = list(optimizer_cov = optimizer,
                                                 max_num_restarts_lbfgs = max_num_restarts,
                                                 delta_rel_conv = delta_rel_conv, maxit = 10000))
          })
          expect_reporting_matches_status(out, gp_model)
        }
      }
    }
  })

  test_that("an unresolved line search failure warns, restarts that establish convergence do not", {
    # On the reference platform, 'lbfgs_linesearch_nocedal_wright' without restarts terminates here with
    # an unsuccessful line search that cannot be resolved (convergence status 2), while the same fit with
    # restarts ends with restarts that no longer improved the objective function (convergence status 0).
    # Both are only checked when they actually occur, since this depends on the build
    fit_with_restarts <- function(max_num_restarts) {
      out <- capture.output({
        gp_model <- fitGPModel(group_data = group, y = y, X = X, likelihood = "gaussian",
                               params = list(optimizer_cov = "lbfgs_linesearch_nocedal_wright",
                                             max_num_restarts_lbfgs = max_num_restarts,
                                             delta_rel_conv = 1E-30, maxit = 10000))
      })
      list(out = out, gp_model = gp_model, status = convergence_status(gp_model))
    }
    no_restarts <- fit_with_restarts(0)
    expect_reporting_matches_status(no_restarts$out, no_restarts$gp_model)
    if (no_restarts$status == 2L) {
      expect_true(has_warning(no_restarts$out, WARNING_LINE_SEARCH))
      # The advice on the restarts is part of the warning
      expect_true(any(grepl("max_num_restarts_lbfgs", no_restarts$out, fixed = TRUE)))
    }
    with_restarts <- fit_with_restarts(2)
    expect_reporting_matches_status(with_restarts$out, with_restarts$gp_model)
    if (with_restarts$status == 0L) {
      # An unsuccessful line search after restarts that no longer improved the objective function is
      # not a non-convergence and must stay at the Debug level
      expect_false(has_warning(with_restarts$out, WARNING_LINE_SEARCH))
      expect_false(has_warning(with_restarts$out, WARNING_MAX_ITER))
    }
  })

  test_that("a non-Gaussian likelihood reports according to the convergence status", {
    probs <- 1 / (1 + exp(-(b[group] + as.vector(X %*% beta) - 2)))
    y_bin <- as.numeric(sim_rand_unif(n=n, init_c=0.978) < probs)
    # Too few iterations to converge
    out <- capture.output({
      gp_model <- fitGPModel(group_data = group, y = y_bin, X = X, likelihood = "bernoulli_probit",
                             params = list(maxit = 2))
    })
    expect_equal(convergence_status(gp_model), 1L)
    expect_true(has_warning(out, WARNING_MAX_ITER))
    # Enough iterations
    out <- capture.output({
      gp_model <- fitGPModel(group_data = group, y = y_bin, X = X, likelihood = "bernoulli_probit")
    })
    expect_reporting_matches_status(out, gp_model)
  })

  test_that("convergence in the last allowed iteration is not a non-convergence", {
    # The number of iterations does not distinguish an optimizer that satisfied its convergence criterion
    # in the last allowed iteration from one that exhausted its iteration budget: both report 'maxit'
    # iterations. The number of iterations of a converged run is measured first and then used as 'maxit',
    # so that the criterion is satisfied in exactly the last allowed iteration
    # 'nelder_mead' and 'adam' come from OptimLib, which reports whether its convergence criterion was
    # satisfied through its return value. 'adam' is experimental but exercises the same path
    for (optimizer in c("gradient_descent", "fisher_scoring", "nelder_mead", "adam",
                        "lbfgs", "lbfgs_linesearch_nocedal_wright")) {
      out <- capture.output({
        gp_full <- fitGPModel(group_data = group, y = y, X = X, likelihood = "gaussian",
                              params = list(optimizer_cov = optimizer, maxit = 1000))
      })
      expect_equal(convergence_status(gp_full), 0L)
      num_it <- gp_full$get_num_optim_iter()
      expect_gt(num_it, 1)
      # Exactly enough iterations
      out <- capture.output({
        gp_exact <- fitGPModel(group_data = group, y = y, X = X, likelihood = "gaussian",
                               params = list(optimizer_cov = optimizer, maxit = num_it))
      })
      expect_equal(gp_exact$get_num_optim_iter(), num_it)
      expect_equal(convergence_status(gp_exact), 0L)
      expect_false(has_warning(out, WARNING_MAX_ITER))
      # One iteration too few: the budget is genuinely exhausted
      out <- capture.output({
        gp_short <- fitGPModel(group_data = group, y = y, X = X, likelihood = "gaussian",
                               params = list(optimizer_cov = optimizer, maxit = num_it - 1))
      })
      expect_equal(convergence_status(gp_short), 1L)
      expect_true(has_warning(out, WARNING_MAX_ITER))
    }
  })

  test_that("the parameter criterion in the last allowed iteration is not a non-convergence", {
    # The optimizers of OptimLib report only one of their two convergence criteria through their return
    # value ('rel_objfn_change' for 'nelder_mead', the norm of the gradient for 'adam'). With
    # 'relative_change_in_parameters' the criterion that GPBoost has selected is the other one
    for (optimizer in c("nelder_mead", "adam")) {
      params <- list(optimizer_cov = optimizer, convergence_criterion = "relative_change_in_parameters")
      out <- capture.output({
        gp_full <- fitGPModel(group_data = group, y = y, X = X, likelihood = "gaussian",
                              params = c(params, list(maxit = 1000)))
      })
      expect_equal(convergence_status(gp_full), 0L)
      num_it <- gp_full$get_num_optim_iter()
      expect_gt(num_it, 1)
      out <- capture.output({
        gp_exact <- fitGPModel(group_data = group, y = y, X = X, likelihood = "gaussian",
                               params = c(params, list(maxit = num_it)))
      })
      expect_equal(convergence_status(gp_exact), 0L)
      expect_false(has_warning(out, WARNING_MAX_ITER))
      out <- capture.output({
        gp_short <- fitGPModel(group_data = group, y = y, X = X, likelihood = "gaussian",
                               params = c(params, list(maxit = num_it - 1)))
      })
      expect_equal(convergence_status(gp_short), 1L)
      expect_true(has_warning(out, WARNING_MAX_ITER))
    }
  })

  test_that("a forced Vecchia neighbour redetermination keeps the reporting consistent", {
    # With an ARD covariance function the nearest neighbours are redetermined in the transformed space, and
    # the redetermination is forced in the iteration in which the optimizer stops, so this exercises that
    # path. Note what this does not do: it does not check that the criterion is re-evaluated after the
    # redetermination, since whether the criterion still holds afterwards depends on the data. Only the
    # invariant between the status and the reported severity is tested, which an implementation that keeps
    # a stale criterion would also satisfy
    coords <- cbind(sim_rand_unif(n = n, init_c = 0.23), sim_rand_unif(n = n, init_c = 0.77))
    y_gp <- b[group] + xi
    fit_ard <- function(maxit) {
      fitGPModel(gp_coords = coords, cov_function = "matern_ard", gp_approx = "vecchia",
                 num_neighbors = 10L, vecchia_ordering = "none", y = y_gp,
                 params = list(optimizer_cov = "lbfgs", maxit = maxit))
    }
    out <- capture.output({ gp_full <- fit_ard(1000) })
    expect_reporting_matches_status(out, gp_full)
    num_it <- gp_full$get_num_optim_iter()
    expect_gt(num_it, 1)
    for (maxit in c(num_it, num_it - 1)) {
      out <- capture.output({ gp <- fit_ard(maxit) })
      expect_reporting_matches_status(out, gp)
      expect_true(all(is.finite(gp$get_cov_pars())))
    }
  })

  test_that("an estimation that starts at the optimum does not warn with maxit = 1", {
    # lbfgs reports one iteration both when the initial point is already a minimizer and when a single
    # iteration was all that was allowed
    for (optimizer in c("lbfgs", "lbfgs_linesearch_nocedal_wright")) {
      out <- capture.output({
        gp_full <- fitGPModel(group_data = group, y = y, likelihood = "gaussian",
                              params = list(optimizer_cov = optimizer, maxit = 1000))
      })
      expect_equal(convergence_status(gp_full), 0L)
      out <- capture.output({
        gp_one <- fitGPModel(group_data = group, y = y, likelihood = "gaussian",
                             params = list(optimizer_cov = optimizer, maxit = 1,
                                           init_cov_pars = as.numeric(gp_full$get_cov_pars())))
      })
      expect_equal(convergence_status(gp_one), 0L)
      expect_false(has_warning(out, WARNING_MAX_ITER))
    }
  })

  test_that("'adam' returns defined results when it starts at the optimum", {
    # 'adam' returns before the number of iterations and the objective function value are written when the
    # gradient criterion already holds at the starting point, so these have to be set for that case as well
    out <- capture.output({
      gp_full <- fitGPModel(group_data = group, y = y, likelihood = "gaussian",
                            params = list(optimizer_cov = "adam", maxit = 1000))
    })
    expect_equal(convergence_status(gp_full), 0L)
    nll_full <- gp_full$get_current_neg_log_likelihood()
    out <- capture.output({
      gp_opt <- fitGPModel(group_data = group, y = y, likelihood = "gaussian",
                           params = list(optimizer_cov = "adam", maxit = 1000,
                                         init_cov_pars = as.numeric(gp_full$get_cov_pars())))
    })
    expect_equal(convergence_status(gp_opt), 0L)
    expect_false(has_warning(out, WARNING_MAX_ITER))
    num_it <- gp_opt$get_num_optim_iter()
    expect_true(is.finite(num_it))
    expect_gte(num_it, 0)
    expect_lte(num_it, 1000)
    expect_true(is.finite(gp_opt$get_current_neg_log_likelihood()))
    expect_equal(gp_opt$get_current_neg_log_likelihood(), nll_full, tolerance = 1E-3)
  })

  test_that("the GPBoost algorithm does not warn about the internal parameter estimations", {
    # The covariance parameters are re-estimated in every boosting iteration, often without converging.
    # These internal estimations stay at the Debug level, also with a very small 'maxit'
    for (maxit in c(2, 1000)) {
      gp_model <- GPModel(group_data = group, likelihood = "gaussian")
      gp_model$set_optim_params(params = list(maxit = maxit))
      out <- capture.output({
        bst <- gpboost(data = X[, 2, drop = FALSE], label = y, gp_model = gp_model,
                       nrounds = 5, learning_rate = 0.1, max_depth = 2,
                       objective = "regression_l2", verbose = 0)
      })
      expect_false(has_warning(out, WARNING_MAX_ITER))
      expect_false(has_warning(out, WARNING_LINE_SEARCH))
    }
  })

}
