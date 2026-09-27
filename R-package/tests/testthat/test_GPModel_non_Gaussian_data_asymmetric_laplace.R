context("GPModel_non_Gaussian_data")

# Avoid being tested on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

  TOLERANCE_ITERATIVE <- 1e-1
  TOLERANCE_LOOSE <- 1E-2
  TOLERANCE_MEDIUM <- 1e-3
  TOLERANCE_STRICT_LOWER <- 1E-5
  TOLERANCE_STRICT <- 1E-6
  # Some of the optimization problems below are non-convex and/or use stochastic (iterative) methods.
  # A different compiler / standard library (e.g. clang + libc++ on Linux, which is used by the sanitizer
  # containers of R-hub and CRAN) can then converge to a DIFFERENT stationary point with practically the
  # same likelihood value (the negative log-likelihoods agree to ~0.1%, the coefficients differ by ~0.1).
  # The tight tolerances are therefore only required on the reference platform on which the expected
  # values below have been calculated.
  # See helper-tolerances.R, which defines this and reports it once per test run
  USE_STRICT_TOLERANCES <- gpb_use_strict_tolerances()
  TOLERANCE_NON_CONVEX <- if (USE_STRICT_TOLERANCES) TOLERANCE_MEDIUM else 0.5
  # Separate helper for the very strict tolerances (1e-6) of comparisons whose expected values cannot be
  # handed to 'relax_tolerance', so that its lower bound cannot be tied to their magnitude. Deviations of
  # a few 1e-6 occur under valgrind in particular, which does not reproduce floating point arithmetic
  # bit-wise (it rounds the 80 bit intermediate results of x87 to 64 bit and its libm differs)
  relax_tolerance_strict <- function(tol) if (USE_STRICT_TOLERANCES) tol else 100 * tol
  # Covariance functions with a general (non-fixed) smoothness need 'std::cyl_bessel_k', which is a C++17
  # feature that is not provided by every standard library (in particular not by libc++, which is used by
  # clang on macOS and in the clang sanitizer containers of R-hub / CRAN)
  SKIP_BESSEL_COV_TESTS <- !gpboost:::has_std_cyl_bessel_k() &&
    Sys.getenv("GPBOOST_RUN_BESSEL_COV_TESTS") != "true"

  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov = "gradient_descent", optimizer_coef = "gradient_descent",
                               use_nesterov_acc = TRUE, lr_cov=0.1, lr_coef = 0.1, maxit = 1000,
                               acc_rate_cov = 0.5, init_coef_aux_pars_from_iid_model = FALSE)
  OPTIM_PARAMS_BFGS <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 1000,
                            init_coef_aux_pars_from_iid_model = FALSE)

  # Function that simulates uniform random variables
  sim_rand_unif <- function(n, init_c=0.1){
    mod_lcg <- 2^32 # modulus for linear congruential generator (random0 used)
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for(i in 2:n) sim[i] <- (22695477 * sim[i-1] + 1) %% mod_lcg
    return(sim / mod_lcg)
  }

  # Simulate data
  n <- 100 # number of samples
  # Simulate locations / features of GP
  d <- 2 # dimension of GP locations
  coords <- matrix(sim_rand_unif(n=n*d, init_c=0.1), ncol=d)
  D <- as.matrix(dist(coords))
  # Simulate GP
  sigma2_1 <- 1^2 # marginal variance of GP
  rho <- 0.1 # range parameter
  Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
  L <- t(chol(Sigma))
  b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.8))
  # GP random coefficients
  Z_SVC <- matrix(sim_rand_unif(n=n*2, init_c=0.6), ncol=2) # covariate data for random coefficients
  colnames(Z_SVC) <- c("var1","var2")
  b_2 <- qnorm(sim_rand_unif(n=n, init_c=0.17))
  b_3 <- qnorm(sim_rand_unif(n=n, init_c=0.42))
  # First grouped random effects model
  m <- 10 # number of categories / levels for grouping variable
  group <- rep(1,n) # grouping variable
  for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
  Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
  b_gr_1 <- qnorm(sim_rand_unif(n=m, init_c=0.565))
  # Second grouped random effect
  n_obs_gr <- n/m # number of samples per group
  group2 <- rep(1,n) # grouping variable
  for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
  Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
  b_gr_2 <- qnorm(sim_rand_unif(n=n_obs_gr, init_c=0.36))
  # Grouped random slope / coefficient
  x <- cos((1:n-n/2)^2*5.5*pi/n) # covariate data for random slope
  Z3 <- diag(x) %*% Z1
  b_gr_3 <- qnorm(sim_rand_unif(n=m, init_c=0.5678))
  # Data for linear mixed effects model
  X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
  beta <- c(0.1,2) # regression coefficients
  # cluster_ids
  cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))
  # GP with multiple observations at the same locations
  coords_multiple <- matrix(sim_rand_unif(n=n*d/4, init_c=0.1), ncol=d)
  coords_multiple <- rbind(coords_multiple,coords_multiple,coords_multiple,coords_multiple)
  D_multiple <- as.matrix(dist(coords_multiple))
  Sigma_multiple <- sigma2_1*exp(-D_multiple/rho)+diag(1E-10,n)
  L_multiple <- t(chol(Sigma_multiple))
  b_multiple <- qnorm(sim_rand_unif(n=n, init_c=0.8))
  # Space-time GP
  time <- (1:n)/n
  rho_time <- 0.1
  coords_ST_scaled <- cbind(time/rho_time, coords/rho)
  D_ST <- as.matrix(dist(coords_ST_scaled))
  Sigma_ST <- sigma2_1 * exp(-D_ST) + diag(1E-20,n)
  C_ST <- t(chol(Sigma_ST))
  b_ST <- qnorm(sim_rand_unif(n=n, init_c=0.86574))
  eps_ST <- as.vector(C_ST %*% b_ST)
  # For CV
  params_cv <- list(learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
                    feature_pre_filter = FALSE, seed = 1, deterministic = TRUE)
  folds <- list()
  nf <- 2
  for(i in 1:nf) folds[[i]] <- as.integer(((1:(n/nf)) -1) * nf + i)

  test_that("asymmetric_laplace likelihood ", {

    params <- OPTIM_PARAMS_BFGS
    likelihood <- "asymmetric_laplace"

    quantile_asym_laplace <- function(q, alpha, lambda) {
      if (length(q) == 1) {
        if (q <= alpha) {
          return(log(q/alpha) * lambda / (1-alpha))
        } else {
          return(-log((1-q)/(1-alpha)) * lambda / alpha)
        }
      } else {
        res <- rep(NA,length(q))
        ind <- q <= alpha
        res[ind] <- log(q[ind]/alpha) * lambda / (1-alpha)
        res[!ind] <- -log((1-q[!ind])/(1-alpha)) * lambda / alpha
        return(res)
      }
    }

    # Single level grouped random effects
    quantile <- 0.5
    quantile_up <- 0.975
    lambda = 0.25
    error <- quantile_asym_laplace(q=sim_rand_unif(n=n, init_c=0.651), alpha=quantile, lambda = lambda)
    y <- Z1 %*% b_gr_1 + X%*%beta + error

    matrix_inversion_method <- "cholesky"
    # matrix_inversion_method_loop <- c("cholesky", "iterative")
    # for (matrix_inversion_method in matrix_inversion_method_loop) {
    if(matrix_inversion_method == "iterative") {
      tolerance_loc_1 <- TOLERANCE_STRICT
      tolerance_loc_2 <- TOLERANCE_LOOSE
      tolerance_loc_3 <- 0.1
    } else {
      tolerance_loc_1 <- TOLERANCE_STRICT
      tolerance_loc_2 <- TOLERANCE_LOOSE
      tolerance_loc_3 <- 0.1
    }

    expect_error(GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = matrix_inversion_method,
                         likelihood_additional_param = 1.1), "must be a quantile q with 0 < q < 1", fixed = TRUE)
    expect_error(GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = matrix_inversion_method,
                         likelihood_additional_param = -0.1), "must be a quantile q with 0 < q < 1", fixed = TRUE)
    expect_error(GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = matrix_inversion_method,
                         likelihood_additional_param = NA_real_), "must be a finite quantile q with 0 < q < 1", fixed = TRUE)
    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    nll_exp <- 273.0138019
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(lambda))
    expect_lt(abs(nll-nll_exp),tolerance_loc_1)
    expect_error(GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = matrix_inversion_method),
                 "No value was provided for 'likelihood_additional_param'", fixed = TRUE)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile_up)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(lambda))
    expect_lt(abs(nll-302.8484089),tolerance_loc_1)
    gp_model <- GPModel(group_data = group, likelihood = "asymmetric_laplace_tkc",
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    nll_exp2 <- 271.2555943
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(lambda))
    expect_lt(abs(nll-nll_exp2),tolerance_loc_1)
    gp_model <- GPModel(group_data = group, likelihood = "asymmetric_laplace_tkc_fisher_mode_finding",
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(lambda))
    expect_lt(abs(nll-nll_exp2),tolerance_loc_1)
    gp_model <- GPModel(group_data = group, likelihood = "asymmetric_laplace_tkc_not_fisher_mode_finding",
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(lambda))
    expect_lt(abs(nll-270.8276752),tolerance_loc_1)

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                           y = y, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.8153285415)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model$get_aux_pars()-0.2688162279 )),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.3044197085, 2.0765502256))),tolerance_loc_1)
    # no standard errors are calculated for quantile regression (the approximate marginal likelihood is a
    #   pseudo-likelihood which is not smooth)
    expect_false(gp_model$can_calculate_standard_errors_coef())
    expect_false(gp_model$can_calculate_standard_errors_cov_pars())
    expect_false(gp_model$can_calculate_standard_errors_aux_pars())
    expect_equal(gp_model$get_coef(std_err = TRUE), gp_model$get_coef(std_err = FALSE))
    expect_equal(gp_model$get_cov_pars(std_err = TRUE), gp_model$get_cov_pars(std_err = FALSE))
    expect_equal(gp_model$get_aux_pars(std_err = TRUE), gp_model$get_aux_pars(std_err = FALSE))
    expect_lt(sum(abs((gp_model$get_current_neg_log_likelihood()-117.1840987))),tolerance_loc_1)
    expect_equal(gp_model$get_num_optim_iter(), 12)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-1.3426948213, -0.2126176767, 0.2026923684, 1.7721305171)
    expected_var <- c(0.02791522088, 0.02791522088, 0.02791522088, 0.81532854155)
    expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
    expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_1)

    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace_var_cor_pred_freq_asym", likelihood_additional_param = quantile,
                                           y = y, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
    expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_1)

    # Estimation with other options
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace_tkc", likelihood_additional_param = quantile,
                                           y = y, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.8230834628)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model$get_aux_pars()-0.2712208559  )),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.1344325423,  2.0358682043))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model$get_current_neg_log_likelihood()-116.1218566))),tolerance_loc_1)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-1.1523666445, -0.1685696229, 0.2386040180, 1.9014356621)
    expected_var <- c(0.03667225147, 0.03667225147, 0.03667225147, 0.82308346280)
    expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
    expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_1)
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace_tkc_not_fisher_mode_finding", likelihood_additional_param = quantile,
                                           y = y, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.7758838459)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model$get_aux_pars()-0.2545291278 )),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.0654253707, 2.0782768412))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model$get_current_neg_log_likelihood()-114.9476588))),tolerance_loc_1)
    expect_equal(gp_model$get_num_optim_iter(), 15)
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace_tkc_var_cor_pred_freq_asym", likelihood_additional_param = quantile,
                                           y = y, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_var_cor <- c(0.02840872148, 0.02840872148, 0.02840872148, 0.82308346280)
    expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
    expect_lt(sum(abs(pred$var-expected_var_cor)),tolerance_loc_1)

    # Initializing coefficients and auxiliary parameters from an iid model
    #   (this is the default, all fits above use 'init_coef_aux_pars_from_iid_model = FALSE')
    params_init_iid <- params
    params_init_iid$init_coef_aux_pars_from_iid_model <- TRUE
    capture.output( gp_model_iid_init <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                    y = y, X=X, params = params_init_iid, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_iid_init$get_cov_pars(std_err = FALSE)-0.4132224030)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_iid_init$get_aux_pars()-0.2690114464)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_iid_init$get_coef(std_err = FALSE))-c(-0.0468007689, 2.0773708217))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_iid_init$get_current_neg_log_likelihood()-116.2091326))),tolerance_loc_1)
    # the initialization from an iid model finds a better optimum than the one from the marginal sample quantile alone
    expect_lt(gp_model_iid_init$get_current_neg_log_likelihood(), 117.1840987)

    # Restarts of lbfgs ('max_num_restarts_lbfgs'). The approximate marginal likelihood of the 'asymmetric_laplace'
    #   likelihood is not smooth. The line search of lbfgs can thus fail, in which case lbfgs terminates without
    #   having converged (this happens for the fits above) and the variance of the random effects is estimated too large
    # "Cold" restarts (the default): the regression coefficients, the auxiliary parameters, and the modes are reset
    #   to their initial values and only the covariance parameters are kept
    params_restart <- params
    params_restart$max_num_restarts_lbfgs <- 4L
    capture.output( gp_model_restart <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                   y = y, X=X, params = params_restart, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_restart$get_cov_pars(std_err = FALSE)-0.4043949065)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_restart$get_aux_pars()-0.2691821831)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_restart$get_coef(std_err = FALSE))-c(-0.1675955800, 2.0823192468))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_restart$get_current_neg_log_likelihood()-116.0987752))),tolerance_loc_1)
    # the restarts find a better optimum than the fit without restarts (nll = 117.1840987, cov_par = 0.8153285415)
    expect_lt(gp_model_restart$get_current_neg_log_likelihood(), 117.1840987)
    # "Warm" restarts: the optimization simply continues from the current parameters with a re-initialized approximate
    #   Hessian. For the data below, the restarts do not find a better optimum (this is not the case in general)
    params_restart$cold_restart_lbfgs <- FALSE
    capture.output( gp_model_restart <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                   y = y, X=X, params = params_restart, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_restart$get_cov_pars(std_err = FALSE)-0.8144100464)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_restart$get_aux_pars()-0.2688516601)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_restart$get_coef(std_err = FALSE))-c(-0.3029737374, 2.0753408270))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_restart$get_current_neg_log_likelihood()-117.1792005))),tolerance_loc_1)
    # no restarts are done by default -> same results as above (for both 'cold_restart_lbfgs' options)
    params_restart$cold_restart_lbfgs <- TRUE
    params_restart$max_num_restarts_lbfgs <- 0L
    capture.output( gp_model_restart <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                   y = y, X=X, params = params_restart, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_restart$get_cov_pars(std_err = FALSE)-0.8153285415)),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_restart$get_current_neg_log_likelihood()-117.1840987))),tolerance_loc_1)
    expect_error(fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                            y = y, X=X, params = list(max_num_restarts_lbfgs = -1L)), "max_num_restarts_lbfgs is not >= 0", fixed = TRUE)

    # Line search of Nocedal and Wright (the default line search of lbfgs is a backtracking one). In contrast to the
    #   backtracking line search, this line search can return the best point found so far instead of the point that
    #   has been evaluated last (namely if the strong Wolfe condition is not satisfied when the maximal number of line
    #   search iterations is reached). The modes of the Laplace approximations then need to be restored accordingly,
    #   otherwise they correspond to a point that has been rejected by the line search (see 'SaveModesLo()' in
    #   optim_utils.h). This matters in particular for non-smooth likelihoods such as this one, for which mode finding
    #   is start-dependent
    params_nw <- params
    params_nw$optimizer_cov <- "lbfgs_linesearch_nocedal_wright"
    params_nw$optimizer_coef <- "lbfgs_linesearch_nocedal_wright"
    capture.output( gp_model_nw <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                              y = y, X=X, params = params_nw, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_nw$get_cov_pars(std_err = FALSE)-0.8064792556)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_nw$get_aux_pars()-0.2679067269)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_nw$get_coef(std_err = FALSE))-c(-0.2678415466, 2.0747656120))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_nw$get_current_neg_log_likelihood()-117.1057618))),tolerance_loc_1)
    expect_equal(gp_model_nw$get_num_optim_iter(), 13)
    # this line search finds a better optimum than the backtracking one for the data below (nll = 117.1840987)
    expect_lt(gp_model_nw$get_current_neg_log_likelihood(), 117.1840987)

    # Non-zero true intercept: the initial intercept is the marginal sample quantile of y and estimation is
    #   thus equivariant under a location shift of y (the results below are those of the fits above with the
    #   intercept shifted by 'shift'). Note: when initializing the intercept with zero (which was done before),
    #   the intercept stays far away from its true value and the estimated variance of the random effects is
    #   much too large (approx. 42 instead of approx. 0.82 for the data below)
    shift <- 9.9 # true intercept becomes 0.1 + 9.9 = 10
    y_shift <- y + shift
    capture.output( gp_model_shift <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                 y = y_shift, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_shift$get_cov_pars(std_err = FALSE)-0.8153285415)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_shift$get_aux_pars()-0.2688162279)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_shift$get_coef(std_err = FALSE))-c(-0.3044197085 + shift, 2.0765502256))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_shift$get_current_neg_log_likelihood()-117.1840987))),tolerance_loc_1)
    # same when initializing from an iid model
    capture.output( gp_model_shift <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                 y = y_shift, X=X, params = params_init_iid, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_shift$get_cov_pars(std_err = FALSE)-0.4132224030)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_shift$get_aux_pars()-0.2690114464)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_shift$get_coef(std_err = FALSE))-c(-0.0468007689 + shift, 2.0773708217))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_shift$get_current_neg_log_likelihood()-116.2091326))),tolerance_loc_1)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    tolerance_gpboost <- 0.16
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.4967306854)),tolerance_gpboost)
    expect_lt(sum(abs(gp_model$get_aux_pars()-0.2378196222)),tolerance_gpboost)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = TRUE)
    # larger tolerance for the same reason as for the random effect means below: most runs reproduce the
    #   values below exactly, but deviations of about 0.18 have been observed
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-0.9585873450, 0.4262452431, 0.9630491993, 2.0000429180))), 0.3)
    # larger tolerance: the GP model is refitted in every boosting iteration and the parallel
    #   reductions in this refit are not bit-wise reproducible, so the random effect means vary
    #   slightly between runs ('deterministic = TRUE' only makes the tree building deterministic).
    #   Deviations of about 0.3 have been observed with the default tolerance of 0.16
    expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(0.2324522564, -0.2957041199, -0.2957041199, 0.0000000000))), 0.5)
    expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c( 0.02163779030, 0.02163779030, 0.02163779030, 0.49673068540))), tolerance_gpboost)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    set.seed(1)
    capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                    nrounds = 100, early_stopping_rounds = 5, metric="test_neg_log_likelihood",
                                    use_gp_model_for_validation = TRUE, folds = folds, verbose = 0), file='NUL')
    # the CV results below vary between runs since the GP model is refitted in every boosting iteration
    #   (see the comment above): scores of 1.460 - 1.585 and best iterations of 23 - 35 have been observed
    expect_lte(cvbst$best_score,1.52*(1+tolerance_loc_3))
    expect_gte(cvbst$best_score,1.52*(1-tolerance_loc_3))
    nit <- 29
    expect_lte(cvbst$best_iter, nit+10)
    expect_gte(cvbst$best_iter, nit-10)

    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    set.seed(1)
    capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                    nrounds = 100, early_stopping_rounds = 5, metric="quantile",
                                    use_gp_model_for_validation = TRUE, folds = folds, verbose = 0), file='NUL')
    # same as above: scores of 0.390 - 0.436 and best iterations of 23 - 42 have been observed.
    # The upper bound on the iteration is wider than the lower one because the boosting
    # trajectory turned out to be sensitive to the ordering of the sparse Cholesky: with CHOLMOD
    # instead of Eigen's SimplicialLLT the early stopping lands beyond nit+15. The score bounds
    # below are the substantive check, the iteration bounds are only a sanity range
    expect_lte(cvbst$best_score,0.413*(1+tolerance_loc_3))
    expect_gte(cvbst$best_score,0.413*(1-tolerance_loc_3))
    nit <- 32
    expect_lte(cvbst$best_iter, nit+30)
    expect_gte(cvbst$best_iter, nit-15)

    # }

  }) # end asymmetric_laplace regression

  test_that("asymmetric_laplace likelihood with the SSN-ALM mode refinement ", {

    # The (Fisher) quasi-Newton mode finding of the asymmetric Laplace likelihood uses the endpoint convention
    #   for the score at the kinks of the check loss and can stall at points that are not the exact non-smooth
    #   MAP. The '_ssn_alm' suffix enables a semismooth Newton method applied to the subproblems of an augmented
    #   Lagrangian method, which is run after the quasi-Newton loop if an exact KKT check fails ('_ssn_alm_always'
    #   skips the check). The refinement can never return a worse mode, so the negative log-likelihood must
    #   decrease (weakly) for every random effects structure and every matrix approximation
    quantile_asym_laplace <- function(q, alpha, lambda) {
      res <- rep(NA, length(q))
      ind <- q <= alpha
      res[ind] <- log(q[ind]/alpha) * lambda / (1-alpha)
      res[!ind] <- -log((1-q[!ind])/(1-alpha)) * lambda / alpha
      return(res)
    }
    quantile <- 0.7
    lambda <- 0.25
    error <- quantile_asym_laplace(q = sim_rand_unif(n = n, init_c = 0.651), alpha = quantile, lambda = lambda)
    y <- as.vector(Z1 %*% b_gr_1 + X %*% beta + error)
    fixed_effects <- as.vector(X %*% beta)
    tol <- relax_tolerance_nll(TOLERANCE_STRICT)

    # 'nll_ssn' returns the negative log-likelihood without and with the refinement. The two variants of the
    #   suffix must agree here: the exact KKT check never certifies a mode that is not the exact MAP
    nll_ssn <- function(cov_pars, base_likelihood = "asymmetric_laplace", optim_params = NULL, ...) {
      vapply(c("", "_ssn_alm", "_ssn_alm_always"), function(sfx) {
        # 'capture.output': the FITC variants below warn that inducing points coincide
        #   with data points, which is expected here and would only clutter the test output
        capture.output( gp_model <- GPModel(likelihood = paste0(base_likelihood, sfx),
                                            likelihood_additional_param = quantile, ...), file = 'NUL')
        if (!is.null(optim_params)) gp_model$set_optim_params(params = optim_params)
        capture.output( nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars, y = y, aux_pars = c(lambda),
                                                           fixed_effects = fixed_effects), file = 'NUL')
        nll
      }, numeric(1))
    }
    # Checks that the refinement lowers the negative log-likelihood by the expected amount and that the
    #   gated and the unconditional variant give the same result
    expect_ssn <- function(nll, expected_base, expected_ssn) {
      expect_lt(abs(nll[[1]] - expected_base), tol)
      expect_lt(abs(nll[[2]] - expected_ssn), tol)
      expect_equal(nll[[2]], nll[[3]])
      expect_lte(nll[[2]], nll[[1]])
    }

    ## One grouped random effect (Z is an incidence matrix, the SSN system is diagonal)
    expect_ssn(nll_ssn(c(0.9), group_data = group), 138.9898225, 138.9648429)
    ## Same with the triangular kernel curvature Laplace approximation: the refinement changes the mode, the
    ##   inferential curvature of the determinant must still be the TKC one and not the SSN active set
    expect_ssn(nll_ssn(c(0.9), base_likelihood = "asymmetric_laplace_tkc", group_data = group),
               141.5811521, 140.6629934)
    ## Two crossed grouped random effects (general sparse Z)
    expect_ssn(nll_ssn(c(0.9, 0.6), group_data = cbind(group, group2)), 148.5871268, 148.3692190)
    expect_ssn(nll_ssn(c(0.9, 0.6), group_data = cbind(group, group2),
                       matrix_inversion_method = "iterative"), 148.4716563, 148.2538536)
    ## Gaussian process, all matrix approximations
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential"),
               174.8555513, 174.3157800)
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                       gp_approx = "vecchia", num_neighbors = 20), 174.6832113, 174.2094239)
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                       gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = "iterative"),
               174.7652059, 174.2813998)
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                       gp_approx = "fitc", num_ind_points = 30), 172.4871945, 172.1303044)
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                       gp_approx = "full_scale_vecchia", num_ind_points = 30, num_neighbors = 20),
               174.7149656, 174.2474945)
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                       gp_approx = "full_scale_vecchia", num_ind_points = 30, num_neighbors = 20,
                       matrix_inversion_method = "iterative",
                       optim_params = list(fitc_piv_chol_preconditioner_rank = 30, seed_rand_vec_trace = 1,
                                           num_rand_vec_trace = 200)),
               174.7950602, 174.3383760)
    ## Grouped random effects combined with a GP
    expect_ssn(nll_ssn(c(0.9, 0.9, 0.2), group_data = group, gp_coords = coords,
                       cov_function = "exponential"), 147.5915422, 146.6669310)

    ## Sample weights, including zero weights (an observation with a zero weight must not enter the active
    ##   set of the SSN system even though its prox value is zero)
    weights <- rep(1, n)
    weights[1:10] <- 0
    weights[11:20] <- 3
    nll_w <- vapply(c("", "_ssn_alm_always"), function(sfx) {
      # 'capture.output': notes that the weights do not sum to the number of data points,
      #   which is intended here and would only clutter the test output
      capture.output( gp_model <- GPModel(group_data = group, likelihood = paste0("asymmetric_laplace", sfx),
                                          likelihood_additional_param = quantile, weights = weights), file = 'NUL')
      capture.output( nll <- gp_model$neg_log_likelihood(cov_pars = c(0.9), y = y, aux_pars = c(lambda),
                                                         fixed_effects = fixed_effects), file = 'NUL')
      nll
    }, numeric(1))
    expect_lt(abs(nll_w[[1]] - 157.4794379), tol)
    expect_lt(abs(nll_w[[2]] - 157.4537383), tol)

    ## Estimation. Solving the mode finding problem exactly makes the approximate marginal likelihood a
    ##   well-defined function of the parameters, which is what the outer optimizer needs
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace",
                                           likelihood_additional_param = quantile, y = y, X = X,
                                           params = OPTIM_PARAMS_BFGS), file = 'NUL')
    capture.output( gp_model_ssn <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace_ssn_alm",
                                               likelihood_additional_param = quantile, y = y, X = X,
                                               params = OPTIM_PARAMS_BFGS), file = 'NUL')
    expect_lte(gp_model_ssn$get_current_neg_log_likelihood(),
               gp_model$get_current_neg_log_likelihood() + relax_tolerance_nll(TOLERANCE_MEDIUM))
    expect_lt(abs(gp_model_ssn$get_current_neg_log_likelihood() - 136.6565088), relax_tolerance_nll(TOLERANCE_STRICT))
    expect_equal(gp_model_ssn$get_likelihood_name(), "asymmetric_laplace")

    ## Many observations exactly on a kink. With Z = I the exact mode has residuals that are exactly zero, and the
    ##   mode is then only stationary for an interior subgradient of the check loss, which the endpoint convention of
    ##   the quasi-Newton phase cannot produce. The refinement publishes the certifying subgradient instead, so that
    ##   'Q b = Z^T first_deriv_ll_' continues to hold and the gradients stay consistent with the mode
    y_kink <- round(y * 4) / 4# lots of duplicated responses
    nll_kink <- vapply(c("", "_ssn_alm", "_ssn_alm_always"), function(sfx) {
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                          likelihood = paste0("asymmetric_laplace", sfx), likelihood_additional_param = quantile)
      gp_model$neg_log_likelihood(cov_pars = c(0.9, 0.2), y = y_kink, aux_pars = c(lambda),
                                  fixed_effects = fixed_effects)
    }, numeric(1))
    expect_lt(abs(nll_kink[[1]] - 174.6327729), tol)
    expect_lt(abs(nll_kink[[2]] - 173.4317231), tol)
    expect_equal(nll_kink[[2]], nll_kink[[3]])
    expect_lte(nll_kink[[2]], nll_kink[[1]])

    ## A conjugate gradient budget that is far too small for the semismooth Newton systems. An inaccurate Newton
    ##   direction must not be reported as a successful solve: the refinement then gives up, keeps the mode of the
    ##   quasi-Newton iteration, and does not advertise a certified score. 'cg_max_num_it' also throttles the mode
    ##   finding itself, so the comparison has to be made against the same setting without the refinement
    nll_cg <- vapply(c("", "_ssn_alm", "_ssn_alm_always"), function(sfx) {
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "vecchia",
                          num_neighbors = 20, matrix_inversion_method = "iterative",
                          likelihood = paste0("asymmetric_laplace", sfx), likelihood_additional_param = quantile)
      gp_model$set_optim_params(params = list(cg_max_num_it = 1, seed_rand_vec_trace = 1))
      gp_model$neg_log_likelihood(cov_pars = c(0.9, 0.2), y = y, aux_pars = c(lambda),
                                  fixed_effects = fixed_effects)
    }, numeric(1))
    expect_true(all(is.finite(nll_cg)))
    expect_equal(nll_cg[[2]], nll_cg[[1]])
    expect_equal(nll_cg[[3]], nll_cg[[1]])

    ## ADMM warm start ('_admm_ssn_alm') and ADMM alone ('_admm'). The warm start first runs an alternating
    ##   direction method of multipliers at a fixed penalty, which costs one factorization in total instead of
    ##   one per semismooth Newton step, and then hands the multiplier and a penalty matched to the accuracy it
    ##   reached over to the semismooth Newton iterations. It solves the same problem and must therefore reach
    ##   the same optimum up to the KKT tolerance. ADMM alone is a baseline for measuring what the warm start
    ##   contributes: it is cheap, but its iterates are dual feasible and primal infeasible, so it can fail to
    ##   improve the exact MAP objective at all, in which case the quasi-Newton mode is returned unchanged
    nll_admm <- function(cov_pars, y_use = y, optim_params = NULL, ...) {
      vapply(c("_admm_ssn_alm", "_admm"), function(sfx) {
        # 'capture.output': see the comment in 'nll_ssn' above
        capture.output( gp_model <- GPModel(likelihood = paste0("asymmetric_laplace", sfx),
                                            likelihood_additional_param = quantile, ...), file = 'NUL')
        if (!is.null(optim_params)) gp_model$set_optim_params(params = optim_params)
        capture.output( nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars, y = y_use, aux_pars = c(lambda),
                                                           fixed_effects = fixed_effects), file = 'NUL')
        nll
      }, numeric(1))
    }
    # Neither variant may return a mode that is worse than the one of the quasi-Newton iteration
    expect_admm <- function(nll, expected_base, expected_admm_ssn, expected_admm) {
      expect_lt(abs(nll[[1]] - expected_admm_ssn), tol)
      expect_lt(abs(nll[[2]] - expected_admm), tol)
      expect_lte(nll[[1]], expected_base + tol)
      expect_lte(nll[[2]], expected_base + tol)
    }
    expect_admm(nll_admm(c(0.9), group_data = group), 138.9898225, 138.9648479, 138.9713257)
    expect_admm(nll_admm(c(0.9, 0.6), group_data = cbind(group, group2)),
                148.5871268, 148.3692208, 148.5871268)
    expect_admm(nll_admm(c(0.9, 0.6), group_data = cbind(group, group2),
                         matrix_inversion_method = "iterative"), 148.4716563, 148.2537400, 148.4716563)
    expect_admm(nll_admm(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential"),
                174.8555513, 174.3158050, 174.3158050)
    expect_admm(nll_admm(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                         gp_approx = "vecchia", num_neighbors = 20),
                174.6832113, 174.2094198, 174.2094198)
    expect_admm(nll_admm(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                         gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = "iterative"),
                174.7652059, 174.2784319, 174.2784312)
    expect_admm(nll_admm(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                         gp_approx = "fitc", num_ind_points = 30), 172.4871945, 172.1302166, 172.1302166)
    expect_admm(nll_admm(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                         gp_approx = "full_scale_vecchia", num_ind_points = 30, num_neighbors = 20),
                174.7149656, 174.2474984, 174.2474984)
    expect_admm(nll_admm(c(0.9, 0.9, 0.2), group_data = group, gp_coords = coords,
                         cov_function = "exponential"), 147.5915422, 146.6669727, 146.6669727)
    ## Many observations exactly on a kink, see the comment above
    expect_admm(nll_admm(c(0.9, 0.2), y_use = y_kink, gp_coords = coords, cov_function = "exponential"),
                174.6327729, 173.4317200, 173.4317200)

    ## The suffixes are only supported for the asymmetric Laplace likelihood
    expect_error(GPModel(group_data = group, likelihood = "poisson_ssn_alm"),
                 "The '_ssn_alm' mode refinement is currently only supported", fixed = TRUE)
    expect_error(GPModel(group_data = group, likelihood = "poisson_admm_ssn_alm"),
                 "The '_ssn_alm' mode refinement is currently only supported", fixed = TRUE)
    expect_error(GPModel(group_data = group, likelihood = "poisson_admm"),
                 "The '_ssn_alm' mode refinement is currently only supported", fixed = TRUE)

  }) # end asymmetric_laplace with SSN-ALM mode refinement

}
