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

  test_that("Binary classification Gaussian process model with VIF (or Full scale Vecchia) approximation", {
    params_vif <- c(DEFAULT_OPTIM_PARAMS, cg_delta_conv = sqrt(1e-6),
                    num_rand_vec_trace = 500, cg_preconditioner_type = "fitc")
    init_cov_pars = c(1,mean(dist(coords))/3)
    params_vif$init_cov_pars = init_cov_pars
    params_vif$fitc_piv_chol_preconditioner_rank = dim(coords)[1] - 1
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars = init_cov_pars
    params_mult <- DEFAULT_OPTIM_PARAMS
    init_cov_pars_mult = c(1,mean(dist(unique(coords_multiple)))/3)
    params_mult$init_cov_pars = init_cov_pars_mult
    params_vif_mult <- params_vif
    params_vif_mult$init_cov_pars = init_cov_pars_mult

    # Simulate data and define expected values
    probs <- pnorm(L %*% b_1) # note: linear predictor is not included in simulation
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.19341) < probs)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    init_cov_pars <- c(1,mean(dist(coords))/3)
    cov_pars_pred_eval = c(1,0.2)
    cov_pars <- c(0.92350821208, 0.05944214192)
    coefs <- c(0.3983333, -0.2653886)
    num_it <- 17
    expected_mu <- c(0.3389905, 0.1512445, -0.1039307)
    expected_cov <- c(0.6193228722, 0.5503216948, -0.0001420698, 0.5503216948,
                      0.6159348965, -0.0001556274, -0.0001420698, -0.0001556274, 0.4291674143)
    expected_mu_resp <- c(0.6050312, 0.5473537, 0.4653610)
    expected_var_resp <- c(0.2389684, 0.2477576, 0.2488001)
    expected_nll <- 67.18342059
    # Estimation, prediction, and likelihood evaluation without VIF approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, X = X, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE,
                    predict_var = TRUE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu_resp)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var_resp)),TOLERANCE_MEDIUM)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
    expect_lt(abs(nll-expected_nll),TOLERANCE_STRICT)
    # No linear regression term without VIF approximation
    cov_pars_no_X <- c(0.6875476, 0.1062862 )
    mu_no_X <- c(0.01874013, 0.01200800, 0.20498871)
    var_no_X <- c(0.6105248, 0.6093745, 0.4235374)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_no_X)),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval)
    expect_lt(sum(abs(pred$mu-mu_no_X)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-var_no_X)),TOLERANCE_MEDIUM)
    # With duplicates and linear regression term without Vecchia approximation
    eps_multiple <- as.vector(L_multiple %*% b_multiple)
    probs_multiple <- pnorm(eps_multiple)
    y_multiple <- as.numeric(sim_rand_unif(n=n, init_c=0.2818) < probs_multiple)
    coord_test_multiple <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    cov_pars_multiple <- c(0.8263711, 0.1240696 )
    coefs_multiple <- c( 0.6168877, 0.1381717)
    num_it_multiple <- 17
    expected_mu_multiple <- c(-0.01076580, 0.07873293, 0.18927032)
    expected_var_multiple <- c(0.5653402, 0.6019163, 0.6019163)
    nll_multiple <- 58.671494
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y_multiple, X = X, params = params_mult), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_multiple)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_multiple)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_multiple)
    pred <- predict(gp_model, y=y_multiple, gp_coords_pred = coord_test_multiple, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred_eval)
    expect_lt(sum(abs(pred$mu-expected_mu_multiple)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var_multiple)),TOLERANCE_MEDIUM)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y_multiple)
    expect_lt(abs(nll-nll_multiple),TOLERANCE_STRICT)
    tolerance_loc_1 <- TOLERANCE_ITERATIVE
    tolerance_loc_2 <- TOLERANCE_ITERATIVE
    tolerance_loc_3 <- 2*TOLERANCE_ITERATIVE
    loop_cg_PC = c("vifdu", "fitc")
    nsim_var_pred <- 10000
    for (cg_preconditioner_type in loop_cg_PC) {
      params_vif$cg_preconditioner_type <- cg_preconditioner_type
      # vif approximation with no ordering
      capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                          likelihood = "bernoulli_probit", gp_approx = "full_scale_vecchia",
                                          num_neighbors = n-1, num_ind_points = 20,vecchia_ordering = "none",
                                          matrix_inversion_method = "iterative"), file='NUL')
      capture.output( fit(gp_model, y = y, X = X, params = params_vif)
                      , file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_1)
      # Prediction
      gp_model$set_prediction_data(num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                      predict_var = TRUE, predict_response = FALSE,
                                      cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
      expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
      # Likelihood evaluation
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
      expect_lt(abs(nll-expected_nll),tolerance_loc_3)


      if (cg_preconditioner_type == "fitc") {# some tests are only run for one preconditioner
        ############################
        # VIF approximation with correlation-based neighbor search
        ############################
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               vecchia_ordering="random", likelihood = "bernoulli_probit",
                                               gp_approx = "full_scale_vecchia_correlation_based",
                                               num_neighbors = n-1, num_ind_points = 20,
                                               y = y, X = X, params = params_vif,
                                               matrix_inversion_method = "iterative"), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_1)
        # Prediction
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_var = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
        # This stochastic predictive variance varies strongly with the number of OpenMP threads: it is 0.04 with 16
        # threads but 0.19 with a single one, so it used to pass by only 4% of a 2*tolerance_loc_1 (= 0.2) budget while
        # the thread-induced change alone is 0.15. Use 4*tolerance_loc_1 so that intermediate thread counts also pass
        expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),4*tolerance_loc_1)
        # Likelihood evaluation
        nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
        expect_lt(abs(nll-expected_nll),tolerance_loc_1)

        #######################
        ## Less neighbors than observations
        #######################
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = "bernoulli_probit", gp_approx = "full_scale_vecchia",
                                            num_neighbors = 10, num_ind_points = 20, vecchia_ordering = "none",
                                            matrix_inversion_method = "iterative"), file='NUL')
        capture.output( fit(gp_model, y = y, X = X, params = params_vif)
                        , file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_3)
        # Prediction
        mu_less_neig <- c(0.3362000,  0.1499488, -0.1014509)
        var_resp_less_neig <- c(0.6036511, 0.6025073, 0.4220153)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_var = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(pred$var)-var_resp_less_neig)),relax_tolerance(2*tolerance_loc_1, var_resp_less_neig))
      }


    }# end loop cg_preconditioner_type in loop_cg_PC
  })

}
