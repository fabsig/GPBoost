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

  test_that("Binary classification Gaussian process model with Vecchia approximation", {
    params_vecchia <- c(DEFAULT_OPTIM_PARAMS, cg_delta_conv = sqrt(1e-6),
                        num_rand_vec_trace = 500, cg_preconditioner_type = "pivoted_cholesky",
                        fitc_piv_chol_preconditioner_rank = dim(coords)[1] - 1 )
    init_cov_pars = c(1,mean(dist(coords))/3)
    params_vecchia$init_cov_pars = init_cov_pars
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars = init_cov_pars
    params_mult <- DEFAULT_OPTIM_PARAMS
    init_cov_pars_mult = c(1,mean(dist(unique(coords_multiple)))/3)
    params_mult$init_cov_pars = init_cov_pars_mult
    params_vecchia_mult <- params_vecchia
    params_vecchia_mult$init_cov_pars = init_cov_pars_mult
    params_vecchia_mult$fitc_piv_chol_preconditioner_rank <- dim(unique(coords_multiple))[1]

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
    # Estimation, prediction, and likelihood evaluation without Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, X = X, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
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
    # No linear regression term without Vecchia approximation
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

    for (inv_method in c("cholesky", "iterative")) {
      if(inv_method == "iterative") {
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
        tolerance_loc_3 <- 2*TOLERANCE_ITERATIVE
        loop_cg_PC = c("pivoted_cholesky", "vadu", "fitc")
      } else {
        tolerance_loc_1 <- TOLERANCE_STRICT
        tolerance_loc_2 <- TOLERANCE_MEDIUM
        tolerance_loc_3 <-TOLERANCE_STRICT
        loop_cg_PC = c("vadu")
      }
      nsim_var_pred <- 10000
      for (cg_preconditioner_type in loop_cg_PC) {
        params_vecchia$cg_preconditioner_type <- cg_preconditioner_type
        # Vecchia approximation with no ordering
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                            num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        capture.output( fit(gp_model, y = y, X = X, params = params_vecchia)
                        , file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),tolerance_loc_1)
        if(inv_method != "iterative") {
          expect_equal(gp_model$get_num_optim_iter(), num_it)
        }
        # Prediction
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                     num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_cov_mat = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_var = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_response = TRUE, predict_var = TRUE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu_resp)),tolerance_loc_1)
          expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
        }
        # Likelihood evaluation
        nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
        expect_lt(abs(nll-expected_nll),tolerance_loc_1)

        if(inv_method == "iterative" && cg_preconditioner_type == "pivoted_cholesky"){
          ## Cannot change cg_preconditioner_type after a model has been fitted
          expect_error( capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                                           lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                                           convergence_criterion = "relative_change_in_parameters",
                                                                           cg_delta_conv = 1e-6, num_rand_vec_trace = 500,
                                                                           cg_preconditioner_type = "vadu", init_coef_aux_pars_from_iid_model = FALSE)), file='NUL'))
        }

        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {# some tests are only run for one preconditioner
          ############################
          # Vecchia approximation with random ordering
          ############################
          capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                 vecchia_ordering="random", likelihood = "bernoulli_probit",
                                                 gp_approx = "vecchia",  num_neighbors = n-1,
                                                 y = y, X = X, params = params_vecchia,
                                                 matrix_inversion_method = inv_method), file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_1)
          if(inv_method != "iterative") {
            expect_equal(gp_model$get_num_optim_iter(), num_it)
          }
          # Prediction
          gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                       num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_cov_mat = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
          expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),tolerance_loc_3)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_var = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
          expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_response = TRUE, predict_var = TRUE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu_resp)),tolerance_loc_1)
          expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
          # Likelihood evaluation
          nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
          expect_lt(abs(nll-expected_nll),2*tolerance_loc_1)

          #######################
          ## Less neighbors than observations
          #######################
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                              num_neighbors = 30, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method), file='NUL')
          capture.output( fit(gp_model, y = y, X = X, params = params_vecchia)
                          , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_2)
          if(inv_method != "iterative") {
            expect_equal(gp_model$get_num_optim_iter(), num_it)
          }
          # Prediction
          gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                       num_neighbors_pred = 30, nsim_var_pred = nsim_var_pred)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_cov_mat = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          mu_less_neig <- c(0.3368557, 0.1492578, -0.1034736)
          cov_less_neig <- c(0.6193174862, 0.5503175873, -0.0001440701, 0.5503175873,
                             0.6159313469, -0.0001546077, -0.0001440701, -0.0001546077, 0.4292547351)
          mu_resp_less_neig <- c(0.6043853, 0.5467346, 0.4655140)
          var_resp_less_neig <- c(0.2391037, 0.2478159, 0.2488107)
          expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$cov)-cov_less_neig)),0.2)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_var = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_1)
          expect_lt(sum(abs(as.vector(pred$var)-cov_less_neig[c(1,5,9)])),tolerance_loc_1)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_response = TRUE, predict_var = TRUE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-mu_resp_less_neig)),tolerance_loc_1)
          expect_lt(sum(abs(pred$var-var_resp_less_neig)),tolerance_loc_1)
          # Use vecchia_pred_type = "order_obs_first_cond_all"
          gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
          pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                          predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
          expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_1)
          expect_lt(sum(abs(as.vector(pred$cov)-cov_less_neig)),tolerance_loc_1)
          pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE,
                          predict_var = TRUE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
          expect_lt(sum(abs(pred$mu-mu_resp_less_neig)),tolerance_loc_1)
          expect_lt(sum(abs(pred$var-var_resp_less_neig)), tolerance_loc_1)
          # Use vecchia_pred_type = "latent_order_obs_first_cond_obs_only"
          gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_obs_only",
                                       nsim_var_pred = 2000)
          pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                          predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
          expected_cov_loc <- c(0.6193174862, 0.2835405301, -0.0001440701, 0.2835405301, 0.6159312648,
                                -0.0001525779, -0.0001440701, -0.0001525779, 0.4292547351)
          expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_loc)),tolerance_loc_1)
          # Use vecchia_pred_type = "order_obs_first_cond_obs_only"
          gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only")
          pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                          predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
          expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_loc)),1.5*tolerance_loc_1)
        }

        ############################
        # Predict training data random effects
        ############################
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                               num_neighbors = 30, vecchia_ordering = "none",
                                               matrix_inversion_method = inv_method,
                                               y = y, params = params_vecchia), file='NUL')
        training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
        gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only")
        preds <- predict(gp_model, gp_coords_pred = coords, predict_response = FALSE,
                         predict_var = TRUE)
        expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),tolerance_loc_1)
        if(inv_method == "iterative"){
          expect_lt(mean(abs(training_data_random_effects[,2] - preds$var)), tolerance_loc_1) #Different RNG-Status
        } else {
          expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)), tolerance_loc_1)
        }

        ############################
        # No linear regression term
        ############################
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                               num_neighbors = n-1, vecchia_ordering = "random",
                                               matrix_inversion_method = inv_method,
                                               y = y, params = params_vecchia), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_no_X)),tolerance_loc_1)
        pred <- capture.output( predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                                        predict_response = FALSE, cov_pars = cov_pars_pred_eval), file='NUL')
        expect_lt(sum(abs(pred$mu-mu_no_X)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$var)-var_no_X)),tolerance_loc_2)

        ############################
        # With duplicates and linear regression term
        ############################
        capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                               likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                               num_neighbors = n-1, vecchia_ordering = "none",
                                               matrix_inversion_method = inv_method,
                                               y = y_multiple, X = X, params = params_vecchia_mult), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_multiple)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_multiple)),tolerance_loc_1)
        if(inv_method != "iterative") {
          expect_equal(gp_model$get_num_optim_iter(), num_it_multiple)
        }
        # Prediction
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                     num_neighbors_pred = n/4+1, nsim_var_pred = nsim_var_pred)
        pred <- predict(gp_model, y=y_multiple, gp_coords_pred = coord_test_multiple, X_pred = X_test,
                        predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred_eval)
        expect_lt(sum(abs(pred$mu-expected_mu_multiple)),tolerance_loc_2)
        expect_lt(sum(abs(pred$var-expected_var_multiple)),tolerance_loc_2)
        # Likelihood evaluation
        nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y_multiple)
        expect_lt(abs(nll-nll_multiple),tolerance_loc_1)
        # Predict training data random effects
        training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
        gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only",
                                     num_neighbors_pred = n/4, nsim_var_pred = nsim_var_pred)
        preds <- predict(gp_model, gp_coords_pred = coords_multiple, predict_response = FALSE,
                         predict_var = TRUE, X_pred = X)
        pred_mu_exp <- preds$mu - X %*% gp_model$get_coef(std_err = FALSE)
        expect_lt(sum(abs(training_data_random_effects[,1] - pred_mu_exp)),tolerance_loc_1)
        if(inv_method == "iterative"){
          expect_lt(mean(abs(training_data_random_effects[,2] - preds$var)), tolerance_loc_1) #Different RNG-Status
        } else {
          expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)), tolerance_loc_1)
        }

      }# end loop cg_preconditioner_type in loop_cg_PC
    }# end loop inv_method in c("cholesky", "iterative")

    ## "vecchia" preconditioner
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                        num_neighbors = 30, vecchia_ordering = "none",
                                        matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params = params_vecchia)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
    expect_lt(abs(nll-expected_nll),tolerance_loc_1)

    #######################
    ## Other covariance functions
    #######################
    cov_pars_matern <- c(0.98944996176, 0.04986090038)
    coefs_matern <- c(0.4250887028, -0.2722344688)
    num_it_matern <- 18
    nll_opt_matern <- 64.59961544
    nll_matern <- 68.10706059
    mu_matern <- c(0.3603830, 0.1577247, -0.1189037)
    var_matern <- c(0.4497997, 0.4460163, 0.2566184)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, X = X, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_matern)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_matern)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_matern)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_matern), TOLERANCE_STRICT)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
    expect_lt(abs(nll-nll_matern),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-mu_matern)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-var_matern)),TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5 + 1E-4,
                                             likelihood = "bernoulli_probit", gp_approx = "none",
                                             y = y, X = X, params = params), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_matern)),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_matern)),TOLERANCE_MEDIUM)
      expect_equal(gp_model$get_num_optim_iter(), num_it_matern)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_matern), TOLERANCE_MEDIUM)
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
      expect_lt(abs(nll-nll_matern),TOLERANCE_MEDIUM)
      pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                      predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
      expect_lt(sum(abs(pred$mu-mu_matern)),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(pred$var)-var_matern)),TOLERANCE_MEDIUM)
    }
    # With Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           likelihood = "bernoulli_probit", gp_approx = "vecchia", num_neighbors = n-1,
                                           y = y, X = X, params = params, matrix_inversion_method = "cholesky"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_matern)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_matern)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_matern)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_matern), TOLERANCE_STRICT)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
    expect_lt(abs(nll-nll_matern),TOLERANCE_STRICT)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test) , file='NUL')
    expect_lt(sum(abs(pred$mu-mu_matern)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-var_matern)),TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5 + 1E-4,
                                             likelihood = "bernoulli_probit", gp_approx = "vecchia", num_neighbors = n-1,
                                             y = y, X = X, params = params, matrix_inversion_method = "cholesky"), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_matern)),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_matern)),TOLERANCE_MEDIUM)
      expect_equal(gp_model$get_num_optim_iter(), num_it_matern)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_matern), TOLERANCE_MEDIUM)
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
      expect_lt(abs(nll-nll_matern),TOLERANCE_MEDIUM)
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                                      predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test) , file='NUL')
      expect_lt(sum(abs(pred$mu-mu_matern)),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(pred$var)-var_matern)),TOLERANCE_MEDIUM)
    }

    ###################
    ## Random coefficient GPs
    ###################
    probs <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3))
    y_rand_coef <- as.numeric(sim_rand_unif(n=n, init_c=0.543) < probs)
    init_cov_pars_RC <- rep(init_cov_pars, 3)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           y = y_rand_coef, likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                           num_neighbors = n-1, vecchia_ordering = "none", matrix_inversion_method = "cholesky",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov=0.5, maxit=1000, init_cov_pars=init_cov_pars_RC, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expected_values <- c(0.3701097, 0.2846740, 2.1160323, 0.3305266, 0.1241462, 0.1846456)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 39)
    # Same estimation without Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           y = y_rand_coef, likelihood = "bernoulli_probit", gp_approx = "none",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov=0.5, maxit=1000, init_cov_pars=init_cov_pars_RC, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 39)
    # Prediction
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC,
                                        cov_function = "exponential", likelihood = "bernoulli_probit",
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none", matrix_inversion_method = "cholesky"), file='NUL')
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- gp_model$predict(y = y_rand_coef, gp_coords_pred = coord_test, gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.18346009, 0.03479259, -0.17247579)
    expected_cov <- c(1.039879e+00, 7.521981e-01, -3.256500e-04, 7.521981e-01,
                      8.907289e-01, -6.719282e-05, -3.256500e-04, -6.719282e-05, 9.147899e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Same prediction without Veccchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC,
                                        cov_function = "exponential", likelihood = "bernoulli_probit",
                                        gp_approx = "none"), file='NUL')
    pred <- gp_model$predict(y = y_rand_coef, gp_coords_pred = coord_test, gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)

    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(1,0.1,0.8,0.15,1.1,0.08),y=y_rand_coef)
    expect_lt(abs(nll-65.1768199),TOLERANCE_MEDIUM)

    ###################
    ##  Multiple cluster IDs
    ###################
    probs <- pnorm(L %*% b_1)
    y_clus <- as.numeric(sim_rand_unif(n=n, init_c=0.2978341) < probs)
    init_cov_pars <- c(1,mean(dist(coords[cluster_ids==1,]))/3)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y_clus, cluster_ids = cluster_ids, likelihood = "bernoulli_probit",
                                           gp_approx = "vecchia", num_neighbors = n-1,
                                           vecchia_ordering = "none", matrix_inversion_method = "cholesky",
                                           params = list(optimizer_cov = "gradient_descent", lr_cov=0.2,
                                                         use_nesterov_acc = FALSE, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(0.5085134, 0.2011667)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 20)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    cluster_ids_pred = c(1,3,1)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        cluster_ids = cluster_ids,likelihood = "bernoulli_probit"), file='NUL')
    pred <- gp_model$predict(y = y_clus, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1509569, 0.0000000, 0.9574946)
    expected_cov <- c(1.2225959453, 0.0000000000, 0.0003074858, 0.0000000000,
                      1.5000000000, 0.0000000000, 0.0003074858, 0.0000000000, 1.0761874845)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)

  })

  test_that("Vecchia approximation with duplicate locations, correlation-based neighbors, and multiple clusters ", {
    # For non-Gaussian likelihoods, duplicate locations are collapsed to unique latent locations, among which the
    # neighbors are selected. With all previous unique locations as neighbors, the Vecchia approximation is exact,
    # and the results are thus compared to the ones without an approximation
    eps_multiple <- as.vector(L_multiple %*% b_multiple)
    y_multiple <- as.numeric(sim_rand_unif(n=n, init_c=0.2818) < pnorm(eps_multiple))
    num_unique <- n / 4
    coords_ST_multiple <- cbind(rep((1:num_unique) / num_unique, 4), coords_multiple)
    cov_pars_gneiting <- c(1, 10, 10, 0.5, 1.5, 0.5, 1)# a smoothness of 1.5 does not require the Bessel function
    cov_pars_exp <- c(1, 0.1)
    nll_model <- function(y, cov_pars, ...) {
      capture.output( gp_model <- GPModel(likelihood = "bernoulli_probit", ...), file='NUL')
      capture.output( nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars, y = y), file='NUL')
      nll
    }
    # 'space_time_gneiting', for which the neighbors are selected based on correlations by default
    nll_exact <- nll_model(y_multiple, cov_pars_gneiting, gp_coords = coords_ST_multiple, cov_function = "space_time_gneiting")
    for (approx in c("vecchia", "vecchia_euclidean")) {
      nll <- nll_model(y_multiple, cov_pars_gneiting, gp_coords = coords_ST_multiple, cov_function = "space_time_gneiting",
                       gp_approx = approx, num_neighbors = num_unique - 1, vecchia_ordering = "none",
                       matrix_inversion_method = "cholesky")
      expect_lt(abs(nll - nll_exact), TOLERANCE_STRICT_LOWER)
    }
    # Correlation-based neighbors for an isotropic covariance function
    nll_exact <- nll_model(y_multiple, cov_pars_exp, gp_coords = coords_multiple, cov_function = "exponential")
    nll <- nll_model(y_multiple, cov_pars_exp, gp_coords = coords_multiple, cov_function = "exponential",
                     gp_approx = "vecchia_correlation_based", num_neighbors = num_unique - 1, vecchia_ordering = "none",
                     matrix_inversion_method = "cholesky")
    expect_lt(abs(nll - nll_exact), TOLERANCE_STRICT_LOWER)
    # Multiple clusters, only the first of which has duplicate locations
    coords_clus <- rbind(coords_multiple[1:50, ], coords[51:100, ])
    coords_ST_clus <- cbind(time, coords_clus)
    cluster_ids_dup <- c(rep(1, 50), rep(2, 50))
    nll_exact <- nll_model(y_multiple, cov_pars_exp, gp_coords = coords_clus, cov_function = "exponential",
                           cluster_ids = cluster_ids_dup)
    for (approx in c("vecchia", "vecchia_correlation_based")) {
      nll <- nll_model(y_multiple, cov_pars_exp, gp_coords = coords_clus, cov_function = "exponential",
                       cluster_ids = cluster_ids_dup, gp_approx = approx, num_neighbors = 49, vecchia_ordering = "none",
                       matrix_inversion_method = "cholesky")
      expect_lt(abs(nll - nll_exact), TOLERANCE_STRICT_LOWER)
    }
    nll_exact <- nll_model(y_multiple, cov_pars_gneiting, gp_coords = coords_ST_clus, cov_function = "space_time_gneiting",
                           cluster_ids = cluster_ids_dup)
    nll <- nll_model(y_multiple, cov_pars_gneiting, gp_coords = coords_ST_clus, cov_function = "space_time_gneiting",
                     cluster_ids = cluster_ids_dup, gp_approx = "vecchia", num_neighbors = 49, vecchia_ordering = "none",
                     matrix_inversion_method = "cholesky")
    expect_lt(abs(nll - nll_exact), TOLERANCE_STRICT_LOWER)
    # Predictions for a new cluster with duplicate locations
    coords_pred <- rbind(coords[1:10, ], coords[1:5, ])
    capture.output( gp_model_exact <- GPModel(gp_coords = coords_multiple[1:50, ], cov_function = "exponential",
                                              likelihood = "bernoulli_probit"), file='NUL')
    capture.output( pred_exact <- gp_model_exact$predict(y = y_multiple[1:50], gp_coords_pred = coords_pred, cluster_ids_pred = rep(2, 15),
                                                         cov_pars = cov_pars_exp, predict_cov_mat = TRUE, predict_response = FALSE), file='NUL')
    for (approx in c("vecchia", "vecchia_correlation_based")) {
      capture.output( gp_model <- GPModel(gp_coords = coords_multiple[1:50, ], cov_function = "exponential",
                                          likelihood = "bernoulli_probit", gp_approx = approx, num_neighbors = num_unique - 1,
                                          vecchia_ordering = "none", matrix_inversion_method = "cholesky"), file='NUL')
      capture.output( pred <- gp_model$predict(y = y_multiple[1:50], gp_coords_pred = coords_pred, cluster_ids_pred = rep(2, 15),
                                               cov_pars = cov_pars_exp, predict_cov_mat = TRUE, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu - pred_exact$mu)), TOLERANCE_STRICT)
      expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_exact$cov))), TOLERANCE_STRICT)
      capture.output( pred <- gp_model$predict(y = y_multiple[1:50], gp_coords_pred = coords_pred, cluster_ids_pred = rep(2, 15),
                                               cov_pars = cov_pars_exp, predict_var = TRUE, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$var - diag(pred_exact$cov))), TOLERANCE_STRICT)
    }
    # Duplicate locations are not supported for 'full_scale_vecchia' and non-Gaussian likelihoods, also not for a new cluster
    capture.output( gp_model <- GPModel(gp_coords = coords[1:50, ], cov_function = "exponential", likelihood = "bernoulli_probit",
                                        gp_approx = "full_scale_vecchia", num_ind_points = 5, num_neighbors = 10), file='NUL')
    expect_error(capture.output( gp_model$predict(y = y_multiple[1:50], gp_coords_pred = coords_pred, cluster_ids_pred = rep(2, 15),
                                                  cov_pars = cov_pars_exp, predict_var = TRUE, predict_response = FALSE), file='NUL'),
                 "Duplicates found in the prediction coordinates")
  })

  test_that("Vecchia approximation: predictions for a new cluster use 'num_neighbors_pred' ", {
    # With all previous prediction points as neighbors, the approximation of the prior of a new cluster is exact. The
    # models use fewer neighbors for estimation, and the neighbors are determined in different ways: Euclidean
    # distances, correlations, scaled coordinates ('matern_space_time'), and residual correlations ('vif')
    y_bin <- as.numeric(sim_rand_unif(n=n, init_c=0.2978341) < pnorm(L %*% b_1))
    num_pred <- 50
    coords_pred <- matrix(sim_rand_unif(n=num_pred*d, init_c=0.27), ncol=d)
    coords_ST <- cbind(time, coords)
    coords_ST_pred <- cbind(sim_rand_unif(n=num_pred, init_c=0.63), coords_pred)
    cases <- list(
      list(model = list(gp_coords = coords, cov_function = "exponential"), approx = list(gp_approx = "vecchia"),
           gp_coords_pred = coords_pred, cov_pars = c(1,0.1)),
      list(model = list(gp_coords = coords, cov_function = "exponential"), approx = list(gp_approx = "vecchia_correlation_based"),
           gp_coords_pred = coords_pred, cov_pars = c(1,0.1)),
      list(model = list(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 1.5), approx = list(gp_approx = "vecchia"),
           gp_coords_pred = coords_ST_pred, cov_pars = c(1,0.2,0.1)),
      list(model = list(gp_coords = coords_ST, cov_function = "space_time_gneiting"), approx = list(gp_approx = "vecchia"),
           gp_coords_pred = coords_ST_pred, cov_pars = c(1,10,10,0.5,1.5,0.5,1)),
      list(model = list(gp_coords = coords, cov_function = "exponential"),
           approx = list(gp_approx = "vif_correlation_based", num_ind_points = 10),
           gp_coords_pred = coords_pred, cov_pars = c(1,0.1))
    )
    for (cs in cases) {
      capture.output( gp_model <- do.call(GPModel, c(cs$model, list(likelihood = "bernoulli_probit"))), file='NUL')
      capture.output( pred_exact <- gp_model$predict(y = y_bin, gp_coords_pred = cs$gp_coords_pred, cluster_ids_pred = rep(2, num_pred),
                                                     cov_pars = cs$cov_pars, predict_cov_mat = TRUE, predict_response = FALSE), file='NUL')
      capture.output( gp_model <- do.call(GPModel, c(cs$model, cs$approx, list(likelihood = "bernoulli_probit", num_neighbors = 5,
                                                                                vecchia_ordering = "none", matrix_inversion_method = "cholesky"))), file='NUL')
      gp_model$set_prediction_data(num_neighbors_pred = num_pred - 1)
      capture.output( pred <- gp_model$predict(y = y_bin, gp_coords_pred = cs$gp_coords_pred, cluster_ids_pred = rep(2, num_pred),
                                               cov_pars = cs$cov_pars, predict_cov_mat = TRUE, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu - pred_exact$mu)), TOLERANCE_STRICT)
      expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_exact$cov))), TOLERANCE_STRICT)
    }
  })

}
