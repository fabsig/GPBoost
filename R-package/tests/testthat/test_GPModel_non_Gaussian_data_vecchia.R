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

  test_that("Binary classification Gaussian process model with Wendland covariance function", {

    probs <- pnorm(L %*% b_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    init_cov_pars <- c(mean(dist(coords))/3)
    # Estimation using gradient descent
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland",
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 0.1,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(0.5553221)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 33)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.05440076, -0.05767809, 0.05060592)
    expected_cov <- c(0.5539199, 0.4080647, 0.0000000, 0.4080647, 0.5533222, 0.0000000, 0.0000000, 0.0000000, 0.5146614)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE)
    expected_mu <- c(0.4825954, 0.4815441, 0.5163995)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
  })

  test_that("Binary classification with linear predictor and grouped random effects model ", {

    probs <- pnorm(Z1 %*% b_gr_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.542) < probs)
    init_cov_pars = c(1)

    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                     optimizer_coef = "gradient_descent", lr_cov = 0.05, lr_coef = 1,
                                                     use_nesterov_acc = TRUE, acc_rate_cov = 0.2, acc_rate_coef = 0.1,
                                                     init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.4072025)
    coef <- c(-0.1113238, 1.5178339)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 43)

    # Estimation using Nelder-Mead
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead", delta_rel_conv=1e-12,
                                                     init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.399973)
    coef <- c(-0.1109516, 1.5149596)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    # init_cov_pars not given
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead", delta_rel_conv=1e-12, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.399973)
    coef <- c(-0.1109516, 1.5149596)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)

    # Estimation using lbfgs
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.3996146704)
    coef <- c(-0.1109363315, 1.5150072519)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 9)

    # Prediction
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                     optimizer_coef = "gradient_descent",
                                                     use_nesterov_acc=FALSE, lr_coef=1, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.81132150, -0.08574588, 0.21768684, 1.40591430)
    expected_cov <- c(0.1380238, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.1688248, 0.1688248,
                      0.0000000, 0.0000000, 0.1688248, 0.1688248, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.4051185)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test, predict_response = TRUE)
    expected_mu <- c(0.2234684, 0.4683923, 0.5797886, 0.8821984)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)

    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences,]
    group_unique <- unique(group)
    X_zero <- cbind(rep(0,length(group_unique)),rep(0,length(group_unique)))
    preds <- predict(gp_model, group_data_pred = group_unique, X_pred = X_zero,
                     predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),TOLERANCE_STRICT)

    # Standard deviations
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                                           y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                                     optimizer_coef = "gradient_descent", init_cov_pars=init_cov_pars,
                                                                     use_nesterov_acc = TRUE, lr_cov = 0.1, lr_coef = 1, init_coef_aux_pars_from_iid_model = FALSE)),
                    file='NUL')
    cov_pars <- c(0.4016599868 )
    coef <- c(-0.1116235586,  0.2568338470 , 1.5161515464,  0.2637361920)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)

    # Providing initial covariance parameters and coefficients
    cov_pars <- c(1)
    coef <- c(2,5)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(maxit=0, init_cov_pars=cov_pars, init_coef=coef,
                                                     optimizer_cov = "gradient_descent", init_coef_aux_pars_from_iid_model = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)

    # Large data
    n_L <- 1e6 # number of samples
    m_L <- n_L/10 # number of categories / levels for grouping variable
    group_L <- rep(1,n_L) # grouping variable
    for(i in 1:m_L) group_L[((i-1)*n_L/m_L+1):(i*n_L/m_L)] <- i
    keps <- 1E-10
    b1_L <- qnorm(sim_rand_unif(n=m_L, init_c=0.671)*(1-keps) + keps/2)
    X_L <- cbind(rep(1,n_L),sim_rand_unif(n=n_L, init_c=0.8671)-0.5) # design matrix / covariate data for fixed effect
    probs_L <- pnorm(b1_L[group_L] + X_L%*%beta)
    y_L <- as.numeric(sim_rand_unif(n=n_L, init_c=0.12378)*(1-keps) + keps/2 < probs_L)
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- fitGPModel(group_data = group_L, likelihood = "bernoulli_probit",
                           y = y_L, X=X_L, params = list(optimizer_cov = "gradient_descent",
                                                         optimizer_coef = "gradient_descent", lr_cov = 0.05, lr_coef = 0.1,
                                                         use_nesterov_acc = TRUE, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.9757876802)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(0.09848153264, 1.99446139138))),TOLERANCE_MEDIUM)

  })

  test_that("Binary classification with linear predictor and Gaussian process model ", {

    probs <- pnorm(L %*% b_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.199) < probs)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- c(1,mean(dist(coords))/3)

    # Estimation
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = params)
    cov_pars <- c(1.2660987164, 0.2854664658)
    coefs <- c(0.2041076447, 1.4663366438)
    nll <- 48.41567975
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)

    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = c(1,0.2))
    expected_mu <- c(-0.6873889499, 0.3334397127, 2.5116340251)
    expected_var <- c(0.7205439641, 0.7196871780, 0.4591627357)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # Estimation using Nelder-Mead
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead",
                                                     maxit=1000, delta_rel_conv=1e-12, init_cov_pars = params$init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(1.2717516, 0.2875537))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(0.1999365, 1.4666199))),TOLERANCE_MEDIUM)

    # Standard deviations
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           y = y, X=X, params = params),
                    file='NUL')
    coef <- c(0.2041076447, 0.5402831971, 1.4663366438, 0.3028191307)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)

  })

  test_that("Tapering for binary classification", {
    probs <- pnorm(L %*% b_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.199) < probs)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- c(1,mean(dist(coords))/3)
    params_mult = DEFAULT_OPTIM_PARAMS
    params_mult$init_cov_pars <- c(1,mean(dist(unique(coords_multiple)))/3)

    # No tapering
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = params)
    cov_pars <- c(1.2660987164, 0.2854664658)
    coefs <- c(0.2041076447, 1.4663366438)
    nll <- 48.41567975
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = c(1, 0.2))
    expected_mu <- c(-0.6873889499, 0.3334397127, 2.5116340251)
    expected_var <- c(0.7205439641, 0.7196871780 ,0.4591627357)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # With tapering and very large tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           gp_approx = "tapering", cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X=X, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = c(1, 0.2))
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # With tapering and small tapering range
    params_25 <- params
    params_25$init_cov_pars <- c(1, mean(dist(coords))/5.9 / 2 * sqrt(5))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", likelihood = "bernoulli_probit",
                                           cov_fct_shape = 2.5,
                                           gp_approx = "tapering", cov_fct_taper_shape = 1, cov_fct_taper_range = 0.5,
                                           y = y, X=X, params = params_25), file='NUL')
    cov_pars <- c(0.8066310, 0.4394089)
    coefs <- c(0.3572927, 1.3868262)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 9)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = c(1,0.2))
    expected_mu <- c( -0.4479216, 0.5456168, 2.4365937)
    expected_var <- c(0.7188332, 0.7297977, 0.3826909)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # Multiple observations at the same location
    eps_multiple <- as.vector(L_multiple %*% b_multiple)
    probs <- pnorm(eps_multiple + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.41) < probs)
    #No tapering
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X=X, params = params_mult), file='NUL')
    cov_pars <- c(1.10087285407, 0.08210071565)
    coefs <- c(0.5546205072, 1.7750831945)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_MEDIUM)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = c(1,0.1))
    expected_mu <- c( -0.4444039547, 0.7819372016, 2.2020037559)
    expected_var <- c(0.8182217994, 0.8492281399, 0.8492281399)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # With tapering and very large tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "tapering",
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X=X, params = params_mult), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),1e-1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),1e-1)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.1)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),1)

    # With tapering and small tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "tapering",
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 0.5,
                                           y = y, X=X, params = params_mult), file='NUL')
    cov_pars <- c(1.1116570369, 0.1605364915)
    coefs <- c(0.5578846856, 1.7690518846)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),1e-2)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c( -0.4310628596 , 0.7944185893 , 2.2096600970)
    expected_var <- c( 0.9198750706, 0.9556430171, 0.9556430171)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

  })

  test_that("FITC for binary classification", {

    probs <- pnorm(L %*% b_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.199) < probs)
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    coord_test_v1 <- rbind(c(0.11,0.45),coords[1:2,])
    coord_test_multiple <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    cov_pars_ll <- c(1, 0.2)
    cov_pars_pred = c(1, 0.2)
    y_multiple <- as.numeric(sim_rand_unif(n=n, init_c=0.6779) < pnorm(L_multiple %*% b_multiple + X%*%beta))
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- c(1,mean(dist(coords))/3)
    params_mult = DEFAULT_OPTIM_PARAMS
    params_mult$init_cov_pars <- c(1,mean(dist(unique(coords_multiple)))/3)
    cluster_ids_ip <- c(rep(1,n/2),rep(2,n/2))
    cluster_ids_pred <- c(1,2,2)
    cluster_ids_pred_new <- c(1,2,99)
    X_test_clus <- cbind(rep(0,3),rep(0,3))

    # Cannot have more inducing points than samples
    expect_error( fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                             y = y, X=X, params = params, gp_approx = "fitc",
                             num_ind_points = n+1, ind_points_selection = "random") )

    ## Evaluate log-likelihood
    gp_model_no_approx <- GPModel(gp_coords = coords, cov_function = "exponential",
                                  likelihood = "bernoulli_probit")
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        likelihood = "bernoulli_probit", gp_approx = "fitc",
                        num_ind_points = n, ind_points_selection = "random")
    expect_lt(abs(gp_model$neg_log_likelihood(y = y, cov_pars = cov_pars_ll) -
                    gp_model_no_approx$neg_log_likelihood(y = y, cov_pars = cov_pars_ll)),TOLERANCE_STRICT)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        likelihood = "bernoulli_probit", gp_approx = "fitc",
                                        num_ind_points  = 50, ind_points_selection = "kmeans++") , file='NUL')
    nll2 <- 63.19375632
    expect_lt(abs(gp_model$neg_log_likelihood(y = y, cov_pars = cov_pars_ll) - nll2),TOLERANCE_STRICT)

    # Estimation without approximation
    gp_model_no_approx <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                     y = y, X=X, params = params)
    nll_exp <- gp_model_no_approx$get_current_neg_log_likelihood() + 0.
    # Nelder-Mead
    params_NM <- params
    params_NM$optimizer_coef <- params_NM$optimizer_cov <- "nelder_mead"
    gp_model_NM_no_approx <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                        y = y, X=X, params = params_NM)
    nll_NM_exp <- gp_model_NM_no_approx$get_current_neg_log_likelihood() + 0.
    # Prediction
    pred_var_no_approx <- predict(gp_model_no_approx, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                                  predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    pred_cov_no_approx <- predict(gp_model_no_approx, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                                  predict_cov = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    pred_resp_no_approx <- predict(gp_model_no_approx, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                                   predict_var = FALSE, predict_response = TRUE, cov_pars = cov_pars_pred)
    X0 <- matrix(0, nrow=nrow(X), ncol=ncol(X))
    pred_train_no_approx <- predict(gp_model_no_approx, y=y, gp_coords_pred = coords, X_pred = X0,
                                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    # duplicate coordinates
    gp_model_mult_no_approx <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", likelihood = "bernoulli_probit",
                                          y = y_multiple, X=X, params = params_mult)
    nll_mult_exp <- gp_model_mult_no_approx$get_current_neg_log_likelihood() + 0.
    pred_mult_no_approx <- predict(gp_model_mult_no_approx, gp_coords_pred = coord_test_multiple, X_pred = X_test,
                                   predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    # cluster_ids
    gp_model_clus_no_approx <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                          cluster_ids = cluster_ids_ip, y = y, X=X, params = params)
    nll_clus_exp <- gp_model_clus_no_approx$get_current_neg_log_likelihood() + 0.
    pred_clus_no_approx <- predict(gp_model_clus_no_approx, y=y, gp_coords_pred = coord_test_v1,
                                   X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred,
                                   predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    pred_clus_no_approx_new <- predict(gp_model_clus_no_approx, y=y, gp_coords_pred = coord_test_v1,
                                       X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred_new,
                                       predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)

    # Fitc and large num_ind_points
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = params, gp_approx = "fitc",
                           num_ind_points = n, ind_points_selection = "random")
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_no_approx$get_cov_pars(std_err = FALSE)))),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_no_approx$get_coef(std_err = FALSE)))),TOLERANCE_STRICT_LOWER)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),TOLERANCE_STRICT_LOWER)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_var_no_approx$mu)),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_approx$var))),TOLERANCE_STRICT_LOWER)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                    predict_cov = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_cov_no_approx$mu)),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_cov_no_approx$cov))),TOLERANCE_STRICT_LOWER)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                    predict_var = FALSE, predict_response = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_resp_no_approx$mu)),TOLERANCE_STRICT)
    # Predict training data
    pred_train_fitc <- predict(gp_model, y=y, gp_coords_pred = coords, X_pred = X0,
                               predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred_train_no_approx$mu - pred_train_fitc$mu)), TOLERANCE_LOOSE)
    expect_lt(sum(abs(pred_train_no_approx$var - pred_train_fitc$var)), TOLERANCE_LOOSE)
    # With duplicate locations
    gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y_multiple, X=X, params = params_mult, gp_approx = "fitc",
                           num_ind_points = dim(unique(coords_multiple))[1], ind_points_selection = "random")
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_mult_no_approx$get_cov_pars(std_err = FALSE)))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_mult_no_approx$get_coef(std_err = FALSE)))),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), gp_model_mult_no_approx$get_num_optim_iter())
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_mult_exp),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test_multiple, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_mult_no_approx$mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_mult_no_approx$var))),TOLERANCE_STRICT_LOWER)
    # cluster_ids
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = params, gp_approx = "fitc", cluster_ids = cluster_ids_ip,
                           num_ind_points = n/2, ind_points_selection = "random")
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_clus_no_approx$get_cov_pars(std_err = FALSE)))),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_clus_no_approx$get_coef(std_err = FALSE)))),TOLERANCE_STRICT_LOWER)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_clus_exp),TOLERANCE_STRICT_LOWER)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1,
                    X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_clus_no_approx$mu)),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_clus_no_approx$var))),TOLERANCE_STRICT_LOWER)
    # Prediction for a new cluster: there is no observed data, hence the prior is used, and the inducing
    # points are determined from the prediction locations of this cluster. The predictive variances differ
    # from the exact ones by the jitter that is added to the diagonal of the inducing point matrix
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1,
                    X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred_new,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_clus_no_approx_new$mu)),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_clus_no_approx_new$var))),TOLERANCE_STRICT_LOWER)

    # Fitc and smaller num_ind_points
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           y = y, X=X, params = params, gp_approx = "fitc",
                                           num_ind_points = 50, ind_points_selection = "kmeans++") , file='NUL')
    cov_pars_2 <- c(1.67443064, 0.23625418)
    coefs_2 <- c(0.28845306, 1.62607540)
    nll_2 <- 48.11931906
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_2)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_2)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_2), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_no_approx$get_cov_pars(std_err = FALSE)))),1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_no_approx$get_coef(std_err = FALSE)))),0.5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),0.5)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    mu_exp <- c(0.16002215, 1.49343290, 1.37457340)
    cov_exp <- c(0.80926655, 0.28066810, -0.00027551, 0.28066810, 0.65993096, -0.00015671, -0.00027551, -0.00015671, 0.39237780)
    expect_lt(sum(abs(pred$mu - mu_exp)),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(pred$var) - cov_exp[c(1,5,9)])),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(pred$mu - pred_var_no_approx$mu)),0.5)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_approx$var))),0.5)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                    predict_cov = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_cov_no_approx$cov))),0.5)
    expect_lt(sum(abs(as.vector(pred$cov) - cov_exp)),TOLERANCE_STRICT_LOWER)
    # Predict training data
    pred_train_fitc <- predict(gp_model, y=y, gp_coords_pred = coords, X_pred = X0,
                               predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred_train_no_approx$mu - pred_train_fitc$mu)), 7)
    expect_lt(sum(abs(pred_train_no_approx$var - pred_train_fitc$var)), 5)
    # With duplicate locations
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           y = y_multiple, X=X, params = params_mult, gp_approx = "fitc",
                                           num_ind_points = 12, ind_points_selection = "kmeans++")  , file='NUL')
    cov_pars_2 <- c(4.58741750, 0.07401628)
    coefs_2 <- c(1.47775348, 4.19935782)
    nll_2 <- 31.49300336
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_2)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_2)),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_2), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_mult_no_approx$get_cov_pars(std_err = FALSE)))),1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_mult_no_approx$get_coef(std_err = FALSE)))),0.5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_mult_exp),0.2)

    # Nelder-Mead for fitc and large num_ind_points
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = params_NM, gp_approx = "fitc",
                           num_ind_points = n, ind_points_selection = "random")
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_NM_no_approx$get_cov_pars(std_err = FALSE)))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_NM_no_approx$get_coef(std_err = FALSE)))),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_NM_exp),TOLERANCE_STRICT_LOWER)
    # Nelder-Mead for fitc and smaller num_ind_points
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           y = y, X=X, params = params_NM, gp_approx = "fitc",
                                           num_ind_points = 50, ind_points_selection = "kmeans++") , file='NUL')
    cov_pars_NM2 <- c(1.6426189413, 0.2444053821)
    coefs_NM2 <- c(0.249596402, 1.609043132)
    nll_NM2 <- 48.11741695
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_NM2)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_NM2)), TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_NM2), TOLERANCE_STRICT_LOWER)

  })

  test_that("Binary classification with Gaussian process model and logit link function", {

    probs <- 1/(1+exp(- L %*% b_1))
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- c(1,mean(dist(coords))/3)
    params$lr_cov=0.01

    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_logit",
                                           y = y, params = params)
                    , file='NUL')
    cov_pars <- c(1.4300136, 0.1891952)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 85)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.7792960, -0.7876208, 0.5476390)
    expected_cov <- c(1.024266883e+00, 9.215203622e-01, 5.561463409e-05, 9.215203622e-01, 1.022897212e+00, 2.028646043e-05, 5.561463409e-05, 2.028646043e-05, 7.395745025e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.3442815, 0.3426873, 0.6159933)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_mu*(1-expected_mu))),TOLERANCE_STRICT)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-66.299571),TOLERANCE_STRICT)
  })

}
