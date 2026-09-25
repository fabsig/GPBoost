context("GPModel_gaussian_process")

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  TOLERANCE_ITERATIVE <- 1E-1
  TOLERANCE_LOOSE <- 1E-2
  TOLERANCE_MEDIUM <- 1e-3
  TOLERANCE_STRICT <- 1E-5
  # Some of the optimization problems below are non-convex, and a different compiler or standard library
  # does not reproduce floating point arithmetic bit-wise. The tight tolerances therefore only hold on the
  # reference platform on which the expected values were calculated. 'relax_tolerance*()' of
  # helper-tolerances.R relaxes them elsewhere and reports once per test run which of the two is in force.
  # Covariance functions with a general (non-fixed) smoothness need 'std::cyl_bessel_k', which is a C++17
  # feature that is not provided by every standard library (in particular not by libc++, which is used by
  # clang on macOS and in the clang sanitizer containers of R-hub / CRAN)
  SKIP_BESSEL_COV_TESTS <- !gpboost:::has_std_cyl_bessel_k() &&
    Sys.getenv("GPBOOST_RUN_BESSEL_COV_TESTS") != "true"
  
  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov = "gradient_descent",
                               lr_cov = 0.1, use_nesterov_acc = TRUE,
                               acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                               optimizer_coef = "gradient_descent", lr_coef = 0.1,
                               convergence_criterion = "relative_change_in_log_likelihood",
                               cg_delta_conv = 1E-6, cg_preconditioner_type = "predictive_process_plus_diagonal",
                               cg_max_num_it = 1000, cg_max_num_it_tridiag = 1000,
                               num_rand_vec_trace = 1000, reuse_rand_vec_trace = TRUE,
                               init_coef_aux_pars_from_iid_model = FALSE)
  DEFAULT_OPTIM_PARAMS_FISHER <- list(optimizer_cov = "fisher_scoring", delta_rel_conv = 1E-6,
                                      optimizer_coef = "gradient_descent", lr_coef = 0.1,
                                      convergence_criterion = "relative_change_in_log_likelihood",
                                      cg_delta_conv = 1E-6, cg_preconditioner_type = "predictive_process_plus_diagonal",
                                      cg_max_num_it = 1000, cg_max_num_it_tridiag = 1000,
                                      num_rand_vec_trace = 1000, reuse_rand_vec_trace = TRUE,
                                      seed_rand_vec_trace = 1,
                                      init_coef_aux_pars_from_iid_model = FALSE)
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
  
  # Create data
  n <- 100 # number of samples
  # Simulate locations / features of GP
  d <- 2 # dimension of GP locations
  coords <- matrix(sim_rand_unif(n=n*d, init_c=0.1), ncol=d)
  D <- as.matrix(dist(coords))
  # Simulate GP
  sigma2_1 <- 1^2 # marginal variance of GP
  rho <- 0.1 # range parameter
  Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
  C <- t(chol(Sigma))
  b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.8))
  eps <- as.vector(C %*% b_1)
  # Random coefficients
  Z_SVC <- matrix(sim_rand_unif(n=n*2, init_c=0.6), ncol=2) # covariate data for random coeffients
  colnames(Z_SVC) <- c("var1","var2")
  b_2 <- qnorm(sim_rand_unif(n=n, init_c=0.17))
  b_3 <- qnorm(sim_rand_unif(n=n, init_c=0.42))
  eps_svc <- as.vector(C %*% b_1 + Z_SVC[,1] * C %*% b_2 + Z_SVC[,2] * C %*% b_3)
  # Error term
  xi <- qnorm(sim_rand_unif(n=n, init_c=0.1)) / 5
  # Data for linear mixed effects model
  X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
  beta <- c(2,2) # regression coefficients
  # cluster_ids 
  cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))
  # GP with multiple observations at the same locations
  coords_multiple <- matrix(sim_rand_unif(n=n*d/4, init_c=0.1), ncol=d)
  coords_multiple <- rbind(coords_multiple,coords_multiple,coords_multiple,coords_multiple)
  D_multiple <- as.matrix(dist(coords_multiple))
  Sigma_multiple <- sigma2_1*exp(-D_multiple/rho)+diag(1E-10,n)
  C_multiple <- t(chol(Sigma_multiple))
  b_multiple <- qnorm(sim_rand_unif(n=n, init_c=0.8))
  eps_multiple <- as.vector(C_multiple %*% b_multiple)
  
  test_that("Gaussian process model ", {
    y <- eps + xi
    params <- DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    
    # Evaluate negative log-likelihood
    cov_pars_eval_nll <- c(0.1,1.6,0.2)
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    nll_exp <- 124.2549533
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    expect_lt(abs(nll - nll_exp), TOLERANCE_STRICT)
    # Other covariance functions: Matern 0.5
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    expect_lt(abs(nll - nll_exp), TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 0.5 + 1E-6)
      nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
      expect_lt(abs(nll - nll_exp), TOLERANCE_STRICT)
      gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 0.5 - 1E-6)
      nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
      expect_lt(abs(nll - nll_exp), TOLERANCE_STRICT)
    }
    # Matern 1.5
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    nll_exp_mat <- 141.3502172
    expect_lt(abs(nll - nll_exp_mat), TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5 + 1E-6)
      nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
      expect_lt(abs(nll - nll_exp_mat), TOLERANCE_MEDIUM)
      gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5 - 1E-6)
      nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
      expect_lt(abs(nll - nll_exp_mat), TOLERANCE_MEDIUM)
    }
    # Matern 2.5
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 2.5)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    nll_exp_mat <- 158.1111626
    expect_lt(abs(nll - nll_exp_mat), TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 2.5 + 1E-6)
      nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
      expect_lt(abs(nll - nll_exp_mat), TOLERANCE_MEDIUM)
      gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 2.5 - 1E-6)
      nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
      expect_lt(abs(nll - nll_exp_mat), TOLERANCE_MEDIUM)
    }
    
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    capture.output( fit(gp_model, y = y, params = params), 
                    file='NUL')
    cov_pars <- c(0.03784221, 0.07943467, 1.07390943, 0.25351519, 0.11451432, 0.03840236)
    num_it <- 59
    nll_opt <- 122.7771373
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[2], 3)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Can switch between likelihoods
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood="gamma")
    gp_model$set_likelihood("gaussian")
    capture.output( fit(gp_model, y = y, params = params), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_STRICT)
    # Gradient descent without Nesterov acceleration
    params_no_acc <- params
    params_no_acc$use_nesterov_acc <- FALSE
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_no_acc) , file='NUL')
    cov_pars_other <- c(0.04040441, 0.08036674, 1.06926607, 0.25360131, 0.11502362, 0.03877014)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_other)),5E-6)
    expect_equal(gp_model$get_num_optim_iter(), 97)
    # Using a too large learning rate
    params_lr <- params
    params_lr$lr_cov <- 1
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_lr) , file='NUL')
    cov_pars_other <- c(0.03738147, 0.07929704, 1.07520000, 0.25359186, 0.11441031, 0.03833048)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_other)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 49)
    # Different terminations criterion
    params_loc <- params
    params_loc$convergence_criterion = "relative_change_in_parameters"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc), file='NUL')
    cov_pars_other_crit <- c(0.03276547, 0.07715343, 1.07617676, 0.25177603, 0.11352557, 0.03770062)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_other_crit)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    ll <- gp_model$neg_log_likelihood(y=y,cov_pars=gp_model$get_cov_pars(std_err = TRUE)[1,])
    expect_lt(abs(ll-122.7752664),TOLERANCE_STRICT)
    # Fisher scoring
    params_loc <- params
    params_loc$optimizer_cov = "fisher_scoring"
    params_loc$lr_cov <- 1
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc), file='NUL')
    cov_pars_fisher <- c(0.03294841, 0.07722844, 1.07591929, 0.25179816, 0.11355958, 0.03772550)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_fisher)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_LOOSE)
    # lbfgs
    params_loc$optimizer_cov = "lbfgs"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc)
                    , file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars(std_err = TRUE))
    expect_lt(sum(abs(cov_pars_est-cov_pars)),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_LOOSE)
    # nelder_mead
    params_loc$optimizer_cov = "nelder_mead"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc)
                    , file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars(std_err = TRUE))
    expect_lt(sum(abs(cov_pars_est-cov_pars)),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_LOOSE)
    # Test default values for delta_rel_conv for nelder_mead
    capture.output( gp_model_default <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                   y = y, params = list(optimizer_cov = "nelder_mead", init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    capture.output( gp_model_8 <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             y = y, params = list(optimizer_cov = "nelder_mead",
                                                                  delta_rel_conv=1e-8, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    expect_false(isTRUE(all.equal(gp_model_default$get_cov_pars(std_err = TRUE), gp_model$get_cov_pars(std_err = TRUE))))
    expect_true(isTRUE(all.equal(gp_model_default$get_cov_pars(std_err = TRUE), gp_model_8$get_cov_pars(std_err = TRUE))))
    # Test default values for delta_rel_conv for gradient_descent
    capture.output( gp_model_default <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                   y = y, params = list(optimizer_cov = "gradient_descent", init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    capture.output( gp_model_8 <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             y = y, params = list(optimizer_cov = "gradient_descent",
                                                                  delta_rel_conv=1e-8, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    capture.output( gp_model_6 <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             y = y, params = list(optimizer_cov = "gradient_descent",
                                                                  delta_rel_conv=1e-6, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    expect_true(isTRUE(all.equal(gp_model_default$get_cov_pars(std_err = TRUE), gp_model_6$get_cov_pars(std_err = TRUE))))
    expect_false(isTRUE(all.equal(gp_model_default$get_cov_pars(std_err = TRUE), gp_model_8$get_cov_pars(std_err = TRUE))))
    # lbfgs
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = list(optimizer_cov = "lbfgs", init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_LOOSE)
    # Adam
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = list(optimizer_cov = "adam", init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars(std_err = FALSE))
    expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    # Newton's method
    params_loc <- params
    params_loc$optimizer_cov = "newton"
    params_loc$lr_cov <- 1
    params_loc$use_nesterov_acc <- FALSE
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc), file='NUL')
    cov_pars_newton <- c(0.03282998, 0.07718279, 1.07612393, 0.25179124, 0.11353614, 0.03770875)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_newton)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 7)
    expect_lt(sum(abs(as.vector(gp_model$get_current_neg_log_likelihood())-nll_opt)),TOLERANCE_LOOSE)
    # fix some covariance parameters
    params_loc <- params
    params_loc$optimizer_cov = "lbfgs"
    params_loc$estimate_cov_par_index <- c(1,1,0)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc), file='NUL')
    nll_opt_fix <- 123.4853915
    cov_pars_fix <- c(0.10273152252, 0.08925506562, 1.23337072589, 0.37123039633, 0.17864807736, 0.07351705425)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_fix)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = TRUE)[1,3]-params_loc$init_cov_pars[3])),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_fix), TOLERANCE_STRICT)
    params_loc$estimate_cov_par_index <- c(1,0,0)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc), file='NUL')
    nll_opt_fix <- 126.5787898
    cov_pars_fix <- c(0.3386923228, 0.1199551188, 0.5170731356, 0.2048828394, 0.1786480774, 0.1038492119)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_fix)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = TRUE)[1,2:3]-params_loc$init_cov_pars[2:3])),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_fix), TOLERANCE_STRICT)
    params_loc$estimate_cov_par_index <- c(0,0,0)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc), file='NUL')
    nll_opt_fix <- 128.132446
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = TRUE)[1,1:3]-params_loc$init_cov_pars[1:3])),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_fix), TOLERANCE_STRICT)
    params_loc$estimate_cov_par_index <- c(0,1,0)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc), file='NUL')
    nll_opt_fix <- 127.9879294
    cov_pars_fix <- c(0.5170731356, 0.1687492120, 0.6088800134, 0.2602195062, 0.1786480774, 0.1112692786)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_fix)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = TRUE)[1,c(1,3)]-params_loc$init_cov_pars[c(1,3)])),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_fix), TOLERANCE_STRICT)
    
    # Prediction from fitted model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y,
                                           params = list(optimizer_cov = "fisher_scoring",
                                                         delta_rel_conv = 1E-6, use_nesterov_acc = FALSE,
                                                         convergence_criterion = "relative_change_in_parameters", init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    expect_error(predict(gp_model))# coord data not provided
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expected_mu <- c(0.06960478, 1.61299381, 0.44053480)
    expected_cov <- c(6.218737e-01, 2.024102e-05, 2.278875e-07, 2.024102e-05,
                      3.535390e-01, 8.479210e-07, 2.278875e-07, 8.479210e-07, 4.202154e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Prediction of variances only
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    preds <- predict(gp_model, gp_coords_pred = coords,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)), TOLERANCE_STRICT)
    
    # Prediction using given parameters
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    cov_pars_pred = c(0.02,1.2,0.9)
    
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    expected_mu <- c(0.08704577, 1.63875604, 0.48513581)
    expected_cov <- c(1.189093e-01, 1.171632e-05, -4.172444e-07, 1.171632e-05,
                      7.427727e-02, 1.492859e-06, -4.172444e-07, 1.492859e-06, 8.107455e-02)
    cov_no_nugget <- expected_cov
    cov_no_nugget[c(1,5,9)] <- expected_cov[c(1,5,9)] - cov_pars_pred[1]
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Prediction of variances only
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE, 
                    cov_pars = cov_pars_pred, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Predict latent process
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-cov_no_nugget)), TOLERANCE_STRICT)
    # Sampling from posterior
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = TRUE, 
                    cov_pars = cov_pars_pred, sample_posterior = TRUE, num_post_samples = 100000)
    Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
    tol_mu <- 0.003
    tol_cov <- 0.003
    expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-expected_mu)), tol_mu)
    expect_lt(sum(abs(as.vector(cov(t(pred$posterior_samples)))-expected_cov)), tol_cov)
    # Sampling from prior
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = TRUE,
                    cov_pars = c(1E-20,sigma2_1,rho), sample_prior = TRUE, num_prior_samples = 100000)
    tol_mean <- 0.003
    tol_cov <- 0.003
    expect_lt(mean(abs(apply(pred$prior_samples,1,mean)-rep(0,5))), tol_mean)
    cov_mat_prior <- cov(t(pred$prior))
    expect_lt(mean(abs(as.vector(cov_mat_prior[lower.tri(cov_mat_prior)])-as.vector(Sigma[lower.tri(Sigma)]))), tol_cov)
    
    # Do optimization using optim
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    capture.output( opt <- optim(par=c(0.1,2,0.2), fn=gp_model$neg_log_likelihood, 
                 y=y, method="L-BFGS-B", lower=1E-10) , file='NUL')
    expect_lt(sum(abs(opt$par-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_lt(abs(opt$value-(122.7752694)),1E-5)
    # expect_equal(as.integer(opt$counts[1]), 35)
    
    # Other covariance functions
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 0.5,
                                           y = y, params = params) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                             cov_fct_shape = 0.5 + 1e-6,
                                             y = y, params = params) , file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_STRICT)
      expect_equal(gp_model$get_num_optim_iter(), num_it)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
      pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                      cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
      expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
      expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    }
    # Matern 1.5
    init_cov_pars_15 <- c(var(y)/2,var(y)/2,mean(dist(coords))/4.7*sqrt(3))
    params_15 = DEFAULT_OPTIM_PARAMS
    params_15$init_cov_pars <- init_cov_pars_15
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 1.5, y = y, params = params_15) , file='NUL')
    cov_pars_other <- c(0.22926543, 0.08486055, 0.87886348, 0.24059253, 0.10726402, 0.02672378)
    num_it_other <- 16
    nll_opt_other <- 123.6388965
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_other)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                             cov_fct_shape = 1.5 - 1E-6, y = y, params = params_15) , file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_other)),TOLERANCE_STRICT)
      expect_equal(gp_model$get_num_optim_iter(), num_it_other)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_STRICT)
    }
    
    # Matern 2.5
    init_cov_pars_25 <- c(var(y)/2,var(y)/2,mean(dist(coords))/5.9*sqrt(5))
    params_25 = DEFAULT_OPTIM_PARAMS
    params_25$init_cov_pars <- init_cov_pars_25
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 2.5, y = y, params = params_25) , file='NUL')
    cov_pars_other <- c(0.27251105, 0.08316755, 0.83205621, 0.23561744, 0.10536460, 0.02375078)
    num_it_other <- 13
    nll_opt_other <- 123.9752771
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_other)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                             cov_fct_shape = 2.5 + 1E-3, y = y, params = params_25) , file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_other)),TOLERANCE_MEDIUM)
      expect_equal(gp_model$get_num_optim_iter(), num_it_other)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_MEDIUM)
    }
    # gaussian
    init_cov_pars_G <- c(var(y)/2,var(y)/2,sqrt((mean(dist(coords))/2)^2 / 3))
    params_G = DEFAULT_OPTIM_PARAMS
    params_G$init_cov_pars <- init_cov_pars_G
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "gaussian",
                                           cov_fct_shape = 2.5, y = y, params = params_G) , file='NUL')
    cov_pars_other <- c(0.33824439, 0.07955527, 0.75776861, 0.22661022, 0.14361521, 0.02589934)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    # Matern with shape estimated
    if (!SKIP_BESSEL_COV_TESTS) {
      params = OPTIM_PARAMS_BFGS
      params$init_cov_pars <- c(init_cov_pars_15, 1.5)
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern_estimate_shape",
                                             cov_fct_shape = 1.5, y = y, params = params) , file='NUL')
      cov_pars_other <- c(0.0001323589, 0.2018696019, 1.1022114804, 0.3153382101, 0.1187387358, 0.0512925409, 0.4181996520, 0.3579762498)
      num_it_other <- 23
      nll_opt_other <- 122.7099697
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_other)),TOLERANCE_STRICT)
      expect_equal(gp_model$get_num_optim_iter(), num_it_other)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_STRICT)
    }
    
    ## Test default initial values
    params <- list(optimizer_cov = "gradient_descent", maxit = 0, optimizer_coef = "gradient_descent", init_coef_aux_pars_from_iid_model = FALSE)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 0.5, y = y, params = params) , file='NUL')
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[1] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[2] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[3] - median(dist(coords))/3/2),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 1.5, y = y, params = params) , file='NUL')
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[1] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[2] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[3] - median(dist(coords))/4.7*sqrt(3)/2),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 2.5, y = y, params = params) , file='NUL')
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[1] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[2] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[3] - median(dist(coords))/5.9*sqrt(5)/2),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "gaussian",
                                           y = y, params = params) , file='NUL')
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[1] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[2] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[3] - sqrt((median(dist(coords))/2)^2 / 3)),TOLERANCE_STRICT)
    #non-Gaussian data
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", likelihood = "gamma",
                                           cov_fct_shape = 0.5, y = exp(y), params = params) , file='NUL')
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[1] - 1),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[2] - median(dist(coords))/3/2),TOLERANCE_STRICT)
    
  })
  
  test_that("Gaussian sample weights work for Gaussian processes ", {
    coords_w <- cbind(c(0.05, 0.18, 0.31, 0.52, 0.74, 0.91),
                      c(0.12, 0.44, 0.27, 0.83, 0.35, 0.66))
    y_w <- c(0.25, -0.40, 1.20, 0.75, -0.15, 1.45)
    weights_w <- c(1.0, 2.0, 3.0, 1.5, 0.7, 2.2)
    cov_pars_w <- c(0.45, 1.20, 0.35)
    D_w <- as.matrix(dist(coords_w))
    Sigma_w <- cov_pars_w[2] * exp(-D_w / cov_pars_w[3]) +
      cov_pars_w[1] * diag(1 / weights_w)
    chol_Sigma_w <- chol(Sigma_w)
    nll_w_manual <- 0.5 * drop(crossprod(y_w, solve(Sigma_w, y_w))) +
      sum(log(diag(chol_Sigma_w))) + length(y_w) / 2 * log(2 * pi)
    
    capture.output( gp_model_w <- GPModel(gp_coords = coords_w, cov_function = "exponential",
                                          weights = weights_w) , file='NUL')
    nll_w <- gp_model_w$neg_log_likelihood(cov_pars = cov_pars_w, y = y_w)
    expect_lt(abs(nll_w - nll_w_manual), TOLERANCE_STRICT)
    
    coords_pred_w <- cbind(c(0.16, 0.60, 0.88), c(0.20, 0.70, 0.40))
    D_pred_obs_w <- as.matrix(dist(rbind(coords_pred_w, coords_w)))[1:nrow(coords_pred_w),
                                                                    -(1:nrow(coords_pred_w))]
    D_pred_w <- as.matrix(dist(coords_pred_w))
    cross_cov_w <- cov_pars_w[2] * exp(-D_pred_obs_w / cov_pars_w[3])
    pred_cov_prior_w <- cov_pars_w[2] * exp(-D_pred_w / cov_pars_w[3]) +
      cov_pars_w[1] * diag(nrow(coords_pred_w))
    pred_mean_manual_w <- as.vector(cross_cov_w %*% solve(Sigma_w, y_w))
    pred_cov_manual_w <- pred_cov_prior_w - cross_cov_w %*% solve(Sigma_w, t(cross_cov_w))
    pred_w <- predict(gp_model_w, y = y_w, gp_coords_pred = coords_pred_w,
                      cov_pars = cov_pars_w, predict_response = TRUE,
                      predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred_w$mu - pred_mean_manual_w)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred_w$cov) - as.vector(pred_cov_manual_w))), TOLERANCE_STRICT)
    
    capture.output( gp_model_fitc_w <- GPModel(gp_coords = coords_w, cov_function = "exponential",
                                               gp_approx = "fitc", num_ind_points = nrow(coords_w),
                                               ind_points_selection = "random", weights = weights_w) , file='NUL')
    nll_fitc_w <- gp_model_fitc_w$neg_log_likelihood(cov_pars = cov_pars_w, y = y_w)
    expect_lt(abs(nll_fitc_w - nll_w_manual), TOLERANCE_STRICT)
    pred_fitc_w <- predict(gp_model_fitc_w, y = y_w, gp_coords_pred = coords_pred_w,
                           cov_pars = cov_pars_w, predict_response = TRUE,
                           predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred_fitc_w$mu - pred_mean_manual_w)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred_fitc_w$cov) - as.vector(pred_cov_manual_w))),
              0.05)
    
    capture.output( gp_model_vecchia_w <- GPModel(gp_coords = coords_w, cov_function = "exponential",
                                                  gp_approx = "vecchia", num_neighbors = nrow(coords_w) - 1,
                                                  vecchia_ordering = "none", weights = weights_w) , file='NUL')
    nll_vecchia_w <- gp_model_vecchia_w$neg_log_likelihood(cov_pars = cov_pars_w, y = y_w)
    expect_lt(abs(nll_vecchia_w - nll_w_manual), TOLERANCE_STRICT)
    gp_model_vecchia_w$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all",
                                           num_neighbors_pred = nrow(coords_w) + nrow(coords_pred_w) - 1)
    pred_vecchia_w <- predict(gp_model_vecchia_w, y = y_w, gp_coords_pred = coords_pred_w,
                              cov_pars = cov_pars_w, predict_response = TRUE,
                              predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred_vecchia_w$mu - pred_mean_manual_w)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred_vecchia_w$cov) - as.vector(pred_cov_manual_w))), TOLERANCE_STRICT)
    gp_model_vecchia_w$set_optim_params(params = list(num_rand_vec_trace = 5000,
                                                      seed_rand_vec_trace = 1,
                                                      reuse_rand_vec_trace = TRUE, init_coef_aux_pars_from_iid_model = FALSE))
    fit(gp_model_vecchia_w, y = y_w, params = list(maxit = 0,
                                                   init_cov_pars = cov_pars_w, init_coef_aux_pars_from_iid_model = FALSE))
    grad_cov_w <- list(diag(1 / weights_w), exp(-D_w / cov_pars_w[3]),
                       cov_pars_w[2] * exp(-D_w / cov_pars_w[3]) * D_w / cov_pars_w[3]^2)
    FI_w <- matrix(0, 3, 3)
    Sigma_inv_w <- solve(Sigma_w)
    for (ii in 1:3) {
      for (jj in ii:3) {
        FI_w[ii, jj] <- 0.5 * sum(diag(Sigma_inv_w %*% grad_cov_w[[ii]] %*%
                                         Sigma_inv_w %*% grad_cov_w[[jj]]))
        FI_w[jj, ii] <- FI_w[ii, jj]
      }
    }
    std_err_vecchia_w <- as.vector(gp_model_vecchia_w$get_cov_pars(std_err = TRUE)["Std. err.", ])
    std_err_manual_w <- c(2.253217120111, 1.775896266058, 0.621203477659)
    expect_lt(sum(abs(sqrt(diag(solve(FI_w))) - std_err_manual_w)), TOLERANCE_STRICT)
    expect_true(all(is.finite(std_err_vecchia_w)))
    
    X_w <- cbind(1, c(-1.0, -0.5, 0.2, 0.7, 1.1, -0.2))
    capture.output( gp_model_w_fit_X <- fitGPModel(gp_coords = coords_w,
                                                   cov_function = "exponential",
                                                   y = y_w, X = X_w,
                                                   weights = weights_w,
                                                   params = list(optimizer_cov = "lbfgs",
                                                                 optimizer_coef = "wls", init_coef_aux_pars_from_iid_model = FALSE)) , file='NUL')
    cov_pars_fit_X <- as.vector(gp_model_w_fit_X$get_cov_pars())
    beta_fit_X <- as.vector(gp_model_w_fit_X$get_coef())
    cov_pars <- c(6.86158387056e-06, 4.57441731777e-01, 1.09279322973e-03)
    coef <- c(0.5149274586446, 0.0348285983218)
    expect_lt(sum(abs(cov_pars_fit_X - cov_pars)), TOLERANCE_STRICT)
    expect_lt(sum(abs(beta_fit_X - coef)), TOLERANCE_STRICT)
    
    capture.output( gp_model_vecchia_fit_w <- fitGPModel(gp_coords = coords_w,
                                                         cov_function = "exponential",
                                                         gp_approx = "vecchia",
                                                         num_neighbors = nrow(coords_w) - 1,
                                                         vecchia_ordering = "none",
                                                         weights = weights_w,
                                                         y = y_w,
                                                         params = list(optimizer_cov = "gradient_descent",
                                                                       maxit = 1,
                                                                       init_cov_pars = cov_pars_w, init_coef_aux_pars_from_iid_model = FALSE)) , file='NUL')
    expect_true(is.finite(gp_model_vecchia_fit_w$get_current_neg_log_likelihood()))
    
    capture.output( gp_model_vif_w <- GPModel(gp_coords = coords_w, cov_function = "exponential",
                                              gp_approx = "vif", num_neighbors = 2,
                                              num_ind_points = 3,
                                              ind_points_selection = "random",
                                              weights = weights_w) , file='NUL')
    nll_vif_w <- gp_model_vif_w$neg_log_likelihood(cov_pars = cov_pars_w, y = y_w)
    expect_true(is.finite(nll_vif_w))
    capture.output( gp_model_vif_exact_w <- GPModel(gp_coords = coords_w,
                                                    cov_function = "exponential",
                                                    gp_approx = "vif",
                                                    num_neighbors = nrow(coords_w) - 1,
                                                    num_ind_points = 3,
                                                    ind_points_selection = "random",
                                                    vecchia_ordering = "none",
                                                    weights = weights_w) , file='NUL')
    nll_vif_exact_w <- gp_model_vif_exact_w$neg_log_likelihood(cov_pars = cov_pars_w,
                                                               y = y_w)
    expect_lt(abs(nll_vif_exact_w - nll_w_manual), TOLERANCE_STRICT)
    gp_model_vif_exact_w$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all",
                                             num_neighbors_pred = nrow(coords_w) + nrow(coords_pred_w) - 1)
    pred_vif_exact_w <- predict(gp_model_vif_exact_w, y = y_w,
                                gp_coords_pred = coords_pred_w,
                                cov_pars = cov_pars_w, predict_response = TRUE,
                                predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred_vif_exact_w$mu - pred_mean_manual_w)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred_vif_exact_w$cov) - as.vector(pred_cov_manual_w))),
              TOLERANCE_STRICT)
    capture.output( gp_model_vif_fit_w <- fitGPModel(gp_coords = coords_w,
                                                     cov_function = "exponential",
                                                     gp_approx = "vif",
                                                     num_neighbors = 2,
                                                     num_ind_points = 3,
                                                     ind_points_selection = "random",
                                                     weights = weights_w,
                                                     y = y_w,
                                                     params = list(optimizer_cov = "gradient_descent",
                                                                   maxit = 1,
                                                                   init_cov_pars = cov_pars_w, init_coef_aux_pars_from_iid_model = FALSE)) , file='NUL')
    expect_true(is.finite(gp_model_vif_fit_w$get_current_neg_log_likelihood()))
    
    capture.output( gp_model_fst_w <- GPModel(gp_coords = coords_w, cov_function = "exponential",
                                              gp_approx = "full_scale_tapering",
                                              num_ind_points = 3,
                                              ind_points_selection = "random",
                                              weights = weights_w) , file='NUL')
    nll_fst_w <- gp_model_fst_w$neg_log_likelihood(cov_pars = cov_pars_w, y = y_w)
    expect_true(is.finite(nll_fst_w))
    capture.output( gp_model_fst_exact_w <- GPModel(gp_coords = coords_w,
                                                    cov_function = "exponential",
                                                    gp_approx = "full_scale_tapering",
                                                    num_ind_points = 3,
                                                    ind_points_selection = "random",
                                                    cov_fct_taper_range = 1e6,
                                                    cov_fct_taper_shape = 2,
                                                    weights = weights_w,
                                                    matrix_inversion_method = "cholesky") , file='NUL')
    nll_fst_exact_w <- gp_model_fst_exact_w$neg_log_likelihood(cov_pars = cov_pars_w,
                                                               y = y_w)
    expect_lt(abs(nll_fst_exact_w - nll_w_manual), TOLERANCE_STRICT)
    pred_fst_exact_w <- predict(gp_model_fst_exact_w, y = y_w,
                                gp_coords_pred = coords_pred_w,
                                cov_pars = cov_pars_w, predict_response = TRUE,
                                predict_cov_mat = FALSE)
    expect_lt(sum(abs(pred_fst_exact_w$mu - pred_mean_manual_w)), TOLERANCE_STRICT)
    
    # with fixed effects
    coords_linear_w <- cbind(1, c(-1.0, -0.3, 0.1, 0.4, 0.9, 1.3))
    cov_pars_linear_w <- c(0.45, 0.8)
    Sigma_linear_w <- cov_pars_linear_w[2] * tcrossprod(coords_linear_w) +
      cov_pars_linear_w[1] * diag(1 / weights_w)
    chol_Sigma_linear_w <- chol(Sigma_linear_w)
    nll_linear_manual_w <- 0.5 * drop(crossprod(y_w, solve(Sigma_linear_w, y_w))) +
      sum(log(diag(chol_Sigma_linear_w))) + length(y_w) / 2 * log(2 * pi)
    capture.output( gp_model_linear_w <- GPModel(gp_coords = coords_linear_w, cov_function = "linear",
                                                 weights = weights_w) , file='NUL')
    nll_linear_w <- gp_model_linear_w$neg_log_likelihood(cov_pars = cov_pars_linear_w,
                                                         y = y_w)
    expect_lt(abs(nll_linear_w - nll_linear_manual_w), TOLERANCE_STRICT)
    
    coords_dup_w <- cbind(c(0.05, 0.05, 0.31, 0.52, 0.52, 0.91),
                          c(0.12, 0.12, 0.27, 0.83, 0.83, 0.66))
    y_dup_w <- c(0.25, -0.40, 1.20, 0.75, -0.15, 1.45)
    weights_dup_w <- c(1.0, 2.0, 3.0, 1.5, 0.7, 2.2)
    D_dup_w <- as.matrix(dist(coords_dup_w))
    Sigma_dup_w <- cov_pars_w[2] * exp(-D_dup_w / cov_pars_w[3]) +
      cov_pars_w[1] * diag(1 / weights_dup_w)
    chol_Sigma_dup_w <- chol(Sigma_dup_w)
    nll_dup_manual_w <- 0.5 * drop(crossprod(y_dup_w, solve(Sigma_dup_w, y_dup_w))) +
      sum(log(diag(chol_Sigma_dup_w))) + length(y_dup_w) / 2 * log(2 * pi)
    
    capture.output( gp_model_dup_w <- GPModel(gp_coords = coords_dup_w, cov_function = "exponential",
                                              weights = weights_dup_w) , file='NUL')
    nll_dup_w <- gp_model_dup_w$neg_log_likelihood(cov_pars = cov_pars_w, y = y_dup_w)
    expect_lt(abs(nll_dup_w - nll_dup_manual_w), TOLERANCE_STRICT)
    
    capture.output( gp_model_vecchia_dup_w <- GPModel(gp_coords = coords_dup_w, cov_function = "exponential",
                                                      gp_approx = "vecchia",
                                                      num_neighbors = nrow(coords_dup_w) - 1,
                                                      vecchia_ordering = "none",
                                                      weights = weights_dup_w) , file='NUL')
    nll_vecchia_dup_w <- gp_model_vecchia_dup_w$neg_log_likelihood(cov_pars = cov_pars_w,
                                                                   y = y_dup_w)
    expect_lt(abs(nll_vecchia_dup_w - nll_dup_manual_w), TOLERANCE_STRICT)
    coords_pred_dup_w <- cbind(c(0.05, 0.60, 0.91), c(0.12, 0.70, 0.66))
    gp_model_vecchia_dup_w$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all",
                                               num_neighbors_pred = nrow(coords_dup_w) + nrow(coords_pred_dup_w) - 1)
    D_pred_obs_dup_w <- as.matrix(dist(rbind(coords_pred_dup_w, coords_dup_w)))[1:nrow(coords_pred_dup_w),
                                                                                -(1:nrow(coords_pred_dup_w))]
    D_pred_dup_w <- as.matrix(dist(coords_pred_dup_w))
    cross_cov_dup_w <- cov_pars_w[2] * exp(-D_pred_obs_dup_w / cov_pars_w[3])
    pred_cov_prior_dup_w <- cov_pars_w[2] * exp(-D_pred_dup_w / cov_pars_w[3]) +
      cov_pars_w[1] * diag(nrow(coords_pred_dup_w))
    pred_mean_manual_dup_w <- as.vector(cross_cov_dup_w %*% solve(Sigma_dup_w, y_dup_w))
    pred_cov_manual_dup_w <- pred_cov_prior_dup_w -
      cross_cov_dup_w %*% solve(Sigma_dup_w, t(cross_cov_dup_w))
    pred_vecchia_dup_w <- predict(gp_model_vecchia_dup_w, y = y_dup_w,
                                  gp_coords_pred = coords_pred_dup_w,
                                  cov_pars = cov_pars_w, predict_response = TRUE,
                                  predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred_vecchia_dup_w$mu - pred_mean_manual_dup_w)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred_vecchia_dup_w$cov) - as.vector(pred_cov_manual_dup_w))), TOLERANCE_STRICT)
    gp_model_vecchia_dup_w$set_optim_params(params = list(num_rand_vec_trace = 5000,
                                                          seed_rand_vec_trace = 1,
                                                          reuse_rand_vec_trace = TRUE, init_coef_aux_pars_from_iid_model = FALSE))
    fit(gp_model_vecchia_dup_w, y = y_dup_w, params = list(maxit = 0,
                                                           init_cov_pars = cov_pars_w, init_coef_aux_pars_from_iid_model = FALSE))
    grad_cov_dup_w <- list(diag(1 / weights_dup_w), exp(-D_dup_w / cov_pars_w[3]),
                           cov_pars_w[2] * exp(-D_dup_w / cov_pars_w[3]) * D_dup_w / cov_pars_w[3]^2)
    FI_dup_w <- matrix(0, 3, 3)
    Sigma_inv_dup_w <- solve(Sigma_dup_w)
    for (ii in 1:3) {
      for (jj in ii:3) {
        FI_dup_w[ii, jj] <- 0.5 * sum(diag(Sigma_inv_dup_w %*% grad_cov_dup_w[[ii]] %*%
                                             Sigma_inv_dup_w %*% grad_cov_dup_w[[jj]]))
        FI_dup_w[jj, ii] <- FI_dup_w[ii, jj]
      }
    }
    std_err_vecchia_dup_w <- as.vector(gp_model_vecchia_dup_w$get_cov_pars(std_err = TRUE)["Std. err.", ])
    std_err_manual_dup_w <- c(0.449734084032, 1.047062462493, 0.608427345960)
    expect_lt(sum(abs(sqrt(diag(solve(FI_dup_w))) - std_err_manual_dup_w)), TOLERANCE_STRICT)
    expect_true(all(is.finite(std_err_vecchia_dup_w)))
    
    capture.output( gp_model_fitc_dup_w <- GPModel(gp_coords = coords_dup_w,
                                                   cov_function = "exponential",
                                                   gp_approx = "fitc",
                                                   num_ind_points = 3,
                                                   ind_points_selection = "random",
                                                   weights = weights_dup_w) , file='NUL')
    expect_lt(abs(gp_model_fitc_dup_w$neg_log_likelihood(cov_pars = cov_pars_w,
                                                         y = y_dup_w) - nll_dup_manual_w),
              TOLERANCE_STRICT)
    
    capture.output( gp_model_full_scale_vecchia_dup_w <- GPModel(gp_coords = coords_dup_w,
                                                                 cov_function = "exponential",
                                                                 gp_approx = "full_scale_vecchia",
                                                                 num_neighbors = 2,
                                                                 num_ind_points = 3,
                                                                 ind_points_selection = "random",
                                                                 weights = weights_dup_w) , file='NUL')
    expect_true(is.finite(gp_model_full_scale_vecchia_dup_w$neg_log_likelihood(cov_pars = cov_pars_w,
                                                                               y = y_dup_w)))
    capture.output( gp_model_full_scale_vecchia_exact_dup_w <- GPModel(gp_coords = coords_dup_w,
                                                                       cov_function = "exponential",
                                                                       gp_approx = "full_scale_vecchia",
                                                                       num_neighbors = nrow(coords_dup_w) - 1,
                                                                       num_ind_points = 3,
                                                                       ind_points_selection = "random",
                                                                       vecchia_ordering = "none",
                                                                       weights = weights_dup_w) , file='NUL')
    expect_lt(abs(gp_model_full_scale_vecchia_exact_dup_w$neg_log_likelihood(cov_pars = cov_pars_w,
                                                                             y = y_dup_w) -
                    nll_dup_manual_w), TOLERANCE_STRICT)
    
    capture.output( gp_model_vif_dup_w <- GPModel(gp_coords = coords_dup_w,
                                                  cov_function = "exponential",
                                                  gp_approx = "vif",
                                                  num_neighbors = 2,
                                                  num_ind_points = 3,
                                                  ind_points_selection = "random",
                                                  weights = weights_dup_w) , file='NUL')
    expect_true(is.finite(gp_model_vif_dup_w$neg_log_likelihood(cov_pars = cov_pars_w,
                                                                y = y_dup_w)))
    capture.output( gp_model_vif_exact_dup_w <- GPModel(gp_coords = coords_dup_w,
                                                        cov_function = "exponential",
                                                        gp_approx = "vif",
                                                        num_neighbors = nrow(coords_dup_w) - 1,
                                                        num_ind_points = 3,
                                                        ind_points_selection = "random",
                                                        vecchia_ordering = "none",
                                                        weights = weights_dup_w) , file='NUL')
    expect_lt(abs(gp_model_vif_exact_dup_w$neg_log_likelihood(cov_pars = cov_pars_w,
                                                              y = y_dup_w) -
                    nll_dup_manual_w), TOLERANCE_STRICT)
    
    capture.output( gp_model_fst_exact_dup_w <- GPModel(gp_coords = coords_dup_w,
                                                        cov_function = "exponential",
                                                        gp_approx = "full_scale_tapering",
                                                        num_ind_points = 3,
                                                        ind_points_selection = "random",
                                                        cov_fct_taper_range = 1e6,
                                                        cov_fct_taper_shape = 2,
                                                        weights = weights_dup_w,
                                                        matrix_inversion_method = "cholesky") , file='NUL')
    expect_lt(abs(gp_model_fst_exact_dup_w$neg_log_likelihood(cov_pars = cov_pars_w,
                                                              y = y_dup_w) -
                    nll_dup_manual_w), TOLERANCE_STRICT)
  })
  
  test_that("Gaussian process model with linear regression term ", {
    y <- eps + X%*%beta + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- init_cov_pars
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X = X,
                           params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls",
                                         delta_rel_conv = 1E-6, use_nesterov_acc = FALSE,
                                         convergence_criterion = "relative_change_in_parameters", init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.008461342, 0.069973492, 1.001562822, 0.214358560, 0.094656409, 0.029400407)
    coef <- c(2.30780026, 0.21365770, 1.89951426, 0.09484768)
    nll <- 121.482402
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)), TOLERANCE_STRICT)
    expect_lt(sum(abs( as.vector(gp_model$get_coef(std_err = TRUE))-coef)), TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    # Prediction 
    coord_test <- cbind(c(0.1,0.2,0.201),c(0.9,0.4,0.401))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(1.196952, 4.063324, 4.446861)
    expected_cov <- c(6.305383e-01, 1.358861e-05, 1.414550e-05, 1.358861e-05, 
                      3.469270e-01, 3.282926e-01, 1.414550e-05, 3.282926e-01, 3.561731e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    cov_pars_est <- gp_model$get_cov_pars()
    expected_cov_lat <- expected_cov
    expected_cov_lat[c(1,5,9)] <- expected_cov_lat[c(1,5,9)] - cov_pars_est[1]
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_lat)), TOLERANCE_STRICT)
    # Sampling from posterior
    pred <- predict(gp_model, gp_coords_pred = coord_test, X_pred = X_test, predict_response = TRUE, 
                    sample_posterior = TRUE, num_post_samples=100000)
    tol_mu <- 0.01
    tol_cov <- 0.03
    expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-expected_mu)), tol_mu)
    expect_lt(sum(abs(as.vector(cov(t(pred$posterior_samples)))-expected_cov)), tol_cov)
    pred <- predict(gp_model, gp_coords_pred = coord_test, X_pred = X_test, predict_response = FALSE, 
                    sample_posterior = TRUE, num_post_samples=100000)
    expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-expected_mu)), tol_mu)
    expect_lt(sum(abs(as.vector(cov(t(pred$posterior_samples)))-expected_cov_lat)), tol_cov)
    # Sampling and covariance jointly
    pred <- predict(gp_model, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_cov_mat = TRUE, sample_posterior = TRUE, num_post_samples=100000)
    expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-expected_mu)), tol_mu)
    expect_lt(sum(abs(as.vector(cov(t(pred$posterior_samples)))-expected_cov)), tol_cov)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Sampling and predictive variance
    pred <- predict(gp_model, gp_coords_pred = coord_test, X_pred = X_test, 
                    predict_var = TRUE, sample_posterior = TRUE, num_post_samples=100000)
    expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-expected_mu)), tol_mu)
    expect_lt(sum(abs(as.vector(cov(t(pred$posterior_samples)))-expected_cov)), tol_cov)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    
    # Gradient descent
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y=y, X = X, params = params)
    cov_pars <- c(0.01621846, 0.99717680, 0.09616230)
    coef <- c(2.305529, 1.899208)
    nll <- 121.4886075
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)), TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 100)
    
    # Nelder-Mead
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X = X, params = list(optimizer_cov = "nelder_mead",
                                                       optimizer_coef = "nelder_mead", 
                                                       maxit=1000, delta_rel_conv = 1e-12, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.008459373, 1.001564796, 0.094655964)
    coef <- c(2.307798, 1.899516)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)), TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 429)
    # lbfgs
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X = X, params = list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", 
                                                        maxit=1000, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.008993586382, 1.000518636089, 0.094683724304)
    coef <- c(2.309738418, 1.899886232)
    nll <- 121.4824924
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 15)
    # lbfgs wit wls for coefficients
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X = X, params = list(optimizer_cov = "lbfgs", maxit=1000, optimizer_coef ="wls", 
                                                        init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    coef <- c(2.307912121, 1.899505576)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    
    # Using init_coef and init_cov_pars
    params <- OPTIM_PARAMS_BFGS
    params$maxit <- 0
    params$init_coef <- c(-1,3)
    params$init_cov_pars <- c(0.5,0.5,0.5)
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", y = y, X = X, params = params)
    cov_pars <- c(0.41656029569, 0.19599141062, 0.48878570609, 0.23430447322, 0.09823632211, 0.05904809303)
    coef <- c(2.2645950057, 0.1696021494, 2.2701341181, 0.1223768465)
    nll <- 191.1725919
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-params$init_cov_pars)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-params$init_coef)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 0)
    
    # Continue training
    params <- list(optimizer_cov = "gradient_descent", lr_cov=0.01, maxit = 2, init_coef_aux_pars_from_iid_model = FALSE)
    invisible(capture.output( gp_model_2 <- fitGPModel(gp_coords = coords, cov_function = "exponential", y = y, X = X, params = params) ))
    expect_equal(gp_model_2$get_num_optim_iter(), 2)
    params$maxit <- 1
    invisible(capture.output( gp_model_1_1 <- fitGPModel(gp_coords = coords, cov_function = "exponential", y = y, X = X, params = params) ))
    invisible(capture.output( fit(gp_model_1_1, y = y, X = X, params = params) ))
    expect_lt(sum(abs(as.vector(gp_model_1_1$get_cov_pars(std_err = TRUE))-as.vector(gp_model_2$get_cov_pars(std_err = TRUE)))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_1_1$get_coef(std_err = TRUE))-as.vector(gp_model_2$get_coef(std_err = TRUE)))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_1_1$get_current_neg_log_likelihood() - gp_model_2$get_current_neg_log_likelihood()), TOLERANCE_MEDIUM)
    expect_equal(gp_model_1_1$get_num_optim_iter(), 1)
  })
  
  test_that("Gaussian process and two random coefficients ", {
    
    y <- eps_svc + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3,var(y)/2,mean(dist(coords))/3,var(y)/2,mean(dist(coords))/3)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_rand_coef_data = Z_SVC, y = y,
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, maxit=10, trace=TRUE, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expected_values <- c(0.25740068, 0.22608704, 0.83503539, 0.41896403, 0.15039055,
                         0.10090869, 1.61010233, 0.84207763, 0.09015444, 0.07106099, 
                         0.25064640, 0.62279880, 0.08720822, 0.32047865)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-expected_values)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Predict training data random effects
    cov_pars <- gp_model$get_cov_pars(std_err = TRUE)[1,]
    training_data_random_effects <- predict_training_data_random_effects(gp_model)
    Z_SVC_test <- cbind(rep(0,length(y)),rep(0,length(y)))
    preds <- predict(gp_model, gp_coords_pred = coords,
                     gp_rand_coef_data_pred=Z_SVC_test,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)), TOLERANCE_STRICT)
    Z_SVC_test <- cbind(rep(1,length(y)),rep(0,length(y)))
    preds2 <- predict(gp_model, gp_coords_pred = coords,
                      gp_rand_coef_data_pred=Z_SVC_test,
                      predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,2] - (preds2$mu - preds$mu))), TOLERANCE_STRICT)
    Z_SVC_test <- cbind(rep(0,length(y)),rep(1,length(y)))
    preds3 <- predict(gp_model, gp_coords_pred = coords,
                      gp_rand_coef_data_pred=Z_SVC_test,
                      predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,3] - (preds3$mu - preds$mu))), TOLERANCE_STRICT)
    
    # Prediction
    gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential")
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    expect_error(gp_model$predict(y = y, gp_coords_pred = coord_test,
                                  cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08)))# random slope data not provided
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
    expected_mu <- c(-0.1669209, 1.6166381, 0.2861320)
    expected_cov <- c(9.643323e-01, 3.536846e-04, -1.783557e-04, 3.536846e-04,
                      5.155009e-01, 4.554321e-07, -1.783557e-04, 4.554321e-07, 7.701614e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Predict variances
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_rand_coef_data = Z_SVC, y = y,
                                           params = list(optimizer_cov = "fisher_scoring",
                                                         use_nesterov_acc= FALSE, maxit=5, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expected_values <- c(9.948069e-06, 2.133237e-01, 1.398126e+00, 5.103201e-01, 1.535385e-01, 7.508804e-02, 1.758062e+00, 7.926720e-01, 3.919317e-02, 
                         3.867593e-02, 3.140238e-01, 6.211919e-01, 2.657551e+00, 1.713120e+01)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-expected_values)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-149.4422184),1E-5)
  })
  
  test_that("Random coefficient Gaussian processes with covariance functions that do not use distances ", {

    # Covariance functions which are not isotropic do not save distances but use the coordinates.
    #   Random coefficient GPs thus need the coordinates of the corresponding intercept GP
    y <- eps_svc + xi
    # An ARD covariance function with equal range parameters is the same as the isotropic version
    gp_model_iso <- GPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC)
    gp_model_ard <- GPModel(gp_coords = coords, cov_function = "exponential_ard", gp_rand_coef_data = Z_SVC)
    nll_iso <- gp_model_iso$neg_log_likelihood(cov_pars = c(0.5, 1.2, 0.2, 0.8, 0.3, 1.1, 0.15), y = y)
    nll_ard <- gp_model_ard$neg_log_likelihood(cov_pars = c(0.5, 1.2, 0.2, 0.2, 0.8, 0.3, 0.3, 1.1, 0.15, 0.15), y = y)
    expect_lt(abs(nll_iso - nll_ard), TOLERANCE_STRICT)
    # Estimation works for such covariance functions
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern_ard", cov_fct_shape = 1.5,
                                           gp_rand_coef_data = Z_SVC, y = y,
                                           params = list(maxit = 2, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expect_equal(length(gp_model$get_cov_pars()), 10)

  })

  test_that("Names of covariance parameters for random coefficient Gaussian processes ", {

    # The names of a random coefficient GP are the name of the random coefficient followed by the
    #   same suffixes as the ones of the corresponding intercept GP. They are checked here on the
    #   model object itself since no estimation is required for this
    Z_SVC_no_names <- Z_SVC
    colnames(Z_SVC_no_names) <- NULL
    model_names <- function(cov_function, gp_rand_coef_data, ...) {
      capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = cov_function,
                                          gp_rand_coef_data = gp_rand_coef_data, ...), file='NUL')
      gp_model$.__enclos_env__$private
    }

    # Covariate data for the random coefficients without column names
    private_model <- model_names("exponential", Z_SVC_no_names)
    expect_equal(private_model$cov_par_names,
                 c("Error_var", "GP_var", "GP_range",
                   "GP_rand_coef_nb_1_var", "GP_rand_coef_nb_1_range",
                   "GP_rand_coef_nb_2_var", "GP_rand_coef_nb_2_range"))
    expect_equal(private_model$re_comp_names,
                 c("GP", "GP_rand_coef_nb_1", "GP_rand_coef_nb_2"))
    # Column names of the covariate data are used if provided
    private_model <- model_names("exponential", Z_SVC)
    expect_equal(private_model$cov_par_names,
                 c("Error_var", "GP_var", "GP_range",
                   "GP_rand_coef_var1_var", "GP_rand_coef_var1_range",
                   "GP_rand_coef_var2_var", "GP_rand_coef_var2_range"))
    expect_equal(private_model$re_comp_names,
                 c("GP", "GP_rand_coef_var1", "GP_rand_coef_var2"))
    # ARD covariance functions: one range parameter per input dimension
    expect_equal(model_names("matern_ard", Z_SVC, cov_fct_shape = 1.5)$cov_par_names,
                 c("Error_var", "GP_var", "GP_range_1", "GP_range_2",
                   "GP_rand_coef_var1_var", "GP_rand_coef_var1_range_1", "GP_rand_coef_var1_range_2",
                   "GP_rand_coef_var2_var", "GP_rand_coef_var2_range_1", "GP_rand_coef_var2_range_2"))
    expect_equal(model_names("exponential_ard", Z_SVC_no_names)$cov_par_names,
                 c("Error_var", "GP_var", "GP_range_1", "GP_range_2",
                   "GP_rand_coef_nb_1_var", "GP_rand_coef_nb_1_range_1", "GP_rand_coef_nb_1_range_2",
                   "GP_rand_coef_nb_2_var", "GP_rand_coef_nb_2_range_1", "GP_rand_coef_nb_2_range_2"))
    # Covariance functions with an estimated smoothness parameter. Note: these require 'std::cyl_bessel_k'
    #   already when the model is created, i.e. also without any estimation
    if (!SKIP_BESSEL_COV_TESTS) {
      expect_equal(model_names("matern_estimate_shape", Z_SVC)$cov_par_names,
                   c("Error_var", "GP_var", "GP_range", "GP_smoothness",
                     "GP_rand_coef_var1_var", "GP_rand_coef_var1_range", "GP_rand_coef_var1_smoothness",
                     "GP_rand_coef_var2_var", "GP_rand_coef_var2_range", "GP_rand_coef_var2_smoothness"))
      expect_equal(model_names("matern_ard_estimate_shape", Z_SVC)$cov_par_names,
                   c("Error_var", "GP_var", "GP_range_1", "GP_range_2", "GP_smoothness",
                     "GP_rand_coef_var1_var", "GP_rand_coef_var1_range_1", "GP_rand_coef_var1_range_2", "GP_rand_coef_var1_smoothness",
                     "GP_rand_coef_var2_var", "GP_rand_coef_var2_range_1", "GP_rand_coef_var2_range_2", "GP_rand_coef_var2_smoothness"))
    }
    # Hurst covariance functions
    expect_equal(model_names("hurst", Z_SVC)$cov_par_names,
                 c("Error_var", "GP_var", "H",
                   "GP_rand_coef_var1_var", "GP_rand_coef_var1_H",
                   "GP_rand_coef_var2_var", "GP_rand_coef_var2_H"))
    expect_equal(model_names("hurst_ard", Z_SVC)$cov_par_names,
                 c("Error_var", "GP_var", "H", "GP_range_2",
                   "GP_rand_coef_var1_var", "GP_rand_coef_var1_H", "GP_rand_coef_var1_range_2",
                   "GP_rand_coef_var2_var", "GP_rand_coef_var2_H", "GP_rand_coef_var2_range_2"))
    # Space-time covariance functions
    expect_equal(model_names("matern_space_time", Z_SVC, cov_fct_shape = 1.5)$cov_par_names,
                 c("Error_var", "GP_var", "GP_range_time", "GP_range_space",
                   "GP_rand_coef_var1_var", "GP_rand_coef_var1_range_time", "GP_rand_coef_var1_range_space",
                   "GP_rand_coef_var2_var", "GP_rand_coef_var2_range_time", "GP_rand_coef_var2_range_space"))
    # Covariance functions with only a variance parameter
    expect_equal(model_names("wendland", Z_SVC, cov_fct_taper_range = 0.5)$cov_par_names,
                 c("Error_var", "GP_var", "GP_rand_coef_var1_var", "GP_rand_coef_var2_var"))
    # Non-Gaussian likelihood: there is no "Error_var" parameter
    expect_equal(model_names("exponential", Z_SVC, likelihood = "bernoulli_probit")$cov_par_names,
                 c("GP_var", "GP_range",
                   "GP_rand_coef_var1_var", "GP_rand_coef_var1_range",
                   "GP_rand_coef_var2_var", "GP_rand_coef_var2_range"))

  })

  test_that("Gaussian process model with cluster_id's not constant ", {
    
    y <- eps + xi
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, cluster_ids = cluster_ids,
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters", init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(0.05414149, 0.08722111, 1.05789166, 0.22886740, 0.12702368, 0.04076914)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 247)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, cluster_ids = cluster_ids,
                                           params = list(optimizer_cov = "fisher_scoring",
                                                         use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters", init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(0.05414149, 0.08722111, 1.05789166, 0.22886740, 0.12702368, 0.04076914)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),1E-5)
    expect_equal(gp_model$get_num_optim_iter(), 20)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    cluster_ids_pred = c(1,3,1)
    cov_pars_pred = c(0.1,1,0.15)
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        cluster_ids = cluster_ids)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    expected_mu <- c(-0.01437506, 0.00000000, 0.93112902)
    expected_cov <- c(0.743055189, 0.000000000, -0.000140644, 0.000000000,
                      1.100000000, 0.000000000, -0.000140644, 0.000000000, 0.565243468)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Predict variances
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = cov_pars_pred, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Sampling from posterior
    cluster_ids_pred2 <- c(1,1,2)
    pred <- predict(gp_model, gp_coords_pred = coord_test, cov_pars = cov_pars_pred,
                    cluster_ids_pred = cluster_ids_pred2, sample_posterior = TRUE, 
                    num_post_samples=100000, predict_cov_mat = TRUE)
    expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-pred$mu)), 0.01)
    expect_lt(sum(abs(cov(t(pred$posterior_samples))-pred$cov)), 0.02)
  })
  
  test_that("Gaussian process model with multiple observations at the same location ", {
    
    y <- eps_multiple + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(unique(coords_multiple)))/3)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- init_cov_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", y = y,
                                           params = params), file='NUL')
    cov_pars <- c(0.037168482, 0.006069406, 1.168105814, 0.445122816, 0.196226850, 0.105105379)
    num_it <- 6
    nll <- 33.43686607
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    # With full_scale_tapering
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", y = y, matrix_inversion_method = "cholesky",
                                           params = params, gp_approx = "full_scale_tapering", num_ind_points  = 25), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5)]-cov_pars[c(1,3,5)])), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", y = y,
                                           params = list(optimizer_cov = "fisher_scoring",
                                                         use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters",
                                                         init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(0.037136462, 0.006064181, 1.153630335, 0.435788570, 0.192080613, 0.102631006)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),1E-5)
    expect_equal(gp_model$get_num_optim_iter(), 15)
    
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    preds <- predict(gp_model, gp_coords_pred = coords_multiple,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)), TOLERANCE_STRICT)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential")
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE)
    expected_mu <- c(-0.1460550, 1.0042814, 0.7840301)
    expected_cov <- c(0.6739502109, 0.0008824337, -0.0003815281, 0.0008824337,
                      0.6060039551, -0.0004157361, -0.0003815281, -0.0004157361, 0.7851787946)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Predict variances
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cov_pars = c(0.1,1,0.15), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Sampling from posterior
    pred <- predict(gp_model, gp_coords_pred = coord_test, cov_pars = c(0.1,1,0.15),
                    sample_posterior = TRUE, num_post_samples=100000, predict_cov_mat = TRUE)
    expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-pred$mu)), 0.01)
    expect_lt(sum(abs(cov(t(pred$posterior_samples))-pred$cov)), 0.02)
  })
  
}
