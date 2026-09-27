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
  
  test_that("Wendland covariance function for Gaussian process model ", {
    
    y <- eps + xi
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "wendland", 
                                        cov_fct_taper_shape = 0, cov_fct_taper_range = 0.1), file='NUL')
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent",
                                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5, init_coef_aux_pars_from_iid_model = FALSE)) , file='NUL')
    cov_pars <- c(0.002911765, 0.116338096, 0.993996193, 0.211276385)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 280)
    # Prediction using given parameters
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "wendland", 
                                        cov_fct_taper_shape = 1, cov_fct_taper_range = 2), 
                    file='NUL')
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_cov_mat = TRUE)
    expected_mu <- c(-0.008405567, 1.493836307, 0.720565199)
    expected_cov <- c(2.933992e-02, 2.223241e-06, 1.352544e-05, 2.223241e-06, 2.496193e-02,
                      1.130906e-05, 1.352544e-05, 1.130906e-05, 2.405649e-02)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Prediction of variances only
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 0.1, y = y,
                                           params = list(optimizer_cov = "fisher_scoring",
                                                         use_nesterov_acc = FALSE,
                                                         delta_rel_conv = 1E-6, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(4.941224e-09, 1.497464e-01, 1.302468e+00, 2.746096e-01)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.02,1.2),y=y)
    expect_lt(abs(nll-136.9508962), TOLERANCE_STRICT)
    
    # Other taper shapes
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 1, cov_fct_taper_range = 0.15, y = y,
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(0.0564441, 0.0497191, 0.9921285, 0.1752661)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 19)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_cov_mat = TRUE)
    expected_mu <- c(-0.007404038, 1.487424320, 0.200022114)
    expected_cov <- c(1.113020e+00, -6.424533e-30, -4.186440e-22, -6.424533e-30, 3.522739e-01,
                      9.018454e-10, -4.186440e-22, 9.018454e-10, 6.092985e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Other taper shapes
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 2, cov_fct_taper_range = 0.08, y = y,
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(0.00327103, 0.06579671, 1.08812978, 0.18151366)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 187)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_cov_mat = TRUE)
    expected_mu <- c(-2.314198e-05, 8.967992e-01, 2.430054e-02)
    expected_cov <- c(1.2200000, 0.0000000, 0.0000000, 0.0000000, 0.9024792, 0.0000000, 0.0000000, 0.0000000, 1.1887157)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
  })
  
  test_that("Tapering ", {
    y <- eps + X%*%beta + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- init_cov_pars
    
    # No tapering
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    nll_exp <- 212.9854341
    nll <- gp_model$neg_log_likelihood(y=y, cov_pars = init_cov_pars)
    expect_lt(abs(nll - nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, X = X, params = params), file='NUL')
    cov_pars <- c(0.01621846, 0.07384498, 0.99717680, 0.21704099, 0.09616230, 0.03034715)
    coef <- c(2.30554610, 0.21565230, 1.89920767, 0.09567547)
    num_it <- 100
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expected_mu <- c(1.195910242, 4.060125034, 3.15963272)
    expected_cov <- c(6.304732e-01, 1.313601e-05, 1.008080e-07, 1.313601e-05, 3.524404e-01, 
                      3.699813e-07, 1.008080e-07, 3.699813e-07, 4.277339e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # With tapering and very large tapering range
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "tapering", cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6), file='NUL')
    nll <- gp_model$neg_log_likelihood(y=y, cov_pars = init_cov_pars)
    expect_lt(abs(nll - nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "tapering", cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X = X, 
                                           params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # With tapering and smaller tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "tapering", cov_fct_taper_shape = 0, cov_fct_taper_range = 0.5,
                                           y = y, X = X,
                                           params = params), file='NUL')
    cov_pars_tap <- c(0.02593993, 0.07560715, 0.99435221, 0.21816716, 0.17712808, 0.09797175)
    coef_tap <- c(2.32410488, 0.20610507, 1.89498931, 0.09533541)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_tap)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_tap)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 75)
    
    # Same thing with Matern covariance
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/4.7)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- init_cov_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           y = y, X = X, params = params), file='NUL')
    cov_pars <- c(0.17383685, 0.07956155, 0.84111654, 0.20895243, 0.08839064, 0.02062892)
    coef <- c(2.34174699, 0.19483212, 1.88055706, 0.09788995)
    num_it <- 21
    nll_opt <- 121.8046544
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expected_mu <- c(1.253044, 4.063322, 3.104536)
    expected_cov <- c(5.883587e-01, 3.736330e-05, 4.435167e-08, 3.736330e-05, 3.631517e-01, 1.492745e-06, 4.435167e-08, 1.492745e-06, 3.799906e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    
    # With tapering and very large tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           gp_approx = "tapering", cov_fct_taper_shape = 1, cov_fct_taper_range = 1e6,
                                           y = y, X = X,
                                           params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5)]-cov_pars[c(1,3,5)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # General shape
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5 + 1E-4,
                                             gp_approx = "tapering", cov_fct_taper_shape = 1, cov_fct_taper_range = 1e6,
                                             y = y, X = X,
                                             params = params), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5)]-cov_pars[c(1,3,5)])),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
      expect_equal(gp_model$get_num_optim_iter(), num_it)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_MEDIUM)
    }
    
    # With tapering and smaller tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           gp_approx = "tapering", cov_fct_taper_shape = 1, cov_fct_taper_range = 0.5,
                                           y = y, X = X,
                                           params = params), file='NUL')
    cov_pars_tap <- c(0.18970609, 0.07263436, 0.80493104, 0.20220891, 0.11212289, 0.02562848)
    coef_tap <- c(2.35889350, 0.17954660, 1.87422223, 0.09831309)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_tap)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_tap)),TOLERANCE_STRICT)
  })
  
}
