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
  
  test_that("fitc", {
    y <- eps + X%*%beta + xi
    coord_test_v1 <- rbind(c(0.11,0.45),coords[1:2,])
    X_test_v1 <- cbind(rep(1,3),rep(0.5,3))
    coord_test_multiple <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    X_test_multiple <- cbind(rep(1,3),c(-0.5,0.2,1))
    cov_pars_pred <- c(0.1,1,0.1)
    y_multiple <- eps_multiple + X%*%beta + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- init_cov_pars
    init_cov_pars_mult <- c(var(y)/2,var(y)/2,mean(dist(unique(coords_multiple)))/3)
    params_mult <- DEFAULT_OPTIM_PARAMS
    params_mult$init_cov_pars <- init_cov_pars_mult
    cluster_ids_ip <- c(rep(1,n/2),rep(2,n/2))
    cluster_ids_pred <- c(1,2,2)
    cluster_ids_pred_new <- c(1,2,99)
    X_test_clus <- cbind(rep(0,3),rep(0,3))
    
    # Cannot have more inducing points than samples
    expect_error( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                         gp_approx = "fitc", num_ind_points = n + 1, ind_points_selection = "random",
                                         y = y, X = X, params = params))
    # No Approximation
    capture.output( gp_model_no_approx <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                     y = y, X = X, params = params), file='NUL')
    nll_exp <- gp_model_no_approx$get_current_neg_log_likelihood() + 0.
    pred_var_no_approx <- predict(gp_model_no_approx, gp_coords_pred = coord_test_v1, cov_pars = cov_pars_pred,
                                  X_pred = X_test_v1, predict_var = TRUE)
    pred_var_lat_no_approx <- predict(gp_model_no_approx, gp_coords_pred = coord_test_v1, cov_pars = cov_pars_pred,
                                      X_pred = X_test_v1, predict_var = TRUE, predict_response = FALSE)
    pred_cov_no_approx <- predict(gp_model_no_approx, gp_coords_pred = coord_test_v1, cov_pars = cov_pars_pred,
                                  X_pred = X_test_v1, predict_cov = TRUE)
    X0 <- matrix(0, nrow=nrow(X), ncol=ncol(X))
    pred_train_no_approx <- predict(gp_model_no_approx, gp_coords_pred = coords, cov_pars = cov_pars_pred, 
                                    X_pred = X0, predict_var = TRUE)
    # duplicate locations
    capture.output( gp_model_mult_no_approx <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                                          y = y_multiple, X = X, params = params_mult), file='NUL')
    nll_mult_exp <- gp_model_mult_no_approx$get_current_neg_log_likelihood() + 0.
    pred_mult_no_approx <- predict(gp_model_mult_no_approx, y=y, gp_coords_pred = coord_test_multiple, X_pred = X_test_multiple,
                                   predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    # cluster_ids
    capture.output( gp_model_clus_no_approx <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                          y = y, X = X, cluster_ids = cluster_ids_ip, params = params), file='NUL')
    nll_cluster_exp <- gp_model_clus_no_approx$get_current_neg_log_likelihood() + 0.
    pred_clus_no_approx <- predict(gp_model_clus_no_approx, y=y, gp_coords_pred = coord_test_v1, 
                                   X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred, 
                                   predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    pred_clus_no_approx_new <- predict(gp_model_clus_no_approx, y=y, gp_coords_pred = coord_test_v1, 
                                       X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred_new, 
                                       predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    
    # With fitc and n inducing points
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "fitc", num_ind_points = n, ind_points_selection = "random",
                                           y = y, X = X, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE)) - as.vector(gp_model_no_approx$get_cov_pars(std_err = TRUE)))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE)) - as.vector(gp_model_no_approx$get_coef(std_err = TRUE)))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), gp_model_no_approx$get_num_optim_iter())
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),TOLERANCE_STRICT)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test_v1, cov_pars = cov_pars_pred,
                    X_pred = X_test_v1, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu - pred_var_no_approx$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_approx$var))),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test_v1, cov_pars = cov_pars_pred,
                    X_pred = X_test_v1, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu - pred_var_lat_no_approx$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_lat_no_approx$var))),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test_v1, cov_pars = cov_pars_pred,
                    X_pred = X_test_v1, predict_cov = TRUE)
    expect_lt(sum(abs(pred$mu - pred_cov_no_approx$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_cov_no_approx$cov))),TOLERANCE_STRICT)
    # Predict training data
    pred_train_fitc <- predict(gp_model, gp_coords_pred = coords, cov_pars = cov_pars_pred, 
                               X_pred = X0, predict_var = TRUE)
    expect_lt(sum(abs(pred_train_no_approx$mu - pred_train_fitc$mu)), TOLERANCE_LOOSE)
    expect_lt(sum(abs(pred_train_no_approx$var - pred_train_fitc$var)), TOLERANCE_LOOSE)
    # With duplicate locations
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "fitc", num_ind_points = dim(unique(coords_multiple))[1], ind_points_selection = "random",
                                           y = y_multiple, X = X, params = params_mult), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE)) - as.vector(gp_model_mult_no_approx$get_cov_pars(std_err = TRUE)))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE)) - as.vector(gp_model_mult_no_approx$get_coef(std_err = TRUE)))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), gp_model_mult_no_approx$get_num_optim_iter())
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_mult_exp),TOLERANCE_STRICT)
    pred_mult_fitc <- predict(gp_model, y=y, gp_coords_pred = coord_test_multiple, X_pred = X_test_multiple,
                              predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred_mult_no_approx$mu - pred_mult_fitc$mu)), TOLERANCE_LOOSE)
    expect_lt(sum(abs(pred_mult_no_approx$var - pred_mult_fitc$var)), TOLERANCE_LOOSE)
    # cluster_ids
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "fitc", num_ind_points = n/2, ind_points_selection = "random",
                                           y = y, X = X, cluster_ids = cluster_ids_ip, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE)[1,]) - as.vector(gp_model_clus_no_approx$get_cov_pars(std_err = TRUE)[1,]))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE)[2,]) - as.vector(gp_model_clus_no_approx$get_cov_pars(std_err = TRUE)[2,]))),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE)) - as.vector(gp_model_clus_no_approx$get_coef(std_err = TRUE)))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), gp_model_clus_no_approx$get_num_optim_iter())
    expect_lt(abs(gp_model_clus_no_approx$get_current_neg_log_likelihood() - nll_cluster_exp),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, 
                    X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred, 
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_clus_no_approx$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_clus_no_approx$var))),TOLERANCE_STRICT)
    # Prediction for a new cluster: there is no observed data, hence the prior is used, and the inducing
    # points are determined from the prediction locations of this cluster
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1,
                    X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred_new,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_clus_no_approx_new$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_clus_no_approx_new$var))),TOLERANCE_STRICT)
    
    # Fisher scoring
    params_FS = DEFAULT_OPTIM_PARAMS_FISHER
    params_FS$init_cov_pars <- init_cov_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "fitc", num_ind_points = n-1, 
                                           y = y, X = X, params = params_FS), file='NUL')
    cov_pars_FS <- c(0.008606874, 0.067462675, 1.001903559, 0.208839567, 0.094773935, 0.028174515)
    ind <- (1:length(init_cov_pars))*2-1
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[ind]-cov_pars_FS[ind])), TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[ind+1]-cov_pars_FS[ind+1])), 0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE)) - as.vector(gp_model_no_approx$get_coef(std_err = TRUE)))),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),TOLERANCE_LOOSE)
    
    # With fitc and less <- ucing points
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "fitc", num_ind_points = 50, 
                                           y = y, X = X, params = params), file='NUL')
    cov_pars_tap <- c(0.01030298, 0.07942118, 0.99809618, 0.22406519, 0.10787353, 0.03374618)
    coef_tap <- c(2.29553776, 0.22988084, 1.89903213, 0.09726784)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_tap)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_tap)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE)) - as.vector(gp_model_no_approx$get_cov_pars(std_err = TRUE)))),0.05)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE)) - as.vector(gp_model_no_approx$get_coef(std_err = TRUE)))),0.05)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),1)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE)
    expected_mu <- c(1.171558, 3.640009, 3.437938)
    expected_var <- c(0.6681653, 0.6396615, 0.5602457)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    # Predict training data
    pred_train_fitc <- predict(gp_model, gp_coords_pred = coords, cov_pars = cov_pars_pred, 
                               X_pred = X0, predict_var = TRUE)
    expect_lt(sum(abs(pred_train_no_approx$mu - pred_train_fitc$mu)), 2)
    expect_lt(sum(abs(pred_train_no_approx$var - pred_train_fitc$var)), 0.5)
    # With duplicate locations
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "fitc", num_ind_points = 12, ind_points_selection = "kmeans++",
                                           y = y_multiple, X = X, params = params_mult), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE)) - as.vector(gp_model_mult_no_approx$get_cov_pars(std_err = TRUE)))),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE)) - as.vector(gp_model_mult_no_approx$get_coef(std_err = TRUE)))),0.05)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_mult_exp),0.1)
    
    # Same thing with Matern covariance
    init_cov_pars_15 <- c(var(y)/2,var(y)/2,mean(dist(coords))/4.7*sqrt(3))
    params_15 = DEFAULT_OPTIM_PARAMS
    params_15$init_cov_pars <- init_cov_pars_15
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                           y = y, X = X, params = params)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           y = y, X = X, params = params_15), file='NUL')
    cov_pars <- c(0.17401588, 0.07960002, 0.84106347, 0.20899707, 0.08841966, 0.02064155)
    coef <- c(2.33980860, 0.19481950, 1.88058081, 0.09786326)
    num_it <- 19
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE)
    expected_mu <- c(1.253044, 4.063322, 3.104536)
    expected_var <- c(5.880651e-01, 3.627280e-01, 3.796592e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # With fitc and n-1 inducing points or very small coverTree radius
    # Different Inducing Point Methods
    ind_point_methods <- c("random","kmeans++","cover_tree")
    for (i in ind_point_methods) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                             gp_approx = "fitc", num_ind_points = n-1, cover_tree_radius = 1e-2,
                                             ind_points_selection = i, y = y, X = X,
                                             params = params_15), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_LOOSE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_LOOSE)
      # Prediction 
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, predict_var = TRUE)
      expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
      expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    }
    
    # With fitc and less inducing points (random)
    num_ind_points <- n - 5
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           gp_approx = "fitc", num_ind_points = num_ind_points, ind_points_selection = "random",
                                           y = y, X = X,
                                           params = params_15), file='NUL')
    cov_pars_ip <- c(0.17399744, 0.07965507, 0.84106802, 0.20842725, 0.08841727, 0.02058474)
    coef_ip <- c(2.33983295, 0.19481861, 1.88057897, 0.09786246)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_ip)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_ip)),TOLERANCE_LOOSE)
    
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE)
    expected_mu <- c(1.108623, 4.063135, 3.104525)
    expected_var <- c(0.6587713, 0.3627189, 0.3796549)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.2)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),0.1)
    
    # With fitc and 50 inducing points (kmeans++)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           gp_approx = "fitc", num_ind_points = 50, ind_points_selection = "kmeans++",
                                           y = y, X = X,
                                           params = params_15), file='NUL')
    cov_pars_tap <- c(0.19684565, 0.09587969, 0.81890989, 0.21870173, 0.09413984, 0.02404176)
    coef_tap <- c(2.3383270, 0.2017728, 1.8559971, 0.1004556)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_tap)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_tap)),TOLERANCE_LOOSE)
    
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE)
    expected_mu <- c(1.261284, 3.720942, 3.427156)
    expected_var <- c(0.6189797, 0.5693002, 0.4932870)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # With fitc and small covertree radius (cover_tree)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           gp_approx = "fitc", num_ind_points = 50, cover_tree_radius = 0.01, ind_points_selection = "cover_tree",
                                           y = y, X = X,
                                           params = params_15), file='NUL')
    cov_pars_tap <- c(0.17283864, 0.07884683, 0.84200101, 0.20800583, 0.08812385, 0.02039472)
    coef_tap <- c(2.34020808, 0.19446642, 1.88063092, 0.09771836)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_tap)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_tap)),TOLERANCE_LOOSE)
    
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE)
    expected_mu <- c(1.253004, 4.063949, 3.103098)
    expected_var <- c(0.5887857, 0.3618276, 0.3794413)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
  })
  
  test_that("FSA", {
    y <- eps + X%*%beta + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    # coord_test <- coords[1:3,] # works also with this
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    cov_pars_pred <- c(0.1,1,0.1)
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- init_cov_pars
    
    vec_chol_or_iterative <- c("cholesky","iterative")
    for (i in vec_chol_or_iterative) {
      if(i == "iterative"){
        TOLERANCE <- TOLERANCE_ITERATIVE
      } else {
        TOLERANCE <- TOLERANCE_LOOSE
      }
      # No Approximation
      capture.output( gp_model_no_approx <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                       y = y, X = X, params = params), file='NUL')
      nll_exp <- gp_model_no_approx$get_current_neg_log_likelihood() + 0.
      pred_var_no_approx <- predict(gp_model_no_approx, gp_coords_pred = coord_test,
                                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      pred_cov_no_approx <- predict(gp_model_no_approx, gp_coords_pred = coord_test, cov_pars = cov_pars_pred,
                                    X_pred = X_test, predict_cov = TRUE)
      capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential"), file='NUL')
      pred_var_no_X_no_approx <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      pred_cov_no_X_no_approx <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_cov = TRUE, cov_pars = cov_pars_pred)
      
      # With FSA and very large tapering range and 60 inducing points
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             gp_approx = "full_scale_tapering",num_ind_points = 60, cov_fct_taper_shape = 2, cov_fct_taper_range = 1e6,
                                             y = y, X = X,  matrix_inversion_method = i,
                                             params = params), file='NUL')
      
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE)) - as.vector(gp_model_no_approx$get_cov_pars(std_err = TRUE)))),TOLERANCE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE)) - as.vector(gp_model_no_approx$get_coef(std_err = TRUE)))),TOLERANCE)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),TOLERANCE)
      if(i == "cholesky"){
        expect_equal(gp_model$get_num_optim_iter(), gp_model_no_approx$get_num_optim_iter())
      }
      # Prediction 
      if(i == "iterative"){
        gp_model$set_prediction_data(cg_delta_conv_pred = 1e-8, nsim_var_pred = 700)
      }
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      expect_lt(sum(abs(pred$mu - pred_var_no_approx$mu)),TOLERANCE)
      expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_approx$var))),0.2)
      
      # Prediction without X
      capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "full_scale_tapering",
                                          num_ind_points = 60, cov_fct_taper_shape = 2, cov_fct_taper_range = 1e6,
                                          matrix_inversion_method = i), file='NUL')
      pred <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      expect_lt(sum(abs(pred$mu - pred_var_no_X_no_approx$mu)),TOLERANCE)
      expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_X_no_approx$var))),0.02)
      if(i == "cholesky") {
        ## TODO: Prediction of covariance matrix is currently wrong for FSA and iterative methods and also cholesky when gp_approx="full_scale_tapering_pred_var_exact"
        pred <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_cov = TRUE, cov_pars = cov_pars_pred)
        expect_lt(sum(abs(pred$mu - pred_cov_no_X_no_approx$mu)),TOLERANCE)
        expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_cov_no_X_no_approx$cov))),0.03) 
        
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "full_scale_tapering_pred_var_exact_stable",
                                            num_ind_points = 60, cov_fct_taper_shape = 2, cov_fct_taper_range = 1e6,
                                            matrix_inversion_method = "cholesky"), file='NUL')
        pred <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_var = TRUE, cov_pars = cov_pars_pred)
        expect_lt(sum(abs(pred$mu - pred_var_no_X_no_approx$mu)),TOLERANCE_STRICT)
        expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_X_no_approx$var))),TOLERANCE_STRICT)
        pred <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_cov = TRUE, cov_pars = cov_pars_pred)
        expect_lt(sum(abs(pred$mu - pred_cov_no_X_no_approx$mu)),TOLERANCE_STRICT)
        expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_cov_no_X_no_approx$cov))),TOLERANCE_STRICT)
        
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "full_scale_tapering_pred_var_exact",
                                            num_ind_points = 60, cov_fct_taper_shape = 2, cov_fct_taper_range = 1e6,
                                            matrix_inversion_method = "cholesky"), file='NUL')
        pred <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_var = TRUE, cov_pars = cov_pars_pred)
        expect_lt(sum(abs(pred$mu - pred_var_no_X_no_approx$mu)),TOLERANCE_STRICT)
        expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_X_no_approx$var))),TOLERANCE_STRICT)
        # pred <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_cov = TRUE, cov_pars = cov_pars_pred) 
        # expect_lt(sum(abs(pred$mu - pred_cov_no_X_no_approx$mu)),TOLERANCE_STRICT)
        # expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_cov_no_X_no_approx$cov))),0.02) # This test currently fails for iterative methods (11.11.2024)
      }
      
      # Fisher scoring
      params_FS <- DEFAULT_OPTIM_PARAMS_FISHER
      params_FS$num_rand_vec_trace <- 100
      params_FS$cg_delta_conv <- 0.01
      params_FS$init_cov_pars <- init_cov_pars
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             gp_approx = "full_scale_tapering", num_ind_points = 60, 
                                             cov_fct_taper_shape = 2, cov_fct_taper_range = 1e6,
                                             y = y, X = X,  matrix_inversion_method = i,
                                             params = params_FS), file='NUL')
      cov_pars_FS <- c(0.01318913, 0.07175457, 0.98649515, 0.21183893, 0.09380920, 0.02929087)
      coef_FS <- c(2.30864975, 0.21098934, 1.89940469, 0.09546584)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_FS)),2*TOLERANCE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_FS)),0.1)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),3*TOLERANCE)
      
      if(i == "cholesky"){
        # With FSA and n-1 inducing points and taper range 0.4
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               gp_approx = "full_scale_tapering",num_ind_points = n-1, 
                                               cov_fct_taper_shape = 2, cov_fct_taper_range = 0.4,
                                               y = y, X = X,matrix_inversion_method = i, 
                                               params = params), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE)) - as.vector(gp_model_no_approx$get_cov_pars(std_err = TRUE)))),TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE)) - as.vector(gp_model_no_approx$get_coef(std_err = TRUE)))),TOLERANCE)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),TOLERANCE)
        expect_equal(gp_model$get_num_optim_iter(), gp_model_no_approx$get_num_optim_iter())
        # Prediction 
        if(i == "iterative"){
          gp_model$set_prediction_data(cg_delta_conv_pred = 1e-6, nsim_var_pred = 500)
        }
        pred <- predict(gp_model, gp_coords_pred = coord_test,
                        X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
        expect_lt(sum(abs(pred$mu - pred_var_no_approx$mu)),TOLERANCE)
        expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_approx$var))),TOLERANCE)
        pred <- predict(gp_model, gp_coords_pred = coord_test, cov_pars = cov_pars_pred,
                        X_pred = X_test, predict_cov = TRUE)
        expect_lt(sum(abs(pred$mu - pred_cov_no_approx$mu)),TOLERANCE)
        expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_cov_no_approx$cov))),TOLERANCE)
        
        # With FSA and 50 inducing points and taper range 0.5
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               gp_approx = "full_scale_tapering", num_ind_points = 50, cov_fct_taper_shape = 2, cov_fct_taper_range = 0.5,
                                               y = y, X = X,matrix_inversion_method = i, 
                                               params = params), file='NUL')
        cov_pars <- c(0.01503776, 0.06968536, 1.00219308, 0.21262000, 0.09835141, 0.02968291)
        coef <- c(2.30508771, 0.21857115, 1.89918852, 0.09536239)
        num_it <- 103
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE)
        if(i == "cholesky"){
          expect_equal(gp_model$get_num_optim_iter(), num_it)
        }
        # Prediction 
        if(i == "iterative"){
          gp_model$set_prediction_data(cg_delta_conv_pred = 1e-6, nsim_var_pred = 500)
        }
        pred <- predict(gp_model, gp_coords_pred = coord_test, X_pred = X_test, predict_var = TRUE)
        expected_mu <- c(1.186786, 4.048299, 3.173789) 
        expected_var <- c(0.6428104, 0.3562637, 0.4344309)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),0.2)
        
        # Same thing with Matern covariance
        init_cov_pars_15 <- c(var(y)/2,var(y)/2,mean(dist(coords))/4.7*sqrt(3))
        params_15 = DEFAULT_OPTIM_PARAMS
        params_15$init_cov_pars <- init_cov_pars_15
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                               y = y, X = X, params = params_15), file='NUL')
        cov_pars <- c(0.17369771, 0.07950745, 0.84098718, 0.20889907, 0.08839526, 0.01190858)
        coef <- c(2.33980860, 0.19481950, 1.88058081, 0.09786326)
        num_it <- 19
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_LOOSE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_LOOSE)
        expect_equal(gp_model$get_num_optim_iter(), num_it)
        if(i == "cholesky"){
          expect_equal(gp_model$get_num_optim_iter(), num_it)
        }
        # Prediction 
        if(i == "iterative"){
          gp_model$set_prediction_data(cg_delta_conv_pred = 1e-6, nsim_var_pred = 500)
        }
        pred <- predict(gp_model, gp_coords_pred = coord_test,
                        X_pred = X_test, predict_var = TRUE)
        expected_mu <- c(1.253044, 4.063322, 3.104536)
        expected_var <- c(5.880651e-01, 3.627280e-01, 3.796592e-01)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
        
        # With FSA and very large tapering range and 60 inducing points
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                               gp_approx = "full_scale_tapering",num_ind_points = 60, cov_fct_taper_shape = 2, cov_fct_taper_range = 1e6,
                                               y = y, X = X,  matrix_inversion_method = i, 
                                               params = params_15), file='NUL')
        ind <- (1:length(params_15$init_cov_pars))*2-1
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[ind]-cov_pars[ind])), TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[ind+1]-cov_pars[ind+1])), TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE)
        if(i == "cholesky"){
          expect_equal(gp_model$get_num_optim_iter(), num_it)
        }
        # Prediction 
        if(i == "iterative"){
          gp_model$set_prediction_data(cg_delta_conv_pred = 1e-6, nsim_var_pred = 500)
        }
        pred <- predict(gp_model, gp_coords_pred = coord_test,
                        X_pred = X_test, predict_var = TRUE)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),2*TOLERANCE)
        
        # With FSA and n-1 inducing points and taper range 0.5
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                               gp_approx = "full_scale_tapering",num_ind_points = n-1, cov_fct_taper_shape = 2, cov_fct_taper_range = 0.5,
                                               y = y, X = X,
                                               params = params_15), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),1.5*TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE)
        expect_equal(gp_model$get_num_optim_iter(), num_it)
        if(i == "cholesky"){
          expect_equal(gp_model$get_num_optim_iter(), num_it)
        }
        # Prediction 
        if(i == "iterative"){
          gp_model$set_prediction_data(cg_delta_conv_pred = 1e-6, nsim_var_pred = 500)
        }
        pred <- predict(gp_model, gp_coords_pred = coord_test,
                        X_pred = X_test, predict_var = TRUE)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE)
        
        # With FSA and 50 inducing points and taper range 0.5
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                               gp_approx = "full_scale_tapering",num_ind_points = 50, cov_fct_taper_shape = 2, cov_fct_taper_range = 0.5,
                                               y = y, X = X,matrix_inversion_method = i, 
                                               params = params_15), file='NUL')
        cov_pars <- c(0.16791734, 0.07920530, 0.84909181, 0.20964697, 0.08810687, 0.02041659)
        coef <- c(2.34257038, 0.19533006, 1.87702082, 0.09749923)
        ind <- (1:length(params_15$init_cov_pars))*2-1
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[ind]-cov_pars[ind])), TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[ind+1]-cov_pars[ind+1])), TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE)
        # Prediction 
        if(i == "iterative"){
          gp_model$set_prediction_data(cg_delta_conv_pred = 1e-6, nsim_var_pred = 500)
        }
        pred <- predict(gp_model, gp_coords_pred = coord_test,
                        X_pred = X_test, predict_var = TRUE)
        expected_mu <- c(1.250332, 4.049631, 3.160899)
        expected_var <- c(0.5981874, 0.3632729, 0.3848723)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE)
        # The predictive variances of the FSA depend more strongly on the compiler than the predictive
        # means, as for the tapering ranges above. The deviation measured with gcc is 0.0113
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),2*TOLERANCE)
      }# end (i == "cholesky")
    }# end loop over i (matrix_inversion_method)
  })# end FSA
  
  test_that("prediction for a new cluster with the FITC and the full-scale approximations ", {

    # There is no observed data for the new cluster, hence the predictive distribution of the latent GP is
    # its prior: the mean is zero and the variance is the marginal variance of the GP. The Vecchia-based
    # approximations return the variance of the observable process, i.e. they additionally contain the
    # nugget effect. For the full-scale approximations, the inducing points of the new cluster are
    # determined from its prediction locations
    n_nc <- 100
    coords_nc <- cbind(sim_rand_unif(n = n_nc, init_c = 0.11), sim_rand_unif(n = n_nc, init_c = 0.22))
    cluster_ids_nc <- c(rep(1, n_nc / 2), rep(2, n_nc / 2))
    y_nc <- qnorm(sim_rand_unif(n = n_nc, init_c = 0.33))
    n_pred_nc <- 30
    coords_pred_nc <- cbind(sim_rand_unif(n = n_pred_nc, init_c = 0.44),
                            sim_rand_unif(n = n_pred_nc, init_c = 0.55))
    cluster_ids_pred_nc <- rep(c(1, 2, 3), length.out = n_pred_nc)# cluster 3 has not been observed
    is_new_nc <- cluster_ids_pred_nc == 3
    cov_pars_nc <- c(0.1, 1.3, 0.2)# error variance, marginal variance, range
    # The predictive distribution of a cluster without observed data is the prior, i.e. the variance of the
    # latent process is the marginal variance for every approximation
    cases_nc <- list(list(gp_approx = "none", var = cov_pars_nc[2], deterministic = TRUE, args = list()),
                     list(gp_approx = "fitc", var = cov_pars_nc[2], deterministic = TRUE,
                          args = list(num_ind_points = 8, ind_points_selection = "random")),
                     list(gp_approx = "full_scale_tapering", var = cov_pars_nc[2], deterministic = TRUE,
                          args = list(num_ind_points = 8, ind_points_selection = "random",
                                      cov_fct_taper_range = 1e6, cov_fct_taper_shape = 2)),
                     list(gp_approx = "vecchia", var = cov_pars_nc[2], deterministic = TRUE,
                          args = list(num_neighbors = 20, vecchia_ordering = "none")),
                     list(gp_approx = "full_scale_vecchia", var = cov_pars_nc[2], deterministic = TRUE,
                          args = list(num_ind_points = 8, ind_points_selection = "random",
                                      num_neighbors = 20, vecchia_ordering = "none")))
    for (case_nc in cases_nc) {
      args_nc <- c(list(gp_coords = coords_nc, cov_function = "exponential", gp_approx = case_nc$gp_approx,
                        cluster_ids = cluster_ids_nc), case_nc$args)
      capture.output(gp_model_nc <- do.call(GPModel, args_nc), file = "NUL")
      capture.output(pred_nc <- predict(gp_model_nc, y = y_nc, cov_pars = cov_pars_nc,
                                        gp_coords_pred = coords_pred_nc,
                                        cluster_ids_pred = cluster_ids_pred_nc,
                                        predict_var = TRUE, predict_response = FALSE), file = "NUL")
      expect_lt(sum(abs(pred_nc$mu[is_new_nc])), TOLERANCE_STRICT,
                label = paste0("predictive mean for the new cluster (", case_nc$gp_approx, ")"))
      expect_lt(sum(abs(pred_nc$var[is_new_nc] - case_nc$var)), TOLERANCE_MEDIUM,
                label = paste0("predictive variance for the new cluster (", case_nc$gp_approx, ")"))
      # The predictions for the observed clusters must not be affected by the new cluster, i.e. making
      # predictions for a new cluster must not change the fitted model
      capture.output(pred_obs_nc <- predict(gp_model_nc, y = y_nc, cov_pars = cov_pars_nc,
                                            gp_coords_pred = coords_pred_nc[!is_new_nc, , drop = FALSE],
                                            cluster_ids_pred = cluster_ids_pred_nc[!is_new_nc],
                                            predict_var = TRUE, predict_response = FALSE), file = "NUL")
      expect_lt(sum(abs(pred_nc$mu[!is_new_nc] - pred_obs_nc$mu)), TOLERANCE_STRICT,
                label = paste0("predictive mean for the observed clusters (", case_nc$gp_approx, ")"))
      if (case_nc$deterministic) {
        expect_lt(sum(abs(pred_nc$var[!is_new_nc] - pred_obs_nc$var)), TOLERANCE_STRICT,
                  label = paste0("predictive variance for the observed clusters (", case_nc$gp_approx, ")"))
      }
    }
  })
  test_that("VIF or Full scale Vecchia", {
    
    y <- eps + X%*%beta + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    # coord_test <- coords[1:3,] # works also with this
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    cov_pars_pred <- c(0.1,1,0.1)
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params = OPTIM_PARAMS_BFGS
    params$init_cov_pars <- init_cov_pars
    
    vec_kNN_search <- c("euclidean-based kNN","correlation-based kNN")
    TOLERANCE <- TOLERANCE_LOOSE
    for (i in vec_kNN_search) {
      if(i == "euclidean-based kNN"){
        gp_approx <- "full_scale_vecchia"
      } else {
        gp_approx <- "full_scale_vecchia_correlation_based"
      }
      # No Approximation
      capture.output( gp_model_no_approx <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                       y = y, X = X, params = OPTIM_PARAMS_BFGS), file='NUL')
      nll_exp <- gp_model_no_approx$get_current_neg_log_likelihood() + 0.
      pred_var_no_approx <- predict(gp_model_no_approx, gp_coords_pred = coord_test,
                                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      pred_cov_no_approx <- predict(gp_model_no_approx, gp_coords_pred = coord_test, cov_pars = cov_pars_pred,
                                    X_pred = X_test, predict_cov = TRUE)
      capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential"), file='NUL')
      pred_var_no_X_no_approx <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      pred_cov_no_X_no_approx <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_cov = TRUE, cov_pars = cov_pars_pred)
      
      # With VIF and a lot of Vecchia neighbors and 60 inducing points
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             gp_approx = gp_approx,num_ind_points = 60,num_neighbors = 50,
                                             y = y, X = X,  matrix_inversion_method = "cholesky",
                                             params = OPTIM_PARAMS_BFGS), file='NUL')
      
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_no_approx$get_cov_pars(std_err = FALSE)))),TOLERANCE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_no_approx$get_coef(std_err = FALSE)))),TOLERANCE)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),TOLERANCE)
      expect_equal(gp_model$get_num_optim_iter(), gp_model_no_approx$get_num_optim_iter())
      # Prediction 
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      expect_lt(sum(abs(pred$mu - pred_var_no_approx$mu)),0.1)
      expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_approx$var))),0.2)
      
      # Prediction without X
      capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", gp_approx = gp_approx,
                                          num_ind_points = 60, num_neighbors = 50,
                                          matrix_inversion_method = "cholesky"), file='NUL')
      pred <- predict(gp_model, y = y, gp_coords_pred = coord_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      expect_lt(sum(abs(pred$mu - pred_var_no_X_no_approx$mu)),TOLERANCE)
      expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_X_no_approx$var))),0.02)
      
      
      # With VIF and n-1 inducing points and 5 Vecchia neighbors
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             gp_approx = gp_approx,num_ind_points = n-1,num_neighbors = 5,
                                             y = y, X = X,  matrix_inversion_method = "cholesky",
                                             params = OPTIM_PARAMS_BFGS), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_no_approx$get_cov_pars(std_err = FALSE)))),TOLERANCE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_no_approx$get_coef(std_err = FALSE)))),TOLERANCE)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),TOLERANCE)
      expect_equal(gp_model$get_num_optim_iter(), gp_model_no_approx$get_num_optim_iter())
      # Prediction 
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      expect_lt(sum(abs(pred$mu - pred_var_no_approx$mu)),TOLERANCE)
      expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_approx$var))),TOLERANCE)
      pred <- predict(gp_model, gp_coords_pred = coord_test, cov_pars = cov_pars_pred,
                      X_pred = X_test, predict_cov = TRUE)
      expect_lt(sum(abs(pred$mu - pred_cov_no_approx$mu)),TOLERANCE)
      expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_cov_no_approx$cov))),TOLERANCE)
      
      # With VIF and 50 inducing points and 15 Vecchia neighbors
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             gp_approx = gp_approx,num_ind_points = 50,num_neighbors = 15,
                                             y = y, X = X,  matrix_inversion_method = "cholesky",
                                             params = OPTIM_PARAMS_BFGS), file='NUL')
      cov_pars <- c(0.009170148 , 1.002068032 , 0.095036760)
      coef <- c(2.305036, 1.899353)
      num_it <- 16
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE)
      # Prediction 
      pred <- predict(gp_model, gp_coords_pred = coord_test, X_pred = X_test, predict_var = TRUE)
      if(i == "euclidean-based kNN"){
        expected_mu <- c(1.197409, 4.062989, 3.156590) 
        expected_var <- c(0.6299383, 0.3469613, 0.4251834)
      } else {
        expected_mu <- c(1.196875, 4.063326, 3.156890)
        expected_var <- c(0.6303005, 0.3469042, 0.4254070)
      }
      expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE)
      expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE)
      # Sampling from posterior
      pred <- predict(gp_model, gp_coords_pred = coord_test, X_pred = X_test, predict_response = TRUE, 
                      sample_posterior = TRUE, num_post_samples = 100000)
      tol_mu <- 0.01
      tol_cov <- 0.01
      expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-expected_mu)), tol_mu)
      expect_lt(sum(abs(as.vector(diag(cov(t(pred$posterior_samples))))-expected_var)), tol_cov)
      # Sampling from prior
      pred <- predict(gp_model, gp_coords_pred = coord_test, X_pred = X_test, predict_response = TRUE,
                      cov_pars = c(1E-20,sigma2_1,rho), sample_prior = TRUE, num_prior_samples = 100000)
      tol_mean <- 0.01
      tol_cov <- 0.01
      expect_lt(mean(abs(apply(pred$prior_samples[1:5,],1,mean)-rep(0,5))), tol_mean)
      cov_mat_prior <- cov(t(pred$prior))
      expect_lt(mean(abs(as.vector(cov_mat_prior[lower.tri(cov_mat_prior)])-as.vector(Sigma[lower.tri(Sigma)]))), tol_cov)
      if (i == "euclidean-based kNN") {
        # Holding some parameters fix
        params_fix <- params
        params_fix$estimate_cov_par_index <- c(1,0,0)
        invisible(capture.output( gp_model_fix <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                             gp_approx = gp_approx, num_ind_points = 50, num_neighbors = 10,
                                                             y = y, X = X,  matrix_inversion_method = "cholesky",
                                                             params = params_fix) ))
        cov_pars_fix <- c(0.05473032413, 1.43524508454, 0.17864807736)
        nll_fix <- 122.9756055
        expect_lt(sum(abs(as.vector(gp_model_fix$get_cov_pars(std_err = FALSE))-cov_pars_fix)), TOLERANCE_LOOSE)
        expect_lt(sum(abs(gp_model_fix$get_cov_pars(std_err = FALSE)[c(2,3)]-params_fix$init_cov_pars[c(2,3)])),TOLERANCE_STRICT)
        # 0.051 has been measured with clang + libc++ in the clang-asan container of R-hub
        expect_lt(sum(abs(gp_model_fix$get_current_neg_log_likelihood()-nll_fix)), relax_tolerance(0.04, nll_fix))
        params_fix$estimate_cov_par_index <- c(1,1,0)
        gp_model_fix <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                   gp_approx = gp_approx, num_ind_points = 50, num_neighbors = 10,
                                   y = y, X = X,  matrix_inversion_method = "cholesky",
                                   params = params_fix)
        expect_lt(sum(abs(gp_model_fix$get_cov_pars(std_err = FALSE)[c(3)]-params_fix$init_cov_pars[c(3)])),TOLERANCE_STRICT)
        params_fix$estimate_cov_par_index <- c(0,1,0)
        gp_model_fix <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                   gp_approx = gp_approx, num_ind_points = 50, num_neighbors = 10,
                                   y = y, X = X,  matrix_inversion_method = "cholesky",
                                   params = params_fix)
        expect_lt(sum(abs(gp_model_fix$get_cov_pars(std_err = FALSE)[c(1,3)]-params_fix$init_cov_pars[c(1,3)])),TOLERANCE_STRICT)
      }
    }# end loop over i (vec_kNN_search)
  })
  
  test_that("Saving a GPModel and loading from file works ", {
    
    y <- eps + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, X = coords, params = DEFAULT_OPTIM_PARAMS), 
                    file='NUL')
    cov_pars_est <- gp_model$get_cov_pars(std_err = TRUE)
    coef_est <- gp_model$get_coef(std_err = TRUE)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = coord_test, predict_cov_mat = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    # Load from file and make predictions again
    capture.output( gp_model_loaded <- loadGPModel(filename = filename), file='NUL')
    pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, X_pred = coord_test, predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    expect_equal(cov_pars_est, gp_model_loaded$get_cov_pars(std_err = TRUE))
    expect_equal(coef_est, gp_model_loaded$get_coef(std_err = TRUE))

    # Before the internal default-value sentinel was changed to -999, any non-positive value of these
    #   parameters was a request for the internal default, and a saved model can thus contain -1 (the
    #   former default), another negative number, or 0
    json_legacy <- paste(readLines(filename, warn = FALSE), collapse = "\n")
    legacy_params <- c("delta_rel_conv", "lr_cov", "fitc_piv_chol_preconditioner_rank",
                       "m_lbfgs", "delta_conv_mode_finding")
    legacy_values <- c(-1, -2, 0, -1, -2)
    for (i in seq_along(legacy_params)) {
      pattern <- paste0('("', legacy_params[i], '"[[:space:]]*:[[:space:]]*)[-+0-9.eE]+')
      expect_true(grepl(pattern, json_legacy))
      json_legacy <- sub(pattern, paste0("\\1", legacy_values[i]), json_legacy)
    }
    filename_legacy <- tempfile(fileext = ".json")
    writeLines(json_legacy, filename_legacy)
    capture.output( gp_model_legacy <- loadGPModel(filename = filename_legacy), file='NUL')
    pred_legacy <- predict(gp_model_legacy, gp_coords_pred = coord_test, X_pred = coord_test,
                           predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_legacy$mu)
    expect_equal(pred$cov, pred_legacy$cov)
    
    # With Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 20,
                                           vecchia_ordering = "none", y = y, params = DEFAULT_OPTIM_PARAMS), 
                    file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    # Load from file and make predictions again
    capture.output( gp_model_loaded <- loadGPModel(filename = filename), file='NUL')
    pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    
    # With Vecchia approximation and random ordering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 20,
                                           vecchia_ordering = "random", y = y, params = DEFAULT_OPTIM_PARAMS), 
                    file='NUL')
    gp_model$set_prediction_data(num_neighbors_pred = 50)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    # Load from file and make predictions again
    capture.output( gp_model_loaded <- loadGPModel(filename = filename), file='NUL')
    gp_model_loaded$set_prediction_data(num_neighbors_pred = 50)
    pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    
    # With Tapering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "tapering", cov_fct_taper_range = 0.5, cov_fct_taper_shape = 1.,
                                           y = y, params = DEFAULT_OPTIM_PARAMS), 
                    file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    # Load from file and make predictions again
    capture.output( gp_model_loaded <- loadGPModel(filename = filename), file='NUL')
    pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
  })
  
}
