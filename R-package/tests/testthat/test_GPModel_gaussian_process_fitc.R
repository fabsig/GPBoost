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
  test_that("Standard errors of covariance parameters for FITC and full-scale tapering with several clusters ", {

    # Both approximations are exact here (FITC with as many inducing points as data points per cluster,
    # full-scale tapering with a taper range that covers all distances), so they have to give the same
    # standard errors as the model without an approximation. The scaling of the Fisher information with
    # the error variance is applied once to the sum over all clusters; applying it inside the loop over
    # the clusters rescaled the contributions of the previous clusters again
    n_cl_fi <- 4
    cluster_ids_fi <- rep(1:n_cl_fi, each = n / n_cl_fi)
    y_fi <- eps + xi
    params_fi <- c(OPTIM_PARAMS_BFGS,
                   list(init_cov_pars = c(var(y_fi) / 2, var(y_fi) / 2, mean(dist(coords)) / 3),
                        num_rand_vec_trace = 1000, reuse_rand_vec_trace = TRUE, seed_rand_vec_trace = 1))
    args_fi <- list(gp_coords = coords, cov_function = "exponential", cluster_ids = cluster_ids_fi,
                    y = y_fi, params = params_fi)
    capture.output( gp_model_exact_fi <- do.call(fitGPModel, args_fi), file = 'NUL')
    cov_pars_exact_fi <- gp_model_exact_fi$get_cov_pars(std_err = TRUE)
    args_approx_fi <- list(
      fitc = c(args_fi, list(gp_approx = "fitc", num_ind_points = n / n_cl_fi,
                             ind_points_selection = "random")),
      full_scale_tapering = c(args_fi, list(gp_approx = "full_scale_tapering",
                                            num_ind_points = n / n_cl_fi - 1,
                                            ind_points_selection = "random",
                                            cov_fct_taper_range = 1e6, cov_fct_taper_shape = 2,
                                            matrix_inversion_method = "cholesky")))
    for (gp_approx_fi in names(args_approx_fi)) {
      capture.output( gp_model_fi <- do.call(fitGPModel, args_approx_fi[[gp_approx_fi]]), file = 'NUL')
      cov_pars_fi <- gp_model_fi$get_cov_pars(std_err = TRUE)
      expect_lt(max(abs(cov_pars_fi[1, ] / cov_pars_exact_fi[1, ] - 1)), TOLERANCE_MEDIUM)
      # The Fisher information of these approximations is calculated with stochastic trace estimation,
      #   so the standard errors are only equal up to a Monte Carlo error
      expect_lt(max(abs(cov_pars_fi[2, ] / cov_pars_exact_fi[2, ] - 1)), TOLERANCE_ITERATIVE)
    }
  })

}
