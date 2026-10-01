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
  
  test_that("CUDA GPU", {
    
    y <- eps + X%*%beta + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    cov_pars_pred <- c(0.1,1,0.1)
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params = OPTIM_PARAMS_BFGS
    params$init_cov_pars <- init_cov_pars
    
    # VIF without GPU
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", GPU_use = FALSE,
                                           gp_approx = "vif_correlation_based",num_ind_points = 50,num_neighbors = 20,
                                           y = y, X = X,  matrix_inversion_method = "cholesky",
                                           params = OPTIM_PARAMS_BFGS), file='NUL')
    cov_pars_without_GPU <- as.vector(gp_model$get_cov_pars(std_err = FALSE))
    coefs_without_GPU <- as.vector(gp_model$get_coef(std_err = FALSE))
    NLL_without_GPU <- gp_model$get_current_neg_log_likelihood()
    Num_optim_iter_without_GPU <- gp_model$get_num_optim_iter()
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    pred_mu_without_GPU <- pred$mu
    pred_var_without_GPU <- as.vector(pred$var)
    
    # VIF with GPU
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", GPU_use = TRUE,
                                           gp_approx = "vif_correlation_based",num_ind_points = 50,num_neighbors = 20,
                                           y = y, X = X,  matrix_inversion_method = "cholesky",
                                           params = OPTIM_PARAMS_BFGS), file='NUL')
    
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - cov_pars_without_GPU)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - coefs_without_GPU)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - NLL_without_GPU),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), Num_optim_iter_without_GPU)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_mu_without_GPU)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var) - pred_var_without_GPU)),TOLERANCE_STRICT)
    
  })

  test_that("CUDA GPU with Vecchia approximation, duplicate locations, and multiple clusters ", {
    # For non-Gaussian likelihoods, duplicate locations are collapsed to unique latent locations. Only the first
    # cluster has duplicate locations
    coords_ST_dup <- cbind(c(rep((1:25)/25, 2), (51:100)/100), rbind(coords_multiple[1:50,], coords[51:100,]))
    cluster_ids_dup <- c(rep(1,50), rep(2,50))
    y_bin <- as.numeric(eps_multiple > 0)
    cov_pars <- c(1, 10, 10, 0.5, 1.5, 0.5, 1)
    nll <- rep(NA, 2)
    for (i in 1:2) {
      capture.output( gp_model <- GPModel(gp_coords = coords_ST_dup, cov_function = "space_time_gneiting",
                                          cluster_ids = cluster_ids_dup, likelihood = "bernoulli_probit",
                                          gp_approx = "vecchia", num_neighbors = 20, vecchia_ordering = "none",
                                          matrix_inversion_method = "cholesky", GPU_use = (i == 2)), file='NUL')
      nll[i] <- gp_model$neg_log_likelihood(cov_pars = cov_pars, y = y_bin)
    }
    expect_lt(abs(nll[2] - nll[1]), TOLERANCE_STRICT)
  })
   
}
