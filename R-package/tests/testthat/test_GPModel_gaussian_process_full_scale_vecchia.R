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
  
}
