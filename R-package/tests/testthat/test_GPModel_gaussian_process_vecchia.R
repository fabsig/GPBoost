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
  
  test_that("Vecchia approximation for Gaussian process model ", {
    
    y <- eps + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params_vecchia <- list(optimizer_cov = "gradient_descent",
                           lr_cov = 0.1, use_nesterov_acc = TRUE,
                           acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                           convergence_criterion = "relative_change_in_parameters",
                           num_rand_vec_trace = 1000, reuse_rand_vec_trace = TRUE,
                           seed_rand_vec_trace = 1, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)
    
    # Evaluate negative log-likelihood
    cov_pars_ll <- c(0.1,1.6,0.2)
    exp_nll <- 124.2549533
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll,y=y)
    expect_lt(abs(nll-exp_nll), TOLERANCE_STRICT)
    # with weights
    weights <- rep(1.000000001, length(y))
    capture.output( gp_model_w <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n-1,
                                        vecchia_ordering = "none", weights = weights), file='NUL') 
    expect_lt(abs(gp_model_w$neg_log_likelihood(cov_pars=cov_pars_ll,y=y)-exp_nll), TOLERANCE_STRICT)
    # "vecchia_latent"
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "cholesky"), file='NUL')
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1]), file='NUL')
    expect_lt(abs(nll-exp_nll), TOLERANCE_STRICT)
    # "vecchia_latent" with iterative methods (pivoted Cholesky preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(num_rand_vec_trace = 1000, init_coef_aux_pars_from_iid_model = FALSE))
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1]), file='NUL')
    expect_lt(abs(nll-exp_nll), 0.25)
    # "vecchia_latent" with iterative methods (FITC preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(num_rand_vec_trace = 1000, cg_preconditioner_type = "predictive_process_plus_diagonal",
                                          fitc_piv_chol_preconditioner_rank=99, init_coef_aux_pars_from_iid_model = FALSE))
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1]), file='NUL')
    expect_lt(abs(nll-exp_nll), 0.25)
    
    # Same thing without Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "none"), file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll,y=y)
    expect_lt(abs(nll-exp_nll), TOLERANCE_STRICT)
    # less neighbhors
    exp_nll_less_nn <- 124.2252524
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30,
                                        vecchia_ordering = "none"), file='NUL')
    expect_lt(abs(gp_model$neg_log_likelihood(cov_pars=cov_pars_ll,y=y)-exp_nll_less_nn), TOLERANCE_STRICT)
    capture.output( gp_model_w <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30,
                                        vecchia_ordering = "none", weights = weights), file='NUL')
    expect_lt(abs(gp_model_w$neg_log_likelihood(cov_pars=cov_pars_ll,y=y)-exp_nll_less_nn), TOLERANCE_STRICT)
    # "vecchia_latent"
    exp_nll_less_nn_lat <- 124.2549533
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "cholesky"), file='NUL')
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1]), file='NUL')
    expect_lt(abs(nll-exp_nll_less_nn_lat), TOLERANCE_STRICT)
    # "vecchia_latent" with iterative methods (pivoted Cholesky)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(num_rand_vec_trace = 1000, init_coef_aux_pars_from_iid_model = FALSE))
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1]), file='NUL')
    expect_lt(abs(nll-exp_nll_less_nn_lat), 0.25)
    
    # "vecchia_latent" with iterative methods (FITC preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(num_rand_vec_trace = 1000, cg_preconditioner_type = "predictive_process_plus_diagonal",
                                          fitc_piv_chol_preconditioner_rank = n-1, init_coef_aux_pars_from_iid_model = FALSE))
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1]), file='NUL')
    expect_lt(abs(nll-exp_nll_less_nn_lat), 0.25)
    
    # Estimation and maximal number of neighbors
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia), file='NUL')
    cov_pars <- c(0.03276547, 0.07544593, 1.07617676, 0.24743617, 0.11352557, 0.03482885)
    nll_est <- 122.7752664
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5)]-cov_pars[c(1,3,5)])), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5)+1]-cov_pars[c(1,3,5)+1])), TOLERANCE_LOOSE)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[2], 3)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), TOLERANCE_STRICT)
    # With "vecchia_latent"
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "cholesky"), file='NUL')
    params_latent <- params_vecchia
    params_latent$init_cov_pars <- NULL
    params_latent$optimizer_cov = "lbfgs"
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(3,5)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), TOLERANCE_MEDIUM)
    # "vecchia_latent" with iterative methods (pivoted Cholesky)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(3,5)])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), 0.3)
    # "vecchia_latent" with iterative methods (FITC preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    params_latent$cg_preconditioner_type = "predictive_process_plus_diagonal"
    params_latent$fitc_piv_chol_preconditioner_rank = n - 1
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(3,5)])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), 0.3)
    
    # Same thing without Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5)]-cov_pars[c(1,3,5)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[2], 3)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), TOLERANCE_LOOSE)
    
    # Random ordering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering="random", y = y,
                                           params = params_vecchia), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5)]-cov_pars[c(1,3,5)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[2], 3)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    
    # Prediction using given parameters
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    cov_pars = c(0.02,1.2,0.9)
    expected_mu <- c(0.08704577, 1.63875604, 0.48513581)
    expected_cov <- c(1.189093e-01, 1.171632e-05, -4.172444e-07, 1.171632e-05,
                      7.427727e-02, 1.492859e-06, -4.172444e-07, 1.492859e-06, 8.107455e-02)
    exp_cov_no_nugget <- expected_cov
    exp_cov_no_nugget[c(1,5,9)] <- expected_cov[c(1,5,9)] - cov_pars[1]
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars, predict_cov_mat = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars, predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-exp_cov_no_nugget)), TOLERANCE_STRICT)
    # Prediction of variances only
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Predict latent process
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_no_nugget[c(1,5,9)])), TOLERANCE_STRICT)
    # vecchia_latent
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "cholesky"), file='NUL')
    gp_model$set_optim_params(params=list(init_aux_pars = cov_pars[1], init_coef_aux_pars_from_iid_model = FALSE))
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    capture.output( pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = TRUE), file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    capture.output( pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = FALSE), file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_no_nugget[c(1,5,9)])), TOLERANCE_STRICT)
    # vecchia_latent and iterative methods (pivoted Cholesky)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(init_aux_pars = cov_pars[1], init_coef_aux_pars_from_iid_model = FALSE))
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    capture.output( pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = TRUE), file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_LOOSE)
    capture.output( pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = FALSE), file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_no_nugget[c(1,5,9)])), TOLERANCE_LOOSE)
    
    # vecchia_latent and iterative methods (FITC preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(init_aux_pars = cov_pars[1], cg_preconditioner_type = "predictive_process_plus_diagonal",
                                          fitc_piv_chol_preconditioner_rank = dim(coords)[1]-1, init_coef_aux_pars_from_iid_model = FALSE))
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    capture.output( pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = TRUE), file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_LOOSE)
    capture.output( pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = FALSE), file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_no_nugget[c(1,5,9)])), TOLERANCE_LOOSE)
    
    # Vecchia approximation with 30 neighbors
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30,
                                        vecchia_ordering = "none"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia) , file='NUL')
    cov_pars_vecchia <- c(0.03297349, 0.07545639, 1.07691542, 0.24785457, 0.11378505, 0.03493878)
    nll_vecchia <- 122.7680889
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE)[c(1,3,5)])-cov_pars_vecchia[c(1,3,5)])), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE)[c(1,3,5)+1])-cov_pars_vecchia[c(1,3,5)+1])), TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 378)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_vecchia)), TOLERANCE_STRICT)
    # Prediction from fitted model
    coord_test <- cbind(c(0.1,0.10001,0.7),c(0.9,0.90001,0.55))
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expected_mu_vecchia <- c(0.06968068, 0.06967750, 0.44208925)
    expected_cov_vecchia <- c(0.6214955, 0.0000000, 0.0000000, 0.0000000, 0.6215069,
                              0.0000000, 0.0000000, 0.0000000, 0.4199531)
    expect_lt(sum(abs(pred$mu-expected_mu_vecchia)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_vecchia)), TOLERANCE_STRICT)
    # Sampling from posterior
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = TRUE, 
                    sample_posterior = TRUE, num_post_samples = 1000000)
    tol_mu <- 0.01
    tol_cov <- 0.01
    expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-expected_mu_vecchia)), tol_mu)
    expect_lt(sum(abs(as.vector(cov(t(pred$posterior_samples)))-expected_cov_vecchia)), tol_cov)
    # Sampling from the posterior of the latent process: the samples must have the predictive variance of
    #   the latent process, i.e. they must not contain the nugget effect of the observed process
    pred_latent <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                           predict_var = TRUE, sample_posterior = TRUE, num_post_samples = 1000000)
    expect_lt(sum(abs(apply(pred_latent$posterior_samples, 1, var) - pred_latent$var)), tol_cov)
    expect_lt(sum(abs(apply(pred_latent$posterior_samples, 1, mean) - expected_mu_vecchia)), tol_mu)
    # the nugget effect is the difference to the predictive variances of the observed process above
    expect_lt(sum(abs(expected_cov_vecchia[c(1, 5, 9)] - pred_latent$var - cov_pars_vecchia[1])), tol_cov)
    # Sampling from prior
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = TRUE,
                    cov_pars = c(1E-20,sigma2_1,rho), sample_prior = TRUE, num_prior_samples = 100000)
    tol_mean <- 0.01
    tol_cov <- 0.01
    expect_lt(mean(abs(apply(pred$prior_samples[1:5,],1,mean)-rep(0,5))), tol_mean)
    cov_mat_prior <- cov(t(pred$prior))
    expect_lt(mean(abs(as.vector(cov_mat_prior[lower.tri(cov_mat_prior)])-as.vector(Sigma[lower.tri(Sigma)]))), tol_cov)
    # with weights
    capture.output( gp_model_w <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30,
                                        vecchia_ordering = "none", weights = weights), file='NUL')
    capture.output( fit(gp_model_w, y = y, params = params_vecchia) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model_w$get_cov_pars(std_err = TRUE)[c(1,3,5)])-cov_pars_vecchia[c(1,3,5)])), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model_w$get_cov_pars(std_err = TRUE)[c(1,3,5)+1])-cov_pars_vecchia[c(1,3,5)+1])), TOLERANCE_LOOSE)
    expect_equal(gp_model_w$get_num_optim_iter(), 378)
    expect_lt(sum(abs(gp_model_w$get_current_neg_log_likelihood()-nll_vecchia)), TOLERANCE_STRICT)
    gp_model_w$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred = 30)
    pred <- predict(gp_model_w, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu_vecchia)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_vecchia)), TOLERANCE_STRICT)
    
    # Holding some parameters fix
    params_fix <- params_vecchia
    params_fix$convergence_criterion <- NULL
    params_fix$optimizer_cov <- "lbfgs"
    params_fix$estimate_cov_par_index <- c(1,0,0)
    capture.output( gp_model_fix <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               gp_approx = "vecchia", num_neighbors = 30,
                                               vecchia_ordering = "none", y = y, params = params_fix), file='NUL')
    cov_pars_fix <- c(0.3380100131, 0.1193829564, 0.5170731356, 0.2047755947, 0.1786480774, 0.1037257539)
    nll_fix <- 126.5919052
    expect_lt(sum(abs(as.vector(gp_model_fix$get_cov_pars(std_err = TRUE)[c(1,3,5)])-cov_pars_fix[c(1,3,5)])), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model_fix$get_cov_pars(std_err = TRUE)[c(1,3,5)+1])-cov_pars_fix[c(1,3,5)+1])), TOLERANCE_LOOSE)
    expect_lt(sum(abs(gp_model_fix$get_cov_pars(std_err = TRUE)[1,c(2,3)]-params_fix$init_cov_pars[c(2,3)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model_fix$get_current_neg_log_likelihood()-nll_fix)), TOLERANCE_STRICT)
    params_fix$estimate_cov_par_index <- c(1,1,0)
    capture.output( gp_model_fix <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               gp_approx = "vecchia", num_neighbors = 30,
                                               vecchia_ordering = "none", y = y, params = params_fix), file='NUL')
    cov_pars_fix <- c(0.10238832994, 0.08839924158, 1.23364920496, 0.37129674965, 0.17864807736, 0.07329261792)
    nll_fix <- 123.4597106
    expect_lt(sum(abs(as.vector(gp_model_fix$get_cov_pars(std_err = TRUE)[c(1,3,5)])-cov_pars_fix[c(1,3,5)])), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model_fix$get_cov_pars(std_err = TRUE)[c(1,3,5)+1])-cov_pars_fix[c(1,3,5)+1])), TOLERANCE_LOOSE)
    expect_lt(sum(abs(gp_model_fix$get_cov_pars(std_err = TRUE)[1,c(3)]-params_fix$init_cov_pars[c(3)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model_fix$get_current_neg_log_likelihood()-nll_fix)), TOLERANCE_STRICT)
    params_fix$estimate_cov_par_index <- c(0,1,0)
    capture.output( gp_model_fix <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               gp_approx = "vecchia", num_neighbors = 30,
                                               vecchia_ordering = "none", y = y, params = params_fix), file='NUL')
    cov_pars_fix <- c(0.5170731356, 0.1659251265, 0.6109062004, 0.2604720524, 0.1786480774, 0.1041249950)
    nll_fix <- 128.005439
    expect_lt(sum(abs(as.vector(gp_model_fix$get_cov_pars(std_err = TRUE)[c(1,3,5)])-cov_pars_fix[c(1,3,5)])), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model_fix$get_cov_pars(std_err = TRUE)[c(1,3,5)+1])-cov_pars_fix[c(1,3,5)+1])), TOLERANCE_LOOSE)
    expect_lt(sum(abs(gp_model_fix$get_cov_pars(std_err = TRUE)[1,c(1,3)]-params_fix$init_cov_pars[c(1,3)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model_fix$get_current_neg_log_likelihood()-nll_fix)), TOLERANCE_STRICT)
    
    # With "vecchia_latent"
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = 30,
                                        vecchia_ordering = "none", matrix_inversion_method = "cholesky"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_vecchia[3:6])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars_vecchia[1])),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_vecchia), TOLERANCE_MEDIUM)
    # "vecchia_latent" with iterative methods (pivoted Cholesky)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = 30,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_vecchia[3:6])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars_vecchia[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_vecchia), 0.2)
    # "vecchia_latent" with iterative methods (FITC preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = 30,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    params_latent$cg_preconditioner_type = "predictive_process_plus_diagonal"
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_vecchia[3:6])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars_vecchia[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_vecchia), 0.2)
    
    # Vecchia approximation with 30 neighbors and random ordering
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30, 
                                        vecchia_ordering="random"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_vecchia)),0.05)
    expect_gt(gp_model$get_num_optim_iter(), 360) # different compilers result in slightly different results
    expect_lt(gp_model$get_num_optim_iter(), 420)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_vecchia)), 0.1)
    
    # Prediction from fitted model
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu_vecchia)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_vecchia)),TOLERANCE_LOOSE)
    
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_obs_only")
    preds <- predict(gp_model, gp_coords_pred = coords, predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-3)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),1E-3)
    
    # Fisher scoring & default ordering
    params_vecchia_FS <- params_vecchia
    params_vecchia_FS$optimizer_cov <- "fisher_scoring"
    params_vecchia_FS$convergence_criterion <- "relative_change_in_log_likelihood"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering="none", y = y,
                                           params = params_vecchia_FS), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_vecchia)), 0.02)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_vecchia)), TOLERANCE_LOOSE)
    
    # Prediction using given parameters
    cov_pars_pred <- c(0.02,1.2,0.9)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08665472, 0.08664854, 0.49011216)
    expected_cov <- c(0.11891, 0.00000000, 0.00000000, 0.00000000,
                      0.1189129, 0.00000000, 0.00000000, 0.00000000, 0.08108126)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Predict latent process
    pred_var2 <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                         cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu - pred_var2$mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_var$var - cov_pars_pred[1] - pred_var2$var)), TOLERANCE_STRICT)
    
    # Prediction with vecchia_pred_type = "order_obs_first_cond_all"
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08665472, 0.08661259, 0.49011216)
    expected_cov <- c(0.11891004, 0.09889262, 0.00000000, 0.09889262, 0.11891291, 
                      0.00000000, 0.00000000, 0.00000000, 0.08108126)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Predict latent process
    pred_var2 <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                         cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu - pred_var2$mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_var$var - cov_pars_pred[1] - pred_var2$var)), TOLERANCE_STRICT)
    
    # Prediction with vecchia_pred_type = "order_pred_first"
    gp_model$set_prediction_data(vecchia_pred_type = "order_pred_first", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08498682, 0.08502034, 0.49572748)
    expected_cov <- c(1.189037e-01, 9.888624e-02, -1.080005e-05, 9.888624e-02, 
                      1.189065e-01, -1.079431e-05, -1.080005e-05, -1.079431e-05, 8.101757e-02)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Predict latent process
    pred_var2 <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                         cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu - pred_var2$mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_var$var - cov_pars_pred[1] - pred_var2$var)), TOLERANCE_STRICT)
    
    # Prediction with vecchia_pred_type = "latent_order_obs_first_cond_obs_only"
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_obs_only", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08616985, 0.08616384, 0.48721314)
    expected_cov <- c(1.189100e-01, 7.324225e-03, -5.851427e-07, 7.324225e-03, 
                      1.189129e-01, -5.850749e-07, -5.851427e-07, -5.850750e-07, 8.107749e-02)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Predict latent process
    pred_var2 <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                         cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu - pred_var2$mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_var$var - cov_pars_pred[1] - pred_var2$var)), TOLERANCE_STRICT)
    
    # Prediction with vecchia_pred_type = "latent_order_obs_first_cond_all"
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08616985, 0.08616377, 0.48721314)
    expected_cov <- c(1.189100e-01, 9.889258e-02, -5.851418e-07, 9.889258e-02,
                      1.189129e-01, -5.850764e-07, -5.851418e-07, -5.850764e-07, 8.107749e-02)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Predict latent process
    pred_var2 <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                         cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu - pred_var2$mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_var$var - cov_pars_pred[1] - pred_var2$var)), TOLERANCE_STRICT)
    
  })
  
  test_that("Vecchia approximation for Gaussian process model with linear regression term ", {
    
    y <- eps + X%*%beta + xi
    params <- OPTIM_PARAMS_BFGS
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params$init_cov_pars <- init_cov_pars
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, X = X,
                                           params = params), file='NUL')
    cov_pars <- c(0.008993586382, 1.000518636089, 0.094683724304)
    coef <- c(2.309738418, 1.899886232)
    nll_est <- 121.4824924
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)), TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_est), TOLERANCE_STRICT)
    
    # Prediction 
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, gp_coords_pred = coord_test, cov_pars = c(0.01,1,0.1),
                                    X_pred = X_test, predict_cov_mat = TRUE, predict_response = TRUE)
                    , file='NUL')
    expected_mu <- c(1.195997959, 4.070808601, 3.156542000)
    expected_cov <- c( 6.070519415e-01, 1.626496907e-05, 1.036432272e-07, 1.626496907e-05, 3.325922699e-01, 3.202788369e-07, 1.036432272e-07, 3.202788369e-07, 4.073092242e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_LOOSE)
    
    # "vecchia_latent"
    params_latent <- params
    params_latent$init_cov_pars <- init_cov_pars[2:3]
    params_latent$init_aux_pars <- init_cov_pars[1]
    params_latent$optimizer_cov <- "lbfgs"
    params_latent$optimizer_coef <- "lbfgs"
    params_latent$cg_preconditioner_type <- NULL
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, X = X,
                                           params = params_latent, matrix_inversion_method = "cholesky"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(2,3)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), TOLERANCE_LOOSE)
    # "vecchia_latent" and iterative methods 
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, X = X,
                                           params = params_latent, matrix_inversion_method = "iterative"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(2,3)])),2*TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),2*TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), 1.5)
    # "vecchia_latent" and iterative methods (FITC preconditioner)
    params_latent_FITC = params_latent
    params_latent_FITC$cg_preconditioner_type = "predictive_process_plus_diagonal"
    params_latent_FITC$fitc_piv_chol_preconditioner_rank = 70
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, X = X,seed = 1,
                                           params = params_latent_FITC, matrix_inversion_method = "iterative"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(2,3)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),2*TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), 0.2)
    
    # Holding some parameters fix
    params_fix <- OPTIM_PARAMS_BFGS
    params_fix$init_cov_pars <- init_cov_pars
    params_fix$estimate_cov_par_index <- c(1,0,0)
    capture.output( gp_model_fix <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               gp_approx = "vecchia", num_neighbors = 30,
                                               vecchia_ordering = "none", y = y, X=X, params = params_fix), file='NUL')
    cov_pars_fix <- c(0.05425206, 1.43524508, 0.17864808)
    nll_fix <- 122.9300717
    expect_lt(sum(abs(as.vector(gp_model_fix$get_cov_pars(std_err = FALSE))-cov_pars_fix)), TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model_fix$get_cov_pars(std_err = FALSE)[c(2,3)]-params_fix$init_cov_pars[c(2,3)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model_fix$get_current_neg_log_likelihood()-nll_fix)), TOLERANCE_STRICT)
    params_fix$estimate_cov_par_index <- c(0,1,0)
    capture.output( gp_model_fix <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               gp_approx = "vecchia", num_neighbors = 30,
                                               vecchia_ordering = "none", y = y, X=X, params = params_fix), file='NUL')
    expect_lt(sum(abs(gp_model_fix$get_cov_pars(std_err = FALSE)[c(1,3)]-params_fix$init_cov_pars[c(1,3)])),TOLERANCE_STRICT)
  })
  
  test_that("Vecchia approximation for Gaussian process model with cluster_id's not constant ", {
    
    y <- eps + xi
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           vecchia_ordering = "none", y = y, cluster_ids = cluster_ids,
                                           params = DEFAULT_OPTIM_PARAMS), file='NUL')
    cov_pars <- c(0.05870373, 0.08817497, 1.05572659, 0.22911532, 0.12775754, 0.03905891)
    nll <- 129.3761486
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)), TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           vecchia_ordering = "none", y = y, cluster_ids = cluster_ids,
                                           params = DEFAULT_OPTIM_PARAMS_FISHER), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)), 0.1)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.2,0.1001),c(0.9,0.4,0.9001))
    cluster_ids_pred = c(1,3,1)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30, 
                                        vecchia_ordering = "none", cluster_ids = cluster_ids), file='NUL')
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred = 30)
    capture.output( pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                                             cluster_ids_pred = cluster_ids_pred,
                                             cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE), file='NUL')
    expected_mu <- c(-0.01438585, 0.00000000, -0.01500132)
    expected_cov <- c(0.7430552, 0.0000000, 0.6423148, 0.0000000,
                      1.1000000, 0.0000000, 0.6423148, 0.0000000, 0.7434589)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
  })
  
  test_that("Vecchia approximation for Gaussian process model with multiple observations at the same location ", {
    
    y <- eps_multiple + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(unique(coords_multiple)))/3)
    params = OPTIM_PARAMS_BFGS
    params$init_cov_pars <- init_cov_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1, y = y,
                                           vecchia_ordering = "none", params = params), file='NUL')
    cov_pars <- c(0.03713823078, 1.15342626349, 0.19206772520 )
    nll <- 33.43573582
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)), TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    # "vecchia_latent"
    params_latent <- params
    params_latent$init_cov_pars <- init_cov_pars[2:3]
    params_latent$init_aux_pars <- init_cov_pars[1]
    params_latent$optimizer_cov <- "lbfgs"
    params_latent$optimizer_coef <- "lbfgs"
    params_latent$cg_preconditioner_type <- NULL
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y,
                                           params = params_latent, matrix_inversion_method = "cholesky"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(2,3)])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll), TOLERANCE_LOOSE)
    # "vecchia_latent" and matrix_inversion_method = "iterative" (pivoted Cholesky)
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y,
                                           params = params_latent, matrix_inversion_method = "iterative"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(2,3)])),0.03)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll), 0.2)
    # "vecchia_latent" and matrix_inversion_method = "iterative" (FITC preconditioner)
    params_latent_FITC = params_latent
    params_latent_FITC$cg_preconditioner_type = "predictive_process_plus_diagonal"
    params_latent_FITC$fitc_piv_chol_preconditioner_rank = 25
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, seed = 1,
                                           params = params_latent_FITC, matrix_inversion_method = "iterative"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(2,3)])),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll), 0.2)
    
    # Fisher scoring
    params_loc = DEFAULT_OPTIM_PARAMS_FISHER
    params_loc$init_cov_pars <- init_cov_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1, y = y, 
                                           vecchia_ordering = "none", params = params_loc), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)), 0.1)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    cluster_ids_pred = c(1,3,1)
    capture.output( gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n+2,
                                        vecchia_ordering = "none"), file='NUL')
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                                             cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE), file='NUL')
    expected_mu <- c(-0.1460550, 1.0042814, 0.7840301)
    expected_cov <- c(0.6739502109, 0.0008824337, -0.0003815281, 0.0008824337,
                      0.6060039551, -0.0004157361, -0.0003815281, -0.0004157361, 0.7851787946)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    
  })
  
  test_that("Vecchia approximation for Gaussian process and two random coefficients ", {
    
    y <- eps_svc + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3,var(y)/2,mean(dist(coords))/3,var(y)/2,mean(dist(coords))/3)
    # Fit model using gradient descent with Nesterov acceleration
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1,
                                           gp_rand_coef_data = Z_SVC, vecchia_ordering = "none", y = y,
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, maxit=10, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expected_values <- c(0.25740068213, 0.23032520957, 0.83503538559, 0.43677516845, 0.15039055133, 0.09649672707, 1.61010233081,
                         0.82929462187, 0.09015443875, 0.07087533969, 0.25064639566, 0.61853380403, 0.08720821575, 0.33715512575)
    ind <- (1:length(init_cov_pars))*2-1
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[ind]-expected_values[ind])), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[ind+1]-expected_values[ind+1])), 0.1)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Prediction
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n+2,
                                        vecchia_ordering = "none"), file='NUL')
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                                             gp_rand_coef_data_pred = Z_SVC_test,
                                             cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
                    , file='NUL')
    expected_mu <- c(-0.1669209, 1.6166381, 0.2861320)
    expected_cov <- c(9.643323e-01, 3.536846e-04, -1.783557e-04, 3.536846e-04,
                      5.155009e-01, 4.554321e-07, -1.783557e-04, 4.554321e-07, 7.701614e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-149.4422184),1E-5)
    
    # Less neighbors
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           gp_rand_coef_data = Z_SVC, vecchia_ordering = "none", y = y,
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 0.1, use_nesterov_acc = FALSE, maxit=10, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expected_values <- c(0.34489931519, 0.23323165919, 0.79813421101, 0.43059398820, 0.15144409082, 0.10221352187, 1.14797483590, 
                         0.76467661294, 0.10321260903, 0.10115338316, 0.32243986621, 0.63783953997, 0.10613523300, 0.30795562497)
    ind <- (1:length(init_cov_pars))*2-1
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[ind]-expected_values[ind])), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[ind+1]-expected_values[ind+1])), 0.1)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Prediction
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, 
                                        cov_function = "exponential", gp_approx = "vecchia", 
                                        num_neighbors = 30, vecchia_ordering = "none"), file='NUL')
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred = 30)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
    expected_mu <- c(-0.1688452, 1.6181756, 0.2849745)
    expected_cov <- c(0.9643376, 0.0000000, 0.0000000, 0.0000000, 0.5155030, 
                      0.0000000, 0.0000000, 0.0000000, 0.7702683)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-149.4840466), TOLERANCE_STRICT)
  })
  
}
