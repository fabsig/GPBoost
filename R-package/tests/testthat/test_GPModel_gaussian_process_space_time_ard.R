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
  
  test_that("Space-time Gaussian process model with linear regression term ", {
    # Simulate data
    time <- 1:n
    us <- sim_rand_unif(n=n, init_c=0.6418)
    nt <- 20
    for (i in 1:nt) {
      time[us<i/nt & us >= (i-1)/nt] <- i
    } 
    rho_time <- 2
    coords_ST_scaled <- cbind(time/rho_time, coords/rho)
    D_ST <- as.matrix(dist(coords_ST_scaled))
    Sigma_ST <- sigma2_1 * exp(-D_ST) + diag(1E-20,n)
    C_ST <- t(chol(Sigma_ST))
    b_ST <- qnorm(sim_rand_unif(n=n, init_c=0.688))
    eps_ST <- as.vector(C_ST %*% b_ST)
    y <- eps_ST + X%*%beta + xi
    
    init_cov_pars_ST <- c(var(y)/2,var(y)/2,mean(dist(time))/3,mean(dist(coords))/3)
    params_ST = OPTIM_PARAMS_BFGS
    params_ST$init_cov_pars <- init_cov_pars_ST
    
    cov_pars_nll <- c(0.1, 1.6, rho_time * 0.5, 2 * rho)
    coord_test <- rbind(c(10000,0.2,0.9), cbind(time, coords)[c(1,10),])
    coord_test[-1,c(2:3)] <- coord_test[-1,c(2:3)] + 0.01
    X_test <- cbind(rep(1,3),c(0,0,0))
    cov_pars_pred <- c(1, 1, rho_time, rho)
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 272.1497719
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                           y = y, X = X, params = params_ST)
    cov_pars <- c(0.0000580483961, 0.2722859690020, 1.0181004093911, 0.3212035580580, 1.3496061931943, 0.7536302404641, 0.1157315017765, 0.0532282672019)
    coef <- c(1.9593121521, 0.1479700951, 2.1693074509, 0.1392691184)
    nrounds <- 36
    nll_opt <- 138.1879339
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu <- c(1.959312152, 1.940440108, 2.566912825)
    expected_cov <- c(2.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 
                      1.5919472301920 , 0.0001229642924, 0.0000000000, 0.0001229642924, 1.5650143857452)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    
    ## Other shape parameters
    init_cov_pars_ST_15 <- c(var(y)/2,var(y)/2,mean(dist(time))/4.7*sqrt(3),mean(dist(coords))/4.7*sqrt(3))
    params_ST_15 = OPTIM_PARAMS_BFGS
    params_ST_15$init_cov_pars <- init_cov_pars_ST_15
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 1.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_1_5_exp <- 288.6072086
    expect_lt(abs(nll-nll_1_5_exp),TOLERANCE_STRICT)
    # General shape
    if (!SKIP_BESSEL_COV_TESTS) {
      gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 1.5 + 1E-5)
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
      expect_lt(abs(nll-nll_1_5_exp),TOLERANCE_MEDIUM)
    }
    # Fit model
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", 
                           cov_fct_shape = 1.5, y = y, X = X, params = params_ST_15)
    cov_pars_1_5 <- c(0.6889137858, 0.1920021471, 0.3249804446, 0.2068533561, 5.0964362212, 4.0329915408, 0.2066662908, 0.1297232407)
    coef_1_5 <- c( 1.9627841913, 0.1881284120, 2.2133311549, 0.1411491729)
    nrounds_1_5 <- 21
    nll_opt_1_5 <- 138.6349682
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_1_5)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_1_5)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_1_5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_1_5), TOLERANCE_STRICT)
    # General shape
    if (!SKIP_BESSEL_COV_TESTS) {
      gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                             cov_fct_shape = 1.5 + 1E-4, y = y, X = X, params = params_ST_15)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_1_5)),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_1_5)),TOLERANCE_MEDIUM)
      expect_equal(gp_model$get_num_optim_iter(), nrounds_1_5)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_1_5), TOLERANCE_MEDIUM)
    }
    # Shape = 2.5: evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 2.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_2_5_exp <- 296.7149408
    expect_lt(abs(nll-nll_2_5_exp),TOLERANCE_STRICT)
    
    ##############
    ## With Vecchia approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X = X, params = params_ST), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)]-cov_pars[c(1,3,5,7)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)+1]-cov_pars[c(1,3,5,7)+1])),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction 
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "order_pred_first", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    
    ## Less neighbors 
    # Evaluate negative log-likelihood
    num_neighbors <- 50
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-272.1376522),TOLERANCE_STRICT)
    # Different orderings
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "time"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-272.1498125),TOLERANCE_LOOSE)
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "time_random_space"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-272.1498202),TOLERANCE_LOOSE)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X = X, params = params_ST), 
                    file='NUL')
    cov_pars_nn <- c(6.369928869e-05, 2.730398462e-01, 1.018337938e+00, 3.223076467e-01, 1.359779342e+00, 7.327419245e-01, 1.155289567e-01, 5.155877935e-02)
    coef_nn <- c(1.9580931653, 0.1481539264, 2.1696897178, 0.1392611361)
    nrounds_nn <- 33
    nll_opt_nn <- 138.1864467
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)]-cov_pars_nn[c(1,3,5,7)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)+1]-cov_pars_nn[c(1,3,5,7)+1])),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
    # Different ordering
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "time",
                                           y = y, X = X, params = params_ST), 
                    file='NUL')
    cov_pars_nn <- c(9.152856134e-05, 2.631819547e-01, 1.017750095e+00, 3.144000322e-01, 1.333863387e+00, 7.214933499e-01, 1.161419522e-01, 5.173674604e-02)
    coef_nn <- c(1.9593629633, 0.1477649750, 2.1698058447, 0.1392772621)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)]-cov_pars_nn[c(1,3,5,7)])), 0.03)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),TOLERANCE_LOOSE)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=num_neighbors)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu_nn <- c(1.958346, 1.939918, 2.566458)
    expected_cov_nn <- c(2.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.591947245758, 0.000120255663, 0.000000000000, 0.000120255663, 1.565014424976)
    expect_lt(sum(abs(pred$mu-expected_mu_nn)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_nn)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov_nn[c(1,5,9)])),TOLERANCE_STRICT)
    
    ##############
    ## Multiple observations at the same location
    ##############
    coords_ST = cbind(time, coords)
    coords_ST[1:5,] <- coords_ST[(n-4):n,]
    init_cov_pars_mult_ST <- c(var(y)/2,var(y)/2,mean(dist(unique(coords_ST)[,1]))/3,mean(dist(unique(coords_ST)[,-1]))/3)
    params_mult_ST <- OPTIM_PARAMS_BFGS
    params_mult_ST$init_cov_pars <- init_cov_pars_mult_ST
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 276.47191976324
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                           y = y, X = X, params = params_mult_ST)
    cov_pars <- c(0.4930726098, 0.2020862417, 0.5269095541, 0.2463829260, 4.0915948605, 3.0601130769, 0.2163649721, 0.1361137661)
    coef <- c(1.9537559557, 0.2156366688, 2.2009251106, 0.1379433393)
    nrounds <- 23
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)]-cov_pars[c(1,3,5,7)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)+1]-cov_pars[c(1,3,5,7)+1])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    ## With Vecchia approximation
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X = X, params = params_mult_ST), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)]-cov_pars[c(1,3,5,7)])),TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)+1]-cov_pars[c(1,3,5,7)+1])),0.3)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    
    #####################
    ## Gneiting space-time covariance
    #####################
    cov_pars_nll_gneiting_fixed_nu <- c(0.1,1,0.2,2,0.5,1.5,0.5,2)
    nll_exp <- 604.779654987741
    # Evaluate negative log-likelihood with fixed smoothness
    gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll_gneiting_fixed_nu,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model without estimating the smoothness parameter. This does not require the Bessel function.
    params_ST_gneiting_fixed_nu <- OPTIM_PARAMS_BFGS
    params_ST_gneiting_fixed_nu$init_cov_pars <- cov_pars_nll_gneiting_fixed_nu
    params_ST_gneiting_fixed_nu$estimate_cov_par_index <- c(1,1,1,1,1,0,1,1)
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting",
                                           y = y, X = X, params = params_ST_gneiting_fixed_nu),
                    file='NUL')
    expect_equal(as.vector(gp_model$get_cov_pars(std_err = FALSE))[6], 1.5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-137.2451317867212), TOLERANCE_STRICT)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, cov_pars = cov_pars_nll_gneiting_fixed_nu)
    expected_mu <- c(1.965547011, 1.856092042, 2.429890300)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)

    if (!SKIP_BESSEL_COV_TESTS) {
      cov_pars_nll_gneiting <- c(0.1,1,0.2,2,0.5,1.5,0.5,2)
      nll_exp <- 604.779654987741
      # Evaluate negative log-likelihood
      gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting", cov_fct_shape = 0.5)
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll_gneiting,y=y)
      expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
      params_ST$init_cov_pars <- cov_pars_nll_gneiting
      # Fit model
      capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting", cov_fct_shape = 0.5,
                                             y = y, X = X, params = params_ST), 
                      file='NUL')
      cov_pars <- c(0.01118145, 0.07067827, 1.01161075,  0.17122660, 0.60008963,   
                    4.96597785, 58.12214002, 389.63570115, 4.12563119, 3.31118654,  
                    11.36187543, 146.57294986, 5.22611159, 6.71388132, 0.24194531, 2.01265616)
      coef <- c(1.9652662, 0.1455411, 2.1144101, 0.1316155)
      nrounds <- 26
      nll_opt <- 137.428674247055
      # Absolute sum over 16 values, the largest of which are the standard errors of the range
      # parameters (up to ~390), so a budget of 1e-3 is a reference-platform tolerance: with gcc on
      # Linux the same stationary point is reached (identical iteration count, and the negative
      # log-likelihood and the coefficients still agree to 1e-5 below) with a sum of ~1.6e-3
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),relax_tolerance(TOLERANCE_MEDIUM, cov_pars))
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
      expect_equal(gp_model$get_num_optim_iter(), nrounds)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
      # Prediction 
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_nll_gneiting)
      expected_mu <- c(1.965266, 1.865088, 2.441091)
      expected_cov <- c(1.100000e+00, -5.406416e-11, -2.934254e-11, 
                        -5.406416e-11,  1.610924e-01,  5.727994e-05, 
                        -2.934254e-11,  5.727994e-05, 1.460070e-01)
      expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
      expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_nll_gneiting)
      expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
      expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
      
      ##############
      ## With Vecchia approximation
      ##############
      # Evaluate negative log-likelihood
      num_neighbors <- 50
      capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting", cov_fct_shape = 0.5,
                                          gp_approx = "vecchia_euclidean_based", num_neighbors = num_neighbors, vecchia_ordering = "none"), 
                      file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll_gneiting,y=y)
      expect_lt(abs(nll-603.189168889409),TOLERANCE_STRICT)
      # Fit model
      capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting", cov_fct_shape = 0.5,
                                             gp_approx = "vecchia_euclidean_based", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                             y = y, X = X, params = params_ST), 
                      file='NUL')
      cov_pars_nn <- c(1.920056e-03, 4.691929e-02, 1.015382e+00, 1.608047e-01, 1.156587e+00, 
                       1.526540e+01, 1.271204e+02, 3.842982e+03, 1.594210e+00, 5.499628e+00, 
                       6.151744e+01, 3.687555e+03, 1.352373e+01, 9.283513e+00, 1.608771e-01, 1.815412e+00)
      coef_nn <- c(1.9676559, 0.1448350, 2.1328759, 0.1315564)
      nrounds_nn <- 29
      nll_opt_nn <- 137.140644557018
      capture.output( expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)]-cov_pars_nn[c(1,3,5,7)])),TOLERANCE_LOOSE), file='NUL')
      expect_lt(sum(abs(as.vector((gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)+1]-cov_pars_nn[c(1,3,5,7)+1])/cov_pars_nn[c(1,3,5,7)+1])),0.2)
      capture.output( expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),TOLERANCE_STRICT), file='NUL')
      expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
      # Prediction 
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, cov_pars = cov_pars_nll_gneiting)
      expected_mu <- c(1.967656, 1.860779, 2.435741)
      expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
      
      ## Correlation-based neighbor search 
      # Evaluate negative log-likelihood
      capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting", cov_fct_shape = 0.5,
                                          gp_approx = "vecchia_correlation_based", num_neighbors = num_neighbors, vecchia_ordering = "none"), 
                      file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll_gneiting,y=y)
      nll_exp <- 602.88672043745
      expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
      # Default should be correlation-based
      capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting", cov_fct_shape = 0.5,
                                          gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none"), 
                      file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll_gneiting,y=y)
      expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
      # Fit model
      capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting", cov_fct_shape = 0.5,
                                             gp_approx = "vecchia_correlation_based", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                             y = y, X = X, params = params_ST), 
                      file='NUL')
      cov_pars_nn <- c(0.02114328, 0.14890136, 1.00313912,0.21692871,0.23757860,1.25912716,
                       55.81628997, 270.55795042, 4.63369088, 2.26817998, 6.59072999,
                       61.29507043, 3.26058344, 5.32202902, 0.25530661, 1.87121857)
      coef_nn <- c(1.9795317, 0.1424944, 2.2360390, 0.1323973)
      nrounds_nn <- 23
      nll_opt_nn <- 138.089095556994
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)]-cov_pars_nn[c(1,3,5,7)])),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector((gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7)+1]-cov_pars_nn[c(1,3,5,7)+1])/cov_pars_nn[c(1,3,5,7)+1])),0.2)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),TOLERANCE_STRICT)
      expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
      # Prediction 
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, cov_pars = cov_pars_nll_gneiting)
      expected_mu <- c(1.979532, 1.836721, 2.405857)
      expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
      
      ##############
      ## With FITC approximation (space-time separated kMeans++)
      ##############
      # Evaluate negative log-likelihood
      capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting", cov_fct_shape = 0.5,
                                          gp_approx = "fitc", ind_points_selection = "space_time_kmeans++", num_ind_points = 30), 
                      file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll_gneiting,y=y)
      expect_lt(abs(nll-339.411253590468),TOLERANCE_STRICT)
      # Fit model
      capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "space_time_gneiting", cov_fct_shape = 0.5,
                                             gp_approx = "fitc", ind_points_selection = "space_time_kmeans++", num_ind_points = 30,
                                             y = y, X = X, params = params_ST), 
                      file='NUL')
      cov_pars_nn <- c(0.14666538, 0.82530619, 0.02268084, 101.37316301, 0.17523760, 154.00191351, 0.01261858, 18.64147777)
      coef_nn <- c(1.9194304, 0.1401412, 2.2149501, 0.1364415)
      nrounds_nn <- 39
      nll_opt_nn <- 137.073147464373
      capture.output( expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_nn)),0.04), file='NUL')
      capture.output( expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),0.001), file='NUL')
      expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
      # Prediction 
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, cov_pars = cov_pars_nll_gneiting)
      expected_mu <- c(1.919430, 1.751229, 2.389440)
      expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    }
  })
  
  test_that("ARD Gaussian process model with linear regression term ", {
    # Simulate data
    d <- 3 # dimension of GP locations
    coords_ARD <- matrix(sim_rand_unif(n=n*d, init_c=0.981), ncol=d)
    sigma2_1 <- 1^2 # marginal variance of GP
    rhos <- c(0.2,0.4,0.3)
    coords_ARD_scaled <- coords_ARD
    for (i in 1:dim(coords_ARD)[2]) coords_ARD_scaled[,i] <- coords_ARD[,i] / rhos[i]
    D_ARD <- as.matrix(dist(coords_ARD_scaled))
    Sigma_ARD <- sigma2_1 * exp(-D_ARD) + diag(1E-20,n)
    C_ARD <- t(chol(Sigma_ARD))
    b_ARD <- qnorm(sim_rand_unif(n=n, init_c=0.978688))
    eps_ARD <- as.vector(C_ARD %*% b_ARD)
    y <- eps_ARD + X%*%beta + xi
    
    init_cov_pars_ARD <- c(var(y)/2,var(y)/2)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars_ARD <- c(init_cov_pars_ARD, mean(dist(coords_ARD[,i])/3))
    params_ARD <- OPTIM_PARAMS_BFGS
    params_ARD$init_cov_pars <- init_cov_pars_ARD
    
    cov_pars_nll <- c(0.1, 1.6, 0.5 * rhos)
    coord_test <- rbind(c(10000,0.2,0.9), coords_ARD[c(1,10),])
    coord_test[-1,c(2:3)] <- coord_test[-1,c(2:3)] + 0.01
    X_test <- cbind(rep(1,3),c(0,0,0))
    cov_pars_pred <- c(1, 1, rhos)
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 249.4821103
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                           y = y, X = X, params = params_ARD)
    cov_pars <- c(1.0739209e-05, 7.6269508e-02, 1.2557353e+00, 4.2757479e-01, 3.5227030e-01, 1.7506071e-01, 5.5749636e-01, 2.8785130e-01, 3.3151485e-01, 1.6483089e-01)
    coef <- c(2.268094879, 0.456234626, 1.721694800, 0.084365857)
    nrounds <- 30
    nll_opt <- 111.19846
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu <- c( 2.2680949, 3.0698811, 3.3288540)
    expected_cov <- c(2.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.4864426774, -0.0012712546, 0.0000000000, -0.0012712546, 1.4070721689)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    
    ## Other shape parameters
    init_cov_pars_ARD <- c(var(y)/2,var(y)/2)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars_ARD <- c(init_cov_pars_ARD, mean(dist(coords_ARD[,i])/4.7*sqrt(3)))
    params_ARD_15 <- OPTIM_PARAMS_BFGS
    params_ARD_15$init_cov_pars <- init_cov_pars_ARD
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 1.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_1_5_exp <- 276.2341252
    expect_lt(abs(nll-nll_1_5_exp),TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 1.5 + 1E-5)
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
      expect_lt(abs(nll-nll_1_5_exp),TOLERANCE_MEDIUM)
    }
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", 
                           cov_fct_shape = 1.5, y = y, X = X, params = params_ARD_15)
    cov_pars_1_5 <- c(0.052223946, 0.041194469, 1.135354760, 0.300760025, 0.238321185, 0.059517930, 0.318967024, 0.081431654, 0.200689095, 0.049796063)
    coef_1_5 <- c( 2.299970861, 0.312730131, 1.731089000, 0.074109573)
    nrounds_1_5 <- 15
    nll_opt_1_5 <- 107.83105
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_1_5)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_1_5)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_1_5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_1_5), TOLERANCE_STRICT)
    # General shape
    if (!SKIP_BESSEL_COV_TESTS) {
      gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", 
                             cov_fct_shape = 1.5 - 1E-4, y = y, X = X, params = params_ARD_15)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_1_5)),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_1_5)),TOLERANCE_MEDIUM)
      expect_equal(gp_model$get_num_optim_iter(), nrounds_1_5)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_1_5), TOLERANCE_MEDIUM)
    }
    # Gaussian covariance: evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "gaussian_ard")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_gaussian_exp <- 322.3104221
    expect_lt(abs(nll-nll_gaussian_exp),TOLERANCE_STRICT)
    # Fit model
    init_cov_pars_gauss <- c(var(y)/2,var(y)/2)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars_gauss <- c(init_cov_pars_gauss, sqrt((mean(dist(coords_ARD[,i])))^2/3))
    params_loc <- OPTIM_PARAMS_BFGS
    params_loc$init_cov_pars <- init_cov_pars_gauss
    gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "gaussian_ard",
                           y = y, X = X, params = params_loc)
    cov_pars_gaussian <- c(0.066978015, 0.030112965, 1.049395096, 0.234467918, 0.238649844, 0.038584533, 0.307654564, 0.038453314, 0.218393133, 0.039438349)
    coef_gaussian <- c(2.33954165, 0.22155905, 1.74671636, 0.06863206)
    nrounds_gaussian <- 26
    nll_opt_gauss <- 106.56845
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_gaussian)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_gaussian)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_gaussian)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_gauss), TOLERANCE_MEDIUM)
    # Matern with shape estimated
    params_ARD_est_shape <- OPTIM_PARAMS_BFGS
    params_ARD_est_shape$init_cov_pars <- c(init_cov_pars_ARD,1.5)
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output(     gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard_estimate_shape",
                                                 y = y, X = X, params = params_ARD_est_shape), 
                      file='NUL')
      cov_pars_est_shape <- c(6.685939690e-02, 3.389668693e-02, 1.050559295e+00, 2.417603051e-01, 1.703677225e-01, 3.619481495e-02, 2.179632368e-01, 4.622495774e-02, 1.544734537e-01, 3.248756329e-02, 1.418090169e+02, 5.839766923e+03)
      coef_est_shape <- c(2.338255631, 0.222555043, 1.746682245, 0.068682910)
      nrounds_est_shape <- 48
      nll_opt_est_shape <- 106.56952
      capture.output(expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[1:10]-cov_pars_est_shape[1:10])),TOLERANCE_LOOSE), file='NUL')
      expect_lt(sum(abs((gp_model$get_cov_pars(std_err = TRUE))[11]-cov_pars_est_shape[11])/cov_pars_est_shape[11]),0.2)
      expect_lt(sum(abs((gp_model$get_cov_pars(std_err = TRUE))[12]-cov_pars_est_shape[12])/cov_pars_est_shape[12]),0.25)
      capture.output(expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_est_shape)),TOLERANCE_LOOSE), file='NUL')
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_est_shape), TOLERANCE_MEDIUM)
    }
    
    ##############
    ## With Vecchia approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X = X, params = params_ARD), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction 
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "order_pred_first", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    if (Sys.getenv("GPBOOST_ADDITIONAL_SLOW_TESTS") == "GPBOOST_ADDITIONAL_SLOW_TESTS" &&
        !SKIP_BESSEL_COV_TESTS) {
      # Estimate shape, slow test
      capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard_estimate_shape", cov_fct_shape = 0.5,
                                             gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                             y = y, X = X, params = params_ARD_est_shape), 
                      file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars_est_shape[c(1,3,5,7,9)])),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars_est_shape[c(1,3,5,7,9)+1])),0.5)
      # Absolute sum over the two coefficients and their standard errors of a model whose shape
      # parameter is estimated, so a budget of 1e-5 is a reference-platform tolerance
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_est_shape)),relax_tolerance(TOLERANCE_STRICT, coef_est_shape))
      expect_equal(gp_model$get_num_optim_iter(), nrounds_est_shape)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_est_shape), TOLERANCE_STRICT)
    }
    
    ## Less neighbors 
    # Evaluate negative log-likelihood
    num_neighbors <- 50
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-249.4121769),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X = X, params = params_ARD), 
                    file='NUL')
    cov_pars_nn <- c(4.8571377e-05, 8.0447133e-02, 1.2405613e+00, 4.0823851e-01, 3.4498495e-01, 1.6571188e-01, 5.5034201e-01, 2.7332050e-01, 3.2562881e-01, 1.5515080e-01)
    coef_nn <- c(2.274632318, 0.448551336, 1.721675310, 0.084580779)
    nrounds_nn <- 32
    nll_opt_nn <- 111.271
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars_nn[c(1,3,5,7,9)+1])),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_ITERATIVE)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=num_neighbors)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu_nn <- c(2.2746323, 3.0707369, 3.3264722)
    expected_cov_nn <- c(2.000000000, 0.000000000, 0.000000000, 0.000000000, 1.4864450, 0.0000000, 0.0000000, 0.0000000, 1.4071153)
    expect_lt(sum(abs(pred$mu-expected_mu_nn)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_nn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.05)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov_nn[c(1,5,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    # Fit model with lbfgs & wls
    params_loc <- params_ARD
    params_loc$optimizer_cov <- "lbfgs"
    params_loc$optimizer_coef <- "wls"
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X = X, params = params_loc), 
                    file='NUL')
    cov_pars_nn <- c(5.4017458e-06, 7.3140556e-02, 1.2474697e+00, 3.6256732e-01, 3.4803261e-01, 1.2928035e-01, 5.5519083e-01, 2.0767558e-01, 3.2678054e-01, 1.1788572e-01)
    coef_nn <- c(2.266088422, 0.451684073, 1.722038529, 0.084540028)
    nll_opt_nn <- 111.2698
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_ITERATIVE)
    # Fit model with lbfgs & only intercept
    params_loc <- params_ARD
    params_loc$optimizer_cov <- "lbfgs"
    params_loc$optimizer_coef <- "lbfgs"
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X = rep(1,n), params = params_loc), 
                    file='NUL')
    cov_pars_nn <- c(1.16206347, 0.63694914, 1.46482515, 0.75275676, 0.15448515, 0.11041991, 0.43153686, 0.35339319, 0.14355878, 0.10018139)
    coef_nn <- c(2.48690635, 0.33191588)
    nll_opt_nn <- 183.819
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 9)
    params_loc$optimizer_coef <- "wls"
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X = rep(1,n), params = params_loc), 
                    file='NUL')
    cov_pars_nn <- c(1.16230918, 0.63851718, 1.46250385, 0.75347404, 0.15389105, 0.11017790, 0.43024530, 0.35290786, 0.14307058, 0.10001178)
    coef_nn <- c(2.48692272, 0.33090192)
    nll_opt_nn <- 183.81901
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    
    ##############
    ## With VIF (Full scale Vecchia) approximation
    ##############
    OPTIM_PARAMS_BFGS$init_cov_pars <- c(1.3224515, 1.3224515, 0.1171273, 0.1189399, 0.1177321)
    #### n-1 Vecchia neighbors
    ### Euclidean-based Neighbor search
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "full_scale_vecchia", 
                                        num_neighbors = n-1, num_ind_points = 20 ,vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "full_scale_vecchia",  num_neighbors = n-1, num_ind_points = 20,
                                           vecchia_ordering = "none",
                                           y = y, X = X, params = OPTIM_PARAMS_BFGS), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(1,3,5,7,9)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef[c(1,3)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction 
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    
    ### Correlation-based Neighbor search
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "full_scale_vecchia_correlation_based",  
                                        num_neighbors = n-1, num_ind_points = 20 ,vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "full_scale_vecchia_correlation_based",  num_neighbors = n-1, num_ind_points = 20,
                                           vecchia_ordering = "none",
                                           y = y, X = X, params = OPTIM_PARAMS_BFGS), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(1,3,5,7,9)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef[c(1,3)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction 
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    
    #### n-1 inducing points
    ### Euclidean-based Neighbor search
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "full_scale_vecchia",  
                                        num_neighbors = 5, num_ind_points = n-1 ,vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "full_scale_vecchia",  num_neighbors = 5, num_ind_points = n-1,
                                           vecchia_ordering = "none",
                                           y = y, X = X, params = OPTIM_PARAMS_BFGS), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(1,3,5,7,9)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef[c(1,3)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction 
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    
    ### Correlation-based Neighbor search
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "full_scale_vecchia_correlation_based",  
                                        num_neighbors = 5, num_ind_points = n-1,vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "full_scale_vecchia_correlation_based",   num_neighbors = 5, num_ind_points = n-1,
                                           vecchia_ordering = "none",
                                           y = y, X = X, params = OPTIM_PARAMS_BFGS), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars[c(1,3,5,7,9)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef[c(1,3)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction 
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    
    ### Less neighbors and inducing points
    ## Euclidean-based Neighbor search
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "full_scale_vecchia", 
                                        num_neighbors = 10, num_ind_points = 20, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-250.051325253846),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "full_scale_vecchia", 
                                           num_neighbors = 10, num_ind_points = 20,vecchia_ordering = "none",
                                           y = y, X = X, params = OPTIM_PARAMS_BFGS), 
                    file='NUL')
    cov_pars_nn <- c(5.649128e-06, 1.270109e+00, 3.806525e-01, 5.835826e-01, 3.327165e-01)
    coef_nn <- c(2.271157, 1.729481)
    nrounds_nn <- 34
    nll_opt_nn <- 110.248660959757
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_nn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_nn)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu_nn <- c(2.271157, 3.068187, 3.313782)
    expected_cov_nn <- c(2.000000000, 0.000000000, 0.000000000, 0.000000000, 1.486501925 , -0.000681858  , 0.0000000, -0.000681858  , 1.408111055)
    expect_lt(sum(abs(pred$mu-expected_mu_nn)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_nn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov_nn[c(1,5,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    
    ## Correlation-based Neighbor search
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "full_scale_vecchia_correlation_based", 
                                        num_neighbors = 10, num_ind_points = 20, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-249.37405899436),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "full_scale_vecchia_correlation_based", 
                                           num_neighbors = 10, num_ind_points = 20,vecchia_ordering = "none",
                                           y = y, X = X, params = OPTIM_PARAMS_BFGS), 
                    file='NUL')
    cov_pars_nn <- c(4.343210e-06, 1.246736e+00, 3.685636e-01, 5.591977e-01, 3.140119e-01)
    coef_nn <- c(2.278199, 1.717998)
    nrounds_nn <- 38
    nll_opt_nn <- 111.014825115058
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_nn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_nn)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu_nn <- c(2.278199, 3.081812, 3.337652)
    expected_cov_nn <- c(2.000000000, 0.000000000, 0.000000000, 0.000000000, 1.4864848946, -0.0009735682, 0.0000000, -0.0009735682, 1.4072311387)
    expect_lt(sum(abs(pred$mu-expected_mu_nn)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_nn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov_nn[c(1,5,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    
    ##############
    ## With FITC approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "fitc", num_ind_points = n, ind_points_selection = "random"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_MEDIUM)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = n, ind_points_selection = "random",
                                           y = y, X = X, params = params_ARD), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_MEDIUM)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    
    ## Less inducing points 
    # Evaluate negative log-likelihood
    num_ind_points <- 50
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "fitc", num_ind_points = num_ind_points, ind_points_selection = "kmeans++"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-250.029401175403),TOLERANCE_MEDIUM)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = num_ind_points, ind_points_selection = "kmeans++",
                                           y = y, X = X, params = params_ARD), 
                    file='NUL')
    cov_pars_nn <- c(1.735005e-05, 8.627363e-02, 1.290560e+00, 4.357872e-01, 3.452861e-01, 1.654098e-01, 6.571689e-01, 3.280626e-01, 3.710781e-01, 1.798495e-01)
    coef_nn <- c(2.27868746, 0.48056816, 1.70536543, 0.08686789)
    nll_opt_nn <- 112.714161295749
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars_nn[c(1,3,5,7,9)+1])),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), 2)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu_nn <- c(2.278687, 2.603583, 3.386486)
    expected_cov_nn <- c(2.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.867991677, -0.002758946, 0.0000000000, -0.002758946, 1.576546577)
    expect_lt(sum(abs(pred$mu-expected_mu_nn)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_nn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$mu-expected_mu)),1)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov_nn[c(1,5,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1)
    # Estimate shape
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard_estimate_shape", 
                                             gp_approx = "fitc", num_ind_points = 10, 
                                             ind_points_selection = "kmeans++",#"random" crashes for some reason
                                             y = y, X = X, params = params_ARD_est_shape),
                      file='NUL')
      cov_pars_nn <- c(5.539052e-04, 9.491968e-02, 1.256650e+00, 2.960151e-01, 1.401322e-01, 3.614054e-02, 2.532974e-01, 7.085940e-02, 3.727976e-01, 1.118897e-01, 2.613374e+01, 3.704829e+02)
      coef_nn <- c(2.2803513, 0.2614949, 1.8147817, 0.1016647)
      nll_opt_nn <- 124.4784
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),TOLERANCE_MEDIUM)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
      expect_equal(gp_model$get_num_optim_iter(), 26)
    }
    
    ##############
    ## Multiple observations at the same location
    ##############
    coords_ARD_mult = coords_ARD
    coords_ARD_mult[1:5,] <- coords_ARD_mult[(n-4):n,]
    init_cov_pars_ARD_mult <- c(var(y)/2,var(y)/2)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars_ARD_mult <- c(init_cov_pars_ARD_mult, mean(dist(unique(coords_ARD_mult)[,i])/3))
    params_ARD_mult <- OPTIM_PARAMS_BFGS
    params_ARD_mult$init_cov_pars <- init_cov_pars_ARD_mult
    
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 268.9672548
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                           y = y, X = X, params = params_ARD_mult)
    cov_pars <- c(0.24243853, 0.10619517, 0.99821242, 0.37686739, 0.28320232, 0.16594982, 0.59983666, 0.37491360, 0.40864020, 0.24825380)
    coef <- c(2.20969325, 0.42366874, 1.70924316, 0.10869877)
    nrounds <- 12
    nll_opt <- 125.68513
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_ITERATIVE)
    # Prediction
    coord_test_mult <- rbind(c(10000,0.2,0.9),c(10000,0.2,0.9), coords_ARD_mult[1,])
    coord_test_mult[-c(1,2),c(2:3)] <- coord_test_mult[-c(1,2),c(2:3)] + 0.01
    exp_mu_mult <- c( 2.2096932, 2.2096932, 2.7995200)
    exp_cov_mult <- c(2.000000, 1.000000, 0.000000, 1.000000, 2.000000, 0.000000, 0.000000, 0.000000, 1.3481487)
    pred <- predict(gp_model, gp_coords_pred = coord_test_mult,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-exp_mu_mult)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-exp_cov_mult)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test_mult,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-exp_mu_mult)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_mult[c(1,5,9)])),TOLERANCE_STRICT)
    
    ## With Vecchia approximation
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X = X, params = params_ARD_mult), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),2*TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_ITERATIVE)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test_mult,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-exp_mu_mult)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-exp_cov_mult)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test_mult,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-exp_mu_mult)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_mult[c(1,5,9)])),TOLERANCE_STRICT)
    ## With fitc approximation
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "fitc", num_ind_points = dim(unique(coords_ARD_mult))[1], ind_points_selection = "random"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = dim(unique(coords_ARD_mult))[1], ind_points_selection = "random",
                                           y = y, X = X, params = params_ARD_mult), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),2*TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test_mult,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-exp_mu_mult)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_mult[c(1,5,9)])),TOLERANCE_STRICT)
  })## end ARD Gaussian process model with linear regression term
 
}
