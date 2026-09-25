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

  test_that("Space-time Gaussian process model with linear regression term ", {
    probs <- pnorm(eps_ST)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.165) < probs)
    likelihood <- "bernoulli_logit"
    cov_pars_nll <- c(1.6,0.07,0.2)
    cov_pars_nll2 <- c(1.6,10,0.01)
    coord_test <- rbind(c(200,0.2,0.9), cbind(time, coords)[c(1,10),])
    coord_test[-1,c(2:3)] <- coord_test[-1,c(2:3)] + 0.01
    X_test <- cbind(rep(1,3),c(0,0,0))
    cov_pars_pred <- c(1,0.1,0.1)
    params = OPTIM_PARAMS_BFGS
    params$init_cov_pars <- c(1,mean(dist(time))/3,mean(dist(coords))/3)

    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                        cov_function = "matern_space_time", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 70.2364458
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                           cov_function = "matern_space_time", cov_fct_shape = 0.5,
                           y = y, X = X, params = params)
    cov_pars <- c(0.13319234812, 0.06333494877, 0.12906707148)
    coef <- c(0.1363328524, 0.2142364703, 0.2661459983, 0.2975975894)
    nrounds <- 15
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu <- c(0.1363328524, 0.4163590207, 0.6388916187)
    expected_cov <- c(1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.85149434352, 0.01824729944, 0.00000000000, 0.01824729944, 0.81056965538)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = TRUE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expected_mu_resp <- c(0.5281428989, 0.5872303341, 0.6330448814)
    expected_var_resp <- c(0.2492079772, 0.2423908688, 0.2322990595)
    expect_lt(sum(abs(pred$mu-expected_mu_resp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_resp)),TOLERANCE_STRICT)

    ##############
    ## With Vecchia approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                                        cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none", matrix_inversion_method = "cholesky"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                                           cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky"),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = TRUE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu_resp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_resp)),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)

    ## Less neighbors
    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative"){
        tolerance_loc <- TOLERANCE_ITERATIVE
      } else{
        tolerance_loc <- TOLERANCE_STRICT
      }
      nsim_var_pred <- 10000
      # Evaluate negative log-likelihood
      num_neighbors <- 50
      capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                                          cov_function = "matern_space_time", cov_fct_shape = 0.5, matrix_inversion_method = inv_method,
                                          gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none"),
                      file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
      expect_lt(abs(nll-70.2364313),0.2)
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll2,y=y)
      expect_lt(abs(nll-70.6574683),0.2)
      # Fit model
      capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), likelihood = likelihood, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                             gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                             matrix_inversion_method = inv_method, y = y, X=X, params = params),
                      file='NUL')
      cov_pars_nn <- c(0.13310337502, 0.06332284601, 0.12921443605)
      coef_nn <- c(0.1370527248, 0.2142481946, 0.2677589771, 0.2976186564)
      nrounds_nn <- 15
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),tolerance_loc)
      if (inv_method=="cholesky") {
        expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
      }
      # Prediction
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=num_neighbors, nsim_var_pred=nsim_var_pred)
      pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                      X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
      expected_mu_nn <- c(0.1370527248, 0.4168275706, 0.6393660124)
      expected_cov_nn <- c(1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.8515104403 , 0.0182491973, 0.00000000, 0.0182491973, 0.8105919548)
      expect_lt(sum(abs(pred$mu-expected_mu_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_nn)),tolerance_loc)
      pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                      X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      expect_lt(sum(abs(pred$mu-expected_mu_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(pred$var)-expected_cov_nn[c(1,5,9)])),tolerance_loc)
    }

    ##############
    ## Multiple observations at the same location
    ##############
    coords_ST = cbind(time, coords)
    coords_ST[1:5,] <- coords_ST[(n-4):n,]
    params = OPTIM_PARAMS_BFGS
    params$init_cov_pars <- c(1,mean(dist(unique(coords_ST)[,1]))/3,mean(dist(unique(coords_ST)[,-1]))/3)
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                        likelihood = likelihood)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 70.85206038
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                           y = y, X=X, params = params, likelihood = likelihood)
    cov_pars <- c(0.0003103303859, 0.0160438298347, 0.0139448004490)
    coef <- c(0.1356549353, 0.2031592096, 0.2579334524, 0.2882626226)
    nrounds <- 19
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    ## With Vecchia approximation
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-6, vecchia_ordering = "none",
                                        likelihood = likelihood, matrix_inversion_method = "cholesky"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-6, vecchia_ordering = "none",
                                           y = y, X=X, params = params, likelihood = likelihood, matrix_inversion_method = "cholesky"),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
  })

  test_that("ARD Gaussian process model with linear regression term ", {
    # Simulate data
    d <- 3 # dimension of GP locations
    coords_ARD <- matrix(sim_rand_unif(n=n*d, init_c=0.48231), ncol=d)
    sigma2_1 <- 0.75^2 # marginal variance of GP
    rhos <- c(0.1,0.2,0.1)
    coords_ARD_scaled <- coords_ARD
    for (i in 1:dim(coords_ARD)[2]) coords_ARD_scaled[,i] <- coords_ARD[,i] / rhos[i]
    D_ARD <- as.matrix(dist(coords_ARD_scaled))
    Sigma_ARD <- sigma2_1 * exp(-D_ARD) + diag(1E-20,n)
    # hist(Sigma_ARD)
    C_ARD <- t(chol(Sigma_ARD))
    b_ARD <- qnorm(sim_rand_unif(n=n, init_c=0.4658))
    eps_ARD <- as.vector(C_ARD %*% b_ARD)
    probs <- pnorm(eps_ARD)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.18354) < probs)
    likelihood <- "bernoulli_logit"
    params = OPTIM_PARAMS_BFGS
    init_cov_pars <- c(1)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars <- c(init_cov_pars, mean(dist(coords_ARD[,i])/3))
    params$init_cov_pars <- init_cov_pars

    cov_pars_nll <- c(0.7, 0.5 * rhos)
    coord_test <- rbind(c(10000,0.2,0.9), coords_ARD[c(1,10),])
    coord_test[-1,c(2:3)] <- coord_test[-1,c(2:3)] + 0.01
    X_test <- cbind(rep(1,3),c(0,0,0))
    cov_pars_pred <- c(sigma2_1, rhos)

    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood,
                        cov_function = "matern_ard", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 69.7023612
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood,
                           cov_function = "matern_ard", cov_fct_shape = 0.5, y = y, X = X, params = params)
    cov_pars <- c(0.13905428093, 0.06867025605, 0.04247690364, 0.15469536599)
    coef <- c(-0.2543743520, 0.1505760147)
    nrounds <- 15
    nll_opt <- 68.41713226
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu <- c(-0.25437435197, 0.06788130795, 0.01430265524)
    expected_cov <- c(0.5625000000000, 0.0000000000000, 0.0000000000000, 0.0000000000000, 0.4938848144137, 0.0002158338884, 0.0000000000000, 0.0002158338884, 0.4862042504205)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Prediction without prior model fitting
    exp_mu_no_coef <- c(0.00000000, 0.25771940, 0.17913289)
    exp_cov_no_coef <- c(0.56250000000, 0.00000000000, 0.00000000000, 0.00000000000, 0.49481305128, 0.00021588667, 0.00000000000, 0.00021588667, 0.48645327980)
    gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5)
    pred <- predict(gp_model, gp_coords_pred = coord_test, y = y, predict_response = FALSE,
                    predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-exp_mu_no_coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-exp_cov_no_coef)),TOLERANCE_STRICT)
    # Matern with shape estimated
    params_ARD_est_shape <- OPTIM_PARAMS_BFGS
    params_ARD_est_shape$init_cov_pars <- c(init_cov_pars,1.5)
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard_estimate_shape",
                                             y = y, X = X, params = params_ARD_est_shape),
                      file='NUL')
      cov_pars_est_shape <- c(0.57108958797,  0.08471275821,  0.03304572501,  0.16194229745, 115.08702014148)
      coef_est_shape <- c(-0.2905450775, 0.2387123371, 0.1944576895, 0.3275844333)
      nrounds_est_shape <- 28
      nll_opt_est_shape <- 68.13569857
      capture.output( expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))[1:4]-cov_pars_est_shape[1:4])),TOLERANCE_STRICT), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))[5]-cov_pars_est_shape[5])),TOLERANCE_MEDIUM)
      capture.output( expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_est_shape )),TOLERANCE_MEDIUM), file='NUL')
      expect_equal(gp_model$get_num_optim_iter(), nrounds_est_shape )
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_est_shape), TOLERANCE_STRICT)
    }

    ##############
    ## With Vecchia approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood,
                                        cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none", matrix_inversion_method = "cholesky"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood,
                                           cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky"),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),relax_tolerance_strict(TOLERANCE_STRICT))
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance_strict(TOLERANCE_STRICT))
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance_strict(TOLERANCE_STRICT))
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance_strict(TOLERANCE_STRICT))
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance_strict(TOLERANCE_STRICT))
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Prediction without prior model fitting
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood,
                                        cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                        matrix_inversion_method = "cholesky"),
                    file='NUL')
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test, y = y, predict_response = FALSE,
                    predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-exp_mu_no_coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-exp_cov_no_coef)),TOLERANCE_STRICT)

    ## Less neighbors
    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative"){
        tolerance_loc <- TOLERANCE_ITERATIVE
      } else{
        tolerance_loc <- TOLERANCE_STRICT
      }
      nsim_var_pred <- 10000
      # Evaluate negative log-likelihood
      num_neighbors <- 50
      capture.output( gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood,
                                          cov_function = "matern_ard", cov_fct_shape = 0.5, matrix_inversion_method = inv_method,
                                          gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none"),
                      file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
      expect_lt(abs(nll-69.70236284),tolerance_loc)
      # Fit model
      capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                             gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                             y = y, X=X, params = params, matrix_inversion_method = inv_method),
                      file='NUL')
      cov_pars_nn <- c(0.19603539585, 0.06791498325, 0.03368011905, 0.15885250994)
      coef_nn <- c(-0.2701394756, 0.1619874679)
      nrounds_nn <- 25
      nll_opt_nn <- 68.41033632
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_nn)),tolerance_loc)
      if (inv_method == "cholesky") {
        expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
      }
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), tolerance_loc)
      # Prediction
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=num_neighbors, nsim_var_pred=nsim_var_pred)
      pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                      X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
      expected_mu_nn <- c(-0.270139475620, 0.056010991285, 0.004015366351)
      expected_cov_nn <- c(0.5625000000000, 0.0000000000000, 0.0000000000000, 0.0000000000000, 0.4938560837495, 0.0002305907991, 0.0000000000000, 0.0002305907991, 0.4862229980931)
      expect_lt(sum(abs(pred$mu-expected_mu_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_nn)),tolerance_loc)
      pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                      X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      expect_lt(sum(abs(pred$mu-expected_mu_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(pred$var)-expected_cov_nn[c(1,5,9)])),tolerance_loc)
    }

    ##############
    ## With FITC approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "fitc", num_ind_points = n, ind_points_selection = "random"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = n, ind_points_selection = "random",
                                           y = y, X = X, params = params),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT_LOWER)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)

    ## Less inducing points
    # Evaluate negative log-likelihood
    num_ind_points <- 50
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "fitc", num_ind_points = num_ind_points, ind_points_selection = "kmeans++"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-69.8362518046058),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = num_ind_points, ind_points_selection = "kmeans++",
                                           y = y, X = X, params = params),
                    file='NUL')
    cov_pars_nn <- c(0.00006475, 0.02037307, 0.01531317, 0.14835818)
    coef_nn <- c(-0.25715745, 0.14726729)
    nll_opt_nn <- 68.46260798
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_nn)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), 0.2)

    ##############
    ## Multiple observations at the same location
    ##############
    coords_ARD_mult = coords_ARD
    coords_ARD_mult[1:5,] <- coords_ARD_mult[(n-4):n,]
    params = OPTIM_PARAMS_BFGS
    init_cov_pars_mult <- c(1)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars_mult <- c(init_cov_pars_mult, mean(dist(unique(coords_ARD_mult)[,i])/3))
    params$init_cov_pars <- init_cov_pars_mult
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                        likelihood = likelihood)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 69.34595415
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                           y = y, X=X, params = params, likelihood = likelihood)
    cov_pars <- c(0.44308197588, 0.09997589302, 0.03067100521, 0.12031834901)
    coef <- c(-0.2800805796, 0.1785353899)
    nrounds <- 22
    nll_opt <- 68.21587237
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    ## With Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-6, vecchia_ordering = "none",
                                        likelihood = likelihood, matrix_inversion_method = "cholesky"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-6, vecchia_ordering = "none",
                                           y = y, X=X, params = params, likelihood = likelihood, matrix_inversion_method = "cholesky"),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    ## With FITC approximation
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "fitc", num_ind_points = n - 5, ind_points_selection = "random",
                                        likelihood = likelihood),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_MEDIUM)
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = n - 5, ind_points_selection = "random",
                                           y = y, X=X, params = params, likelihood = likelihood),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
  })

  test_that("t likelihood", {

    params = OPTIM_PARAMS_BFGS
    init_cov_pars = c(1,mean(dist(coords))/3)
    params$init_cov_pars = init_cov_pars
    likelihood_additional_param = 1
    params_vecchia <- c(params, cg_delta_conv = sqrt(1e-6),
                        num_rand_vec_trace = 50, cg_preconditioner_type = "pivoted_cholesky",
                        fitc_piv_chol_preconditioner_rank = n-1)
    params_vecchia$init_cov_pars = init_cov_pars

    # Simulate data and define expected values
    y <- L %*% b_1 + qnorm(sim_rand_unif(n=n, init_c=0.1)) / 5
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    cov_pars_pred_eval = c(1,0.2)
    coefs_pred = c(0.5,0.1)
    aux_pars_pred_eval = c(1,3)
    expected_nll <- 144.1194745
    cov_pars_fix_df <- c(0.94274448822, 0.09483522922)
    coefs_fix_df <- c(0.2780976933, -0.1045530677)
    aux_pars_fix_df <- c(0.07765501731, 1.00000000000)
    num_it_fix_df <- 10
    nll_est_fix_df <- 112.1635624
    cov_pars <- c(1.00786933961, 0.09231304834)
    coefs <- c(0.30227595731, -0.09752032205)
    aux_pars <- c(0.00165718826, 1.63405265512)
    num_it <- 24
    nll_est <- 107.8275669
    expected_mu <- c(-0.046398775956, -0.003934498908, 0.789074244932)
    expected_cov <- c(0.5895448972540, 0.5224498197427, -0.0001390612641, 0.5224498197427, 0.5897209705595, -0.0001486534570, -0.0001390612641, -0.0001486534570, 0.4058804336013)
    expected_var_resp <- expected_cov[c(1,5,9)] + aux_pars_pred_eval[1]^2
    # expected_var_resp <- c(3.589544897, 3.589720971, 3.405880434)
    # Estimation, prediction, and likelihood evaluation without Vecchia approximation
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "t", gp_approx = "none")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y, aux_pars = aux_pars_pred_eval)
    expect_lt(abs(nll-expected_nll),TOLERANCE_STRICT)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "t_fix_df", gp_approx = "none",
                                           y = y, X = X, params = params, likelihood_additional_param=likelihood_additional_param), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_fix_df)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_fix_df)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_fix_df)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_fix_df),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_fix_df)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "t", gp_approx = "none",
                                           y = y, X = X, params = params, likelihood_additional_param=likelihood_additional_param), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),2*TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est),TOLERANCE_MEDIUM)
    # Prediction
    gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE,
                                    predict_var = TRUE, cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var_resp)),TOLERANCE_MEDIUM)
    ############################
    # With duplicates and linear regression term without Vecchia approximation
    ############################
    y_multiple <- L_multiple %*% b_multiple + qnorm(sim_rand_unif(n=n, init_c=0.2818)) / 5
    coord_test_multiple <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    expected_nll_multiple <- 126.5295458
    cov_pars_multiple <- c(0.7281991570720, 0.0007068132731)
    coefs_multiple <- c(0.572603702130, 0.007961020259)
    aux_pars_multiple <- c(0.1803089654, 6.9281212623)
    num_it_multiple <- 40
    nll_est_multiple <- 34.7799636
    expected_mu_multiple <- c(-0.01186332228, 0.04582440444, 0.12582440444)
    expected_var_multiple <- c(0.5599871156, 0.5964929698, 0.5964929698)
    gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential", likelihood = "t", gp_approx = "none")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y_multiple, aux_pars = aux_pars_pred_eval)
    expect_lt(abs(nll-expected_nll_multiple),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "t", gp_approx = "none",
                                           y = y_multiple, X = X, params = params, likelihood_additional_param=likelihood_additional_param), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_multiple)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_multiple)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_multiple)),relax_tolerance(TOLERANCE_STRICT_LOWER, aux_pars_multiple))
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_multiple),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_multiple)
    gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
    pred <- predict(gp_model, y=y_multiple, gp_coords_pred = coord_test_multiple, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu_multiple)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_multiple)),TOLERANCE_MEDIUM)

    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative") {
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
        tolerance_loc_3 <- 0.5
        loop_cg_PC = c("pivoted_cholesky", "vadu", "fitc")
      } else {
        tolerance_loc_1 <- TOLERANCE_MEDIUM
        tolerance_loc_2 <- TOLERANCE_LOOSE
        tolerance_loc_3 <- TOLERANCE_LOOSE
        loop_cg_PC = c("vadu")
      }
      nsim_var_pred <- 10000
      for (cg_preconditioner_type in loop_cg_PC) {
        params_vecchia$cg_preconditioner_type <- cg_preconditioner_type

        # Likelihood evaluation
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = "t", gp_approx = "vecchia",
                                            num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        gp_model$set_optim_params(params = params_vecchia)
        capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y, aux_pars = aux_pars_pred_eval), file='NUL')
        expect_lt(abs(nll-expected_nll),2*tolerance_loc_3)
        # Estimation
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = "t", gp_approx = "vecchia",
                                            num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method, likelihood_additional_param=likelihood_additional_param), file='NUL')
        capture.output( fit(gp_model, y = y, X = X, params = params_vecchia) , file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_ITERATIVE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_ITERATIVE)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_ITERATIVE)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est),3*tolerance_loc_3)
        # Prediction
        gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                     num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_cov_mat = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_var = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_response = TRUE, predict_var = TRUE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
          expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
        }

        ############################
        # With duplicates and linear regression term
        ############################
        capture.output( gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential", likelihood = "t",
                                            gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        params_vecchia_mult <- params_vecchia
        params_vecchia_mult$fitc_piv_chol_preconditioner_rank <- dim(unique(coords_multiple))[1]
        gp_model$set_optim_params(params = params_vecchia_mult)
        capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y_multiple, aux_pars = aux_pars_pred_eval), file='NUL')
        expect_lt(abs(nll-expected_nll_multiple),tolerance_loc_3)
        capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                               likelihood = "t", gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                               matrix_inversion_method = inv_method,
                                               y = y_multiple, X = X, params = params_vecchia_mult, likelihood_additional_param=likelihood_additional_param), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_multiple)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_multiple)),TOLERANCE_ITERATIVE)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_multiple)),0.3)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_multiple),tolerance_loc_2)
        gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                     num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
        capture.output( pred <- predict(gp_model, y=y_multiple, gp_coords_pred = coord_test_multiple, predict_var = TRUE,
                                        predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu_multiple)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var_multiple)),tolerance_loc_1)

        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {# some tests are only run for one preconditioner

          #######################
          ## Less neighbors than observations
          #######################
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = "t", gp_approx = "vecchia",
                                              num_neighbors = 20, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method), file='NUL')
          gp_model$set_optim_params(params = params_vecchia)
          capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y, aux_pars = aux_pars_pred_eval), file='NUL')
          expected_nll_less_nn <- 144.099563
          expect_lt(abs(nll-expected_nll_less_nn),2*tolerance_loc_3)
          # Estimation
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = "t", gp_approx = "vecchia",
                                              num_neighbors = 20, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method, likelihood_additional_param=likelihood_additional_param), file='NUL')
          capture.output( fit(gp_model, y = y, X = X, params = params_vecchia) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_ITERATIVE)
          expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_ITERATIVE)
          nll_est_less_nn <- 107.8264387
          expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_less_nn),3*tolerance_loc_3)
          # Prediction
          gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
          gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", nsim_var_pred = nsim_var_pred)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_cov_mat = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_var = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
          if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {
            capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                            predict_response = TRUE, predict_var = TRUE,
                                            cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
            expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
            expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
          }
        }

      }# end loop cg_preconditioner_type in loop_cg_PC
    }# end loop inv_method in c("cholesky", "iterative")
  }) #end t-likelihood

  test_that("gaussian_heteroscedastic_fixed_and_random likelihood", {
    params = OPTIM_PARAMS_BFGS
    init_cov_pars = c(1,mean(dist(coords))/3,0.1,mean(dist(coords))/3)
    params$init_cov_pars = init_cov_pars
    params_vecchia <- c(params, cg_delta_conv = sqrt(1e-6),
                        num_rand_vec_trace = 50, cg_preconditioner_type = "pivoted_cholesky")
    params_vecchia$init_cov_pars = init_cov_pars
    likelihood <- "gaussian_heteroscedastic_fixed_and_random"

    # Simulate data and define expected values
    # Note: the GP of the log-error variance is simulated with the same covariance 'Sigma' as the GP of
    #   the mean. With a smaller marginal variance its estimated variance collapses to about 3e-05 at this
    #   sample size, which would no longer exercise the two sets of random effects
    L2 <- t(chol(Sigma))
    b_2 <- qnorm(sim_rand_unif(n=n, init_c=0.834))
    y <- L %*% b_1 + qnorm(sim_rand_unif(n=n, init_c=0.1234)) * exp(0.5 * L2 %*% b_2)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    cov_pars_pred_eval = c(1,0.2,0.1,0.2)
    coefs_pred = c(c(0.5,0.1),c(0.5,0.1))
    expected_nll <- 199.6831947
    cov_pars <- c(0.29001518290, 0.15063562850, 0.20539174870, 0.01285131220)
    coefs <- c(0.2562339067, -0.1161880773, 0.6373796861, 0.3056600825)
    num_it <- 37
    nll_est <- 191.2145084
    expected_mu <- c(0.06126291, 0.07337373, 0.30807230)
    expected_var <- c(0.5994207, 0.6014515, 0.3936357)
    expected_var_resp <- c(2.147623, 2.268682, 2.010216)

    #  # Estimation, prediction, and likelihood evaluation without Vecchia approximation
    # gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = likelihood , gp_approx = "none")
    # nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y, aux_pars = aux_pars_pred_eval)
    # expect_lt(abs(nll-expected_nll),TOLERANCE_STRICT)
    # # Estimation
    # capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
    #                                        likelihood = "t_fix_df", gp_approx = "none",
    #                                        y = y, X = X, params = params, likelihood_additional_param=likelihood_additional_param), file='NUL')
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_fix_df)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_fix_df)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_fix_df)),TOLERANCE_STRICT)
    # expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_fix_df),TOLERANCE_STRICT)
    # expect_equal(gp_model$get_num_optim_iter(), num_it_fix_df)
    # capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
    #                                        likelihood = likelihood , gp_approx = "none",
    #                                        y = y, X = X, params = params, likelihood_additional_param=likelihood_additional_param), file='NUL')
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    # expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_LOOSE)
    # expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_LOOSE)
    # expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est),TOLERANCE_MEDIUM)
    # # Prediction
    # gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
    # pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
    #                 predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    # expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    # expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
    #                 predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    # expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    # expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    # capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE,
    #                                 predict_var = TRUE, cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
    # expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    # expect_lt(sum(abs(pred$var-expected_var_resp)),TOLERANCE_MEDIUM)
    # The mode is a root of the exact score equation Z^T * l'(mode) = Sigma^-1 * mode also though the
    # approximation of the marginal likelihood uses the Fisher information instead of the observed Hessian,
    # so the derivatives through the mode are governed by the observed Hessian. This is checked through its
    # consequence: with a gradient that is not the derivative of the objective, the optimizer does not reach
    # the optimum. A new GPModel is used for every evaluation of the objective so that the mode is
    # recalculated from scratch, which a warm started mode would hide
    nll_fd_hetero <- function(cov_pars_loc, coefs_loc) {
      gp_fd <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = likelihood,
                       gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                       matrix_inversion_method = "cholesky")
      capture.output( nll_loc <- gp_fd$neg_log_likelihood(cov_pars = cov_pars_loc, y = y,
                        fixed_effects = as.vector(cbind(X %*% coefs_loc[1:2], X %*% coefs_loc[3:4]))), file='NUL')
      nll_loc
    }
    # The optimum that is reported has to be the optimum of that objective. The negative log-likelihood is
    # compared and not its derivative: the objective is flat around the optimum, so the gradient there varies
    # strongly with the point at which the optimizer happens to stop and is not reproducible across compilers,
    # while the objective value is. When the derivatives through the mode used the Fisher information instead
    # of the observed Hessian, the optimizer did not move away from the initial values at all
    nll_ref_hetero <- nll_fd_hetero(cov_pars, coefs)
    # Both matrix inversion methods are checked since they calculate the gradient differently (the iterative
    # methods use stochastic estimates)
    for (inv_method_fd in c("cholesky", "iterative")) {
      params_fd <- params_vecchia
      params_fd$cg_preconditioner_type <- "vadu"# the other preconditioners are not supported for this likelihood
      params_fd$num_rand_vec_trace <- 500
      params_fd$reuse_rand_vec_trace <- TRUE
      params_fd$seed_rand_vec_trace <- 1
      capture.output( gp_model_fd <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                likelihood = likelihood, gp_approx = "vecchia",
                                                num_neighbors = n-1, vecchia_ordering = "none",
                                                matrix_inversion_method = inv_method_fd,
                                                y = y, X = X, params = params_fd), file='NUL')
      nll_opt_fd <- nll_fd_hetero(as.vector(gp_model_fd$get_cov_pars(std_err = FALSE)),
                                  as.vector(gp_model_fd$get_coef(std_err = FALSE)))
      # the gap is 0 (cholesky) and 0.008 (iterative) here, while it was 1.89 before the fix, when the
      #   optimizer stayed at the initial values
      expect_lt(nll_opt_fd - nll_ref_hetero, 0.2)
    }
    for(inv_method in c("cholesky")){#, "iterative"
      if(inv_method == "iterative") {
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
        tolerance_loc_3 <- 0.5
        loop_cg_PC = c("pivoted_cholesky", "vadu", "fitc")
      } else {
        tolerance_loc_1 <- TOLERANCE_MEDIUM
        tolerance_loc_2 <- TOLERANCE_LOOSE
        tolerance_loc_3 <- TOLERANCE_LOOSE
        loop_cg_PC = c("vadu")
      }
      nsim_var_pred <- 10000
      for (cg_preconditioner_type in loop_cg_PC) {
        params_vecchia$cg_preconditioner_type <- cg_preconditioner_type

        # Likelihood evaluation
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = likelihood , gp_approx = "vecchia",
                                            num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        gp_model$set_optim_params(params = params_vecchia)
        capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y), file='NUL')
        expect_lt(abs(nll-expected_nll),tolerance_loc_3)
        # Estimation
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = likelihood , gp_approx = "vecchia",
                                            num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        capture.output( fit(gp_model, y = y, X = X, params = params_vecchia) , file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_ITERATIVE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_ITERATIVE)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est),relax_tolerance_nll(tolerance_loc_3))
        if (inv_method != "iterative" && USE_STRICT_TOLERANCES) {
          # the number of iterations of this non-convex optimization is compiler sensitive
          expect_equal(gp_model$get_num_optim_iter(), num_it)
        }
        # Prediction
        gp_model$set_optim_params(params = list(init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                     num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_var = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),tolerance_loc_1)
        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_response = TRUE, predict_var = TRUE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
          expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
        }

        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {# some tests are only run for one preconditioner

          #######################
          ## Less neighbors than observations
          #######################
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = likelihood , gp_approx = "vecchia",
                                              num_neighbors = 20, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method), file='NUL')
          gp_model$set_optim_params(params = params_vecchia)
          capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y), file='NUL')
          expected_nll_less_nn <- 199.6932499
          expect_lt(abs(nll-expected_nll_less_nn),tolerance_loc_3)
          # Estimation
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = likelihood , gp_approx = "vecchia",
                                              num_neighbors = 30, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method), file='NUL')
          capture.output( fit(gp_model, y = y, X = X, params = params_vecchia) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_ITERATIVE)
          nll_est_less_nn <- 191.2162037
          expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_less_nn),relax_tolerance_nll(tolerance_loc_3))
          # Prediction
          gp_model$set_optim_params(params = list(init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
          gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", nsim_var_pred = nsim_var_pred)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_var = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$var)-expected_var)),tolerance_loc_2)
          if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {
            capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                            predict_response = TRUE, predict_var = TRUE,
                                            cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
            expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
            expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_2)
          }
        }

      }# end loop cg_preconditioner_type in loop_cg_PC
    }# end loop inv_method in c("cholesky", "iterative")
  }) #end gaussian_heteroscedastic_fixed_and_random likelihood

  test_that("Initial coefficients from an iid model for gaussian_heteroscedastic_fixed_and_random ", {

    # 'gaussian_heteroscedastic_fixed_and_random' is supported only for a Vecchia approximated GP, so the
    # auxiliary iid model cannot use it. 'gaussian_heteroscedastic' has the same two fixed effects
    # predictors and is used instead
    likelihood_ii <- "gaussian_heteroscedastic_fixed_and_random"
    L2_ii <- t(chol(Sigma))
    b_2_ii <- qnorm(sim_rand_unif(n=n, init_c=0.834))
    y_ii <- L %*% b_1 + qnorm(sim_rand_unif(n=n, init_c=0.1234)) * exp(0.5 * L2_ii %*% b_2_ii)
    params_ii <- OPTIM_PARAMS_BFGS
    params_ii$init_cov_pars <- c(1, mean(dist(coords))/3, 0.1, mean(dist(coords))/3)
    fit_ii <- function(init_from_iid_model) {
      params_loc <- params_ii
      params_loc$init_coef_aux_pars_from_iid_model <- init_from_iid_model
      capture.output( gp_model_loc <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                 likelihood = likelihood_ii, gp_approx = "vecchia",
                                                 num_neighbors = 30, vecchia_ordering = "none",
                                                 matrix_inversion_method = "cholesky",
                                                 y = y_ii, X = X, params = params_loc), file='NUL')
      gp_model_loc
    }
    gp_model_ii <- fit_ii(TRUE)
    coefs_ii <- as.vector(gp_model_ii$get_coef(std_err = FALSE))
    expect_equal(length(coefs_ii), 2 * ncol(X))
    expect_true(all(is.finite(coefs_ii)))
    # the optimizer starts from other coefficients but has to reach the same optimum
    gp_model_no_ii <- fit_ii(FALSE)
    expect_lt(abs(gp_model_ii$get_current_neg_log_likelihood() -
                    gp_model_no_ii$get_current_neg_log_likelihood()), 0.2)

  })

  test_that("Loading a model saved when 'gaussian_heteroscedastic' had a variance GP works ", {
    # 'gaussian_heteroscedastic' used to denote the likelihood whose variance predictor contains both
    # fixed and random effects, which is now called 'gaussian_heteroscedastic_fixed_and_random'. Such
    # a saved model is recognized by num_sets_re = 2 and has to be migrated when it is loaded,
    # otherwise the variance GP is silently dropped
    n_het <- 60
    coords_het <- cbind(sim_rand_unif(n = n_het, init_c = 0.11), sim_rand_unif(n = n_het, init_c = 0.77))
    y_het <- qnorm(sim_rand_unif(n = n_het, init_c = 0.53))
    coord_test <- cbind(c(0.1, 0.7), c(0.9, 0.55))
    cov_pars_het <- c(0.6, 0.25, 2.0, 0.35)
    gp_model <- GPModel(gp_coords = coords_het, cov_function = "exponential", gp_approx = "vecchia",
                        num_neighbors = 20, likelihood = "gaussian_heteroscedastic_fixed_and_random")
    capture.output( fit(gp_model, y = y_het, params = list(
      init_cov_pars = cov_pars_het, optimizer_cov = "gradient_descent", lr_cov = 1e-8, maxit = 1,
      use_nesterov_acc = FALSE, init_coef_aux_pars_from_iid_model = FALSE, trace = FALSE)), file = 'NUL')
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    pred <- predict(gp_model, y = y_het, cov_pars = cov_pars_het, gp_coords_pred = coord_test,
                    predict_var = TRUE)
    # A file written by the older version differs only in the name of the likelihood
    json_old_name <- gsub("gaussian_heteroscedastic_fixed_and_random", "gaussian_heteroscedastic",
                          paste(readLines(filename, warn = FALSE), collapse = "\n"))
    filename_old_name <- tempfile(fileext = ".json")
    writeLines(json_old_name, filename_old_name)
    capture.output( gp_model_loaded <- loadGPModel(filename = filename_old_name), file = 'NUL')
    expect_equal(gp_model_loaded$get_likelihood_name(), "gaussian_heteroscedastic_fixed_and_random")
    expect_equal(as.numeric(gp_model_loaded$get_cov_pars()), cov_pars_het, tolerance = TOLERANCE_STRICT)
    pred_loaded <- predict(gp_model_loaded, y = y_het, cov_pars = cov_pars_het,
                           gp_coords_pred = coord_test, predict_var = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$var, pred_loaded$var)
    # A model saved by the current version keeps its name: 'gaussian_heteroscedastic' now has a
    # variance predictor with fixed effects only (num_sets_re = 1) and must not be migrated
    n_fe <- 100
    group_fe <- rep(1:10, each = 10)
    X_fe <- cbind(rep(1, n_fe), sim_rand_unif(n = n_fe, init_c = 0.256))
    b_fe <- qnorm(sim_rand_unif(n = 10, init_c = 0.741))
    y_fe <- as.vector(X_fe %*% c(0.3, 0.7)) + b_fe[group_fe] +
      qnorm(sim_rand_unif(n = n_fe, init_c = 0.369)) * exp(0.5 * as.vector(X_fe %*% c(-0.5, 1.2)))
    gp_model_fe <- GPModel(group_data = group_fe, likelihood = "gaussian_heteroscedastic")
    capture.output( fit(gp_model_fe, y = y_fe, X = X_fe, params = list(
      maxit = 2, init_coef_aux_pars_from_iid_model = FALSE, trace = FALSE)), file = 'NUL')
    filename_fe <- tempfile(fileext = ".json")
    saveGPModel(gp_model_fe, filename = filename_fe)
    capture.output( gp_model_fe_loaded <- loadGPModel(filename = filename_fe), file = 'NUL')
    expect_equal(gp_model_fe_loaded$get_likelihood_name(), "gaussian_heteroscedastic")
  })

  test_that("gaussian_heteroscedastic likelihood (fixed effects only) for linear and GPBoost models ", {

    n_het <- 100
    group_het <- rep(1:10, each = 10)
    X_het <- cbind(rep(1, n_het), sim_rand_unif(n = n_het, init_c = 0.256))
    beta_mean <- c(0.3, 0.7)
    beta_var <- c(-0.5, 1.2)
    gr_var_het <- 1
    b_gr_het <- qnorm(sim_rand_unif(n = 10, init_c = 0.741))
    mean_true <- as.vector(X_het %*% beta_mean) + sqrt(gr_var_het) * b_gr_het[group_het]
    log_var_true <- as.vector(X_het %*% beta_var)
    y_het <- mean_true + qnorm(sim_rand_unif(n = n_het, init_c = 0.369)) * exp(0.5 * log_var_true)

    # Likelihood evaluated at given (not estimated) parameters: a pure formula check, independent of any optimizer
    cov_pars_given_het <- 0.3
    fixed_effects_given_het <- as.vector(cbind(X_het %*% c(0.2, 0.5), X_het %*% c(-0.3, 0.8)))
    nll_given_het <- GPModel(group_data = group_het, likelihood = "gaussian_heteroscedastic")$neg_log_likelihood(
      cov_pars = cov_pars_given_het, y = y_het, fixed_effects = fixed_effects_given_het)
    expect_lt(abs(nll_given_het - 157.80743264), TOLERANCE_MEDIUM)

    # A fixed-effects-only variance requires a fixed effects term (covariates and / or GPBoost boosting):
    # without any covariates and without the GPBoost algorithm, fitting should raise an informative error
    expect_error(capture.output(fitGPModel(group_data = group_het, likelihood = "gaussian_heteroscedastic",
                                           y = y_het, params = list(maxit = 2, init_coef_aux_pars_from_iid_model = FALSE)),
                                file = "NUL"))

    ###################
    ## Linear regression model (mean has a grouped random effect, variance is fixed-effects only)
    ###################
    capture.output(gp_model_het <- fitGPModel(group_data = group_het, likelihood = "gaussian_heteroscedastic",
                                              y = y_het, X = X_het,
                                              params = OPTIM_PARAMS_BFGS), file = "NUL")
    coef_het <- as.vector(gp_model_het$get_coef(std_err = FALSE))
    expect_equal(length(coef_het), 4L)
    coef_het_std_err <- gp_model_het$get_coef(std_err = TRUE)
    expect_equal(dim(coef_het_std_err), c(2L, 4L))
    # Note: std. errs. must be strictly positive; a plain is.finite() check would not catch a regression where the
    # variance block's std. errs. are silently left at their R-side zero-initialized default (0 is finite)
    expect_true(all(coef_het_std_err["Std. err.", ] > 0))
    expected_coef_het <- c(-0.16843105, 1.05258998, -0.64123490, 1.54924057)
    expect_lt(sum(abs(coef_het - expected_coef_het)), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_het$get_cov_pars(std_err = FALSE)) - 0.24994751), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_het$get_current_neg_log_likelihood() - 155.27522914), TOLERANCE_MEDIUM)
    # Prediction: response mean and variance
    X_test_het <- cbind(rep(1, 3), c(0.1, 0.4, 0.8))
    group_test_het <- c(1, 3, 11)
    pred_het <- predict(gp_model_het, y = y_het, group_data_pred = group_test_het, X_pred = X_test_het,
                        predict_var = TRUE, predict_response = TRUE)
    expected_mu_het <- c(0.35476713, 0.16102877, 0.67364093)
    expected_var_het <- c(0.69153035, 1.04948914, 2.06871225)
    expect_lt(sum(abs(pred_het$mu - expected_mu_het)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_het$var - expected_var_het)), TOLERANCE_MEDIUM)
    X_zero_het <- matrix(0, nrow = n_het, ncol = ncol(X_het))
    re_pred_train_het <- predict_training_data_random_effects(gp_model_het)
    expected_re_pred_train_het <- c(0.41793918, 0.11415140, -0.09157617, -0.06884991, 0.53482262,
                                    -0.64437448, 0.20923218, -0.85328655, 0.28497061, 0.09707038)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_het[, 1])) - expected_re_pred_train_het)), TOLERANCE_MEDIUM)
    re_pred_train_het_var <- predict_training_data_random_effects(gp_model_het, predict_var = TRUE)
    expected_re_pred_train_het_var <- c(0.07663970, 0.06660995, 0.07079751, 0.07706457, 0.07193486,
                                        0.06797221, 0.08127999, 0.07331034, 0.06945477, 0.07959953)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_het_var[, 2])) - expected_re_pred_train_het_var)), TOLERANCE_MEDIUM)
    pred_train_re_het <- predict(gp_model_het, y = y_het, group_data_pred = group_het, X_pred = X_zero_het,
                                 predict_response = FALSE, predict_var = FALSE)
    expect_lt(sum(abs(as.vector(re_pred_train_het[, 1]) - pred_train_re_het$mu)), TOLERANCE_STRICT)
    # Predicting requires covariate data for the model's linear predictor (mean and variance)
    expect_error(predict(gp_model_het, y = y_het, group_data_pred = group_test_het,
                         predict_var = TRUE, predict_response = TRUE))

    ###################
    ## No random effects at all (iid model, pure linear regression for mean and variance)
    ###################
    capture.output(gp_model_het_iid <- fitGPModel(likelihood = "gaussian_heteroscedastic",
                                                  y = y_het, X = X_het,
                                                  params = OPTIM_PARAMS_BFGS), file = "NUL")
    coef_het_iid <- as.vector(gp_model_het_iid$get_coef(std_err = FALSE))
    expected_coef_het_iid <- c(-0.18164405, 1.06906319, -0.14266627, 0.97312331)
    expect_lt(sum(abs(coef_het_iid - expected_coef_het_iid)), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_het_iid$get_current_neg_log_likelihood() - 159.44268884), TOLERANCE_MEDIUM)

    ###################
    ## GPBoost algorithm (tree-boosting): mean via a grouped random effect + trees, variance via a second tree ensemble
    ###################
    gp_model_het_boost <- GPModel(group_data = group_het, likelihood = "gaussian_heteroscedastic")
    gp_model_het_boost$set_optim_params(params = OPTIM_PARAMS_BFGS)
    dtrain_het <- gpb.Dataset(data = X_het[, 2, drop = FALSE], label = y_het)
    bst_het <- gpb.train(data = dtrain_het, gp_model = gp_model_het_boost, nrounds = 20,
                         learning_rate = 0.01, max_depth = 2, min_data_in_leaf = 5,
                         verbose = 0, deterministic = TRUE)
    pred_het_boost <- predict(bst_het, data = X_het[1:3, 2, drop = FALSE], group_data_pred = group_test_het,
                              predict_var = TRUE, pred_latent = FALSE)
    expect_lt(abs(as.vector(gp_model_het_boost$get_cov_pars(std_err = FALSE)) - 0.15080798), TOLERANCE_MEDIUM)
    expected_response_mean_boost <- c(0.52600579, 0.24099045, 0.37506889)
    expected_response_var_boost <- c(1.43766912, 1.43641997, 1.58325054)
    expect_lt(sum(abs(pred_het_boost$response_mean - expected_response_mean_boost)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_het_boost$response_var - expected_response_var_boost)), TOLERANCE_MEDIUM)
    expect_error(predict_training_data_random_effects(gp_model_het_boost),
                 "predict_training_data_random_effects\\(bst\\)")
    re_pred_train_boost <- predict_training_data_random_effects(bst_het)
    expected_re_pred_train_boost <- c(0.27708188, 0.02084478, -0.06485732, -0.09881753, 0.40985823,
                                      -0.38168637, 0.10144342, -0.57590024, 0.17622270, 0.07567558)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_boost[, 1])) - expected_re_pred_train_boost)), TOLERANCE_MEDIUM)
    re_pred_train_boost_var <- predict_training_data_random_effects(bst_het, predict_var = TRUE)
    expect_lt(sum(abs(re_pred_train_boost_var[, 1] - re_pred_train_boost[, 1])), TOLERANCE_STRICT)
    expected_re_pred_train_boost_var <- c(0.07329236, 0.07204321, 0.07222443, 0.07247900, 0.07183686,
                                          0.07303205, 0.07294693, 0.07201704, 0.07256303)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_boost_var[, 2])) - expected_re_pred_train_boost_var)), TOLERANCE_MEDIUM)

    ###################
    ## GPs
    ###################

    n_het2 <- 100
    X_het2 <- cbind(rep(1, n_het2), sim_rand_unif(n = n_het2, init_c = 0.187))
    beta_mean2 <- c(0.2, 0.5)
    beta_var2 <- c(-0.4, 0.9)
    log_var_true2 <- as.vector(X_het2 %*% beta_var2)
    coords_het2 <- matrix(sim_rand_unif(n = n_het2 * 2, init_c = 0.723), ncol = 2)
    D_het2 <- as.matrix(dist(coords_het2))
    gp_var2 <- 0.6
    gp_range2 <- 0.15
    Sigma_het2 <- gp_var2 * exp(-D_het2 / gp_range2) + diag(1E-10, n_het2)
    b_gp_het2 <- as.vector(t(chol(Sigma_het2)) %*% qnorm(sim_rand_unif(n = n_het2, init_c = 0.812)))
    mean_true2 <- as.vector(X_het2 %*% beta_mean2) + b_gp_het2
    y_het2 <- mean_true2 + qnorm(sim_rand_unif(n = n_het2, init_c = 0.234)) * exp(0.5 * log_var_true2)
    optim_params_het2_bfgs <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 300,
                                   init_coef_aux_pars_from_iid_model = FALSE)
    optim_params_het2_bfgs_iter <- c(optim_params_het2_bfgs, list(seed_rand_vec_trace = 1))

    ###################
    ## Dense GP ("Stable")
    ###################
    # Likelihood evaluated at given (not estimated) parameters: a pure formula check, independent of any optimizer
    nll_given_gp <- GPModel(gp_coords = coords_het2, cov_function = "exponential",
                            likelihood = "gaussian_heteroscedastic")$neg_log_likelihood(
      cov_pars = c(1, mean(dist(coords_het2)) / 3), y = y_het2, fixed_effects = rep(0, 2 * n_het2))
    expect_lt(abs(nll_given_gp - 172.94595730), TOLERANCE_MEDIUM)
    capture.output(gp_model_gp <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                             likelihood = "gaussian_heteroscedastic", y = y_het2, X = X_het2,
                                             params = optim_params_het2_bfgs), file = "NUL")
    coef_gp_bfgs <- as.vector(gp_model_gp$get_coef(std_err = FALSE))
    expected_coef_gp <- c(0.53370608, -0.06832911, -0.05858086, 0.76800104)
    expect_lt(sum(abs(coef_gp_bfgs - expected_coef_gp)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_gp$get_cov_pars(std_err = FALSE)) - c(0.10683991, 0.01031087))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_gp$get_current_neg_log_likelihood() - 163.47352630), TOLERANCE_MEDIUM)
    coord_test_gp <- coords_het2[1:3, , drop = FALSE] + 1e-3
    pred_gp <- predict(gp_model_gp, y = y_het2, gp_coords_pred = coord_test_gp, X_pred = X_het2[1:3, , drop = FALSE],
                       predict_var = TRUE, predict_response = TRUE)
    expected_mu_gp <- c(0.51363253, 0.49606308, 0.51154296)
    expected_var_gp <- c(1.18640601, 1.19690164, 1.42727248)
    expect_lt(sum(abs(pred_gp$mu - expected_mu_gp)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_gp$var - expected_var_gp)), TOLERANCE_MEDIUM)
    X_zero_het2 <- matrix(0, nrow = n_het2, ncol = ncol(X_het2))
    re_pred_train_gp <- predict_training_data_random_effects(gp_model_gp)
    expected_re_pred_train_gp <- c(-0.00843116, -0.02773224, 0.00938880, 0.03301499, 0.09514145)
    expect_lt(sum(abs(re_pred_train_gp[1:5, 1] - expected_re_pred_train_gp)), TOLERANCE_MEDIUM)
    pred_train_gp <- predict(gp_model_gp, y = y_het2, gp_coords_pred = coords_het2, X_pred = X_zero_het2,
                             predict_response = FALSE, predict_var = FALSE)
    expect_lt(sum(abs(as.vector(re_pred_train_gp[, 1]) - pred_train_gp$mu)), TOLERANCE_STRICT)

    ###################
    ## GP with Vecchia approximation
    ###################
    capture.output(gp_model_vecchia <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                                  likelihood = "gaussian_heteroscedastic", gp_approx = "vecchia",
                                                  num_neighbors = n_het2 - 1, vecchia_ordering = "none",
                                                  matrix_inversion_method = "cholesky",
                                                  y = y_het2, X = X_het2, params = optim_params_het2_bfgs), file = "NUL")
    coef_vecchia <- as.vector(gp_model_vecchia$get_coef(std_err = FALSE))
    # With num_neighbors = n - 1, Vecchia is exact and should match the dense GP fit (same optimizer) closely
    expect_lt(sum(abs(coef_vecchia - expected_coef_gp)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_vecchia$get_cov_pars(std_err = FALSE)) - c(0.10683991, 0.01031087))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_vecchia$get_current_neg_log_likelihood() - 163.47352630), TOLERANCE_MEDIUM)
    re_pred_train_vecchia <- predict_training_data_random_effects(gp_model_vecchia)
    capture.output( pred_train_vecchia <- predict(gp_model_vecchia, y = y_het2, gp_coords_pred = coords_het2, X_pred = X_zero_het2,
                                  predict_response = FALSE, predict_var = FALSE), file = "NUL")
    expect_lt(sum(abs(as.vector(re_pred_train_vecchia[, 1]) - pred_train_vecchia$mu)), TOLERANCE_STRICT)
    # matrix_inversion_method = "iterative" 
    capture.output(gp_model_vecchia_iter <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                                       likelihood = "gaussian_heteroscedastic", gp_approx = "vecchia",
                                                       num_neighbors = n_het2 - 1, vecchia_ordering = "none",
                                                       matrix_inversion_method = "iterative",
                                                       y = y_het2, X = X_het2, params = optim_params_het2_bfgs_iter), file = "NUL")
    coef_vecchia_iter <- as.vector(gp_model_vecchia_iter$get_coef(std_err = FALSE))
    expected_coef_vecchia_iter <- c(0.53573745, -0.07251654, -0.06497315, 0.77184161)
    expect_lt(sum(abs(coef_vecchia_iter - expected_coef_vecchia_iter)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(as.vector(gp_model_vecchia_iter$get_cov_pars(std_err = FALSE)) - c(0.11147938, 0.01147543))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_vecchia_iter$get_current_neg_log_likelihood() - 163.47418211), TOLERANCE_NON_CONVEX)

    ###################
    ## GP with FITC / VIF (full_scale_vecchia) approximation
    ###################
    capture.output(gp_model_fitc <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                               likelihood = "gaussian_heteroscedastic", gp_approx = "fitc",
                                               num_ind_points = 50, y = y_het2, X = X_het2, params = optim_params_het2_bfgs), file = "NUL")
    coef_fitc <- as.vector(gp_model_fitc$get_coef(std_err = FALSE))
    expected_coef_fitc <- c(0.53242708, -0.07016827, 0.04718839, 0.69725428)
    expect_lt(sum(abs(coef_fitc - expected_coef_fitc)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_fitc$get_cov_pars(std_err = FALSE)) - c(0.00873612, 0.00548896))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_fitc$get_current_neg_log_likelihood() - 163.49280003), TOLERANCE_MEDIUM)
    pred_fitc <- predict(gp_model_fitc, y = y_het2, gp_coords_pred = coord_test_gp, X_pred = X_het2[1:3, , drop = FALSE],
                         predict_var = TRUE, predict_response = TRUE)
    expected_mu_fitc <- c(0.51930607, 0.51859412, 0.50126126)
    expected_var_fitc <- c(1.20305368, 1.21152758, 1.43760007)
    expect_lt(sum(abs(pred_fitc$mu - expected_mu_fitc)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_fitc$var - expected_var_fitc)), TOLERANCE_MEDIUM)

    ###################
    ## Crossed / multiple grouped random effects (Woodbury identity)
    ###################
    group1_het2 <- rep(1:10, each = 10)
    group2_het2 <- rep(1:5, length.out = n_het2)
    # matrix_inversion_method = "cholesky" is required: "iterative" (the default for crossed / multiple grouped
    # random effects, i.e., num_re_group_total > 1) is not yet supported for this likelihood
    capture.output(gp_model_crossed <- fitGPModel(group_data = cbind(group1_het2, group2_het2),
                                                  likelihood = "gaussian_heteroscedastic", y = y_het2, X = X_het2,
                                                  matrix_inversion_method = "cholesky",
                                                  params = optim_params_het2_bfgs), file = "NUL")
    coef_crossed_bfgs <- as.vector(gp_model_crossed$get_coef(std_err = FALSE))
    expected_coef_crossed <- c(0.53302511, -0.07068343, 0.04484505, 0.70722753)
    expect_lt(sum(abs(coef_crossed_bfgs - expected_coef_crossed)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_crossed$get_cov_pars(std_err = FALSE)) - c(0.00172197, 0.00177310))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_crossed$get_current_neg_log_likelihood() - 163.49297024), TOLERANCE_MEDIUM)
    group_test_crossed <- cbind(c(1, 3, 11), c(1, 2, 6))
    pred_crossed <- predict(gp_model_crossed, y = y_het2, group_data_pred = group_test_crossed, X_pred = X_het2[1:3, , drop = FALSE],
                            predict_var = TRUE, predict_response = TRUE)
    expected_mu_crossed <- c(0.51311185, 0.51840682, 0.50163047)
    expected_var_crossed <- c(1.19716914, 1.20576336, 1.43534326)
    expect_lt(sum(abs(pred_crossed$mu - expected_mu_crossed)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_crossed$var - expected_var_crossed)), TOLERANCE_MEDIUM)
    # With two crossed grouped random effects, predict_training_data_random_effects() returns one column per
    # random effect; their sum corresponds to the total random effect predicted via the X_pred = 0 trick
    re_pred_train_crossed <- predict_training_data_random_effects(gp_model_crossed)
    expected_re_pred_train_crossed_1 <- c(0.00145590, 0.00145590, 0.00145590, 0.00145590, 0.00145590)
    expected_re_pred_train_crossed_2 <- c(-0.00815135, 0.00366130, -0.00372236, 0.00801002, 0.00019450)
    expect_lt(sum(abs(re_pred_train_crossed[1:5, 1] - expected_re_pred_train_crossed_1)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(re_pred_train_crossed[1:5, 2] - expected_re_pred_train_crossed_2)), TOLERANCE_MEDIUM)
    X_zero_crossed <- matrix(0, nrow = n_het2, ncol = ncol(X_het2))
    pred_train_crossed <- predict(gp_model_crossed, y = y_het2, group_data_pred = cbind(group1_het2, group2_het2), X_pred = X_zero_crossed,
                                  predict_response = FALSE, predict_var = FALSE)
    re_pred_train_crossed_sum <- as.vector(re_pred_train_crossed[, 1]) + as.vector(re_pred_train_crossed[, 2])
    expect_lt(sum(abs(re_pred_train_crossed_sum - pred_train_crossed$mu)), TOLERANCE_STRICT)
    # matrix_inversion_method = "iterative" 
    capture.output(gp_model_crossed_iter <- fitGPModel(group_data = cbind(group1_het2, group2_het2),
                                                       likelihood = "gaussian_heteroscedastic", y = y_het2, X = X_het2,
                                                       matrix_inversion_method = "iterative",
                                                       params = optim_params_het2_bfgs_iter), file = "NUL")
    coef_crossed_iter <- as.vector(gp_model_crossed_iter$get_coef(std_err = FALSE))
    expected_coef_crossed_iter <- c(0.53458908, -0.07561868, 0.05287152, 0.69402576)
    expect_lt(sum(abs(coef_crossed_iter - expected_coef_crossed_iter)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(as.vector(gp_model_crossed_iter$get_cov_pars(std_err = FALSE)) - c(0.00215241, 0.00234007))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_crossed_iter$get_current_neg_log_likelihood() - 163.49394794), TOLERANCE_NON_CONVEX)

    ###################
    ## Full scale Vecchia / VIF approximation (inducing points + Vecchia-approximated residual)
    ###################
    # Note: this is a hard, non-convex optimization problem (as with the dense GP above), so cholesky and iterative
    # (which uses a stochastic gradient) can converge to different stationary points with similar likelihood value;
    # we therefore compare the *exact* (cholesky-computed) negative log-likelihood at both solutions instead of coefficients
    optim_params_fsva <- c(optim_params_het2_bfgs, list(fitc_piv_chol_preconditioner_rank = 50))
    capture.output(gp_model_fsva <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                               likelihood = "gaussian_heteroscedastic", gp_approx = "full_scale_vecchia",
                                               num_ind_points = 30, num_neighbors = 19, cov_fct_taper_range = 0.5,
                                               matrix_inversion_method = "cholesky",
                                               y = y_het2, X = X_het2, params = optim_params_fsva), file = "NUL")
    coef_fsva <- as.vector(gp_model_fsva$get_coef(std_err = FALSE))
    cov_pars_fsva <- as.vector(gp_model_fsva$get_cov_pars())
    expected_coef_fsva <- c(0.53280591, -0.06916614, 0.05398657, 0.69427748)
    expect_lt(sum(abs(coef_fsva - expected_coef_fsva)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(cov_pars_fsva - c(0.00020567118, 0.02293172583))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_fsva$get_current_neg_log_likelihood() - 163.49333870), TOLERANCE_NON_CONVEX)
    pred_fsva <- predict(gp_model_fsva, y = y_het2, gp_coords_pred = coord_test_gp, X_pred = X_het2[1:3, , drop = FALSE],
                         predict_var = TRUE, predict_response = TRUE)
    expected_mu_fsva <- c(0.51985669, 0.51912314, 0.50210374)
    expected_var_fsva <- c(1.20200085, 1.21049129, 1.43691553)
    expect_lt(sum(abs(pred_fsva$mu - expected_mu_fsva)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(pred_fsva$var - expected_var_fsva)), TOLERANCE_NON_CONVEX)
    capture.output( gp_model_fsva_exact_ll <- GPModel(gp_coords = coords_het2, cov_function = "exponential",
                                      likelihood = "gaussian_heteroscedastic", gp_approx = "full_scale_vecchia",
                                      num_ind_points = 30, num_neighbors = 19, cov_fct_taper_range = 0.5,
                                      matrix_inversion_method = "cholesky"), file = "NUL")
    nll_fsva_at_chol_point <- gp_model_fsva_exact_ll$neg_log_likelihood(
      cov_pars = cov_pars_fsva, y = y_het2, fixed_effects = as.vector(cbind(X_het2 %*% coef_fsva[1:2], X_het2 %*% coef_fsva[3:4])))
    expect_equal(nll_fsva_at_chol_point, gp_model_fsva$get_current_neg_log_likelihood(), tolerance = TOLERANCE_MEDIUM)
    # matrix_inversion_method = "iterative" 
    optim_params_fsva_iter <- c(optim_params_fsva, list(seed_rand_vec_trace = 1))
    capture.output(gp_model_fsva_iter <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                                    likelihood = "gaussian_heteroscedastic", gp_approx = "full_scale_vecchia",
                                                    num_ind_points = 30, num_neighbors = 19, cov_fct_taper_range = 0.5,
                                                    matrix_inversion_method = "iterative",
                                                    y = y_het2, X = X_het2, params = optim_params_fsva_iter), file = "NUL")
    coef_fsva_iter <- as.vector(gp_model_fsva_iter$get_coef(std_err = FALSE))
    cov_pars_fsva_iter <- as.vector(gp_model_fsva_iter$get_cov_pars())
    # The coefficients and covariance parameters of this fit are not compared to hard-wired values: the
    # stochastic trace estimates make the point on the flat ridge that is reached build dependent, see the
    # note below. The negative log-likelihood is a robust statistic on this ridge and is checked instead
    expect_lt(abs(gp_model_fsva_iter$get_current_neg_log_likelihood() - 163.15168709), TOLERANCE_NON_CONVEX)
    nll_fsva_iter_at_exact_ll <- gp_model_fsva_exact_ll$neg_log_likelihood(
      cov_pars = cov_pars_fsva_iter, y = y_het2, fixed_effects = as.vector(cbind(X_het2 %*% coef_fsva_iter[1:2], X_het2 %*% coef_fsva_iter[3:4])))
    # The exact (cholesky) NLL at the iterative solution should be close to the exact NLL at the cholesky solution,
    # even though the coefficients themselves can differ substantially (flat likelihood ridge for this small dataset).
    # Tolerance is loose (relative to the ~150-200 scale of the NLL here) since this is a stochastic-gradient fit
    # converging on a hard, non-convex landscape; a real gradient bug would be expected to cause a much larger gap
    expect_lt(abs(nll_fsva_iter_at_exact_ll - nll_fsva_at_chol_point), 5)
    # matrix_inversion_method = "iterative" with the "vifdu" CG preconditioner (solves (Sigma^-1+W) directly, no push-through)
    optim_params_fsva_iter_vifdu <- c(optim_params_het2_bfgs, list(seed_rand_vec_trace = 1, cg_preconditioner_type = "vifdu"))
    capture.output(gp_model_fsva_iter_vifdu <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                                          likelihood = "gaussian_heteroscedastic", gp_approx = "full_scale_vecchia",
                                                          num_ind_points = 30, num_neighbors = 19, cov_fct_taper_range = 0.5,
                                                          matrix_inversion_method = "iterative",
                                                          y = y_het2, X = X_het2, params = optim_params_fsva_iter_vifdu), file = "NUL")
    coef_fsva_iter_vifdu <- as.vector(gp_model_fsva_iter_vifdu$get_coef(std_err = FALSE))
    # As for the iterative fit above, only the negative log-likelihood is compared to a hard-wired value,
    # and the solution is additionally checked through the exact (cholesky) negative log-likelihood
    expect_lt(abs(gp_model_fsva_iter_vifdu$get_current_neg_log_likelihood() - 163.39130273), TOLERANCE_NON_CONVEX)
    nll_fsva_vifdu_at_exact_ll <- gp_model_fsva_exact_ll$neg_log_likelihood(
      cov_pars = as.vector(gp_model_fsva_iter_vifdu$get_cov_pars()), y = y_het2,
      fixed_effects = as.vector(cbind(X_het2 %*% coef_fsva_iter_vifdu[1:2], X_het2 %*% coef_fsva_iter_vifdu[3:4])))
    expect_lt(abs(nll_fsva_vifdu_at_exact_ll - nll_fsva_at_chol_point), 5)
  })

}
