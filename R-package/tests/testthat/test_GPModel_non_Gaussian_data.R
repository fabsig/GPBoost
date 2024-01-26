context("GPModel_non_Gaussian_data")

# Avoid being tested on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  TOLERANCE_LOOSE <- 1e-3
  TOLERANCE_STRICT <- 1E-6
  TOLERANCE_ITERATIVE <- 1e-1
  
  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov = "gradient_descent", optimizer_coef = "gradient_descent",
                               use_nesterov_acc = TRUE, lr_cov=0.1, lr_coef = 0.1, maxit = 1000,
                               acc_rate_cov = 0.5)
  DEFAULT_OPTIM_PARAMS_STD <- c(DEFAULT_OPTIM_PARAMS, list(std_dev = TRUE))
  
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
  
  test_that("Binary classification with Gaussian process model ", {
    
    probs <- pnorm(L %*% b_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    init_cov_pars <- c(1,mean(dist(coords))/3)
    # Label needs to have correct format
    expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential",
                            likelihood = "bernoulli_probit",
                            y = b_1, params = list(optimizer_cov = "gradient_descent")))
    yw <- y
    yw[3] <- yw[3] + 1E-6
    expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential",
                            likelihood = "bernoulli_probit",
                            y = yw, params = list(optimizer_cov = "gradient_descent")))
    # Only gradient descent can be used
    expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential",
                            likelihood = "bernoulli_probit",
                            y = y, params = list(optimizer_cov = "fisher_scoring")))
    # Estimation using gradient descent
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                       convergence_criterion = "relative_change_in_parameters",
                                                       init_cov_pars = init_cov_pars)), file='NUL')
    cov_pars <- c(0.9419234, 0.1866877)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 40)
    # Can switch between likelihoods
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
    gp_model$set_likelihood("gaussian")
    gp_model$set_likelihood("bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                       convergence_criterion = "relative_change_in_parameters",
                                                       init_cov_pars = init_cov_pars)), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                                       lr_cov = 0.01, use_nesterov_acc = TRUE, 
                                                       acc_rate_cov = 0.5, init_cov_pars = init_cov_pars)), file='NUL')
    cov_pars2 <- c(0.9646422, 0.1844797)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars2)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 26)
    # Estimation using Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "nelder_mead", delta_rel_conv=1e-6, 
                                                       init_cov_pars = init_cov_pars))
                    , file='NUL')
    cov_pars3 <- c(0.9998047, 0.1855072)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars3)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Estimation using BFGS
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "bfgs", init_cov_pars = init_cov_pars)), file='NUL')
    cov_pars3 <- c(0.9419084, 0.1866882)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars3)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 4)
    # Estimation using Adam
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "adam", init_cov_pars = init_cov_pars)), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars3)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 200)
    
    # Prediction
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           y = y, params = list(optimizer_cov = "gradient_descent", 
                                                                lr_cov=0.01, use_nesterov_acc=FALSE, init_cov_pars = init_cov_pars))
                    , file='NUL')
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.6595663, -0.6638940, 0.4997690)
    expected_cov <- c(0.6482224576, 0.5765285950, -0.0001030520, 0.5765285950,
                      0.6478191338, -0.0001163496, -0.0001030520, -0.0001163496, 0.4435551436)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.3037139, 0.3025143, 0.6612807)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_mu*(1-expected_mu))),TOLERANCE_STRICT)
    
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    preds <- predict(gp_model, gp_coords_pred = coords, 
                     predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),TOLERANCE_STRICT)
    
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-63.6205917),TOLERANCE_STRICT)
    
    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
    opt <- optim(par=c(1,0.1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
    cov_pars <- c(0.9419234, 0.1866877)
    expect_lt(sum(abs(opt$par-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(abs(opt$value-(63.6126363)),TOLERANCE_LOOSE)
    expect_equal(as.integer(opt$counts[1]), 47)
    
    ###################
    ## Random coefficient GPs
    ###################
    probs <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3))
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.543) < probs)
    init_cov_pars_RC <- rep(init_cov_pars, 3)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 1, use_nesterov_acc = TRUE, 
                                                         acc_rate_cov=0.5, maxit=1000, init_cov_pars=init_cov_pars_RC))
                    , file='NUL')
    expected_values <- c(0.3701097, 0.2846740, 2.1160325, 0.3305266, 0.1241462, 0.1846456)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 39)
    # Prediction
    gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential", likelihood = "bernoulli_probit")
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(1,0.1,0.8,0.15,1.1,0.08),
                             predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.18346008, 0.03479258, -0.17247579)
    expected_cov <- c(1.039879e+00, 7.521981e-01, -3.256500e-04, 7.521981e-01,
                      8.907289e-01, -6.719282e-05, -3.256500e-04, -6.719282e-05, 9.147899e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-65.1768199),TOLERANCE_LOOSE)
    
    ###################
    ##  Multiple cluster IDs
    ###################
    probs <- pnorm(L %*% b_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2978341) < probs)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, cluster_ids = cluster_ids,likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", lr_cov=0.2, 
                                                         use_nesterov_acc=FALSE, init_cov_pars=init_cov_pars))
                    , file='NUL')
    cov_pars <- c(0.5085134, 0.2011667)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 20)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        cluster_ids = cluster_ids,likelihood = "bernoulli_probit")
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1509569, 0.0000000, 0.9574946)
    expected_cov <- c(1.2225959453, 0.0000000000, 0.0003074858, 0.0000000000,
                      1.5000000000, 0.0000000000, 0.0003074858, 0.0000000000, 1.0761874845)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
  })
  
  test_that("Binary classification with Gaussian process model with multiple observations at the same location", {
    
    eps_multiple <- as.vector(L_multiple %*% b_multiple)
    probs <- pnorm(eps_multiple)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.9341) < probs)
    
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           y = y,likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov=0.1, use_nesterov_acc = TRUE)), file='NUL')
    cov_pars <- c(0.6857065, 0.2363754)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.2633282, -0.2637633, -0.2637633)
    expected_cov <- c(0.9561355, 0.8535206, 0.8535206, 0.8535206, 1.0180227,
                      1.0180227, 0.8535206, 1.0180227, 1.0180227)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred_resp <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                                  cov_pars = c(1.5,0.15), predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_resp$mu-c(0.4253296, 0.4263502, 0.4263502))),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_resp$var-c(0.2444243, 0.2445757, 0.2445757))),TOLERANCE_STRICT)
    
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    preds <- predict(gp_model, gp_coords_pred = coords_multiple, 
                     predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),TOLERANCE_STRICT)
    
    # Multiple cluster IDs and multiple observations
    coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    cluster_ids_pred = c(0L,3L,3L)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_cov <- c(0.9561355, 0.0000000, 0.0000000, 0.0000000, 1.5000000,
                      1.5000000, 0.0000000, 1.5000000, 1.5000000)
    expect_lt(sum(abs(pred$mu-c(-0.2633282, rep(0,2)))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred_resp <- gp_model$predict(y = y, gp_coords_pred = coord_test, cluster_ids_pred = cluster_ids_pred,
                                  cov_pars = c(1.5,0.15), predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_resp$mu-c(0.4253296, 0.5000000, 0.5000000))),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_resp$var-c(0.2444243, 0.2500000, 0.2500000))),TOLERANCE_STRICT)
    
    # With linear regression term
    probs <- pnorm(eps_multiple + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.67981) < probs)
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", 
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS), file='NUL')
    cov_pars <- c(0.74629148, 0.05008149 )
    coefs <- c(0.8545045, 1.7286020)
    num_it <- 39
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(0.07791956, 1.27273974, 2.65562131)
    expected_var <- c(0.7267896, 0.7329027, 0.7329027)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2), y=y)
    expect_lt(abs(nll-59.9183192),TOLERANCE_STRICT)
    # With fixed effects
    fixed_effects <- as.numeric(X%*%beta)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2), y=y, fixed_effects=fixed_effects)
    expect_lt(abs(nll-42.8518187),TOLERANCE_STRICT)
  })
  
  test_that("Binary classification with one grouped random effects ", {
    
    probs <- pnorm(Z1 %*% b_gr_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.823431) < probs)
    init_cov_pars <- c(1)
    
    # Estimation using gradient descent
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters", 
                                       init_cov_pars=init_cov_pars))
    cov_pars <- c(0.40255)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 62)
    # Can switch between likelihoods
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    gp_model$set_likelihood("gaussian")
    gp_model$set_likelihood("bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters", 
                                       init_cov_pars=init_cov_pars))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                       lr_cov = 0.1, use_nesterov_acc = TRUE, 
                                       acc_rate_cov = 0.5, init_cov_pars=init_cov_pars))
    cov_pars2 <- c(0.4012595)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars2)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Estimation using gradient descent and too large learning rate
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                       lr_cov = 10, use_nesterov_acc = FALSE, init_cov_pars=init_cov_pars))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    
    # Prediction
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, params = list(optimizer_cov = "gradient_descent", 
                                                use_nesterov_acc = FALSE, lr_cov = 0.1, init_cov_pars=init_cov_pars))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.000000, -0.796538, -0.796538, 0.000000)
    expected_cov <- c(0.1133436, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
                      0.1407783, 0.1407783, 0.0000000, 0.0000000, 0.1407783,
                      0.1407783, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.4070775)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,6,11,16)])),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_response = TRUE)
    expected_mu <- c(0.5000000, 0.2279027, 0.2279027, 0.5000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    # Prediction for only new groups
    group_test <- c(-1,-1,-2,-2)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-rep(0,4))),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-rep(0,0.4070775))),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-rep(0.5,4))),TOLERANCE_STRICT)
    # Prediction for only new cluster_ids
    cluster_ids_pred <- c(-1L,-1L,-2L,-2L)
    group_test <- c(1,99999,3,3)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, cluster_ids_pred = cluster_ids_pred,
                    predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-rep(0,4))),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-rep(0.4070771,4))),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, cluster_ids_pred = cluster_ids_pred,
                    predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-rep(0.5,4))),TOLERANCE_STRICT)
    
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences,] 
    group_unique <- unique(group)
    preds <- predict(gp_model, group_data_pred = group_unique, 
                     predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-6)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),1E-6)
    
    # Estimation using Nelder-Mead
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "nelder_mead", delta_rel_conv=1e-6, init_cov_pars=init_cov_pars))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.4027452)),TOLERANCE_STRICT)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-c(0.0000000, -0.7935873, -0.7935873, 0.0000000))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-c(0.1130051, 0.1401125, 0.1401125, 0.4027452))),TOLERANCE_STRICT)
    
    # Estimation using BFGS
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "bfgs", init_cov_pars=init_cov_pars))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.4025483)),TOLERANCE_STRICT)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-c(0.0000000, -0.7934523, -0.7934523, 0.0000000))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-c(0.1129896, 0.1400821, 0.1400821, 0.4025483))),TOLERANCE_STRICT)
    
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-65.8590638),TOLERANCE_STRICT)
    
    # Do optimization using optim
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    opt <- optim(par=c(2), fn=gp_model$neg_log_likelihood, y=y, method="Brent", lower=0, upper=1E9)
    cov_pars <- c(0.40255)
    expect_lt(sum(abs(opt$par-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(abs(opt$value-(65.2599674)),TOLERANCE_LOOSE)
  })
  
  test_that("Binary classification with one grouped random effects and offset", {
  
    n <- 250000 # number of samples
    m <- n / 500 # number of categories / levels for grouping variable
    group <- rep(1,n) # grouping variable
    for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
    b_gr <- sqrt(0.5) * qnorm(sim_rand_unif(n=m, init_c=0.5455))
    probs <- pnorm(b_gr[group])
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.23431) < probs)
    offset <- (2*(sim_rand_unif(n=m, init_c=0.54) - 0.5))[group]
    probs_o <- pnorm(b_gr[group] + offset)
    y_o <- as.numeric(sim_rand_unif(n=n, init_c=0.23431) < probs_o)
    group_test <- c(1,3,9999)

    nrounds <- 5
    cov_pars <- c(0.4872743)
    expected_mu <- c(0.03985967, -0.42595831, 0.00000000)
    expected_cov <- c(0.003123268, 0.000000000, 0.000000000, 0.000000000, 
                      0.003334890, 0.000000000, 0.000000000, 0.000000000, 0.487274305)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, params = DEFAULT_OPTIM_PARAMS)
    pred <- predict(gp_model, group_data_pred = group_test, 
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y_o, params = DEFAULT_OPTIM_PARAMS, fixed_effects = offset)
    pred <- predict(gp_model, group_data_pred = group_test, fixed_effects = offset, 
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.03)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    
    # With linear predictor and offset
    X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
    X_test <- cbind(rep(1,3),c(-0.5,0.4,1))
    beta <- c(0.1,2) # regression coefficients
    probs <- pnorm(b_gr[group] + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.23431) < probs)
    probs_o <- pnorm(b_gr[group] + offset + X%*%beta)
    y_o <- as.numeric(sim_rand_unif(n=n, init_c=0.23431) < probs_o)
    
    nrounds <- 8
    cov_pars <- c(0.4784317)
    coefs <- c(0.032651058, 0.031126881, 2.006988705, 0.006637149)
    expected_mu <- c(-0.8417404, 0.5597127, 2.0396398)
    expected_cov <- c(0.005222087, 0.000000000, 0.000000000, 0.000000000, 
                      0.005740345, 0.000000000, 0.000000000, 0.000000000, 0.478431721)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD)
    pred <- predict(gp_model, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y_o, X=X, params = DEFAULT_OPTIM_PARAMS_STD, fixed_effects = offset)
    pred <- predict(gp_model, group_data_pred = group_test, X_pred = X_test, fixed_effects = offset, 
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),0.01)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs)),0.4)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.15)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.01)
  })
  
  test_that("Binary classification with multiple grouped random effects ", {
    
    probs <- pnorm(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.57341) < probs)
    init_cov_pars <- rep(1,3)
    
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                         lr_cov = 0.2, use_nesterov_acc = FALSE, maxit=100))
                    , file='NUL')
    expected_values <- c(0.3060671, 0.9328884, 0.3146682)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 37)
    
    # Predict training data random effects
    cov_pars <- gp_model$get_cov_pars()
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    first_occurences_1 <- match(unique(group), group)
    first_occurences_2 <- match(unique(group2), group2)
    pred_random_effects <- all_training_data_random_effects[first_occurences_1,c(1,4)]
    pred_random_slopes <- all_training_data_random_effects[first_occurences_1,c(3,6)]
    head(pred_random_slopes)
    pred_random_effects_crossed <- all_training_data_random_effects[first_occurences_2,c(2,5)] 
    group_unique <- unique(group)
    group_data_pred = cbind(group_unique,rep(-1,length(group_unique)))
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr, 
                     predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(pred_random_effects[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_random_effects[,2] - (preds$var-cov_pars[2]))),TOLERANCE_STRICT)
    # Check whether random slopes are correct
    x_pr = rep(1,length(group_unique))
    preds2 <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr, 
                      predict_response = FALSE)
    expect_lt(sum(abs(pred_random_slopes[,1] - (preds2$mu-preds$mu))),TOLERANCE_STRICT)
    # Check whether crossed random effects are correct
    group_unique <- unique(group2)
    group_data_pred = cbind(rep(-1,length(group_unique)),group_unique)
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr, 
                     predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(pred_random_effects_crossed[,1] - preds$mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(pred_random_effects_crossed[,2] - (preds$var-cov_pars[1]))),TOLERANCE_STRICT)
    
    # Prediction
    gp_model <- GPModel(likelihood = "bernoulli_probit", group_data = cbind(group,group2),
                        group_rand_coef_data = x, ind_effect_group_rand_coef = 1)
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.5195889, -0.6411954, 0.0000000)
    expected_cov <- c(0.3422367, 0.1554011, 0.0000000, 0.1554011,
                      0.3457334, 0.0000000, 0.0000000, 0.0000000, 1.8080000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Multiple random effects: training with Nelder-Mead
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "nelder_mead", delta_rel_conv=1e-6, init_cov_pars=init_cov_pars))
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.3055487, 0.9300562, 0.3048811))),TOLERANCE_STRICT)
    # Multiple random effects: training with BFGS
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "bfgs", init_cov_pars=init_cov_pars)), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.3030693, 0.9293106, 0.3037503))),TOLERANCE_STRICT)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.8,1.2),y=y)
    expect_lt(abs(nll-60.6422359),TOLERANCE_LOOSE)
    
    # Multiple cluster_ids
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y, cluster_ids = cluster_ids, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                         lr_cov = 0.2, use_nesterov_acc = FALSE, maxit=100)), file='NUL')
    expected_values <- c(0.1634433, 0.8952201, 0.3219087)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 42)
    # Prediction
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                        cluster_ids = cluster_ids, likelihood = "bernoulli_probit")
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), cluster_ids_pred = cluster_ids_pred, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.2159939, 0.0000000, 0.0000000)
    expected_cov <- c(0.4547941, 0.0000000, 0.0000000, 0.0000000,
                      1.7120000, 0.0000000, 0.0000000, 0.0000000, 1.8080000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # Only one RE and random coefficient
    probs <- pnorm(Z1 %*% b_gr_1 + Z3 %*% b_gr_3)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.957341) < probs)
    init_cov_pars <- c(1,1)
    capture.output( gp_model <- fitGPModel(group_data = group, group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE, maxit=100))
                    , file='NUL')
    expected_values <- c(1.00742383, 0.02612587)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 100)
    
    # Random coefficients with intercept random effect dropped
    probs <- pnorm(Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.8341) < probs)
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, 
                                           ind_effect_group_rand_coef = 1, drop_intercept_group_rand_effect = c(TRUE,FALSE),
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars)), file='NUL')
    expected_values <- c(1.0044712, 0.6549656)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 18)
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
    first_occurences_1 <- match(unique(group), group)
    first_occurences_2 <- match(unique(group2), group2)
    pred_random_slopes <- all_training_data_random_effects[first_occurences_1,2]
    pred_random_effects_crossed <- all_training_data_random_effects[first_occurences_2,1] 
    group_unique <- unique(group)
    group_data_pred = cbind(group_unique,rep(-1,length(group_unique)))
    # Check whether random slopes are correct
    x_pr = rep(1,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr, predict_response = FALSE)
    expect_lt(sum(abs(pred_random_slopes - preds$mu)),TOLERANCE_LOOSE)
    # Check whether crossed random effects are correct
    group_unique <- unique(group2)
    group_data_pred = cbind(rep(-1,length(group_unique)),group_unique)
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr, predict_response = FALSE)
    expect_lt(sum(abs(pred_random_effects_crossed - preds$mu)),TOLERANCE_LOOSE)
    # Prediction
    gp_model <- GPModel(likelihood = "bernoulli_probit", group_data = cbind(group,group2),
                        group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                        drop_intercept_group_rand_effect = c(TRUE,FALSE))
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.8,1.2), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.8493404, -0.2338359, 0.0000000)
    expected_cov <- c(0.206019606, -0.001276366, 0.0000000, -0.001276366,
                      0.155209578, 0.0000000, 0.0000000, 0.0000000, 0.908000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.8,1.2), predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    
    # Including linear fixed effects
    probs <- pnorm(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3 + X%*%beta)
    y_lin <- as.numeric(sim_rand_unif(n=n, init_c=0.41) < probs)
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y_lin, X=X, likelihood = "bernoulli_probit",
                                           params = DEFAULT_OPTIM_PARAMS)
                    , file='NUL')
    cov_pars <- c(0.7135209, 1.4289386, 1.6037208)
    coef <- c(-0.382657, 2.413484)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 19)
    # Prediction
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    X_test <- cbind(rep(1,3),c(-0.5,0.4,1))
    pred <- gp_model$predict(group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             X_pred = X_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.5136463, 0.8644346, 2.0308268)
    expected_cov <- c(0.5546899, 0.1847860, 0.0000000, 0.1847860, 0.5615866, 
                      0.0000000, 0.0000000, 0.0000000, 2.2867943)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
  })
  
  test_that("Binary classification for combined Gaussian process and grouped random effects ", {
    
    probs <- pnorm(L %*% b_1 + Z1 %*% b_gr_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.67341) < probs)
    init_cov_pars <- c(1,1,mean(dist(coords))/3)
    
    # Estimation using gradient descent
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        group_data = group, likelihood = "bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                       lr_cov = 0.2, use_nesterov_acc = FALSE,
                                                       convergence_criterion = "relative_change_in_parameters"))
                    , file='NUL')
    cov_pars <- c(0.3181509, 1.2788456, 0.1218680)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 55)
    
    # Prediction
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           group_data = group, y = y, params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars, 
                                                                                    use_nesterov_acc = FALSE, lr_cov = 0.2))
                    , file='NUL')
    coord_test <- cbind(c(0.1,0.21,0.7),c(0.9,0.91,0.55))
    group_test <- c(1,3,9999)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1217634, -0.9592585, -0.2694489)
    expected_cov <- c(1.0745455607, 0.2190063794, 0.0040797451, 0.2190063794,
                      1.0089298170, 0.0000629706, 0.0040797451, 0.0000629706, 1.0449941968)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test,
                    predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test, predict_response = TRUE)
    expected_mu <- c(0.5336859, 0.2492699, 0.4252731)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model)
    pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]), predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,2] - pred_GP$mu)),1E-6)
    # Grouped REs
    preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords, predict_response = FALSE)
    pred_RE <- preds$mu - pred_GP$mu
    expect_lt(sum(abs(training_data_random_effects[,1] - pred_RE)),1E-6)
    
    # Estimation using Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        group_data = group, likelihood = "bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "nelder_mead", delta_rel_conv=1E-8, init_cov_pars=init_cov_pars)), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.3181320, 1.2795124, 0.1218866))),TOLERANCE_STRICT)
    
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(1.1,0.9,0.2),y=y)
    expect_lt(abs(nll-65.7219266),TOLERANCE_STRICT)
    
    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        group_data = group, likelihood = "bernoulli_probit")
    capture.output( opt <- optim(par=c(1.5,1,0.1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead"), file='NUL')
    cov_pars <- c(0.3181509, 1.2788456, 0.1218680)
    expect_lt(sum(abs(opt$par-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(abs(opt$value-(63.7432077)),TOLERANCE_LOOSE)
    expect_equal(as.integer(opt$counts[1]), 164)
  })
  
  test_that("Combined GP and grouped random effects model with random coefficients ", {
    
    probs <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3) + 
                     Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.9867234) < probs)
    init_cov_pars <- c(rep(1,3),rep(c(1,mean(dist(coords))/3),3))
    
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                         lr_cov = 0.2, use_nesterov_acc = FALSE, maxit=10))
                    , file='NUL')
    expected_values <- c(0.09859312, 0.35813763, 0.50164573, 0.67372019,
                         0.08825524, 0.77807532, 0.10896128, 1.03921290, 0.09538707)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Prediction
    gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential", likelihood = "bernoulli_probit",
                        group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1)
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    group_data_pred = cbind(c(1,1,7),c(2,1,3))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(1.612451, 1.147407, -1.227187)
    expected_cov <- c(1.63468526, 1.02982815, -0.01916993, 1.02982815,
                      1.43601348, -0.03404720, -0.01916993, -0.03404720, 1.55017397)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-71.4286594),TOLERANCE_LOOSE)
  })
  
  test_that("Combined GP and grouped random effects model with cluster_id's not constant ", {
    
    probs <- pnorm(L %*% b_1 + Z1 %*% b_gr_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    init_cov_pars <- c(1,1,mean(dist(coords[cluster_ids==1,]))/3)
    
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                                           y = y, cluster_ids = cluster_ids,likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", lr_cov=0.2, use_nesterov_acc = FALSE,
                                                         init_cov_pars=init_cov_pars))
                    , file='NUL')
    cov_pars <- c(0.276476226, 0.007278016, 0.132195703)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 261)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    group_data_pred = c(1,1,9999)
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                        cluster_ids = cluster_ids,likelihood = "bernoulli_probit")
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,1,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1074035, 0.0000000, 0.2945508)
    expected_cov <- c(0.98609786, 0.00000000, -0.02013244, 0.00000000,
                      2.50000000, 0.00000000, -0.02013244, 0.00000000, 2.28927616)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
  })
  
  test_that("Binary classification Gaussian process model with Vecchia approximation", {
    params_vecchia <- c(DEFAULT_OPTIM_PARAMS, cg_delta_conv = sqrt(1e-6), 
                        num_rand_vec_trace = 500, cg_preconditioner_type = "piv_chol_on_Sigma")
    # Simulate data and define expected values
    probs <- pnorm(L %*% b_1) # note: linear predictor is not included in simulation
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.19341) < probs)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    init_cov_pars <- c(1,mean(dist(coords))/3)
    cov_pars_pred_eval = c(1,0.2)
    cov_pars <- c(0.92350730, 0.05944216)
    coefs <- c(0.3983333, -0.2653886)
    num_it <- 17
    expected_mu <- c(0.3389905, 0.1512445, -0.1039307)
    expected_cov <- c(0.6193228722, 0.5503216948, -0.0001420698, 0.5503216948, 
                      0.6159348965, -0.0001556274, -0.0001420698, -0.0001556274, 0.4291674143)
    expected_mu_resp <- c(0.6050312, 0.5473537, 0.4653610)
    expected_var_resp <- c(0.2389684, 0.2477576, 0.2488001)
    expected_nll <- 67.1834222
    # Estimation, prediction, and likelihood evaluation without Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, X = X, params = DEFAULT_OPTIM_PARAMS), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, 
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, 
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_LOOSE)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE, 
                    predict_var = TRUE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu_resp)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(pred$var-expected_var_resp)),TOLERANCE_LOOSE)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
    expect_lt(abs(nll-expected_nll),TOLERANCE_STRICT)
    # No linear regression term without Vecchia approximation
    cov_pars_no_X <- c(0.6875476, 0.1062862 )
    mu_no_X <- c(0.01874013, 0.01200800, 0.20498871)
    var_no_X <- c(0.6105248, 0.6093745, 0.4235374)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, params = DEFAULT_OPTIM_PARAMS), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_no_X)),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, 
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval)
    expect_lt(sum(abs(pred$mu-mu_no_X)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-var_no_X)),TOLERANCE_LOOSE)
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
                                           y = y_multiple, X = X, params = DEFAULT_OPTIM_PARAMS), file='NUL')
    
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_multiple)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs_multiple)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_multiple)
    pred <- predict(gp_model, y=y_multiple, gp_coords_pred = coord_test_multiple, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred_eval)
    expect_lt(sum(abs(pred$mu-expected_mu_multiple)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(pred$var-expected_var_multiple)),TOLERANCE_LOOSE)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y_multiple)
    expect_lt(abs(nll-nll_multiple),TOLERANCE_STRICT)
    
    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative") {
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
        loop_cg_PC = c("piv_chol_on_Sigma", "Sigma_inv_plus_BtWB")
      } else {
        tolerance_loc_1 <- TOLERANCE_STRICT
        tolerance_loc_2 <- TOLERANCE_LOOSE
        loop_cg_PC = c("Sigma_inv_plus_BtWB")
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
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_1)
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
        expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),tolerance_loc_1)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, 
                                        predict_var = TRUE, predict_response = FALSE, 
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
        if (inv_method != "iterative" || cg_preconditioner_type == "Sigma_inv_plus_BtWB") {
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, 
                                          predict_response = TRUE, predict_var = TRUE, 
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu_resp)),tolerance_loc_1)
          expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
        }        
        # Likelihood evaluation
        nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
        expect_lt(abs(nll-expected_nll),tolerance_loc_1)
        
        if(inv_method == "iterative" && cg_preconditioner_type == "piv_chol_on_Sigma"){
          ## Cannot change cg_preconditioner_type after a model has been fitted
          expect_error( capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                                           lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                                           convergence_criterion = "relative_change_in_parameters",
                                                                           cg_delta_conv = 1e-6, num_rand_vec_trace = 500,
                                                                           cg_preconditioner_type = "Sigma_inv_plus_BtWB")), file='NUL'))
        }
        
        if (inv_method != "iterative" || cg_preconditioner_type == "Sigma_inv_plus_BtWB") {# some tests are only run for one preconditioner
          ############################
          # Vecchia approximation with random ordering
          ############################
          capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                                 vecchia_ordering="random", likelihood = "bernoulli_probit", 
                                                 gp_approx = "vecchia",  num_neighbors = n-1,
                                                 y = y, X = X, params = params_vecchia, 
                                                 matrix_inversion_method = inv_method), file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_1)
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
          expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),tolerance_loc_1)
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
          expect_lt(abs(nll-expected_nll),tolerance_loc_1)
          
          #######################
          ## Less neighbors than observations
          #######################
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = "bernoulli_probit", gp_approx = "vecchia", 
                                              num_neighbors = 30, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method), file='NUL')
          capture.output( fit(gp_model, y = y, X = X, params = params_vecchia)
                          , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_2)
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
          expect_lt(sum(abs(as.vector(pred$cov)-cov_less_neig)),tolerance_loc_1)
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
          expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_loc)),tolerance_loc_1)
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
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_no_X)),tolerance_loc_1)
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
                                               y = y_multiple, X = X, params = params_vecchia), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_multiple)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs_multiple)),tolerance_loc_1)
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
        pred_mu_exp <- preds$mu - X %*% gp_model$get_coef()
        expect_lt(sum(abs(training_data_random_effects[,1] - pred_mu_exp)),tolerance_loc_1)
        if(inv_method == "iterative"){
          expect_lt(mean(abs(training_data_random_effects[,2] - preds$var)), tolerance_loc_1) #Different RNG-Status
        } else {
          expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)), tolerance_loc_1)  
        }
        
      }# end loop cg_preconditioner_type in loop_cg_PC
    }# end loop inv_method in c("cholesky", "iterative")
    
    ###################
    ## Random coefficient GPs
    ###################
    probs <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3))
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.543) < probs)
    init_cov_pars_RC <- rep(init_cov_pars, 3)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           y = y, likelihood = "bernoulli_probit", gp_approx = "vecchia", 
                                           num_neighbors = n-1, vecchia_ordering = "none",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 1, use_nesterov_acc = TRUE, 
                                                         acc_rate_cov=0.5, maxit=1000, init_cov_pars=init_cov_pars_RC)), file='NUL')
    expected_values <- c(0.3701097, 0.2846740, 2.1160323, 0.3305266, 0.1241462, 0.1846456)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 39)
    # Same estimation without Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           y = y, likelihood = "bernoulli_probit", gp_approx = "none",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 1, use_nesterov_acc = TRUE, 
                                                         acc_rate_cov=0.5, maxit=1000, init_cov_pars=init_cov_pars_RC)), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 39)
    # Prediction
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC,
                                        cov_function = "exponential", likelihood = "bernoulli_probit",
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none"), file='NUL')
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, gp_rand_coef_data_pred=Z_SVC_test,
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
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-65.1768199),TOLERANCE_LOOSE)
    
    ###################
    ##  Multiple cluster IDs
    ###################
    probs <- pnorm(L %*% b_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2978341) < probs)
    init_cov_pars <- c(1,mean(dist(coords[cluster_ids==1,]))/3)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, cluster_ids = cluster_ids, likelihood = "bernoulli_probit",
                                           gp_approx = "vecchia", num_neighbors = n-1,
                                           vecchia_ordering = "none",
                                           params = list(optimizer_cov = "gradient_descent", lr_cov=0.2, 
                                                         use_nesterov_acc = FALSE, init_cov_pars=init_cov_pars)), file='NUL')
    cov_pars <- c(0.5085134, 0.2011667)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 20)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    cluster_ids_pred = c(1,3,1)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        cluster_ids = cluster_ids,likelihood = "bernoulli_probit"), file='NUL')
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1509569, 0.0000000, 0.9574946)
    expected_cov <- c(1.2225959453, 0.0000000000, 0.0003074858, 0.0000000000,
                      1.5000000000, 0.0000000000, 0.0003074858, 0.0000000000, 1.0761874845)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
  })
  
  test_that("Binary classification Gaussian process model with Wendland covariance function", {
    
    probs <- pnorm(L %*% b_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    init_cov_pars <- c(mean(dist(coords))/3)
    # Estimation using gradient descent
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 0.1,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, init_cov_pars=init_cov_pars)), file='NUL')
    cov_pars <- c(0.5553221)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
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
                                                     init_cov_pars=init_cov_pars))
    cov_pars <- c(0.4072025)
    coef <- c(-0.1113238, 1.5178339)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 43)
    
    # Estimation using Nelder-Mead
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead", delta_rel_conv=1e-12,
                                                     init_cov_pars=init_cov_pars))
    cov_pars <- c(0.399973)
    coef <- c(-0.1109516, 1.5149596)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 188)
    # init_cov_pars not given
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead", delta_rel_conv=1e-12))
    cov_pars <- c(0.399973)
    coef <- c(-0.1109516, 1.5149596)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 194)
    
    # # Estimation using BFGS
    # Does not converge to correct solution (version 0.7.2, 21.02.2022)
    # gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
    #                        y = y, X=X, params = list(optimizer_cov = "bfgs",
    #                                                  optimizer_coef = "bfgs"))
    # cov_pars <- c(0.3999729)
    # coef <- c(-0.1109532, 1.5149592)
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    # expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    # expect_equal(gp_model$get_num_optim_iter(), 17)
    
    # Prediction
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                     optimizer_coef = "gradient_descent",
                                                     use_nesterov_acc=FALSE, lr_coef=1, init_cov_pars=init_cov_pars))
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.81132150, -0.08574588, 0.21768684, 1.40591430)
    expected_cov <- c(0.1380238, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.1688248, 0.1688248, 
                      0.0000000, 0.0000000, 0.1688248, 0.1688248, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.4051185)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test, predict_response = TRUE)
    expected_mu <- c(0.2234684, 0.4683923, 0.5797886, 0.8821984)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    
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
                                           y = y, X=X, params = list(std_dev = TRUE, optimizer_cov = "gradient_descent",
                                                                     optimizer_coef = "gradient_descent", init_cov_pars=init_cov_pars,
                                                                     use_nesterov_acc = TRUE, lr_cov = 0.1, lr_coef = 1)),
                    file='NUL')
    cov_pars <- c(0.4006247)
    coef <- c(-0.1112284, 0.2566224, 1.5160319, 0.2636918)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    
    # Providing initial covariance parameters and coefficients
    cov_pars <- c(1)
    coef <- c(2,5)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(maxit=0, init_cov_pars=cov_pars, init_coef=coef))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    
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
                                                         use_nesterov_acc = TRUE, init_cov_pars=init_cov_pars))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.9759329)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-c(0.0971936, 1.9950664))),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    
    # Estimation using Nelder-Mead
    gp_model <- fitGPModel(group_data = group_L, likelihood = "bernoulli_probit",
                           y = y_L, X=X_L, params = list(optimizer_cov = "nelder_mead",
                                                         optimizer_coef = "nelder_mead", 
                                                         delta_rel_conv=1e-6, init_cov_pars=init_cov_pars))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.9712287)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-c(0.09758098, 1.99473498))),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 109)
    
  })
  
  test_that("Binary classification with linear predictor and Gaussian process model ", {
    
    probs <- pnorm(L %*% b_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.199) < probs)
    # Estimation
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1.3992407, 0.261976))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-c(0.2764603, 1.5556477))),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.7480014, 0.3297389, 2.7039005)
    expected_var <- c(0.8596074, 0.8574038, 0.5016189)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # Estimation using Nelder-Mead
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead", 
                                                     maxit=1000, delta_rel_conv=1e-12))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1.2717516, 0.2875537))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-c(0.1999365, 1.4666199))),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 231)
    
    # Standard deviations
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit", 
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD),
                    file='NUL')
    coef <- c(0.2764603, 0.5420554, 1.5556477, 0.3146670)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    
  })
  
  test_that("Tapering for binary classification", {
    
    probs <- pnorm(L %*% b_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.199) < probs)
    # No tapering
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS)
    cov_pars <- c(1.3992407, 0.261976)
    coefs <- c(0.2764603, 1.5556477)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.7480014, 0.3297389, 2.7039005)
    expected_var <- c(0.8596074, 0.8574038, 0.5016189)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # With tapering and very large tapering range 
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           gp_approx = "tapering", cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # With tapering and small tapering range 
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", likelihood = "bernoulli_probit",
                                           cov_fct_shape = 2.5,
                                           gp_approx = "tapering", cov_fct_taper_shape = 1, cov_fct_taper_range = 0.5,
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS), file='NUL')
    cov_pars <- c(0.9397216, 0.1193397)
    coefs <- c(0.4465991, 1.4398973)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 16)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.3887768, 0.6451925, 2.5398402)
    expected_var <- c(0.7738142, 0.7868789, 0.4606071)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # Multiple observations at the same location
    eps_multiple <- as.vector(L_multiple %*% b_multiple)
    probs <- pnorm(eps_multiple + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.41) < probs)
    #No tapering
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", 
                                           likelihood = "bernoulli_probit", gp_approx = "none", 
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS), file='NUL')
    cov_pars <- c(1.09933629, 0.08239163)
    coefs <- c(0.5652697, 1.7696475)
    num_it <- 15
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.4170239, 0.8119773, 2.2276953)
    expected_var <- c(0.9473460, 0.9779394, 0.9779394)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # With tapering and very large tapering range 
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", 
                                           likelihood = "bernoulli_probit", gp_approx = "tapering", 
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs)),1e-2)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1e-2)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),1e-2)
    
    # With tapering and smalltapering range 
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", 
                                           likelihood = "bernoulli_probit", gp_approx = "tapering", 
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 0.5,
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS), file='NUL')
    cov_pars <- c(1.1062329, 0.1479239)
    coefs <- c(0.5562694, 1.7715742)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coefs)),1e-2)
    expect_equal(gp_model$get_num_optim_iter(), 48)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.4272906, 0.8014820, 2.2187414)
    expected_var <- c(0.9295132, 0.9637862, 0.9637862)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
  })
  
  test_that("Binary classification with Gaussian process model and logit link function", {
    
    probs <- 1/(1+exp(- L %*% b_1))
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    # Estimation 
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_logit",
                                           y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE, lr_cov=0.01))
                    , file='NUL')
    cov_pars <- c(1.4300136, 0.1891952)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 85)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.7792960, -0.7876208, 0.5476390)
    expected_cov <- c(1.024267e+00, 9.215206e-01, 5.561435e-05, 9.215206e-01, 1.022897e+00,
                      2.028618e-05, 5.561435e-05, 2.028618e-05, 7.395747e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.3442815, 0.3426873, 0.6159933)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_mu*(1-expected_mu))),TOLERANCE_STRICT)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-66.299571),TOLERANCE_STRICT)
  })
  
  test_that("Poisson regression ", {
    
    # Single level grouped random effects
    mu <- exp(Z1 %*% b_gr_1)
    y <- qpois(sim_rand_unif(n=n, init_c=0.04532), lambda = mu)
    # Estimation 
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "poisson",
                                           y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE, lr_cov=0.1))
                    , file='NUL')
    cov_pars <- c(0.4033406)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.07765297, -0.87488533, -0.87488533, 0.00000000)
    expected_cov <- c(0.07526284, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                      0.15041230, 0.15041230, 0.00000000, 0.00000000, 0.15041230,
                      0.15041230, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.40334058)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(1.1221925, 0.4494731, 0.4494731, 1.2234446)
    expected_var <- c(1.2206301, 0.4822647, 0.4822647, 1.9670879)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-140.4554806),TOLERANCE_LOOSE)
    
    # Multiple random effects
    mu <- exp(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- qpois(sim_rand_unif(n=n, init_c=0.74532), lambda = mu)
    # Estimation 
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "poisson",
                                           y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE, lr_cov=0.1))
                    , file='NUL')
    cov_pars <- c(0.4069344, 1.6988978, 1.3415016)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 7)
    # Prediction
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.92620057, -0.08200469, 0.00000000)
    expected_cov <- c(0.07730896, 0.04403442, 0.00000000, 0.04403442, 0.11600469,
                      0.00000000, 0.00000000, 0.00000000, 1.80800000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # Gaussian process model
    mu <- exp(L %*% b_1)
    y <- qpois(sim_rand_unif(n=n, init_c=0.435), lambda = mu)
    # Estimation 
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "poisson",
                                           y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE))
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1.1853922, 0.1500197))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.4329068, 0.4042531, 0.6833738)
    expected_cov <- c(6.550626e-01, 5.553938e-01, -8.406290e-06, 5.553938e-01, 6.631295e-01, -7.658261e-06, -8.406290e-06, -7.658261e-06, 4.170417e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(2.139213, 2.087188, 2.439748)
    expected_var <- c(6.373433, 6.185895, 5.519896)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_LOOSE)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-195.03708036),TOLERANCE_STRICT)
    
    ## Grouped random effects model with a linear predictor
    mu_lin <- exp(Z1 %*% b_gr_1 + X%*%beta)
    y_lin <- qpois(sim_rand_unif(n=n, init_c=0.84532), lambda = mu_lin)
    gp_model <- fitGPModel(group_data = group, likelihood = "poisson",
                           y = y_lin, X=X, params = list(optimizer_cov = "gradient_descent",
                                                         optimizer_coef = "gradient_descent", lr_cov = 0.1, lr_coef = 0.1,
                                                         use_nesterov_acc = TRUE, acc_rate_cov = 0.5))
    cov_pars <- c(0.2993371)
    coef <- c(-0.1526089, 2.1267601)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 29)
  })
  
  test_that("Gamma regression ", {
    params <- list(optimizer_cov = "gradient_descent", optimizer_coef = "gradient_descent", 
                   estimate_aux_pars = FALSE, init_aux_pars = 1.,
                   lr_cov = 0.1, lr_coef = 0.1,
                   use_nesterov_acc = TRUE, acc_rate_cov = 0.5)
    params_shape <- params
    params_shape$estimate_aux_pars <- TRUE
    shape <- 1
    
    # Single level grouped random effects
    mu <- exp(Z1 %*% b_gr_1)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.04532), scale = mu/shape, shape = shape)
    # Cannot have 0 in response variable
    y_zero <- y
    y_zero[1] <- 0
    expect_error(gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                        y = y_zero, params = params))
    # Estimation 
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params)
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.5174554)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.2095341, -0.9170767, -0.9170767, 0.0000000)
    expected_cov <- c(0.08105393, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                      0.09842279, 0.09842279, 0.00000000, 0.00000000, 0.09842279,
                      0.09842279, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.51745540)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(1.2841038, 0.4198468, 0.4198468, 1.2952811)
    expected_var <- c(1.9273575, 0.2127346, 0.2127346, 3.9519573)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-105.676137),TOLERANCE_LOOSE)
    # Also estimate shape parameter
    params_shape$optimizer_cov <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.5141632)
    aux_pars <- c(0.9719373)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 45)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.5198431)
    aux_pars <- c(0.970999)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 23)
    # Can set learning rate for auxiliary parameters via lr_cov
    params_temp <- params_shape
    params_temp$maxit = 1
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params_temp), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.9058829)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-0.9297985)),TOLERANCE_STRICT)
    params_temp$lr_cov = 0.001
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params_temp), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.998025)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-0.9985453)),TOLERANCE_STRICT)
    
    # Multiple random effects
    mu <- exp(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.04532), scale = mu/shape, shape = shape)
    # Estimation 
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params)
                    , file='NUL')
    cov_pars <- c(0.5050690, 1.2043329, 0.5280103)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    # Prediction
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1121777, 0.1972216, 0.0000000)
    expected_var <- c(0.2405621, 0.2259258, 1.8080000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    # Also estimate shape parameter
    params_shape$optimizer_cov <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.5050897, 1.2026241, 0.5232070)
    aux_pars <- c(0.9819755)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 215)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.5065183, 1.2028488, 0.5360939)
    aux_pars <- c(0.9827199)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 31)
    # Also estimate shape parameter with adam
    params_shape$optimizer_cov <- "adam"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.5052794, 1.2018843, 0.5230190)
    aux_pars <- c(0.9820493)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 279)
    # Also estimate shape parameter with bfgs
    params_shape$optimizer_cov <- "bfgs"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.5052794, 1.2018842, 0.5230190)
    aux_pars <- c(0.9820493)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    # Also estimate shape parameter with gradient descent using internal initialization
    params_shape_no_init <- params_shape
    params_shape_no_init$init_aux_pars <- NULL
    params_shape_no_init$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_shape_no_init), file='NUL')
    cov_pars <- c(0.5064068, 1.2028118, 0.5355322)
    aux_pars <- c(0.9826897)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 34)
    
    # Gaussian process model
    mu <- exp(L %*% b_1)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.435), scale = mu/shape, shape = shape)
    # Estimation 
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                           likelihood = "gamma", y = y, params = params)
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1.0649094, 0.2738999))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.3376250, 0.3023855, 0.7810425)
    expected_cov <- c(0.4567916157, 0.4033257822, -0.0002256179, 0.4033257822, 0.4540419202, 
                      -0.0002258048, -0.0002256179, -0.0002258048, 0.3368598330)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-154.4561783),TOLERANCE_STRICT)
    # Also estimate shape parameter
    params_shape$optimizer_cov <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                           likelihood = "gamma", y = y, params = params_shape)
                    , file='NUL')
    cov_pars <- c(1.0447764, 0.2973086)
    aux_pars <- c(0.9402551)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 104)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                           likelihood = "gamma", y = y, params = params_shape)
                    , file='NUL')
    cov_pars <- c(1.0323441, 0.2898717)
    aux_pars <- c(0.9413081)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 26)
    
    ## Grouped random effects model with a linear predictor
    mu_lin <- exp(Z1 %*% b_gr_1 + X%*%beta)
    y_lin <- qgamma(sim_rand_unif(n=n, init_c=0.532), scale = mu_lin/shape, shape = shape)
    gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                           y = y_lin, X=X, params = params)
    cov_pars <- c(0.474273)
    coef <- c(-0.07802971, 1.89766436)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    # Also estimate shape parameter
    params_shape$optimizer_cov <- "nelder_mead"
    gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                           y = y_lin, X=X, params = params_shape)
    cov_pars <- c(0.5097316)
    coef <- c(-0.08623548, 1.90033132)
    aux_pars <- c(1.350364)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 258)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                           y = y_lin, X=X, params = params_shape)
    cov_pars <- c(0.5179147)
    coef <- c(-0.08646342, 1.90053164)
    aux_pars <- c(1.351008)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 42)
    
    ## Combined grouped random effects and Gaussian process model
    mu <- exp(L %*% b_1 + Z1 %*% b_gr_1)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.987), scale = mu/shape, shape = shape)
    # Estimation 
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           group_data = group, likelihood = "gamma", y = y, params = params)
                    , file='NUL')
    cov_pars <- c(0.56585185, 0.62507125, 0.08278787)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 9)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    group_test <- c(1,3,3)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred=group_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.28574903, -0.67562130, 0.08821624)
    expected_cov <- c(0.649420831, 0.448952853, 0.007415143, 0.448952853, 0.683363103, 
                      0.126645556, 0.007415143, 0.126645556, 0.531015480)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.6,0.9,0.2),y=y)
    expect_lt(abs(nll-123.3965571),TOLERANCE_STRICT)
    # Also estimate shape parameter
    params_shape$optimizer_cov <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                           group_data = group, likelihood = "gamma", y = y, params = params_shape)
                    , file='NUL')
    cov_pars <- c(0.62184856, 0.98925230, 0.07441182)
    aux_pars <- c(1.702741)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 222)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                           group_data = group, likelihood = "gamma", y = y, params = params_shape)
                    , file='NUL')
    cov_pars <- c(0.62143448, 0.98703748, 0.07443428)
    aux_pars <- c(1.707991)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 27)
    
    # Gaussian process model with Vecchia approximation
    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative"){
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
      } else{
        tolerance_loc_1 <- 0.01
        tolerance_loc_2 <- 0.01
      }
      mu <- exp(0.75 * L %*% b_1)
      y <- qgamma(sim_rand_unif(n=n, init_c=0.7654), scale = mu/shape, shape = shape)
      # Estimation 
      if(inv_method=="iterative"){
        params$cg_delta_conv = 1e-6
        params$num_rand_vec_trace=500
      }
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                             likelihood = "gamma", y = y, params = params,
                                             gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering = "random",
                                             matrix_inversion_method = inv_method), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.9484890, 0.0731435))),tolerance_loc_2)
      if(inv_method != "iterative"){
        expect_lt(gp_model$get_num_optim_iter(), 11)
        expect_gt(gp_model$get_num_optim_iter(), 8) 
      }
      # Prediction
      coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
      gp_model$set_prediction_data(nsim_var_pred = 10000)
      pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
      expected_mu <- c(-0.1159426, -0.1028064, -0.3223582)
      expected_cov <- c(8.091398e-01, 1.079958e-01, -4.403387e-07, 1.079958e-01, 
                        8.055727e-01, -4.442709e-07, -4.403387e-07, -4.442709e-07, 6.957873e-01)
      expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
      adjust_tol <- 2
      if (inv_method == "iterative") adjust_tol <- 1.5
      expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),adjust_tol*tolerance_loc_1)
      # Evaluate approximate negative marginal log-likelihood
      nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
      nll_exp <- 159.9221359
      if(inv_method=="iterative"){
        expect_lt(abs(nll-nll_exp),0.2)
      } else{
        expect_lt(abs(nll-nll_exp),0.05)
      }
      # Also estimate shape parameter
      params_shape$optimizer_cov <- "nelder_mead"
      if(inv_method=="iterative"){
        params_shape$cg_delta_conv = 1e-6
        params_shape$num_rand_vec_trace=500
      }
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                             likelihood = "gamma", y = y, params = params_shape,
                                             gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering = "random")
                      , file='NUL')
      cov_pars <- c(1.14184253, 0.03605877)
      aux_pars <- c(1.328749)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_2)
      if(inv_method!="iterative"){
        expect_gt(gp_model$get_num_optim_iter(), 115)
        expect_lt(gp_model$get_num_optim_iter(), 135)
      }
      # Also estimate shape parameter with gradient descent
      params_shape$optimizer_cov <- "gradient_descent"
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                             likelihood = "gamma", y = y, params = params_shape,
                                             gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering = "random")
                      , file='NUL')
      cov_pars <- c(1.13722505, 0.03706853)
      aux_pars <- c(1.321834)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_2)
      expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),tolerance_loc_2)
      if(inv_method!="iterative"){
        expect_gt(gp_model$get_num_optim_iter(), 55)
        expect_lt(gp_model$get_num_optim_iter(), 60)
      }
    }
  }) # end Gamma regression
  
  test_that("negative binomial regression regression ", {
    params <- DEFAULT_OPTIM_PARAMS
    params$estimate_aux_pars <- TRUE
    params$init_aux_pars <- 1.
    params_shape <- params
    params_shape$estimate_aux_pars <- TRUE
    shape <- 1.8
    likelihood <- "negative_binomial"
    
    # Single level grouped random effects
    mu <- exp(Z1 %*% b_gr_1)
    y <- qnbinom(sim_rand_unif(n=n, init_c=0.156), mu = mu, size = shape)
    # Estimation 
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, params = params)
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.3356339)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1856629, -0.4022728, -0.4022728, 0.0000000)
    expected_cov <- c(0.09849537, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 
                      0.13548864, 0.13548864, 0.00000000, 0.00000000, 0.13548864, 
                      0.13548864, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.33563392)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(1.2647957, 0.7156755, 0.7156755, 1.18272011)
    expected_var <- c(2.508242, 1.148106, 1.148106, 2.935353)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-145.8511891),TOLERANCE_LOOSE)
    # Also estimate shape parameter
    params_shape$optimizer_cov <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.3371432)
    aux_pars <- c(1.735066)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 46)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.3356339)
    aux_pars <- c(1.637772)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    
    # Multiple random effects
    mu <- exp(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- qnbinom(sim_rand_unif(n=n, init_c=0.1468), mu = mu, size = shape)
    # Estimation 
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = likelihood,
                                           y = y, params = params)
                    , file='NUL')
    cov_pars <- c(0.5503418, 2.7228365, 0.6656752)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Prediction
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(0.365256, -1.618925, 0.000000)
    expected_var <- c(0.2766743, 0.4021417, 1.8080000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = likelihood,
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.5503418, 2.7228365, 0.6656752)
    aux_pars <- c(2.180879)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    
    # Also estimate shape parameter with gradient descent using internal initialization
    params_shape_no_init <- params_shape
    params_shape_no_init$init_aux_pars <- NULL
    params_shape_no_init$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = likelihood,
                                           y = y, params = params_shape_no_init), file='NUL')
    cov_pars <- c(0.5486444, 2.7506274, 0.6688556)
    aux_pars <- c(2.231622)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    
    # Gaussian process model
    mu <- exp(L %*% b_1)
    y <- qnbinom(sim_rand_unif(n=n, init_c=0.546), mu = mu, size = shape)
    # Estimation 
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                           likelihood = likelihood, y = y, params = params)
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1.1615931, 0.1677131))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 2)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.3683090, -0.3657134, 0.5807237)
    expected_var <- c(0.8239828, 0.8174733, 0.5547150)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-180.3098483),TOLERANCE_STRICT)
    # Also estimate shape parameter
    params_shape$optimizer_cov <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                           likelihood = likelihood, y = y, params = params_shape)
                    , file='NUL')
    cov_pars <- c(1.3238396, 0.1613754 )
    aux_pars <- c(1.091764 )
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 102)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                           likelihood = likelihood, y = y, params = params_shape)
                    , file='NUL')
    cov_pars <- c(1.1615931, 0.1677131 )
    aux_pars <- c(0.8235968)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 2)
    
    ## Grouped random effects model with a linear predictor
    mu_lin <- exp(Z1 %*% b_gr_1 + X%*%beta)
    y_lin <- qnbinom(sim_rand_unif(n=n, init_c=0.13278), mu = mu_lin, size = shape)
    gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                           y = y_lin, X=X, params = params)
    cov_pars <- c(0.2465099)
    coef <- c(-0.0265488, 2.2445935 )
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 21)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                           y = y_lin, X=X, params = params_shape)
    cov_pars <- c(0.2465099 )
    coef <- c( -0.0265488, 2.2445935)
    aux_pars <- c(1.859462 )
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 21)
    
    # Gaussian process model with Vecchia approximation
    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative"){
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
      } else{
        tolerance_loc_1 <- 0.01
        tolerance_loc_2 <- 0.01
      }
      mu <- exp(0.75 * L %*% b_1)
      y <- qnbinom(sim_rand_unif(n=n, init_c=0.4819), mu = mu, size = shape)
      # Estimation 
      if(inv_method=="iterative"){
        params$cg_delta_conv = 1e-6
        params$num_rand_vec_trace=500
      }
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                             likelihood = likelihood, y = y, params = params,
                                             gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering = "random",
                                             matrix_inversion_method = inv_method), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.400761, 0.143670))),tolerance_loc_2)
      # Prediction
      coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
      gp_model$set_prediction_data(nsim_var_pred = 10000)
      pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
      expected_mu <- c(-0.1924198, -0.2171927, 0.4252168)
      expected_cov <- c(0.3457035342, 0.1671072475, 0.0001336443, 0.1671072475, 0.3481552180, 
                        0.0001349335, 0.0001336443, 0.0001349335, 0.2529442560)
      expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
      adjust_tol <- 2
      if (inv_method == "iterative") adjust_tol <- 1.5
      expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),adjust_tol*tolerance_loc_1)
      # Evaluate approximate negative marginal log-likelihood
      nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
      nll_exp <- 164.4182898
      if(inv_method=="iterative"){
        expect_lt(abs(nll-nll_exp),0.2)
      } else{
        expect_lt(abs(nll-nll_exp),0.05)
      }
      # Also estimate shape parameter with gradient descent
      params_shape$optimizer_cov <- "gradient_descent"
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                             likelihood = likelihood, y = y, params = params_shape,
                                             gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering = "random")
                      , file='NUL')
      cov_pars <- c(0.400761, 0.143670 )
      aux_pars <- c(0.9492465)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_2)
      expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),tolerance_loc_2)
      
    }
  }) # end negative binomial regression
  
  test_that("Saving a GPModel and loading from file works for non-Gaussian data", {
    
    # Binary regression
    probs <- pnorm(Z1 %*% b_gr_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.542) < probs)
    # Train model
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                     optimizer_coef = "gradient_descent",
                                                     use_nesterov_acc = TRUE))
    # Make predictions
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    # Predict response
    pred_resp <- predict(gp_model, y=y, group_data_pred = group_test,
                         X_pred = X_test, predict_var = TRUE, predict_response = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model,filename = filename)
    # Delete model
    rm(gp_model)
    # Load from file and make predictions again
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, group_data_pred = group_test,
                           X_pred = X_test, predict_cov_mat = TRUE, predict_response = FALSE)
    pred_resp_loaded <- predict(gp_model_loaded, y=y, group_data_pred = group_test,
                                X_pred = X_test, predict_var = TRUE, predict_response = TRUE)
    
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    expect_equal(pred_resp$mu, pred_resp_loaded$mu)
    expect_equal(pred_resp$var, pred_resp_loaded$var)
    
    # Gamma regression
    mu <- exp(Z1 %*% b_gr_1 + X%*%beta)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.146), scale = mu, shape = 10)
    # Train model
    gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                           y = y, X = X, params = DEFAULT_OPTIM_PARAMS)
    # Make predictions
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    # Predict response
    pred_resp <- predict(gp_model, y=y, group_data_pred = group_test,
                         X_pred = X_test, predict_var = TRUE, predict_response = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    # Delete model
    rm(gp_model)
    # Load from file and make predictions again
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, group_data_pred = group_test,
                           X_pred = X_test, predict_cov_mat = TRUE, predict_response = FALSE)
    pred_resp_loaded <- predict(gp_model_loaded, y=y, group_data_pred = group_test,
                                X_pred = X_test, predict_var = TRUE, predict_response = TRUE)
    
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    expect_equal(pred_resp$mu, pred_resp_loaded$mu)
    expect_equal(pred_resp$var, pred_resp_loaded$var)
  })
  
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
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), likelihood = likelihood, 
                        cov_function = "matern_space_time")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 70.2364458
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), likelihood = likelihood, 
                           cov_function = "matern_space_time",
                           y = y, X = X, params = DEFAULT_OPTIM_PARAMS_STD)
    cov_pars <- c(0.1732524, 0.0705018, 0.1248471)
    coef <- c(0.1361989, 0.2177408, 0.2689329, 0.3004328)
    nrounds <- 85
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu <- c(0.1361989, 0.4160904, 0.6387114)
    expected_cov <- c(1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.8514930, 
                      0.0182482, 0.0000000, 0.0182482, 0.8105661)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = TRUE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expected_mu_resp <- c(0.5281153, 0.5871751, 0.6330091)
    expected_var_resp <- c(0.2492095, 0.2424005, 0.2323086)
    expect_lt(sum(abs(pred$mu-expected_mu_resp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_resp)),TOLERANCE_STRICT)
    
    ##############
    ## With Vecchia approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                                        cov_function = "matern_space_time",
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), likelihood = likelihood, 
                                           cov_function = "matern_space_time",
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
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
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
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
                                          cov_function = "matern_space_time", matrix_inversion_method = inv_method,
                                          gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none"), 
                      file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
      expect_lt(abs(nll-70.2364313),0.2)
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll2,y=y)
      expect_lt(abs(nll-70.6574683),0.2)
      # Fit model
      capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), likelihood = likelihood, cov_function = "matern_space_time",
                                             gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                             y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD), 
                      file='NUL')
      cov_pars_nn <- c(0.17325163, 0.07047597, 0.12485021)
      coef_nn <- c(0.1361984, 0.2177389, 0.2689333, 0.3004326)
      nrounds_nn <- 85
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),tolerance_loc)
      expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
      # Prediction
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=num_neighbors, nsim_var_pred=nsim_var_pred)
      pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                      X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
      expected_mu_nn <- c(0.1361984, 0.4161447, 0.6388071)
      expected_cov_nn <- c(1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.85149451, 0.01824841, 0.00000000, 0.01824841, 0.81056946)
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
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time",
                        likelihood = likelihood)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 70.85206038
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time",
                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD, likelihood = likelihood)
    cov_pars <- c(0.07478838338, 0.06647974215, 0.10436422267)
    coef <- c(0.1364351905, 0.2089784079, 0.2630968184, 0.2936127737)
    nrounds <- 148
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    ## With Vecchia approximation
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time",
                                        gp_approx = "vecchia", num_neighbors = n-6, vecchia_ordering = "none",
                                        likelihood = likelihood), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time",
                                           gp_approx = "vecchia", num_neighbors = n-6, vecchia_ordering = "none",
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD, likelihood = likelihood), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
  })
  
}

