context("GPModel_gaussian_process")

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  TOLERANCE_ITERATIVE <- 1E-1
  TOLERANCE_LOOSE <- 1E-2
  TOLERANCE_MEDIUM <- 1e-3
  TOLERANCE_STRICT <- 1E-5
  
  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov = "gradient_descent",
                               lr_cov = 0.1, use_nesterov_acc = TRUE,
                               acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                               optimizer_coef = "gradient_descent", lr_coef = 0.1,
                               convergence_criterion = "relative_change_in_log_likelihood",
                               cg_delta_conv = 1E-6, cg_preconditioner_type = "predictive_process_plus_diagonal",
                               cg_max_num_it = 1000, cg_max_num_it_tridiag = 1000,
                               num_rand_vec_trace = 1000, reuse_rand_vec_trace = TRUE)
  DEFAULT_OPTIM_PARAMS_STD <- c(DEFAULT_OPTIM_PARAMS, list(std_dev = TRUE))
  DEFAULT_OPTIM_PARAMS_FISHER <- list(optimizer_cov = "fisher_scoring", delta_rel_conv = 1E-6,
                                      optimizer_coef = "gradient_descent", lr_coef = 0.1,
                                      convergence_criterion = "relative_change_in_log_likelihood",
                                      cg_delta_conv = 1E-6, cg_preconditioner_type = "predictive_process_plus_diagonal",
                                      cg_max_num_it = 1000, cg_max_num_it_tridiag = 1000,
                                      num_rand_vec_trace = 1000, reuse_rand_vec_trace = TRUE,
                                      seed_rand_vec_trace = 1)
  DEFAULT_OPTIM_PARAMS_FISHER_STD <- c(DEFAULT_OPTIM_PARAMS_FISHER, list(std_dev = TRUE))
  OPTIM_PARAMS_BFGS <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 1000)
  OPTIM_PARAMS_BFGS_STD <- c(OPTIM_PARAMS_BFGS, list(std_dev = TRUE))
  
  
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
    params <- DEFAULT_OPTIM_PARAMS_STD
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
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 0.5 + 1E-6)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    expect_lt(abs(nll - nll_exp), TOLERANCE_STRICT)
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 0.5 - 1E-6)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    expect_lt(abs(nll - nll_exp), TOLERANCE_STRICT)
    # Matern 1.5
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    nll_exp_mat <- 141.3502172
    expect_lt(abs(nll - nll_exp_mat), TOLERANCE_STRICT)
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5 + 1E-6)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    expect_lt(abs(nll - nll_exp_mat), TOLERANCE_MEDIUM)
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5 - 1E-6)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    expect_lt(abs(nll - nll_exp_mat), TOLERANCE_MEDIUM)
    # Matern 2.5
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 2.5)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    nll_exp_mat <- 158.1111626
    expect_lt(abs(nll - nll_exp_mat), TOLERANCE_STRICT)
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 2.5 + 1E-6)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    expect_lt(abs(nll - nll_exp_mat), TOLERANCE_MEDIUM)
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 2.5 - 1E-6)
    nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars_eval_nll, y = y)
    expect_lt(abs(nll - nll_exp_mat), TOLERANCE_MEDIUM)
    
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    capture.output( fit(gp_model, y = y, params = params), 
                    file='NUL')
    cov_pars <- c(0.03784221, 0.07943467, 1.07390943, 0.25351519, 0.11451432, 0.03840236)
    num_it <- 59
    nll_opt <- 122.7771373
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Can switch between likelihoods
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    gp_model$set_likelihood("gamma")
    gp_model$set_likelihood("gaussian")
    capture.output( fit(gp_model, y = y, params = params), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    # Gradient descent without Nesterov acceleration
    params_no_acc <- params
    params_no_acc$use_nesterov_acc <- FALSE
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_no_acc) , file='NUL')
    cov_pars_other <- c(0.04040441, 0.08036674, 1.06926607, 0.25360131, 0.11502362, 0.03877014)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),5E-6)
    expect_equal(gp_model$get_num_optim_iter(), 97)
    # Using a too large learning rate
    params_lr <- params
    params_lr$lr_cov <- 1
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_lr) , file='NUL')
    cov_pars_other <- c(0.03738147, 0.07929704, 1.07520000, 0.25359186, 0.11441031, 0.03833048)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 49)
    # Different terminations criterion
    params_loc <- params
    params_loc$convergence_criterion = "relative_change_in_parameters"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc), file='NUL')
    cov_pars_other_crit <- c(0.03276547, 0.07715343, 1.07617676, 0.25177603, 0.11352557, 0.03770062)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other_crit)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    ll <- gp_model$neg_log_likelihood(y=y,cov_pars=gp_model$get_cov_pars()[1,])
    expect_lt(abs(ll-122.7752664),TOLERANCE_STRICT)
    # Fisher scoring
    params_loc <- params
    params_loc$optimizer_cov = "fisher_scoring"
    params_loc$lr_cov <- 1
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc), file='NUL')
    cov_pars_fisher <- c(0.03294841, 0.07722844, 1.07591929, 0.25179816, 0.11355958, 0.03772550)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_fisher)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_LOOSE)
    # lbfgs
    params_loc$optimizer_cov = "lbfgs"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc)
                    , file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars)),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_LOOSE)
    # nelder_mead
    params_loc$optimizer_cov = "nelder_mead"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc)
                    , file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars)),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_LOOSE)
    # Test default values for delta_rel_conv for nelder_mead
    capture.output( gp_model_default <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                   y = y, params = list(optimizer_cov = "nelder_mead"))
                    , file='NUL')
    capture.output( gp_model_8 <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             y = y, params = list(optimizer_cov = "nelder_mead",
                                                                  delta_rel_conv=1e-8))
                    , file='NUL')
    expect_false(isTRUE(all.equal(gp_model_default$get_cov_pars(), gp_model$get_cov_pars())))
    expect_true(isTRUE(all.equal(gp_model_default$get_cov_pars(), gp_model_8$get_cov_pars())))
    # Test default values for delta_rel_conv for gradient_descent
    capture.output( gp_model_default <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                   y = y, params = list(optimizer_cov = "gradient_descent"))
                    , file='NUL')
    capture.output( gp_model_8 <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             y = y, params = list(optimizer_cov = "gradient_descent",
                                                                  delta_rel_conv=1e-8))
                    , file='NUL')
    capture.output( gp_model_6 <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             y = y, params = list(optimizer_cov = "gradient_descent",
                                                                  delta_rel_conv=1e-6))
                    , file='NUL')
    expect_true(isTRUE(all.equal(gp_model_default$get_cov_pars(), gp_model_6$get_cov_pars())))
    expect_false(isTRUE(all.equal(gp_model_default$get_cov_pars(), gp_model_8$get_cov_pars())))
    # lbfgs
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = list(optimizer_cov = "lbfgs")), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_LOOSE)
    # Adam
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = list(optimizer_cov = "adam")), file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    # Newton's method
    params_loc <- params
    params_loc$optimizer_cov = "newton"
    params_loc$lr_cov <- 1
    params_loc$use_nesterov_acc <- FALSE
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params_loc), file='NUL')
    cov_pars_newton <- c(0.03282998, 0.07718279, 1.07612393, 0.25179124, 0.11353614, 0.03770875)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_newton)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 7)
    expect_lt(sum(abs(as.vector(gp_model$get_current_neg_log_likelihood())-nll_opt)),TOLERANCE_LOOSE)
    
    # Prediction from fitted model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y,
                                           params = list(optimizer_cov = "fisher_scoring",
                                                         delta_rel_conv = 1E-6, use_nesterov_acc = FALSE,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
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
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    expected_mu <- c(0.08704577, 1.63875604, 0.48513581)
    expected_cov <- c(1.189093e-01, 1.171632e-05, -4.172444e-07, 1.171632e-05,
                      7.427727e-02, 1.492859e-06, -4.172444e-07, 1.492859e-06, 8.107455e-02)
    cov_no_nugget <- expected_cov
    cov_no_nugget[c(1,5,9)] <- expected_cov[c(1,5,9)] - cov_pars_pred[1]
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Prediction of variances only
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, 
                    cov_pars = cov_pars_pred, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    # Predict latent process
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-cov_no_nugget)), TOLERANCE_STRICT)
    
    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    opt <- optim(par=c(0.1,2,0.2), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
    expect_lt(sum(abs(opt$par-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_lt(abs(opt$value-(122.7752694)),1E-5)
    expect_equal(as.integer(opt$counts[1]), 198)
    
    # Other covariance functions
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 0.5,
                                           y = y, params = params) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 0.5 + 1e-6,
                                           y = y, params = params) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Matern 1.5
    init_cov_pars_15 <- c(var(y)/2,var(y)/2,mean(dist(coords))/4.7*sqrt(3))
    params_15 = DEFAULT_OPTIM_PARAMS_STD
    params_15$init_cov_pars <- init_cov_pars_15
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 1.5, y = y, params = params_15) , file='NUL')
    cov_pars_other <- c(0.22926543, 0.08486055, 0.87886348, 0.24059253, 0.10726402, 0.02672378)
    num_it_other <- 16
    nll_opt_other <- 123.6388965
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_other)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 1.5 - 1E-6, y = y, params = params_15) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_other)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_STRICT)
    
    # Matern 2.5
    init_cov_pars_25 <- c(var(y)/2,var(y)/2,mean(dist(coords))/5.9*sqrt(5))
    params_25 = DEFAULT_OPTIM_PARAMS_STD
    params_25$init_cov_pars <- init_cov_pars_25
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 2.5, y = y, params = params_25) , file='NUL')
    cov_pars_other <- c(0.27251105, 0.08316755, 0.83205621, 0.23561744, 0.10536460, 0.02375078)
    num_it_other <- 13
    nll_opt_other <- 123.9752771
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_other)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 2.5 + 1E-3, y = y, params = params_25) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), num_it_other)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_MEDIUM)
    # gaussian
    init_cov_pars_G <- c(var(y)/2,var(y)/2,sqrt((mean(dist(coords))/2)^2 / 3))
    params_G = DEFAULT_OPTIM_PARAMS_STD
    params_G$init_cov_pars <- init_cov_pars_G
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "gaussian",
                                           cov_fct_shape = 2.5, y = y, params = params_G) , file='NUL')
    cov_pars_other <- c(0.33824439, 0.07955527, 0.75776861, 0.22661022, 0.14361521, 0.02589934)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    # Matern with shape estimated
    params = OPTIM_PARAMS_BFGS_STD
    params$init_cov_pars <- c(init_cov_pars_15, 1.5)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern_estimate_shape",
                                           cov_fct_shape = 1.5, y = y, params = params) , file='NUL')
    cov_pars_other <- c(0.0001323589, 0.2018696019, 1.1022114804, 0.3153382101, 0.1187387358, 0.0512925409, 0.4181996520, 0.3579762498)
    num_it_other <- 23
    nll_opt_other <- 122.7099697
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_other)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_STRICT)
    
    ## Test default initial values
    params <- list(optimizer_cov = "gradient_descent", maxit = 0, optimizer_coef = "gradient_descent")
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 0.5, y = y, params = params) , file='NUL')
    expect_lt(abs(gp_model$get_cov_pars()[1] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars()[2] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars()[3] - median(dist(coords))/3/2),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 1.5, y = y, params = params) , file='NUL')
    expect_lt(abs(gp_model$get_cov_pars()[1] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars()[2] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars()[3] - median(dist(coords))/4.7*sqrt(3)/2),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 2.5, y = y, params = params) , file='NUL')
    expect_lt(abs(gp_model$get_cov_pars()[1] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars()[2] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars()[3] - median(dist(coords))/5.9*sqrt(5)/2),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "gaussian",
                                           y = y, params = params) , file='NUL')
    expect_lt(abs(gp_model$get_cov_pars()[1] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars()[2] - var(y)/2),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars()[3] - sqrt((median(dist(coords))/2)^2 / 3)),TOLERANCE_STRICT)
    #non-Gaussian data
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", likelihood = "gamma",
                                           cov_fct_shape = 0.5, y = exp(y), params = params) , file='NUL')
    expect_lt(abs(gp_model$get_cov_pars()[1] - 1),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_cov_pars()[2] - median(dist(coords))/3/2),TOLERANCE_STRICT)
    
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
                                         delta_rel_conv = 1E-6, use_nesterov_acc = FALSE, std_dev = TRUE,
                                         convergence_criterion = "relative_change_in_parameters", init_cov_pars=init_cov_pars))
    cov_pars <- c(0.008461342, 0.069973492, 1.001562822, 0.214358560, 0.094656409, 0.029400407)
    coef <- c(2.30780026, 0.21365770, 1.89951426, 0.09484768)
    nll <- 121.482402
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_STRICT)
    expect_lt(sum(abs( as.vector(gp_model$get_coef())-coef)), TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    # Prediction 
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expected_mu <- c(1.196952, 4.063324, 3.156427)
    expected_cov <- c(6.305383e-01, 1.358861e-05, 8.317903e-08, 1.358861e-05,
                      3.469270e-01, 2.686334e-07, 8.317903e-08, 2.686334e-07, 4.255400e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    
    # Gradient descent
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y=y, X = X, params = params)
    cov_pars <- c(0.01621846, 0.99717680, 0.09616230)
    coef <- c(2.305529, 1.899208)
    nll <- 121.4886075
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)), TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 100)
    
    # Nelder-Mead
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X = X, params = list(optimizer_cov = "nelder_mead",
                                                       optimizer_coef = "nelder_mead",
                                                       maxit=1000, delta_rel_conv = 1e-12, init_cov_pars=init_cov_pars))
    cov_pars <- c(0.008459373, 1.001564796, 0.094655964)
    coef <- c(2.307798, 1.899516)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)), TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 429)
    # lbfgs
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X = X, params = list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit=1000, init_cov_pars=init_cov_pars))
    cov_pars <- c(0.008993586382, 1.000518636089, 0.094683724304)
    coef <- c(2.309738418, 1.899886232)
    nll <- 121.4824924
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 15)
    # lbfgs wit wls for coefficients
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X = X, params = list(optimizer_cov = "lbfgs", maxit=1000, optimizer_coef ="wls", init_cov_pars=init_cov_pars))
    coef <- c(2.307912121, 1.899505576)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    
  })
  
  test_that("Gaussian process and two random coefficients ", {
    
    y <- eps_svc + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3,var(y)/2,mean(dist(coords))/3,var(y)/2,mean(dist(coords))/3)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_rand_coef_data = Z_SVC, y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, maxit=10, trace=TRUE, init_cov_pars=init_cov_pars)), file='NUL')
    expected_values <- c(0.25740068, 0.22608704, 0.83503539, 0.41896403, 0.15039055,
                         0.10090869, 1.61010233, 0.84207763, 0.09015444, 0.07106099, 
                         0.25064640, 0.62279880, 0.08720822, 0.32047865)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Predict training data random effects
    cov_pars <- gp_model$get_cov_pars()[1,]
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
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc= FALSE, maxit=5, init_cov_pars=init_cov_pars)), file='NUL')
    expected_values <- c(9.948069e-06, 2.133237e-01, 1.398126e+00, 5.103201e-01, 1.535385e-01, 7.508804e-02, 1.758062e+00, 7.926720e-01, 3.919317e-02, 
                         3.867593e-02, 3.140238e-01, 6.211919e-01, 2.657551e+00, 1.713120e+01)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-149.4422184),1E-5)
  })
  
  test_that("Gaussian process model with cluster_id's not constant ", {
    
    y <- eps + xi
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, cluster_ids = cluster_ids,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    cov_pars <- c(0.05414149, 0.08722111, 1.05789166, 0.22886740, 0.12702368, 0.04076914)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 247)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, cluster_ids = cluster_ids,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    cov_pars <- c(0.05414149, 0.08722111, 1.05789166, 0.22886740, 0.12702368, 0.04076914)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
    expect_equal(gp_model$get_num_optim_iter(), 20)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        cluster_ids = cluster_ids)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE)
    expected_mu <- c(-0.01437506, 0.00000000, 0.93112902)
    expected_cov <- c(0.743055189, 0.000000000, -0.000140644, 0.000000000,
                      1.100000000, 0.000000000, -0.000140644, 0.000000000, 0.565243468)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)), TOLERANCE_STRICT)
    # Predict variances
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(0.1,1,0.15), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
  })
  
  test_that("Gaussian process model with multiple observations at the same location ", {
    
    y <- eps_multiple + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(unique(coords_multiple)))/3)
    params = DEFAULT_OPTIM_PARAMS_STD
    params$init_cov_pars <- init_cov_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", y = y,
                                           params = params), file='NUL')
    cov_pars <- c(0.037168482, 0.006069406, 1.168105814, 0.445122816, 0.196226850, 0.105105379)
    num_it <- 6
    nll <- 33.43686607
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    # With full_scale_tapering
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", y = y,
                                           params = params, gp_approx = "full_scale_tapering", num_ind_points  = 25), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5)]-cov_pars[c(1,3,5)])), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", y = y,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters",
                                                         init_cov_pars=init_cov_pars)), file='NUL')
    cov_pars <- c(0.037136462, 0.006064181, 1.153630335, 0.435788570, 0.192080613, 0.102631006)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
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
  })
  
  test_that("Vecchia approximation for Gaussian process model ", {
    
    y <- eps + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params_vecchia <- list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                           lr_cov = 0.1, use_nesterov_acc = TRUE,
                           acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                           convergence_criterion = "relative_change_in_parameters",
                           num_rand_vec_trace = 1000, reuse_rand_vec_trace = TRUE,
                           seed_rand_vec_trace = 1, init_cov_pars=init_cov_pars)
    
    # Evaluate negative log-likelihood
    cov_pars_ll <- c(0.1,1.6,0.2)
    exp_nll <- 124.2549533
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll,y=y)
    expect_lt(abs(nll-exp_nll), TOLERANCE_STRICT)
    # "vecchia_latent"
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1])
    expect_lt(abs(nll-exp_nll), TOLERANCE_STRICT)
    # "vecchia_latent" with iterative methods (pivoted Cholesky preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(num_rand_vec_trace = 1000))
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1])
    expect_lt(abs(nll-exp_nll), 0.2)
    # "vecchia_latent" with iterative methods (FITC preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(num_rand_vec_trace = 1000, cg_preconditioner_type = "predictive_process_plus_diagonal"))
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1]), file='NUL')
    expect_lt(abs(nll-exp_nll), 0.2)
    
    # Same thing without Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "none"), file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll,y=y)
    expect_lt(abs(nll-exp_nll), TOLERANCE_STRICT)
    # less neighbhors
    exp_nll_less_nn <- 124.2252524
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30,
                                        vecchia_ordering = "none"), file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll,y=y)
    expect_lt(abs(nll-exp_nll_less_nn), TOLERANCE_STRICT)
    # "vecchia_latent"
    exp_nll_less_nn_lat <- 124.2549533
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1])
    expect_lt(abs(nll-exp_nll_less_nn_lat), TOLERANCE_STRICT)
    # "vecchia_latent" with iterative methods (pivoted Cholesky)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(num_rand_vec_trace = 1000))
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1])
    expect_lt(abs(nll-exp_nll_less_nn_lat), 0.2)
    
    # "vecchia_latent" with iterative methods (FITC preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(num_rand_vec_trace = 1000, cg_preconditioner_type = "predictive_process_plus_diagonal"))
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_ll[-1],y=y,aux_pars=cov_pars_ll[1]), file='NUL')
    expect_lt(abs(nll-exp_nll_less_nn_lat), 0.2)
    
    # Estimation and maximal number of neighbors
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia), file='NUL')
    cov_pars <- c(0.03276547, 0.07544593, 1.07617676, 0.24743617, 0.11352557, 0.03482885)
    nll_est <- 122.7752664
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_STRICT)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), TOLERANCE_STRICT)
    # With "vecchia_latent"
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    params_latent <- params_vecchia
    params_latent$std_dev = FALSE
    params_latent$init_cov_pars <- NULL
    params_latent$optimizer_cov = "lbfgs"
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(3,5)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), TOLERANCE_MEDIUM)
    # "vecchia_latent" with iterative methods (pivoted Cholesky)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(3,5)])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), 0.2)
    
    # "vecchia_latent" with iterative methods (FITC preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    params_latent$cg_preconditioner_type = "predictive_process_plus_diagonal"
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(3,5)])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), 0.2)
    
    # Same thing without Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5)]-cov_pars[c(1,3,5)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), TOLERANCE_LOOSE)
    
    # Random ordering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering="random", y = y,
                                           params = params_vecchia), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5)]-cov_pars[c(1,3,5)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
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
                                        vecchia_ordering = "none"), file='NUL')
    gp_model$set_optim_params(params=list(init_aux_pars = cov_pars[1]))
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_STRICT)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_no_nugget[c(1,5,9)])), TOLERANCE_STRICT)
    # vecchia_latent and iterative methods (pivoted Cholesky)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(init_aux_pars = cov_pars[1]))
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_LOOSE)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_no_nugget[c(1,5,9)])), TOLERANCE_LOOSE)
    
    # vecchia_latent and iterative methods (FITC preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = n-1,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params=list(init_aux_pars = cov_pars[1], cg_preconditioner_type = "predictive_process_plus_diagonal"))
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    capture.output( pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = TRUE), file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])), TOLERANCE_LOOSE)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars[-1], predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_no_nugget[c(1,5,9)])), TOLERANCE_LOOSE)
    
    # Vecchia approximation with 30 neighbors
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30,
                                        vecchia_ordering = "none"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia) , file='NUL')
    cov_pars_vecchia <- c(0.03297349, 0.07545639, 1.07691542, 0.24785457, 0.11378505, 0.03493878)
    nll_vecchia <- 122.7680889
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_vecchia)), TOLERANCE_STRICT)
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
    
    # With "vecchia_latent"
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = 30,
                                        vecchia_ordering = "none"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_vecchia[c(3,5)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars_vecchia[1])),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_vecchia), TOLERANCE_MEDIUM)
    # "vecchia_latent" with iterative methods (pivoted Cholesky)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = 30,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_vecchia[c(3,5)])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars_vecchia[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_vecchia), 0.2)
    
    # "vecchia_latent" with iterative methods (FITC preconditioner)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia_latent", num_neighbors = 30,
                                        vecchia_ordering = "none", matrix_inversion_method = "iterative"), file='NUL')
    params_latent$cg_preconditioner_type = "predictive_process_plus_diagonal"
    capture.output( fit(gp_model, y = y, params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_vecchia[c(3,5)])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars_vecchia[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_vecchia), 0.2)
    
    # Vecchia approximation with 30 neighbors and random ordering
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30, 
                                        vecchia_ordering="random"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_vecchia)),0.05)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_vecchia)), 0.02)
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
    params <- DEFAULT_OPTIM_PARAMS_FISHER_STD
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params$init_cov_pars <- init_cov_pars
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, X = X,
                                           params = params), file='NUL')
    cov_pars <- c(0.003310954, 0.066230954, 1.005761204, 0.209944716, 0.093313847, 0.026835292)
    coef <- c(2.3058764, 0.2119560, 1.8996884, 0.0944677)
    nll_est <- 121.4854824
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)), TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_est), TOLERANCE_LOOSE)

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
    params_latent$std_dev = FALSE
    params_latent$init_cov_pars <- NULL
    params_latent$optimizer_cov <- "lbfgs"
    params_latent$optimizer_coef <- "lbfgs"
    params_latent$cg_preconditioner_type <- NULL
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, X = X,
                                           params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(3,5)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), TOLERANCE_LOOSE)
    # "vecchia_latent" and iterative methods (pivoted Cholesky)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, X = X,
                                           params = params_latent, matrix_inversion_method = "iterative"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(3,5)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), 0.2)
    
    # "vecchia_latent" and iterative methods (FITC preconditioner)
    params_latent_FITC = params_latent
    params_latent_FITC$cg_preconditioner_type = "predictive_process_plus_diagonal"
    params_latent_FITC$piv_chol_rank = 70
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, X = X,seed = 1,
                                           params = params_latent_FITC, matrix_inversion_method = "iterative"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(3,5)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est), 0.2)
  })
  
  test_that("Vecchia approximation for Gaussian process model with cluster_id's not constant ", {
    
    y <- eps + xi
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           vecchia_ordering = "none", y = y, cluster_ids = cluster_ids,
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars <- c(0.05870373, 0.08817497, 1.05572659, 0.22911532, 0.12775754, 0.03905891)
    nll <- 129.3761486
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           vecchia_ordering = "none", y = y, cluster_ids = cluster_ids,
                                           params = DEFAULT_OPTIM_PARAMS_FISHER_STD), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), 0.1)
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
    params = DEFAULT_OPTIM_PARAMS_STD
    params$init_cov_pars <- init_cov_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1, y = y,
                                           vecchia_ordering = "none", params = params), file='NUL')
    cov_pars <- c(0.037167165666, 0.006064865481, 1.165197180621, 0.435972318447, 0.196301820444, 0.100993102176)
    nll <- 33.43685834
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    # "vecchia_latent"
    params_latent <- params
    params_latent$std_dev = FALSE
    params_latent$init_cov_pars <- NULL
    params_latent$optimizer_cov <- "lbfgs"
    params_latent$optimizer_coef <- "lbfgs"
    params_latent$cg_preconditioner_type <- NULL
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y,
                                           params = params_latent), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(3,5)])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll), TOLERANCE_LOOSE)
    # "vecchia_latent" and matrix_inversion_method = "iterative" (pivoted Cholesky)
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y,
                                           params = params_latent, matrix_inversion_method = "iterative"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(3,5)])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll), 0.2)
    
    # "vecchia_latent" and matrix_inversion_method = "iterative" (FITC preconditioner)
    params_latent_FITC = params_latent
    params_latent_FITC$cg_preconditioner_type = "predictive_process_plus_diagonal"
    params_latent_FITC$piv_chol_rank = 25
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "vecchia_latent", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, seed = 1,
                                           params = params_latent_FITC, matrix_inversion_method = "iterative"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(3,5)])),0.02)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars[1])),0.02)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll), 0.2)
    
    # Fisher scoring
    params_loc = DEFAULT_OPTIM_PARAMS_FISHER_STD
    params_loc$init_cov_pars <- init_cov_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1, y = y, 
                                           vecchia_ordering = "none", params = params_loc), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), 0.1)
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
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, maxit=10, init_cov_pars=init_cov_pars)), file='NUL')
    expected_values <- c(0.25740068213, 0.21395398553, 0.83503538559, 0.32160635543, 0.15039055133, 0.07486033339, 1.61010233081,
                         0.64221278485, 0.09015443875, 0.04966428794, 0.25064639566, 0.46210156876, 0.08720821575, 0.22278416599)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)), TOLERANCE_LOOSE)
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
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = FALSE, maxit=10, init_cov_pars=init_cov_pars)), file='NUL')
    expected_values <- c(0.34489931519, 0.22107902729, 0.79813421101, 0.33185791805, 0.15144409082, 0.08062499175, 1.14797483590, 
                         0.59294272114, 0.10321260903, 0.07092979340, 0.32243986621, 0.48546238572, 0.10613523300, 0.20756237999)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)), TOLERANCE_STRICT)
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
  
  test_that("Wendland covariance function for Gaussian process model ", {
    
    y <- eps + xi
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "wendland", 
                                        cov_fct_taper_shape = 0, cov_fct_taper_range = 0.1), file='NUL')
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5)) , file='NUL')
    cov_pars <- c(0.002911765, 0.116338096, 0.993996193, 0.211276385)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_STRICT)
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
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc = FALSE,
                                                         delta_rel_conv = 1E-6)), file='NUL')
    cov_pars <- c(4.941224e-09, 1.497464e-01, 1.302468e+00, 2.746096e-01)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.02,1.2),y=y)
    expect_lt(abs(nll-136.9508962), TOLERANCE_STRICT)
    
    # Other taper shapes
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 1, cov_fct_taper_range = 0.15, y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5)), file='NUL')
    cov_pars <- c(0.0564441, 0.0497191, 0.9921285, 0.1752661)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_STRICT)
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
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5)), file='NUL')
    cov_pars <- c(0.00327103, 0.06579671, 1.08812978, 0.18151366)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)), TOLERANCE_STRICT)
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
    params = DEFAULT_OPTIM_PARAMS_STD
    params$init_cov_pars <- init_cov_pars
    
    # No tapering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, X = X, params = params), file='NUL')
    cov_pars <- c(0.01621846, 0.07384498, 0.99717680, 0.21704099, 0.09616230, 0.03034715)
    coef <- c(2.30554610, 0.21565230, 1.89920767, 0.09567547)
    num_it <- 100
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_MEDIUM)
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
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "tapering", cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X = X, 
                                           params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_MEDIUM)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_tap)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_tap)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 75)
    
    # Same thing with Matern covariance
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/4.7)
    params = DEFAULT_OPTIM_PARAMS_STD
    params$init_cov_pars <- init_cov_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           y = y, X = X, params = params), file='NUL')
    cov_pars <- c(0.17383685, 0.07956155, 0.84111654, 0.20895243, 0.08839064, 0.02062892)
    coef <- c(2.34174699, 0.19483212, 1.88055706, 0.09788995)
    num_it <- 21
    nll_opt <- 121.8046544
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_MEDIUM)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5)]-cov_pars[c(1,3,5)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # General shape
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5 + 1E-4,
                                           gp_approx = "tapering", cov_fct_taper_shape = 1, cov_fct_taper_range = 1e6,
                                           y = y, X = X,
                                           params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5)]-cov_pars[c(1,3,5)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_MEDIUM)
    
    # With tapering and smaller tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           gp_approx = "tapering", cov_fct_taper_shape = 1, cov_fct_taper_range = 0.5,
                                           y = y, X = X,
                                           params = params), file='NUL')
    cov_pars_tap <- c(0.18970609, 0.07263436, 0.80493104, 0.20220891, 0.11212289, 0.02562848)
    coef_tap <- c(2.35889350, 0.17954660, 1.87422223, 0.09831309)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_tap)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_tap)),TOLERANCE_STRICT)
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
    params = DEFAULT_OPTIM_PARAMS_STD
    params$init_cov_pars <- init_cov_pars
    init_cov_pars_mult <- c(var(y)/2,var(y)/2,mean(dist(unique(coords_multiple)))/3)
    params_mult <- DEFAULT_OPTIM_PARAMS_STD
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars()) - as.vector(gp_model_no_approx$get_cov_pars()))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef()) - as.vector(gp_model_no_approx$get_coef()))),TOLERANCE_STRICT)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars()) - as.vector(gp_model_mult_no_approx$get_cov_pars()))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef()) - as.vector(gp_model_mult_no_approx$get_coef()))),TOLERANCE_STRICT)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars()[1,]) - as.vector(gp_model_clus_no_approx$get_cov_pars()[1,]))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars()[2,]) - as.vector(gp_model_clus_no_approx$get_cov_pars()[2,]))),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef()) - as.vector(gp_model_clus_no_approx$get_coef()))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), gp_model_clus_no_approx$get_num_optim_iter())
    expect_lt(abs(gp_model_clus_no_approx$get_current_neg_log_likelihood() - nll_cluster_exp),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, 
                    X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred, 
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_clus_no_approx$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_clus_no_approx$var))),TOLERANCE_STRICT)
    ## TODO: Prediction of new clusters crashes
    # pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, 
    #                 X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred_new, 
    #                 predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    # expect_lt(sum(abs(pred$mu - pred_clus_no_approx_new$mu)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_clus_no_approx_new$var))),TOLERANCE_STRICT)
    
    # Fisher scoring
    params_FS = DEFAULT_OPTIM_PARAMS_FISHER_STD
    params_FS$init_cov_pars <- init_cov_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "fitc", num_ind_points = n-1, 
                                           y = y, X = X, params = params_FS), file='NUL')
    cov_pars_FS <- c(0.008606874, 0.067462675, 1.001903559, 0.208839567, 0.094773935, 0.028174515)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_FS)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef()) - as.vector(gp_model_no_approx$get_coef()))),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),TOLERANCE_LOOSE)
    
    # With fitc and less inducing points
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "fitc", num_ind_points = 50, 
                                           y = y, X = X, params = params), file='NUL')
    cov_pars_tap <- c(0.01030298, 0.07942118, 0.99809618, 0.22406519, 0.10787353, 0.03374618)
    coef_tap <- c(2.29553776, 0.22988084, 1.89903213, 0.09726784)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_tap)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_tap)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars()) - as.vector(gp_model_no_approx$get_cov_pars()))),0.05)
    expect_lt(sum(abs(as.vector(gp_model$get_coef()) - as.vector(gp_model_no_approx$get_coef()))),0.05)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars()) - as.vector(gp_model_mult_no_approx$get_cov_pars()))),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef()) - as.vector(gp_model_mult_no_approx$get_coef()))),0.05)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_mult_exp),0.1)
    
    # Same thing with Matern covariance
    init_cov_pars_15 <- c(var(y)/2,var(y)/2,mean(dist(coords))/4.7*sqrt(3))
    params_15 = DEFAULT_OPTIM_PARAMS_STD
    params_15$init_cov_pars <- init_cov_pars_15
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                           y = y, X = X, params = params)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           y = y, X = X, params = params_15), file='NUL')
    cov_pars <- c(0.17401588, 0.07960002, 0.84106347, 0.20899707, 0.08841966, 0.02064155)
    coef <- c(2.33980860, 0.19481950, 1.88058081, 0.09786326)
    num_it <- 19
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
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
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_ip)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_ip)),TOLERANCE_LOOSE)
    
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_tap)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_tap)),TOLERANCE_LOOSE)
    
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_tap)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_tap)),TOLERANCE_LOOSE)
    
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
    params = DEFAULT_OPTIM_PARAMS_STD
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
      
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars()) - as.vector(gp_model_no_approx$get_cov_pars()))),TOLERANCE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef()) - as.vector(gp_model_no_approx$get_coef()))),TOLERANCE)
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
      params_FS <- DEFAULT_OPTIM_PARAMS_FISHER_STD
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
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_FS)),2*TOLERANCE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_FS)),0.1)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),3*TOLERANCE)
      
      if(i == "cholesky"){
        # With FSA and n-1 inducing points and taper range 0.4
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               gp_approx = "full_scale_tapering",num_ind_points = n-1, 
                                               cov_fct_taper_shape = 2, cov_fct_taper_range = 0.4,
                                               y = y, X = X,matrix_inversion_method = i, 
                                               params = params), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars()) - as.vector(gp_model_no_approx$get_cov_pars()))),TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef()) - as.vector(gp_model_no_approx$get_coef()))),TOLERANCE)
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
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
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
        params_15 = DEFAULT_OPTIM_PARAMS_STD
        params_15$init_cov_pars <- init_cov_pars_15
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                               y = y, X = X, params = params_15), file='NUL')
        cov_pars <- c(0.17369771, 0.07950745, 0.84098718, 0.20889907, 0.08839526, 0.01190858)
        coef <- c(2.33980860, 0.19481950, 1.88058081, 0.09786326)
        num_it <- 19
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
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
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
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
        
        # With FSA and n-1 inducing points and taper range 0.5
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                               gp_approx = "full_scale_tapering",num_ind_points = n-1, cov_fct_taper_shape = 2, cov_fct_taper_range = 0.5,
                                               y = y, X = X,
                                               params = params_15), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
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
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
        # Prediction 
        if(i == "iterative"){
          gp_model$set_prediction_data(cg_delta_conv_pred = 1e-6, nsim_var_pred = 500)
        }
        pred <- predict(gp_model, gp_coords_pred = coord_test,
                        X_pred = X_test, predict_var = TRUE)
        expected_mu <- c(1.250332, 4.049631, 3.160899) 
        expected_var <- c(0.5981874, 0.3632729, 0.3848723)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE)
      }# end (i == "cholesky")
    }# end loop over i (matrix_inversion_method)
  })
  
  test_that("Saving a GPModel and loading from file works ", {
    
    y <- eps + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    # Load from file and make predictions again
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    
    # With Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 20,
                                           vecchia_ordering = "none", y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
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
                                           vecchia_ordering = "random", y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
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
                                           y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
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
    params_ST = DEFAULT_OPTIM_PARAMS_STD
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
    cov_pars <- c(0.01316765, 0.28736684, 1.00918678, 0.33462814, 1.37748568, 
                  0.78252561, 0.11561567, 0.05410341)
    coef <- c(1.9583409, 0.1484610, 2.1707779, 0.1397487)
    nrounds <- 341
    nll_opt <- 138.2152667
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu <- c(1.958341, 1.939895, 2.566657)
    expected_cov <- c(2.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 
                      1.5919472302, 0.0001229643, 0.0000000000, 0.0001229643, 1.5650143857)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    
    ## Other shape parameters
    init_cov_pars_ST_15 <- c(var(y)/2,var(y)/2,mean(dist(time))/4.7*sqrt(3),mean(dist(coords))/4.7*sqrt(3))
    params_ST_15 = DEFAULT_OPTIM_PARAMS_STD
    params_ST_15$init_cov_pars <- init_cov_pars_ST_15
    init_cov_pars_ST_25 <- c(var(y)/2,var(y)/2,mean(dist(time))/5.9*sqrt(5),mean(dist(coords))/5.9*sqrt(5))
    params_ST_25 = DEFAULT_OPTIM_PARAMS_STD
    params_ST_25$init_cov_pars <- init_cov_pars_ST_25
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 1.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_1_5_exp <- 288.6072086
    expect_lt(abs(nll-nll_1_5_exp),TOLERANCE_STRICT)
    # General shape
    gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 1.5 + 1E-5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_1_5_exp),TOLERANCE_MEDIUM)
    # Fit model
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", 
                           cov_fct_shape = 1.5, y = y, X = X, params = params_ST_15)
    cov_pars_1_5 <- c(0.6848963042, 0.1933549581, 0.3277401095, 0.2084189663, 5.0137397237, 3.9479801602, 0.2044812593, 0.1278899286)
    coef_1_5 <- c(1.9622516576, 0.1869044588, 2.2128306066, 0.1411218434)
    nrounds_1_5 <- 30
    nll_opt_1_5 <- 138.6352032
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_1_5)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_1_5)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_1_5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_1_5), TOLERANCE_STRICT)
    # General shape
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                           cov_fct_shape = 1.5 + 1E-4, y = y, X = X, params = params_ST_15)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_1_5)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_1_5)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_1_5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_1_5), TOLERANCE_MEDIUM)
    # Shape = 2.5: evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 2.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_2_5_exp <- 296.7149408
    expect_lt(abs(nll-nll_2_5_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", 
                           cov_fct_shape = 2.5, y = y, X = X, params = params_ST_25)
    cov_pars_2_5 <- c(0.7257248556, 0.1684675325, 0.2886941955, 0.1806777760, 5.5493183649, 4.2083079994, 0.2209773518, 0.1285873155)
    coef_2_5 <- c(1.9636912837, 0.1907138557, 2.2156509961, 0.1409511004)
    nrounds_2_5 <- 22
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_2_5)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_2_5)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_2_5)
    
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)]-cov_pars[c(1,3,5,7)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)+1]-cov_pars[c(1,3,5,7)+1])),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
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
    cov_pars_nn <- c(0.01328420, 0.28788276, 1.00911528, 0.33509917, 1.38403453, 0.78663837, 0.11543238, 0.05402744)
    coef_nn <- c(1.9581608, 0.1485425, 2.1709711, 0.1397423)
    nrounds_nn <- 339
    nll_opt_nn <- 138.2135269
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)]-cov_pars_nn[c(1,3,5,7)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)+1]-cov_pars_nn[c(1,3,5,7)+1])),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
    # Fit model with lbfgs
    params_loc <- params_ST
    params_loc$optimizer_cov <- "lbfgs"
    params_loc$optimizer_coef <- "lbfgs"
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X = X, params = params_loc), 
                    file='NUL')
    cov_pars_nn <- c(2.749467691e-05, 2.694698768e-01, 1.017810359e+00, 3.194002659e-01, 1.348567120e+00, 7.322411707e-01, 1.155315662e-01, 5.170540454e-02)
    coef_nn <- c(1.958063246, 0.147834061, 2.169119150, 0.139267337)
    nll_opt_nn <- 138.186315
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)]-cov_pars_nn[c(1,3,5,7)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
    params_loc$optimizer_coef <- "wls"
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X = X, params = params_loc), 
                    file='NUL')
    cov_pars_nn <- c(6.271540365e-05, 1.017780689e+00, 1.348867128e+00, 1.155285078e-01)
    coef_nn <- c(1.9580436607, 0.1478391986, 2.1693944861, 0.1392675585)
    nll_opt_nn <- 138.1863872
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)]-cov_pars_nn)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
    # Different ordering
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "time",
                                           y = y, X = X, params = params_ST), 
                    file='NUL')
    cov_pars_nn <- c(0.01312706166, 1.00922409471, 1.37624596395, 0.11566309664)
    coef_nn <- c(1.9583457, 0.1484580, 2.1707320, 0.1397486)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)]-cov_pars_nn)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_LOOSE)
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
    params_mult_ST <- DEFAULT_OPTIM_PARAMS_STD
    params_mult_ST$init_cov_pars <- init_cov_pars_mult_ST
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 276.47191976324
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                           y = y, X = X, params = params_mult_ST)
    cov_pars <- c(0.48244729, 0.20133860, 0.53677606, 0.24652176, 3.84944066, 2.82763607, 0.21590375, 0.13357978)
    coef <- c(1.95425156, 0.21356119, 2.19640126, 0.13803044)
    nrounds <- 41
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)]-cov_pars[c(1,3,5,7)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)+1]-cov_pars[c(1,3,5,7)+1])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)]-cov_pars[c(1,3,5,7)])),TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7)+1]-cov_pars[c(1,3,5,7)+1])),2*TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
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
    params_ARD <- DEFAULT_OPTIM_PARAMS_STD
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
    cov_pars <- c(0.001816805721, 0.077142671804, 1.252297430929, 0.425967774570, 0.351443931951, 0.174922830926, 0.557170908900, 0.288167856581, 0.330248050235, 0.164425600271)
    coef <- c(2.26972331453, 0.45501400197, 1.72180055812, 0.08468301947)
    nrounds <- 285
    nll_opt <- 111.2311351
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu <- c(2.269723315, 3.070293327, 3.328852164)
    expected_cov <- c(2.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.486442677357, -0.001271254606, 0.000000000000, -0.001271254606, 1.407072168927)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    
    ## Other shape parameters
    init_cov_pars_ARD <- c(var(y)/2,var(y)/2)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars_ARD <- c(init_cov_pars_ARD, mean(dist(coords_ARD[,i])/4.7*sqrt(3)))
    params_ARD_15 <- DEFAULT_OPTIM_PARAMS_STD
    params_ARD_15$init_cov_pars <- init_cov_pars_ARD
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 1.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_1_5_exp <- 276.2341252
    expect_lt(abs(nll-nll_1_5_exp),TOLERANCE_STRICT)
    gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 1.5 + 1E-5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_1_5_exp),TOLERANCE_MEDIUM)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", 
                           cov_fct_shape = 1.5, y = y, X = X, params = params_ARD_15)
    cov_pars_1_5 <- c(0.05233868105, 0.04128232576, 1.13744973880, 0.30125631273, 0.23835608472, 0.05953168824, 0.31939322949, 0.08155727437, 0.20026304712, 0.04967975332)
    coef_1_5 <- c( 2.29478935491, 0.31293284267, 1.73132555841, 0.07420053431)
    nrounds_1_5 <- 12
    nll_opt_1_5 <- 107.8313403
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_1_5)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_1_5)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_1_5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_1_5), TOLERANCE_STRICT)
    # General shape
    gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", 
                           cov_fct_shape = 1.5 - 1E-4, y = y, X = X, params = params_ARD_15)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_1_5)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_1_5)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_1_5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_1_5), TOLERANCE_MEDIUM)
    # Gaussian covariance: evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "gaussian_ard")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_gaussian_exp <- 322.3104221
    expect_lt(abs(nll-nll_gaussian_exp),TOLERANCE_STRICT)
    # Fit model
    init_cov_pars_gauss <- c(var(y)/2,var(y)/2)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars_gauss <- c(init_cov_pars_gauss, sqrt((mean(dist(coords_ARD[,i])))^2/3))
    params_loc <- DEFAULT_OPTIM_PARAMS_STD
    params_loc$init_cov_pars <- init_cov_pars_gauss
    gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "gaussian_ard",
                           y = y, X = X, params = params_loc)
    cov_pars_gaussian <- c(0.06710171016, 0.03015477695, 1.04891057462, 0.23438713906, 0.23914312280, 0.03870310755, 0.30755189174, 0.03854926585, 0.21800215962, 0.03952282820)
    coef_gaussian <- c(2.33227066097, 0.22150895229, 1.74680670719, 0.06867352121)
    nrounds_gaussian <- 22
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_gaussian)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_gaussian)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_gaussian)
    
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
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
    cov_pars_nn <- c(0.001973118585, 0.072564950723, 1.245276359068, 0.366067407680, 0.347288304828, 0.132750895168, 0.555816885789, 0.212960451005, 0.326004253016, 0.121561125822)
    coef_nn <- c(2.26556701188, 0.45100256735, 1.72217739431, 0.084886089323)
    nrounds_nn <- 257
    nll_opt_nn <- 111.3062494
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars_nn[c(1,3,5,7,9)+1])),0.1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_ITERATIVE)
    # Fit model with lbfgs
    params_loc <- params_ARD
    params_loc$optimizer_cov <- "lbfgs"
    params_loc$optimizer_coef <- "lbfgs"
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X = X, params = params_loc), 
                    file='NUL')
    cov_pars_nn <- c(7.149460001e-06, 7.173166542e-02, 1.247842820e+00, 3.663302806e-01, 3.490157798e-01, 1.329923436e-01, 5.538784275e-01, 2.115090534e-01, 3.271811573e-01, 1.216508143e-01)
    coef_nn <- c(2.26487752788, 0.45198926378, 1.72216287214, 0.08451697745)
    nll_opt_nn <- 111.2617595
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_ITERATIVE)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=num_neighbors)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu_nn <- c(2.266689375, 3.068640353, 3.326451545)
    expected_cov_nn <- c(2.000000000, 0.000000000, 0.000000000, 0.000000000, 1.486444967, 0.000000000, 0.000000000, 0.000000000, 1.407115273)
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
    cov_pars_nn <- c(1.426301818e-05, 7.188102934e-02, 1.242135603e+00, 3.639657087e-01, 3.450491269e-01, 1.315589237e-01, 5.513548895e-01, 2.107407405e-01, 3.264769976e-01, 1.214618349e-01)
    coef_nn <- c(2.26712698762, 0.44926280067, 1.72219843751, 0.08456816674)
    nll_opt_nn <- 111.2700643
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_ITERATIVE)
    
    # Fit model with lbfgs & only intercept
    params_loc <- params_ARD
    params_loc$optimizer_cov <- "lbfgs"
    params_loc$optimizer_coef <- "lbfgs"
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X = rep(1,n), params = params_loc), 
                    file='NUL')
    cov_pars_nn <- c(1.1631054674, 0.6296342975, 1.4630675526, 0.7420499027, 0.1544979531, 0.1121469466, 0.4315307216, 0.3525193357, 0.1434031479, 0.1022578368)
    coef_nn <- c(2.4868650232, 0.3316730676)
    nll_opt_nn <- 183.8189996
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    params_loc$optimizer_coef <- "wls"
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X = rep(1,n), params = params_loc), 
                    file='NUL')
    cov_pars_nn <- c(1.1620507589, 0.6306093239, 1.4632829121, 0.7427317436, 0.1543495050, 0.1120748022, 0.4303008330, 0.3515338350, 0.1428942899, 0.1019068110)
    coef_nn <- c(2.4866827321, 0.3311283532)
    nll_opt_nn <- 183.8192443
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.01)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_MEDIUM)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    
    ## Less inducing points 
    # Evaluate negative log-likelihood
    num_ind_points <- 50
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "fitc", num_ind_points = num_ind_points, ind_points_selection = "kmeans++"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-250.6295576),TOLERANCE_MEDIUM)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = num_ind_points, ind_points_selection = "kmeans++",
                                           y = y, X = X, params = params_ARD), 
                    file='NUL')
    cov_pars_nn <- c(0.001184793866, 0.083612432493, 1.256681616590, 0.434080128994, 0.298454496109, 0.148129998459, 0.644575205672, 0.346228121873, 0.400022510570, 0.207684924596)
    coef_nn <- c(2.29283911727, 0.46402346934, 1.72424805719, 0.08716921877)
    nll_opt_nn <- 112.7148326
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars_nn[c(1,3,5,7,9)+1])),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), 2)
    # Fit model with lbfgs
    params_loc <- params_ARD
    params_loc$optimizer_cov <- "lbfgs"
    params_loc$optimizer_coef <- "lbfgs"
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = num_ind_points, ind_points_selection = "kmeans++",
                                           y = y, X = X, params = params_loc), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars_nn[c(1,3,5,7,9)])),0.05)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),0.05)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), 0.05)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), 2)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu_nn <- c(2.295399576, 2.579995176, 3.399045713)
    expected_cov_nn <- c(2.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.875031106805, -0.002854441728, 0.000000000000, -0.002854441728, 1.611740670157)
    expect_lt(sum(abs(pred$mu-expected_mu_nn)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_nn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$mu-expected_mu)),1)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov_nn[c(1,5,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1)
    
    ##############
    ## Multiple observations at the same location
    ##############
    coords_ARD_mult = coords_ARD
    coords_ARD_mult[1:5,] <- coords_ARD_mult[(n-4):n,]
    init_cov_pars_ARD_mult <- c(var(y)/2,var(y)/2)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars_ARD_mult <- c(init_cov_pars_ARD_mult, mean(dist(unique(coords_ARD_mult)[,i])/3))
    params_ARD_mult <- DEFAULT_OPTIM_PARAMS_STD
    params_ARD_mult$init_cov_pars <- init_cov_pars_ARD_mult
    
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 268.9672548
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                           y = y, X = X, params = params_ARD_mult)
    cov_pars <- c(0.2426975279, 0.1061137976, 1.0027358655, 0.3802453878, 0.2858790841, 0.1679271974, 0.6040050853, 0.3783277462, 0.4110281274, 0.2501942989)
    coef <- c( 2.2192616509, 0.4265375022, 1.7086751823, 0.1086737770)
    nrounds <- 23
    nll_opt <- 125.6855962
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_ITERATIVE)
    # Prediction
    coord_test_mult <- rbind(c(10000,0.2,0.9),c(10000,0.2,0.9), coords_ARD_mult[1,])
    coord_test_mult[-c(1,2),c(2:3)] <- coord_test_mult[-c(1,2),c(2:3)] + 0.01
    exp_mu_mult <- c(2.219261651, 2.219261651, 2.801233150)
    exp_cov_mult <- c(2.000000, 1.000000, 0.000000, 1.000000, 2.000000, 0.000000, 0.000000, 0.000000, 1.348149)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),2*TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
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
                                        gp_approx = "fitc", num_ind_points = dim(unique(coords_ARD_mult)), ind_points_selection = "random"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = dim(unique(coords_ARD_mult)), ind_points_selection = "random",
                                           y = y, X = X, params = params_ARD_mult), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)]-cov_pars[c(1,3,5,7,9)])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())[c(1,3,5,7,9)+1]-cov_pars[c(1,3,5,7,9)+1])),2*TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test_mult,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-exp_mu_mult)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_mult[c(1,5,9)])),TOLERANCE_STRICT)
  })
  
}

