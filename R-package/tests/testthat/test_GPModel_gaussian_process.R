context("GPModel_gaussian_process")

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  TOLERANCE_LOOSE <- 1E-2
  TOLERANCE_STRICT <- 1E-6
  TOLERANCE_ITERATIVE <- 1E-1
  
  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov = "gradient_descent",
                               lr_cov = 0.1, use_nesterov_acc = TRUE,
                               acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                               optimizer_coef = "gradient_descent", lr_coef = 0.1,
                               convergence_criterion = "relative_change_in_log_likelihood")
  DEFAULT_OPTIM_PARAMS_STD <- c(DEFAULT_OPTIM_PARAMS, list(std_dev = TRUE))
  DEFAULT_OPTIM_PARAMS_iterative <- list(optimizer_cov = "gradient_descent",
                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                         acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                         optimizer_coef = "gradient_descent", lr_coef = 0.1,
                                         convergence_criterion = "relative_change_in_log_likelihood",
                                         cg_delta_conv = 1E-6,
                                         cg_preconditioner_type = "predictive_process_plus_diagonal",
                                         cg_max_num_it = 1000,
                                         cg_max_num_it_tridiag = 1000,
                                         num_rand_vec_trace = 1000,
                                         reuse_rand_vec_trace = TRUE)
  DEFAULT_OPTIM_PARAMS_STD_iterative <- c(DEFAULT_OPTIM_PARAMS_iterative, list(std_dev = TRUE))
  
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
  # Space-time GP
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
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    capture.output( fit(gp_model, y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    cov_pars <- c(0.03784221, 0.07943467, 1.07390943, 0.25351519, 0.11451432, 0.03840236)
    num_it <- 59
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Can switch between likelihoods
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    gp_model$set_likelihood("gamma")
    gp_model$set_likelihood("gaussian")
    capture.output( fit(gp_model, y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    # Gradient descent without Nesterov acceleration
    params <- DEFAULT_OPTIM_PARAMS_STD
    params$use_nesterov_acc <- FALSE
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params) , file='NUL')
    cov_pars_other <- c(0.04040441, 0.08036674, 1.06926607, 0.25360131, 0.11502362, 0.03877014)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),5E-6)
    expect_equal(gp_model$get_num_optim_iter(), 97)
    # Using a too large learning rate
    params <- DEFAULT_OPTIM_PARAMS_STD
    params$lr_cov <- 1
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params) , file='NUL')
    cov_pars_other <- c(0.04487369, 0.08285696, 1.07537253, 0.25676579, 0.11566763, 0.03928107)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 27)
    # Different terminations criterion
    params <- DEFAULT_OPTIM_PARAMS_STD
    params$convergence_criterion = "relative_change_in_parameters"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params), file='NUL')
    cov_pars_other_crit <- c(0.03276547, 0.07715343, 1.07617676, 0.25177603, 0.11352557, 0.03770062)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other_crit)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    ll <- gp_model$neg_log_likelihood(y=y,cov_pars=gp_model$get_cov_pars()[1,])
    expect_lt(abs(ll-122.7752664),TOLERANCE_STRICT)
    # Fisher scoring
    params <- DEFAULT_OPTIM_PARAMS_STD
    params$optimizer_cov = "fisher_scoring"
    params$lr_cov <- 1
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params), file='NUL')
    cov_pars_fisher <- c(0.03300593, 0.07725225, 1.07584118, 0.25180563, 0.11357012, 0.03773325)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_fisher)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    # Nelder-mead
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = list(optimizer_cov = "nelder_mead",
                                                                delta_rel_conv=1e-6))
                    , file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 40)
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
    # BFGS
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = list(optimizer_cov = "bfgs")), file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 16)
    # Adam
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = list(optimizer_cov = "adam")), file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 498)
    
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
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Prediction of variances only
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
    
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    preds <- predict(gp_model, gp_coords_pred = coords,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-6)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),1E-6)
    
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
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Prediction of variances only
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, 
                    cov_pars = cov_pars_pred, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
    # Predict latent process
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-cov_no_nugget)),1E-6)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1.6,0.2),y=y)
    expect_lt(abs(nll-124.2549533),1E-6)
    
    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    opt <- optim(par=c(0.1,2,0.2), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
    expect_lt(sum(abs(opt$par-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_lt(abs(opt$value-(122.7752694)),1E-5)
    expect_equal(as.integer(opt$counts[1]), 198)
    
    # Other covariance functions
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 0.5,
                                           y = y, params = DEFAULT_OPTIM_PARAMS_STD) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 1.5,
                                           y = y, params = DEFAULT_OPTIM_PARAMS_STD) , file='NUL')
    cov_pars_other <- c(0.22926543, 0.08486055, 0.87886348, 0.24059253, 0.10726402, 0.01542898)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 16)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 2.5,
                                           y = y, params = DEFAULT_OPTIM_PARAMS_STD) , file='NUL')
    cov_pars_other <- c(0.27251105, 0.08316755, 0.83205621, 0.23561744, 0.10536460, 0.01062167)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 13)
  })
  
  test_that("Gaussian process model with linear regression term ", {
    
    y <- eps + X%*%beta + xi
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X=X,
                           params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls",
                                         delta_rel_conv = 1E-6, use_nesterov_acc = FALSE, std_dev = TRUE,
                                         convergence_criterion = "relative_change_in_parameters"))
    cov_pars <- c(0.008461342, 0.069973492, 1.001562822, 0.214358560, 0.094656409, 0.029400407)
    coef <- c(2.30780026, 0.21365770, 1.89951426, 0.09484768)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_lt(sum(abs( as.vector(gp_model$get_coef())-coef)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    # Prediction 
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expected_mu <- c(1.196952, 4.063324, 3.156427)
    expected_cov <- c(6.305383e-01, 1.358861e-05, 8.317903e-08, 1.358861e-05,
                      3.469270e-01, 2.686334e-07, 8.317903e-08, 2.686334e-07, 4.255400e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    
    # Gradient descent
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y=y, X=X,
                           params = list(optimizer_cov = "gradient_descent", optimizer_coef = "gradient_descent",
                                         delta_rel_conv = 1E-6, use_nesterov_acc = TRUE, std_dev = FALSE, lr_coef=1))
    cov_pars <- c(0.01624576, 0.99717015, 0.09616822)
    coef <- c(2.305484, 1.899207)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 99)
    
    # Nelder-Mead
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead",
                                                     maxit=1000, delta_rel_conv = 1e-12))
    cov_pars <- c(0.008459373, 1.001564796, 0.094655964)
    coef <- c(2.307798, 1.899516)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 429)
    # BFGS
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X=X, params = list(optimizer_cov = "bfgs",
                                                     maxit=1000))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-2)
    expect_gt(gp_model$get_num_optim_iter(), 24)# different compilers result in slightly different results
    expect_lt(gp_model$get_num_optim_iter(), 28)
    # Adam
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X=X, params = list(optimizer_cov = "adam",
                                                     maxit=5000))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-2)
    expect_gt(gp_model$get_num_optim_iter(), 1950) # different compilers result in slightly different results
    expect_lt(gp_model$get_num_optim_iter(), 2010)
    
  })
  
  test_that("Gaussian process and two random coefficients ", {
    
    y <- eps_svc + xi
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_rand_coef_data = Z_SVC, y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, maxit=10)), file='NUL')
    expected_values <- c(0.25740068, 0.22608704, 0.83503539, 0.41896403, 0.15039055,
                         0.10090869, 1.61010233, 0.84207763, 0.09015444, 0.07106099, 
                         0.25064640, 0.62279880, 0.08720822, 0.32047865)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Predict training data random effects
    cov_pars <- gp_model$get_cov_pars()[1,]
    training_data_random_effects <- predict_training_data_random_effects(gp_model)
    Z_SVC_test <- cbind(rep(0,length(y)),rep(0,length(y)))
    preds <- predict(gp_model, gp_coords_pred = coords,
                     gp_rand_coef_data_pred=Z_SVC_test,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-6)
    Z_SVC_test <- cbind(rep(1,length(y)),rep(0,length(y)))
    preds2 <- predict(gp_model, gp_coords_pred = coords,
                      gp_rand_coef_data_pred=Z_SVC_test,
                      predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,2] - (preds2$mu - preds$mu))),1E-6)
    Z_SVC_test <- cbind(rep(0,length(y)),rep(1,length(y)))
    preds3 <- predict(gp_model, gp_coords_pred = coords,
                      gp_rand_coef_data_pred=Z_SVC_test,
                      predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,3] - (preds3$mu - preds$mu))),1E-6)
    
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
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Predict variances
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_rand_coef_data = Z_SVC, y = y,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc= FALSE, maxit=5)), file='NUL')
    expected_values <- c(0.000242813, 0.176573955, 1.008181385, 0.397341267, 0.141084495, 
                         0.070671768, 1.432715033, 0.708039197, 0.055598038, 0.048988825, 
                         0.430573036, 0.550644708, 0.038976112, 0.106110593)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
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
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Predict variances
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(0.1,1,0.15), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
  })
  
  test_that("Gaussian process model with multiple observations at the same location ", {
    
    y <- eps_multiple + xi
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                         delta_rel_conv = 1E-6, maxit = 500)), file='NUL')
    cov_pars <- c(0.037145465, 0.006065652, 1.151982610, 0.434770575, 0.191648634, 0.102375515)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 12)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", y = y,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    cov_pars <- c(0.037136462, 0.006064181, 1.153630335, 0.435788570, 0.192080613, 0.102631006)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
    expect_equal(gp_model$get_num_optim_iter(), 14)
    
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    preds <- predict(gp_model, gp_coords_pred = coords_multiple,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-6)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),1E-6)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential")
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE)
    expected_mu <- c(-0.1460550, 1.0042814, 0.7840301)
    expected_cov <- c(0.6739502109, 0.0008824337, -0.0003815281, 0.0008824337,
                      0.6060039551, -0.0004157361, -0.0003815281, -0.0004157361, 0.7851787946)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Predict variances
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cov_pars = c(0.1,1,0.15), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
  })
  
  test_that("Vecchia approximation for Gaussian process model ", {
    
    y <- eps + xi
    params_vecchia <- list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                           lr_cov = 0.1, use_nesterov_acc = TRUE,
                           acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                           convergence_criterion = "relative_change_in_parameters")
    # Maximal number of neighbors
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia), file='NUL')
    cov_pars <- c(0.03276544, 0.07715339, 1.07617623, 0.25177590, 0.11352557, 0.03770062)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    
    # Same thing without Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    
    # Random ordering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering="random", y = y,
                                           params = params_vecchia), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    
    # Prediction using given parameters
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    cov_pars = c(0.02,1.2,0.9)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08704577, 1.63875604, 0.48513581)
    expected_cov <- c(1.189093e-01, 1.171632e-05, -4.172444e-07, 1.171632e-05,
                      7.427727e-02, 1.492859e-06, -4.172444e-07, 1.492859e-06, 8.107455e-02)
    exp_cov_no_nugget <- expected_cov
    exp_cov_no_nugget[c(1,5,9)] <- expected_cov[c(1,5,9)] - cov_pars[1]
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars, predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-exp_cov_no_nugget)),1E-6)
    # Prediction of variances only
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
    # Predict latent process
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_no_nugget[c(1,5,9)])),1E-6)
    
    # Vecchia approximation with 30 neighbors
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30,
                                        vecchia_ordering = "none"), file='NUL')
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                                       maxit=1000, convergence_criterion = "relative_change_in_parameters"))
                    , file='NUL')
    cov_pars_vecchia <- c(0.03297349, 0.07716447, 1.07691542, 0.25221730, 0.11378505, 0.03782172)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_vecchia)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 378)
    
    # Prediction from fitted model
    coord_test <- cbind(c(0.1,0.10001,0.7),c(0.9,0.90001,0.55))
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expected_mu_vecchia <- c(0.06968068, 0.06967750, 0.44208925)
    expected_cov_vecchia <- c(0.6214955, 0.0000000, 0.0000000, 0.0000000, 0.6215069,
                              0.0000000, 0.0000000, 0.0000000, 0.4199531)
    expect_lt(sum(abs(pred$mu-expected_mu_vecchia)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_vecchia)),1E-6)
    
    # Vecchia approximation with 30 neighbors and random ordering
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30, 
                                        vecchia_ordering="random"), file='NUL')
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                                       maxit=1000, convergence_criterion = "relative_change_in_parameters"))
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_vecchia)),0.05)
    expect_gt(gp_model$get_num_optim_iter(), 360) # different compilers result in slightly different results
    expect_lt(gp_model$get_num_optim_iter(), 420)
    
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
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering="none", y = y,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc = FALSE,
                                                         delta_rel_conv = 1E-6, maxit=100,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_vecchia)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 16)
    # expect_gt(gp_model$get_num_optim_iter(), 16) # different compilers result in slightly different results
    # expect_lt(gp_model$get_num_optim_iter(), 21)
    
    # Prediction using given parameters
    cov_pars_pred <- c(0.02,1.2,0.9)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08665472, 0.08664854, 0.49011216)
    expected_cov <- c(0.11891, 0.00000000, 0.00000000, 0.00000000,
                      0.1189129, 0.00000000, 0.00000000, 0.00000000, 0.08108126)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])),1E-6)
    # Predict latent process
    pred_var2 <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                         cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu - pred_var2$mu)),1E-6)
    expect_lt(sum(abs(pred_var$var - cov_pars_pred[1] - pred_var2$var)),1E-6)
    
    # Prediction with vecchia_pred_type = "order_obs_first_cond_all"
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08665472, 0.08661259, 0.49011216)
    expected_cov <- c(0.11891004, 0.09889262, 0.00000000, 0.09889262, 0.11891291, 
                      0.00000000, 0.00000000, 0.00000000, 0.08108126)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])),1E-6)
    # Predict latent process
    pred_var2 <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                         cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu - pred_var2$mu)),1E-6)
    expect_lt(sum(abs(pred_var$var - cov_pars_pred[1] - pred_var2$var)),1E-6)
    
    # Prediction with vecchia_pred_type = "order_pred_first"
    gp_model$set_prediction_data(vecchia_pred_type = "order_pred_first", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08498682, 0.08502034, 0.49572748)
    expected_cov <- c(1.189037e-01, 9.888624e-02, -1.080005e-05, 9.888624e-02, 
                      1.189065e-01, -1.079431e-05, -1.080005e-05, -1.079431e-05, 8.101757e-02)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])),1E-6)
    # Predict latent process
    pred_var2 <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                         cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu - pred_var2$mu)),1E-6)
    expect_lt(sum(abs(pred_var$var - cov_pars_pred[1] - pred_var2$var)),1E-6)
    
    # Prediction with vecchia_pred_type = "latent_order_obs_first_cond_obs_only"
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_obs_only", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08616985, 0.08616384, 0.48721314)
    expected_cov <- c(1.189100e-01, 7.324225e-03, -5.851427e-07, 7.324225e-03, 
                      1.189129e-01, -5.850749e-07, -5.851427e-07, -5.850750e-07, 8.107749e-02)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])),1E-6)
    # Predict latent process
    pred_var2 <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                         cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu - pred_var2$mu)),1E-6)
    expect_lt(sum(abs(pred_var$var - cov_pars_pred[1] - pred_var2$var)),1E-6)
    
    # Prediction with vecchia_pred_type = "latent_order_obs_first_cond_all"
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", num_neighbors_pred = 30)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = TRUE)
    expected_mu <- c(0.08616985, 0.08616377, 0.48721314)
    expected_cov <- c(1.189100e-01, 9.889258e-02, -5.851418e-07, 9.889258e-02,
                      1.189129e-01, -5.850764e-07, -5.851418e-07, -5.850764e-07, 8.107749e-02)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])),1E-6)
    # Predict latent process
    pred_var2 <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                         cov_pars = cov_pars_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu - pred_var2$mu)),1E-6)
    expect_lt(sum(abs(pred_var$var - cov_pars_pred[1] - pred_var2$var)),1E-6)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1.6,0.2),y=y)
    expect_lt(abs(nll-124.2252524),1E-6)
  })
  
  test_that("Vecchia approximation for Gaussian process model with linear regression term ", {
    
    y <- eps + X%*%beta + xi
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, X=X,
                                           params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls", std_dev = TRUE,
                                                         delta_rel_conv = 1E-6, use_nesterov_acc = FALSE,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    cov_pars <- c(0.008461342, 0.069973492, 1.001562822, 0.214358560, 0.094656409, 0.029400407)
    coef <- c(2.30780026, 0.21365770, 1.89951426, 0.09484768)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    
    # Prediction 
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, gp_coords_pred = coord_test,
                                    X_pred = X_test, predict_cov_mat = TRUE, predict_response = TRUE)
                    , file='NUL')
    expected_mu <- c(1.196952, 4.063324, 3.156427)
    expected_cov <- c(6.305383e-01, 1.358861e-05, 8.317903e-08, 1.358861e-05,
                      3.469270e-01, 2.686334e-07, 8.317903e-08, 2.686334e-07, 4.255400e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  })
  
  test_that("Vecchia approximation for Gaussian process model with cluster_id's not constant ", {
    
    y <- eps + xi
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           vecchia_ordering = "none", y = y, cluster_ids = cluster_ids,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.05, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    cov_pars <- c(0.05374602, 0.08709594, 1.05800024, 0.22867128, 0.12680152, 0.04066888)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 474)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           vecchia_ordering = "none", y = y, cluster_ids = cluster_ids,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
    expect_equal(gp_model$get_num_optim_iter(), 20)
    
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
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  })
  
  test_that("Vecchia approximation for Gaussian process model 
            with multiple observations at the same location ", {
              
              y <- eps_multiple + xi
              capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                                     gp_approx = "vecchia", num_neighbors = n-1, y = y,
                                                     vecchia_ordering = "none", 
                                                     params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                                   lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                                   delta_rel_conv = 1E-6, maxit = 500)), file='NUL')
              cov_pars <- c(0.037145465, 0.006065652, 1.151982610, 0.434770575, 0.191648634, 0.102375515)
              expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-4)
              expect_equal(gp_model$get_num_optim_iter(), 12)
              
              # Fisher scoring
              capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                                     gp_approx = "vecchia", num_neighbors = n-1, y = y, 
                                                     vecchia_ordering = "none", 
                                                     params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                                   use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                                   convergence_criterion = "relative_change_in_parameters")), file='NUL')
              cov_pars <- c(0.037136462, 0.006064181, 1.153630335, 0.435788570, 0.192080613, 0.102631006)
              expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
              expect_equal(gp_model$get_num_optim_iter(), 14)
              
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
              expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
              expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
              
            })
  
  test_that("Vecchia approximation for Gaussian process and two random coefficients ", {
    
    y <- eps_svc + xi
    # Fit model using gradient descent with Nesterov acceleration
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1,
                                           gp_rand_coef_data = Z_SVC, vecchia_ordering = "none", y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, maxit=10)), file='NUL')
    expected_values <- c(0.25740068, 0.22608704, 0.83503539, 0.41896403, 0.15039055,
                         0.10090869, 1.61010233, 0.84207763, 0.09015444, 0.07106099,
                         0.25064640, 0.62279880, 0.08720822, 0.32047865)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
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
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-149.4422184),1E-5)
    
    # Fit model using gradient descent with Nesterov acceleration
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           gp_rand_coef_data = Z_SVC, vecchia_ordering = "none", y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = FALSE, maxit=10)), file='NUL')
    expected_values <- c(0.3448993, 0.2302483, 0.7981342, 0.4158840, 0.1514441, 0.1074046, 
                         1.1479748, 0.7761315, 0.1032126, 0.1012417, 0.3224399, 0.6417908, 0.1061352, 0.2955655)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
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
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           gp_rand_coef_data = Z_SVC, vecchia_ordering = "none", y = y,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc= FALSE, maxit=5)), file='NUL')
    expected_values <- c(0.0004631625, 0.2001592837, 1.1329783638, 0.4500150650, 0.1466853248, 
                         0.0745943630, 1.6392349806, 0.7922744312, 0.0565169535, 0.0483411364, 0.4393511638, 0.5909479447, 0.0321612593, 0.1046076251)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-149.4840466),1E-6)
  })
  
  test_that("Wendland covariance function for Gaussian process model ", {
    
    y <- eps + xi
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "wendland", 
                                        cov_fct_taper_shape = 0, cov_fct_taper_range = 0.1), file='NUL')
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5)) , file='NUL')
    cov_pars <- c(0.002911765, 0.116338096, 0.993996193, 0.211276385)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
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
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Prediction of variances only
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 0.1, y = y,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc = FALSE,
                                                         delta_rel_conv = 1E-6)), file='NUL')
    cov_pars <- c(2.946448e-08, 1.599928e-01, 1.391589e+00, 2.933997e-01)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.02,1.2),y=y)
    expect_lt(abs(nll-136.9508962),1E-6)
    
    # Other taper shapes
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 1, cov_fct_taper_range = 0.15, y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5)), file='NUL')
    cov_pars <- c(0.0564441, 0.0497191, 0.9921285, 0.1752661)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 19)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_cov_mat = TRUE)
    expected_mu <- c(-0.007404038, 1.487424320, 0.200022114)
    expected_cov <- c(1.113020e+00, -6.424533e-30, -4.186440e-22, -6.424533e-30, 3.522739e-01,
                      9.018454e-10, -4.186440e-22, 9.018454e-10, 6.092985e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Other taper shapes
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 2, cov_fct_taper_range = 0.08, y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5)), file='NUL')
    cov_pars <- c(0.00327103, 0.06579671, 1.08812978, 0.18151366)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 187)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_cov_mat = TRUE)
    expected_mu <- c(-2.314198e-05, 8.967992e-01, 2.430054e-02)
    expected_cov <- c(1.2200000, 0.0000000, 0.0000000, 0.0000000, 0.9024792, 0.0000000, 0.0000000, 0.0000000, 1.1887157)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  })
  
  test_that("Tapering ", {
    
    y <- eps + X%*%beta + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    # No tapering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, X = X, params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars <- c(0.01621846, 0.07384498, 0.99717680, 0.21704099, 0.09616230, 0.03034715)
    coef <- c(2.30554610, 0.21565230, 1.89920767, 0.09567547)
    num_it <- 99
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
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
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
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
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars_tap <- c(0.02593993, 0.07560715, 0.99435221, 0.21816716, 0.17712808, 0.09797175)
    coef_tap <- c(2.32410488, 0.20610507, 1.89498931, 0.09533541)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_tap)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_tap)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 75)
    
    # Same thing with Matern covariance
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           y = y, X = X, params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars <- c(0.17369771, 0.07950745, 0.84098718, 0.20889907, 0.08839526, 0.01190858)
    coef <- c(2.33980860, 0.19481950, 1.88058081, 0.09786326)
    num_it <- 21
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expected_mu <- c(1.253044, 4.063322, 3.104536)
    expected_cov <- c(5.880651e-01, 3.732173e-05, 4.443229e-08, 3.732173e-05, 
                      3.627280e-01, 1.497245e-06, 4.443229e-08, 1.497245e-06, 3.796592e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # With tapering and very large tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           gp_approx = "tapering", cov_fct_taper_shape = 1, cov_fct_taper_range = 1e6,
                                           y = y, X = X,
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # With tapering and smaller tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           gp_approx = "tapering", cov_fct_taper_shape = 1, cov_fct_taper_range = 0.5,
                                           y = y, X = X,
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars_tap <- c(0.21355413, 0.08709305, 0.80448797, 0.20623554, 0.12988850, 0.03404038)
    coef_tap <- c(2.3533920, 0.1896204, 1.8720682, 0.0988744)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_tap)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_tap)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 25)
  })
  
  test_that("fitc", {
    
    y <- eps + X%*%beta + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    
    # No Approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, X = X, params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars <- c(0.01621846, 0.07384498, 0.99717680, 0.21704099, 0.09616230, 0.03034715)
    coef <- c(2.30554610, 0.21565230, 1.89920767, 0.09567547)
    num_it <- 99
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE)
    expected_mu <- c(1.195910242, 4.060125034, 3.15963272)
    expected_var <- c(6.304732e-01, 3.524404e-01, 4.277339e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # With fitc and n-1 inducing points
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "fitc",num_ind_points = n-1, 
                                           y = y, X = X,
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # With fitc and 50 inducing points
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "fitc", num_ind_points = 50, 
                                           y = y, X = X,
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars_tap <- c(0.01030298, 0.07650375, 0.99809618, 0.21799976, 0.10787353, 0.03236256)
    coef_tap <- c(2.29553776, 0.22988084, 1.89903213, 0.09726784)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_tap)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_tap)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 128)
    
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE)
    expected_mu <- c(1.171558, 3.640009, 3.437938)
    expected_var <- c(0.6681653, 0.6396615, 0.5602457)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # Same thing with Matern covariance
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           y = y, X = X, params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars <- c(0.17369771, 0.07950745, 0.84098718, 0.20889907, 0.08839526, 0.01190858)
    coef <- c(2.33980860, 0.19481950, 1.88058081, 0.09786326)
    num_it <- 21
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE)
    expected_mu <- c(1.253044, 4.063322, 3.104536)
    expected_var <- c(5.880651e-01, 3.627280e-01, 3.796592e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
    
    # With fitc and n-1 inducing points or very small coverTree radius
    # Different Inducing Point Methods
    ind_point_methods <- c("random","kmeans++","cover_tree")
    for (i in ind_point_methods) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                             gp_approx = "fitc",num_ind_points = n-1,cover_tree_radius = 1e-2,
                                             ind_points_selection = i, y = y, X = X,
                                             params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
      expect_equal(gp_model$get_num_optim_iter(), num_it)
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
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars_ip <- c(0.17369144, 0.07854085, 0.84099141, 0.20572443, 0.08839449, 0.01162549)
    coef_ip <- c(2.33983295, 0.19481861, 1.88057897, 0.09786246)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_ip)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_ip)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 21)
    
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
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars_tap <- c(0.19682949, 0.09401528, 0.81880087, 0.21614994, 0.09415915, 0.01354960)
    coef_tap <- c(2.3383270, 0.2017728, 1.8559971, 0.1004556)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_tap)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_tap)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 19)
    
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
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars_tap <- c(0.17256176, 0.07770770, 0.84191496, 0.20527780, 0.08810525, 0.01151530)
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
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    
    vec_chol_or_iterative <- c("cholesky","iterative")
    for (i in vec_chol_or_iterative) {
      if(i == "iterative"){
        DEFAULT_OPTIM_PARAMS_STD <- DEFAULT_OPTIM_PARAMS_STD_iterative
        TOLERANCE <- TOLERANCE_ITERATIVE
      } else {
        TOLERANCE <- TOLERANCE_LOOSE
      }
      # No Approximation
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             y = y, X = X, params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
      cov_pars <- c(0.01621846, 0.07384498, 0.99717680, 0.21704099, 0.09616230, 0.03034715)
      coef <- c(2.30554610, 0.21565230, 1.89920767, 0.09567547)
      num_it <- 99
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
      expect_equal(gp_model$get_num_optim_iter(), num_it)
      # Prediction 
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, predict_var = TRUE)
      expected_mu <- c(1.195910242, 4.060125034, 3.15963272)
      expected_var <- c(6.304732e-01, 3.524404e-01, 4.277339e-01)
      expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_LOOSE)
      expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
      
      # With FSA and very large tapering range and 60 inducing points
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             gp_approx = "full_scale_tapering",num_ind_points = 60, cov_fct_taper_shape = 2, cov_fct_taper_range = 1e6,
                                             y = y, X = X,  matrix_inversion_method = i,
                                             params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
      expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
      if(i == "cholesky"){
        expect_equal(gp_model$get_num_optim_iter(), num_it)
      }
      # Prediction 
      if(i == "iterative"){
        gp_model$set_prediction_data(cg_delta_conv_pred = 1e-8, nsim_var_pred = 700)
      }
      pred <- predict(gp_model, gp_coords_pred = coord_test,
                      X_pred = X_test, predict_var = TRUE)
      expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE)
      expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE)
      
      if(i == "cholesky"){
        # With FSA and n-1 inducing points and taper range 0.4
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               gp_approx = "full_scale_tapering",num_ind_points = n-1, cov_fct_taper_shape = 2, cov_fct_taper_range = 0.4,
                                               y = y, X = X,matrix_inversion_method = i, 
                                               params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
        expect_equal(gp_model$get_num_optim_iter(), num_it)
        # Prediction 
        if(i == "iterative"){
          gp_model$set_prediction_data(cg_delta_conv_pred = 1e-6, nsim_var_pred = 500)
        }
        pred <- predict(gp_model, gp_coords_pred = coord_test,
                        X_pred = X_test, predict_var = TRUE)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE)
        
        # With FSA and 50 inducing points and taper range 0.5
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               gp_approx = "full_scale_tapering", num_ind_points = 50, cov_fct_taper_shape = 2, cov_fct_taper_range = 0.5,
                                               y = y, X = X,matrix_inversion_method = i, 
                                               params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
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
        pred <- predict(gp_model, gp_coords_pred = coord_test,
                        X_pred = X_test, predict_var = TRUE)
        expected_mu <- c(1.186786, 4.048299, 3.173789) 
        expected_var <- c(0.6428104, 0.3562637, 0.4344309)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE)
        
        # Same thing with Matern covariance
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                               y = y, X = X, params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
        cov_pars <- c(0.17369771, 0.07950745, 0.84098718, 0.20889907, 0.08839526, 0.01190858)
        coef <- c(2.33980860, 0.19481950, 1.88058081, 0.09786326)
        num_it <- 21
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
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_LOOSE)
        
        # With FSA and very large tapering range and 60 inducing points
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                               gp_approx = "full_scale_tapering",num_ind_points = 60, cov_fct_taper_shape = 2, cov_fct_taper_range = 1e6,
                                               y = y, X = X,  matrix_inversion_method = i, 
                                               params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
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
                                               params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
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
                                               params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
        cov_pars <- c(0.16783429, 0.07818710, 0.84903511, 0.20697391, 0.08811171, 0.01153137)
        coef <- c(2.34108249, 0.19532774, 1.87704014, 0.09748291)
        num_it <- 21
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
    
    y <- eps_ST + X%*%beta + xi
    cov_pars_nll <- c(0.1, 1.6, rho_time * 0.5, 2 * rho)
    coord_test <- rbind(c(10000,0.2,0.9), cbind(time, coords)[c(1,10),])
    coord_test[-1,c(2:3)] <- coord_test[-1,c(2:3)] + 0.01
    X_test <- cbind(rep(1,3),c(0,0,0))
    cov_pars_pred <- c(1, 1, rho_time, rho)
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 272.1497719
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD)
    cov_pars <- c(0.01316765, 0.28736684, 1.00918678, 0.33462814, 1.37748568, 
                  0.78252561, 0.11561567, 0.05410341)
    coef <- c(1.9583409, 0.1484610, 2.1707779, 0.1397487)
    nrounds <- 341
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
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
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", 
                        cov_fct_shape = 1.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_1_5_exp <- 288.6072086
    expect_lt(abs(nll-nll_1_5_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                           cov_fct_shape = 1.5, y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD)
    cov_pars_1_5 <- c(0.6841968698, 0.1935694746, 0.3282571084, 0.2086812255, 
                      5.0002988197, 2.2710888937, 0.2041394663, 0.0736608087)
    coef_1_5 <- c(1.9623139055, 0.1867224223, 2.2128240026, 0.1411173075)
    nrounds_1_5 <- 23
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_1_5)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_1_5)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_1_5)
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time", 
                        cov_fct_shape = 2.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_2_5_exp <- 296.7149408
    expect_lt(abs(nll-nll_2_5_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                           cov_fct_shape = 2.5, y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD)
    cov_pars_2_5 <- c(0.72525339436, 0.16854275975, 0.28900674649, 0.18079699539, 
                      5.53898014923, 1.87693496198, 0.22077648810, 0.05742157744)
    coef_2_5 <- c(1.9637702081, 0.1906062163, 2.2151163284, 0.1409464531)
    nrounds_2_5 <- 16
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_2_5)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_2_5)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_2_5)
    
    ##############
    ## With Vecchia approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
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
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                                        gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-272.1376522),TOLERANCE_STRICT)
    # Different orderings
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                                        gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "time"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-272.1498125),TOLERANCE_LOOSE)
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                                        gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "time_random_space"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-272.1498202),TOLERANCE_LOOSE)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    cov_pars_nn <- c(0.4352792979, 0.3483560718, 0.5764632615, 0.3742805059, 3.3684298629, 2.7677409072, 0.1492689415, 0.1038979210)
    coef_nn <- c(1.9611419162, 0.1768231706, 2.2062141280, 0.1410012118)
    nrounds_nn <- 9
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_nn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
    # Fit model with bfgs & nelder_mead
    params_loc <- DEFAULT_OPTIM_PARAMS_STD
    params_loc$optimizer_cov <- "bfgs"
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X=X, params = params_loc), 
                    file='NUL')
    cov_pars_nn <- c(5.257439607e-09, 2.727649112e-01, 1.017927054e+00, 3.215946495e-01, 1.350712536e+00, 7.550462399e-01, 1.155108830e-01, 5.317131560e-02)
    coef_nn <- c(1.9580348238, 0.1478844664, 2.1694023578, 0.1392658891)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_nn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_STRICT)
    params_loc$optimizer_cov <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                           y = y, X=X, params = params_loc), 
                    file='NUL')
    cov_pars_nn <- c(1.7588095447, 0.8616670911, 1.7580723356, 0.9775093445, 2.6720944053, 2.1292725842, 0.3134624056, 0.2313693929)
    coef_nn <- c(2.2786043463538, 0.4147515045451, 0.0008302855905, 0.2626314668631)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_nn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_STRICT)
    # Different ordering
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), cov_function = "matern_space_time",
                                           gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "time",
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    cov_pars_nn <- c(0.41094648686, 0.35479036337, 0.60200316918, 0.38191826343, 3.19138814849, 2.55633124550, 0.14611908707, 0.09919444919)
    coef_nn <- c(1.9610792081, 0.1751682848, 2.2042968795, 0.1410400655)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_nn)),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_nn)),TOLERANCE_LOOSE)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=num_neighbors)
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu_nn <- c(1.961079208, 1.935536314, 2.566095948)
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
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 276.47191976324
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time",
                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD)
    cov_pars <- c(0.48244729, 0.20133860, 0.53677606, 0.24652176, 3.84944066, 2.82763607, 0.21590375, 0.13357978)
    coef <- c(1.95425156, 0.21356119, 2.19640126, 0.13803044)
    nrounds <- 41
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    ## With Vecchia approximation
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time",
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none"), 
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time",
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
  })
  
}

