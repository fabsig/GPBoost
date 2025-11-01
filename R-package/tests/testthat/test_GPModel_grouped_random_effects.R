if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  context("GPModel_grouped_random_effects")
  
  TOLERANCE_STRICT <- 1E-6
  TOLERANCE_MEDIUM <- 1E-3
  TOLERANCE_LOOSE <- 1E-2
  TOL_VERY_LOOSE <- 1E-1
  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov = "gradient_descent",
                               lr_cov = 0.1, use_nesterov_acc = TRUE,
                               acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                               optimizer_coef = "gradient_descent", lr_coef = 0.1,
                               convergence_criterion = "relative_change_in_log_likelihood",
                               std_dev = FALSE)
  DEFAULT_OPTIM_PARAMS_STD <- DEFAULT_OPTIM_PARAMS
  DEFAULT_OPTIM_PARAMS_STD$std_dev = TRUE
  
  # Function that simulates uniform random variables
  sim_rand_unif <- function(n, init_c=0.1){
    mod_lcg <- 134456 # modulus for linear congruential generator (random0 used)
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for(i in 2:n) sim[i] <- (8121 * sim[i-1] + 28411) %% mod_lcg
    return(sim / mod_lcg)
  }
  
  # Create data
  n <- 1000 # number of samples
  # First grouped random effects model
  m <- 100 # number of categories / levels for grouping variable
  group <- rep(1,n) # grouping variable
  for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
  Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
  b1 <- qnorm(sim_rand_unif(n=m, init_c=0.546))
  # Second random effects
  n_gr <- n/20 # number of groups
  group2 <- rep(1,n) # grouping variable
  for(i in 1:(n/n_gr)) group2[(1:n_gr)+n_gr*(i-1)] <- 1:n_gr
  Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
  b2 <- qnorm(sim_rand_unif(n=length(unique(group2)), init_c=0.46))
  # Random slope / coefficient
  x <- cos((1:n-n/2)^2*5.5*pi/n) # covariate data for random slope
  Z3 <- diag(x) %*% Z1
  b3 <- qnorm(sim_rand_unif(n=m, init_c=0.69))
  # Error term
  xi <- sqrt(0.5) * qnorm(sim_rand_unif(n=n, init_c=0.1))
  # Data for linear mixed effects model
  X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
  beta <- c(2,2) # regression coefficients
  # cluster_ids 
  cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))
  
  test_that("single level grouped random effects model ", {
    
    y <- as.vector(Z1 %*% b1) + xi
    # Estimation using Fisher scoring
    gp_model <- GPModel(group_data = group)
    fit(gp_model, y = y, params = list(std_dev = TRUE, optimizer_cov = "fisher_scoring",
                                       convergence_criterion = "relative_change_in_parameters"))
    cov_pars <- c(0.49348532, 0.02326312, 1.22299521, 0.17995161)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(dim(gp_model$get_cov_pars())[2], 2)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    # Can switch between likelihoods
    gp_model <- GPModel(group_data = group)
    gp_model$set_likelihood("gamma")
    gp_model$set_likelihood("gaussian")
    fit(gp_model, y = y, params = list(std_dev = TRUE, optimizer_cov = "fisher_scoring",
                                       convergence_criterion = "relative_change_in_parameters"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    # Using gradient descent instead of Fisher scoring
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE,
                                         lr_cov = 0.1, use_nesterov_acc = FALSE, maxit = 1000,
                                         convergence_criterion = "relative_change_in_parameters"))
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3)])),1E-5)
    expect_equal(class(cov_pars_est), "numeric")
    expect_equal(length(cov_pars_est), 2)
    # Using gradient descent with Nesterov acceleration
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE,
                                         lr_cov = 0.2, use_nesterov_acc = TRUE,
                                         acc_rate_cov = 0.1, maxit = 1000,
                                         convergence_criterion = "relative_change_in_parameters"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(1,3)])),1E-5)
    # Using gradient descent and a too large learning rate
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE,
                                         lr_cov = 10, use_nesterov_acc = FALSE,
                                         maxit = 1000, convergence_criterion = "relative_change_in_parameters"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(1,3)])),TOLERANCE_STRICT)
    # Different termination criterion
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                         convergence_criterion = "relative_change_in_log_likelihood"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    # Nelder-Mead
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "nelder_mead", std_dev = TRUE, delta_rel_conv=1e-6))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 12)
    # Adam
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "adam", std_dev = TRUE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 275)
    # fix some covariance parameters
    params_loc <- list(optimizer_cov = "lbfgs", std_dev = TRUE,
                       estimate_cov_par_index = c(1,0), init_cov_pars=c(0.23, 0.45))
    gp_model_fix <- fitGPModel(group_data = group, y = y, params = params_loc)
    nll_opt_fix <- 1229.514733
    cov_pars_fix <- c(0.50600551128, 0.02385332856, 0.45000000000, 0.07083578226)
    expect_lt(sum(abs(as.vector(gp_model_fix$get_cov_pars())-cov_pars_fix)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model_fix$get_cov_pars()[1,2]-params_loc$init_cov_pars[2])),TOLERANCE_STRICT)
    expect_lt(abs(gp_model_fix$get_current_neg_log_likelihood()-nll_opt_fix), TOLERANCE_STRICT)
    
    # Evaluation of likelihood
    ll <- gp_model$neg_log_likelihood(y=y,cov_pars=gp_model$get_cov_pars()[1,])
    expect_lt(abs(ll-(1228.293)),1E-2)
    
    # Prediction 
    gp_model <- GPModel(group_data = group)
    group_test <- c(1,2,m+1)
    expect_error(predict(gp_model, y=y, cov_pars = c(0.5,1.5)))# group data not provided
    pred <- predict(gp_model, y=y, group_data_pred = group_test,
                    cov_pars = c(0.5,1.5), predict_cov_mat = TRUE)
    expected_mu <- c(-0.1553877, -0.3945731, 0)
    expected_cov <- c(0.5483871, 0.0000000, 0.0000000, 0.0000000,
                      0.5483871, 0.0000000, 0.0000000, 0.0000000, 2)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- predict(gp_model, y=y, group_data_pred = group_test,
                    cov_pars = c(0.5,1.5), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    
    # Prediction from fitted model
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "fisher_scoring",
                                         convergence_criterion = "relative_change_in_parameters"))
    group_test <- c(1,2,m+1)
    pred <- predict(gp_model, group_data_pred = group_test, predict_cov_mat = TRUE)
    expected_mu <- c(-0.1543396, -0.3919117, 0.0000000)
    expected_cov <- c(0.5409198 , 0.0000000000, 0.0000000000, 0.0000000000,
                      0.5409198 , 0.0000000000, 0.0000000000, 0.0000000000, 1.7164805)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    ## Cannot provide X for prediction when not provided for estimation
    expect_error(pred <- predict(gp_model, group_data_pred = group_test, X = group_test))
    
    # Predict training data random effects
    gp_model <- fitGPModel(group_data = group, y = y)
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences,] 
    group_unique <- unique(group)
    pred_random_effects <- predict(gp_model, group_data_pred = group_unique, 
                                   predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - pred_random_effects$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(training_data_random_effects[,2] - pred_random_effects$var)),TOLERANCE_STRICT)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1), y=y)
    expect_lt(abs(nll-2282.073),1E-2)
    nll <- gpboost::neg_log_likelihood(gp_model, cov_pars=c(0.1,1), y=y)
    expect_lt(abs(nll-2282.073),1E-2)
    fixed_effects <- rep(1, length(y))
    nll1 <- gpboost::neg_log_likelihood(gp_model, cov_pars=c(0.1,1), y=(y-fixed_effects))
    nll2 <- gpboost::neg_log_likelihood(gp_model, cov_pars=c(0.1,1), y=y, fixed_effects=fixed_effects)
    expect_lt(abs(nll1-nll2),1E-6)
    
    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(group_data = group)
    opt <- optim(par=c(1,1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
    expect_lt(sum(abs(opt$par-cov_pars[c(1,3)])),TOLERANCE_MEDIUM)
    expect_lt(abs(opt$value-(1228.293)),1E-2)
    expect_equal(as.integer(opt$counts[1]), 49)
    
    # Use non-ordered grouping data
    set.seed(1)
    shuffle_ind <- sample.int(n=n,size=n,replace=FALSE)
    gp_model <- GPModel(group_data = group[shuffle_ind])
    fit(gp_model, y = y[shuffle_ind], params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                    convergence_criterion = "relative_change_in_parameters"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    
    # Inf in response variable data gives error
    y_Inf <- y
    y_Inf[1] = -Inf
    expect_error(gp_model <- fitGPModel(group_data = group, y = y_Inf))
    
    # With offset
    offset <- 20 * sim_rand_unif(n=n, init_c=0.354)
    y_o <- y + offset
    cov_pars_pred = c(0.5,1.5)
    gp_model_no_offset <- fitGPModel(group_data = group,
                                     y = y,  params = DEFAULT_OPTIM_PARAMS_STD)
    pred_no_offset <- predict(gp_model_no_offset, group_data_pred = group_test,
                              cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    gp_model <- fitGPModel(group_data = group, y = y_o, offset = offset,
                           params = DEFAULT_OPTIM_PARAMS_STD)
    pred <- predict(gp_model, group_data_pred = group_test, offset = offset,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    pred_offset_not_provided <- predict(gp_model, group_data_pred = group_test,
                                        cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-as.vector(gp_model_no_offset$get_cov_pars()))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), gp_model_no_offset$get_num_optim_iter())
    expect_lt(sum(abs(pred$mu-pred_no_offset$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-as.vector(pred_no_offset$cov))),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$mu-pred_offset_not_provided$mu)),TOLERANCE_STRICT)
    # with lbfgs
    params = DEFAULT_OPTIM_PARAMS_STD
    params$optimizer_cov = "lbfgs"
    gp_model_no_offset <- fitGPModel(group_data = group, y = y,  params = params)
    pred_no_offset <- predict(gp_model_no_offset, group_data_pred = group_test,
                              cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    gp_model <- fitGPModel(group_data = group, y = y_o, offset = offset, params = params)
    pred <- predict(gp_model,group_data_pred = group_test, offset = offset,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-as.vector(gp_model_no_offset$get_cov_pars()))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), gp_model_no_offset$get_num_optim_iter())
    expect_lt(sum(abs(pred$mu-pred_no_offset$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-as.vector(pred_no_offset$cov))),TOLERANCE_STRICT)
  })
  
  test_that("linear mixed effects model with grouped random effects ", {
    
    y <- Z1 %*% b1 + X%*%beta + xi
    # Fitting with trace = TRUE works
    expect_error( capture.output( 
      gp_model <- fitGPModel(group_data = group, y = y, X = X,
                             params = list(std_dev = TRUE, maxit = 1, trace = TRUE))
      , file='NUL'), NA)
    # Fit model
    gp_model <- fitGPModel(group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "fisher_scoring",
                                         optimizer_coef = "wls", std_dev = TRUE,
                                         convergence_criterion = "relative_change_in_parameters"))
    cov_pars <- c(0.49205230, 0.02319557, 1.22064076, 0.17959832)
    coef <- c(2.07499902, 0.11269252, 1.94766255, 0.03382472)
    nll <- 1226.885947
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    
    # Prediction
    group_test <- c(1,2,m+1)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    expect_error(predict(gp_model,group_data_pred = group_test))# covariate data not provided
    pred <- predict(gp_model, group_data_pred = group_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expected_mu <- c(0.886494, 2.043259, 2.854064)
    expected_cov <- c(0.5393509 , 0.0000000000, 0.0000000000, 0.0000000000,
                      0.5393509 , 0.0000000000, 0.0000000000, 0.0000000000, 1.712693)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences] 
    group_unique <- unique(group)
    X_zero <- cbind(rep(0,length(group_unique)),rep(0,length(group_unique)))
    pred_random_effects <- predict(gp_model, group_data_pred = group_unique, X_pred = X_zero)
    expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),TOLERANCE_STRICT)
    
    # Fit model using gradient descent instead of wls for regression coefficients
    gp_model <- fitGPModel(group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "gradient_descent", maxit=1000, std_dev = TRUE,
                                         optimizer_coef = "gradient_descent", lr_coef=1, use_nesterov_acc=TRUE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    # Fit model using Nelder-Mead
    gp_model <- fitGPModel(group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "nelder_mead",
                                         optimizer_coef = "nelder_mead", std_dev = FALSE, delta_rel_conv=1e-6))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(1,3)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef[c(1,3)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 125)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    # Fit model using Adam
    gp_model <- fitGPModel(group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "adam", std_dev = FALSE, delta_rel_conv=1e-6))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(1,3)])),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef[c(1,3)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 354)
    
    # Fit model using lbfgs
    gp_model <- fitGPModel(group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", std_dev = TRUE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_LOOSE)
    
    # With offset / fixed_effects
    offset <- 20 * sim_rand_unif(n=n, init_c=0.1354)
    y_o <- y + offset
    group_test <- c(1,2,m+1)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    gp_model_no_offset <- fitGPModel(group_data = group, y = y, X = X, params = DEFAULT_OPTIM_PARAMS_STD)
    pred_no_offset <- predict(gp_model_no_offset, group_data_pred = group_test,
                              X_pred = X_test, predict_cov_mat = TRUE)
    gp_model <- fitGPModel(group_data = group, y = y_o, X = X, 
                           offset = offset, params = DEFAULT_OPTIM_PARAMS_STD)
    pred <- predict(gp_model, group_data_pred = group_test, offset = offset,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-as.vector(gp_model_no_offset$get_cov_pars()))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-as.vector(gp_model_no_offset$get_coef()))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), gp_model_no_offset$get_num_optim_iter())
    expect_lt(sum(abs(pred$mu-pred_no_offset$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-as.vector(pred_no_offset$cov))),TOLERANCE_STRICT)
    # with lbfgs
    params = DEFAULT_OPTIM_PARAMS_STD
    params$optimizer_cov = "lbfgs"
    params$optimizer_coef = "wls"
    gp_model_no_offset <- fitGPModel(group_data = group, y = y, X = X, params = params)
    pred_no_offset <- predict(gp_model_no_offset, group_data_pred = group_test,
                              X_pred = X_test, predict_cov_mat = TRUE)
    gp_model <- fitGPModel(group_data = group, y = y_o, X = X, offset = offset, params = params)
    pred <- predict(gp_model, group_data_pred = group_test, offset = offset,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-as.vector(gp_model_no_offset$get_cov_pars()))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-as.vector(gp_model_no_offset$get_coef()))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), gp_model_no_offset$get_num_optim_iter())
    expect_lt(sum(abs(pred$mu-pred_no_offset$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-as.vector(pred_no_offset$cov))),TOLERANCE_STRICT)
    
    # Large data
    n_L <- 1e6 # number of samples
    m_L <- n_L/10 # number of categories / levels for grouping variable
    group_L <- rep(1,n_L) # grouping variable
    for(i in 1:m_L) group_L[((i-1)*n_L/m_L+1):(i*n_L/m_L)] <- i
    keps <- 1E-10
    b1_L <- qnorm(sim_rand_unif(n=m_L, init_c=0.846)*(1-keps) + keps/2)
    X_L <- cbind(rep(1,n_L),sim_rand_unif(n=n_L, init_c=0.341)) # design matrix / covariate data for fixed effect
    beta <- c(2,2) # regression coefficients
    xi_L <- sqrt(0.5) * qnorm(sim_rand_unif(n=m_L, init_c=0.321)*(1-keps) + keps/2)
    y_L <- b1_L[group_L] + X_L%*%beta + xi_L
    # Fit model using gradient descent instead of wls for regression coefficients
    gp_model <- fitGPModel(group_data = group_L,
                           y = y_L, X = X_L, params = DEFAULT_OPTIM_PARAMS_STD)
    cov_pars <- c(0.5004967636668, 0.0007460965241, 0.9983817688906, 0.0046893215444)
    coef <- c(1.995473876130, 0.003484883983, 2.001625713274, 0.002577124175)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood() - 1224957.387)),TOLERANCE_MEDIUM)
    
    gp_model <- fitGPModel(group_data = group_L, y = y_L, X = X_L,
                           params = list(optimizer_cov = "nelder_mead", maxit=1000, delta_rel_conv=1e-6, std_dev = FALSE))
    cov_pars <- c(0.5004681, 0.9988288)
    coef <- c(2.000747, 1.999343)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood() - 1224958.6057028)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 152)
    
  })
  
  test_that("Multiple grouped random effects ", {
    
    vec_chol_or_iterative <- c("cholesky","iterative")
    for (inv_method in vec_chol_or_iterative) {
      if(inv_method == "iterative") {
        tolerance_loc_1 <- TOLERANCE_LOOSE
        tolerance_loc_2 <- 1E-4
        tolerance_loc_3 <- 1E-2
        tolerance_loc_4 <- 0.5
        loop_cg_PC = c("ssor", "zic")
      } else {
        tolerance_loc_1 <- TOLERANCE_MEDIUM
        tolerance_loc_2 <- TOLERANCE_STRICT
        tolerance_loc_3 <- 1E-4
        tolerance_loc_4 <- TOLERANCE_MEDIUM
        loop_cg_PC = c("ssor")
      }
      for (cg_preconditioner_type in loop_cg_PC) {
        
        ## Two crossed random effects
        y <- Z1%*%b1 + Z2%*%b2 + xi
        
        # Fisher scoring
        gp_model <- fitGPModel(group_data = cbind(group,group2), y = y, matrix_inversion_method = inv_method,
                               params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE, 
                                             cg_preconditioner_type=cg_preconditioner_type, num_rand_vec_trace=100))
        expected_values <- c(0.49792062, 0.02408196, 1.21972166, 0.18357646, 1.06962710, 0.22567292)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),tolerance_loc_1)
        expect_lte(gp_model$get_num_optim_iter(), 6)
        expect_gte(gp_model$get_num_optim_iter(), 4)
        
        # fix some covariance parameters
        params_loc <- list(optimizer_cov = "lbfgs", std_dev = TRUE, 
                           cg_preconditioner_type=cg_preconditioner_type, num_rand_vec_trace=100, 
                           init_cov_pars=c(0.23, 0.1, 0.5), estimate_cov_par_index = c(1,1,0))
        gp_model_fix <- fitGPModel(group_data = cbind(group,group2), y = y, matrix_inversion_method = inv_method,
                                   params = params_loc)
        nll_opt_fix <- 1328.897384
        cov_pars_fix <- c(0.52972794645, 0.02562029714, 1.21929637610, 0.18314784671, 0.50000000000, 0.10972005168)
        expect_lt(sum(abs(as.vector(gp_model_fix$get_cov_pars())-cov_pars_fix)),tolerance_loc_1)
        expect_lt(sum(abs(gp_model_fix$get_cov_pars()[1,3]-params_loc$init_cov_pars[3])),TOLERANCE_STRICT)
        expect_lt(abs(gp_model_fix$get_current_neg_log_likelihood()-nll_opt_fix), tolerance_loc_4)
        
        # Predict training data random effects
        if(inv_method == "cholesky"){
          cov_pars <- gp_model$get_cov_pars()[1,]
          all_training_data_random_effects <- predict_training_data_random_effects(gp_model, 
                                                                                   predict_var = TRUE)
          first_occurences_1 <- match(unique(group), group)
          first_occurences_2 <- match(unique(group2), group2)
          pred_random_effects <- all_training_data_random_effects[first_occurences_1,c(1,3)]
          pred_random_effects_crossed <- all_training_data_random_effects[first_occurences_2,c(2,4)] 
          group_unique <- unique(group)
          group_data_pred = cbind(group_unique,rep(-1,length(group_unique)))
          preds <- predict(gp_model, group_data_pred=group_data_pred,
                           predict_var = TRUE, predict_response = FALSE)
          expect_lt(sum(abs(pred_random_effects[,1] - preds$mu)),TOLERANCE_STRICT)
          expect_lt(sum(abs(pred_random_effects[,2] - (preds$var - cov_pars[3]))),TOLERANCE_STRICT)
          # Check whether crossed random effects are correct
          group_unique <- unique(group2)
          group_data_pred = cbind(rep(-1,length(group_unique)),group_unique)
          preds <- predict(gp_model, group_data_pred=group_data_pred,
                           predict_var = TRUE, predict_response = FALSE)
          expect_lt(sum(abs(pred_random_effects_crossed[,1] - preds$mu)),TOLERANCE_STRICT)
          expect_lt(sum(abs(pred_random_effects_crossed[,2] - (preds$var - cov_pars[2]))),TOLERANCE_STRICT)
        }
        # Prediction after training
        group_data_pred = cbind(c(1,1,m+1),c(2,1,length(group2)+1))
        pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, predict_var = TRUE)
        expected_mu <- c(0.7509175, -0.4208015, 0.0000000)
        expected_var <- c(0.5677178, 0.5677178, 2.7872694)
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),tolerance_loc_3) 
        # Prediction without training and parameters given
        gp_model <- GPModel(group_data = cbind(group,group2), matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params = list(cg_preconditioner_type=cg_preconditioner_type))
        pred <- gp_model$predict(y = y, group_data_pred=group_data_pred,
                                 cov_pars = c(0.1,1,2), predict_cov_mat = TRUE)
        expected_mu <- c(0.7631462, -0.4328551, 0.000000000)
        expected_cov <- c(0.114393721, 0.009406189, 0.0000000, 0.009406189,
                          0.114393721 , 0.0000000, 0.0000000, 0.0000000, 3.100000000)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
        expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),tolerance_loc_3)
        # Prediction for only existing random effects
        group_data_pred_in = cbind(c(1,1),c(2,1))
        pred <- gp_model$predict(y = y, group_data_pred=group_data_pred_in,
                                 cov_pars = c(0.1,1,2), predict_cov_mat = TRUE)
        expected_mu <- c(0.7631462, -0.4328551)
        expected_cov <- c(0.114393721, 0.009406189, 0.009406189, 0.114393721)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
        expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),tolerance_loc_1)
        # Prediction for only new random effects
        group_data_pred_out = cbind(c(m+1,m+1,m+1),c(length(group2)+1,length(group2)+2,length(group2)+1))
        pred <- gp_model$predict(y = y, group_data_pred=group_data_pred_out,
                                 cov_pars = c(0.1,1,2), predict_cov_mat = TRUE)
        expected_mu <- c(rep(0,3))
        expected_cov <- c(3.1, 1.0, 3.0, 1.0, 3.1, 1.0, 3.0, 1.0, 3.1)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
        expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
      }# end loop PC
    } #end loop matrix inversion method
    
    ## Two crossed random effects and a random slope
    y <- Z1%*%b1 + Z2%*%b2 + Z3%*%b3 + xi
    gp_model <- fitGPModel(group_data = cbind(group,group2),
                           group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1,
                           y = y, matrix_inversion_method = "cholesky",
                           params = list(optimizer_cov = "fisher_scoring", maxit=5, std_dev = TRUE))
    expected_values <- c(0.49554952, 0.02546769, 1.24880860, 0.18983953, 1.05505134, 0.22337199, 1.13840014, 0.17950490)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    
    # Predict training data random effects
    cov_pars <- gp_model$get_cov_pars()[1,]
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    first_occurences_1 <- match(unique(group), group)
    first_occurences_2 <- match(unique(group2), group2)
    pred_random_effects <- all_training_data_random_effects[first_occurences_1,c(1,4)]
    pred_random_slopes <- all_training_data_random_effects[first_occurences_1,c(3,6)]
    pred_random_effects_crossed <- all_training_data_random_effects[first_occurences_2,c(2,5)] 
    group_unique <- unique(group)
    group_data_pred = cbind(group_unique,rep(-1,length(group_unique)))
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr, 
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_random_effects[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_random_effects[,2] - (preds$var - cov_pars[3]))),TOLERANCE_STRICT)
    # Check whether random slopes are correct
    x_pr = rep(1,length(group_unique))
    preds2 <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr,
                      predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_random_slopes[,1] - (preds2$mu-preds$mu))),TOLERANCE_STRICT)
    # Check whether crossed random effects are correct
    group_unique <- unique(group2)
    group_data_pred = cbind(rep(-1,length(group_unique)),group_unique)
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_random_effects_crossed[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_random_effects_crossed[,2] - (preds$var - cov_pars[2]))),TOLERANCE_STRICT)
    
    # Prediction
    gp_model <- GPModel(group_data = cbind(group,group2),
                        group_rand_coef_data = x, ind_effect_group_rand_coef = 1, matrix_inversion_method = "cholesky")
    group_data_pred = cbind(c(1,1,m+1),c(2,1,length(group2)+1))
    group_rand_coef_data_pred = c(0,10,0.3)
    expect_error(gp_model$predict(group_data_pred = group_data_pred,
                                  cov_pars = c(0.1,1,2,1.5), y=y))# random slope data not provided
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred,
                             group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.1,1,2,1.5), predict_cov_mat = TRUE)
    expected_mu <- c(0.7579961, -0.2868530, 0.000000000)
    expected_cov <- c(0.11534086, -0.01988167, 0.0000000, -0.01988167, 2.4073302,
                      0.0000000, 0.0000000, 0.0000000, 3.235)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # Predict variances
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred,
                             group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.1,1,2,1.5), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    
    # Gradient descent
    init_cov_pars <- c(var(y)/2,rep(var(y)/2,3))
    gp_model <- fitGPModel(group_data = cbind(group,group2),
                           group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1,
                           y = y, matrix_inversion_method = "cholesky",
                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE, init_cov_pars=init_cov_pars))
    cov_pars <- c(0.4958303, 1.2493181, 1.0543343, 1.1388604 )
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(),8)
    
    # Nelder-Mead
    gp_model <- fitGPModel(group_data = cbind(group,group2),
                           group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1,
                           y = y, matrix_inversion_method = "cholesky",
                           params = list(optimizer_cov = "nelder_mead", std_dev = FALSE, delta_rel_conv=1e-6))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOL_VERY_LOOSE)
    # lbfgs
    gp_model <- fitGPModel(group_data = cbind(group,group2),  group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1, y = y, matrix_inversion_method = "cholesky",
                           params = list(optimizer_cov = "lbfgs", std_dev = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    # Adam
    gp_model <- fitGPModel(group_data = cbind(group,group2),
                           group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1,
                           y = y, matrix_inversion_method = "cholesky",
                           params = list(optimizer_cov = "adam", std_dev = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_LOOSE)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,2,1.5),y=y)
    expect_lt(abs(nll-2335.803),1E-2)
  })
  
  test_that("Random coefficients with intercept random effect dropped ", {
    
    ## A random effect and a random slope without a corresponding intercept effect
    y <- Z2%*%b2 + Z3%*%b3 + xi
    init_cov_pars <- c(var(y)/2,rep(var(y)/2,2))
    gp_model <- fitGPModel(group_data = cbind(group,group2),
                           group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                           drop_intercept_group_rand_effect = c(TRUE,FALSE), y = y, matrix_inversion_method = "cholesky",
                           params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars))
    expected_values <- c(0.5017205, 1.0818474, 1.1157430)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 7)
    
    # Predict training data random effects
    cov_pars <- gp_model$get_cov_pars()
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    first_occurences_1 <- match(unique(group), group)
    first_occurences_2 <- match(unique(group2), group2)
    pred_random_slopes <- all_training_data_random_effects[first_occurences_1,c(2,4)]
    pred_random_effects_crossed <- all_training_data_random_effects[first_occurences_2,c(1,3)] 
    group_unique <- unique(group)
    group_data_pred = cbind(group_unique,rep(-1,length(group_unique)))
    # Check whether random slopes are correct
    x_pr = rep(1,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_random_slopes[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_random_slopes[,2] - (preds$var - cov_pars[2]))),TOLERANCE_STRICT)
    # Check whether crossed random effects are correct
    group_unique <- unique(group2)
    group_data_pred = cbind(rep(-1,length(group_unique)),group_unique)
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_random_effects_crossed[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_random_effects_crossed[,2] - preds$var)),TOLERANCE_STRICT)
    
    # Prediction
    gp_model <- GPModel(group_data = cbind(group,group2),
                        group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                        drop_intercept_group_rand_effect = c(TRUE,FALSE), matrix_inversion_method = "cholesky")
    group_data_pred = cbind(c(1,1,m+1),c(2,1,length(group2)+1))
    group_rand_coef_data_pred = c(0,10,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred,
                             group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.1,2,1.5), predict_cov_mat = TRUE)
    expected_mu <- c(0.8426751, -0.5964363, 0.000000000)
    expected_cov <- c(0.10558205, -0.01269261, 0.00000000, -0.01269261, 2.40180871,
                      0.00000000, 0.00000000, 0.00000000, 2.23500000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # Predict variances
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred,
                             group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.1,2,1.5), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    
  })
  
  test_that("not constant cluster_id's for grouped random effects ", {
    
    y <- Z1 %*% b1 + xi
    capture.output( gp_model <- fitGPModel(group_data = group, cluster_ids = cluster_ids,
                                           y = y,
                                           params = list(optimizer_cov = "fisher_scoring", maxit=100, std_dev = TRUE,
                                                         convergence_criterion = "relative_change_in_parameters"))
                    , file='NUL')
    expected_values <- c(0.49348532, 0.02326312, 1.22299521, 0.17995161)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    
    # gradient descent
    capture.output( gp_model <- fitGPModel(group_data = group, cluster_ids = cluster_ids,
                                           y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = FALSE, maxit = 1000,
                                                         convergence_criterion = "relative_change_in_parameters"))
                    , file='NUL')
    cov_pars_expected <- c(0.49348532, 0.02326312, 1.22299520, 0.17995161)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_expected)),TOLERANCE_STRICT)
    
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences] 
    group_unique <- unique(group)
    pred_random_effects <- predict(gp_model, group_data_pred = group_unique, 
                                   cluster_ids_pred = cluster_ids[first_occurences])
    expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),TOLERANCE_STRICT)
    
    # Prediction
    group_data_pred = c(1,1,m+1)
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(group_data = group, cluster_ids = cluster_ids)
    expect_error(gp_model$predict(group_data_pred = group_data_pred,
                                  cov_pars = c(0.75,1.25), y=y))# cluster_id's not provided
    pred <- gp_model$predict(y = y, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(0.75,1.25), predict_cov_mat = TRUE)
    expected_mu <- c(-0.1514786, 0.000000, 0.000000)
    expected_cov <- c(0.8207547, 0.000000, 0.000000, 0.000000,
                      2.000000, 0.000000, 0.000000, 0.000000, 2.000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # cluster_ids can be string 
    cluster_ids_string <- paste0(as.character(cluster_ids),"_s")
    capture.output( gp_model <- fitGPModel(group_data = group, cluster_ids = cluster_ids_string,
                                           y = y,
                                           params = list(optimizer_cov = "fisher_scoring", maxit=100, std_dev = TRUE,
                                                         convergence_criterion = "relative_change_in_parameters"))
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_expected)),TOLERANCE_STRICT)
    # Prediction
    group_data_pred = c(1,1,m+1)
    cluster_ids_pred_string = paste0(as.character(c(1,3,1)),"_s")
    pred <- gp_model$predict(y = y, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred_string,
                             cov_pars = c(0.75,1.25), predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # Prediction when cluster_ids are not in ascending order
    group_data_pred = c(1,1,m,m)
    cluster_ids_pred = c(2,2,1,2)
    gp_model <- GPModel(group_data = group, cluster_ids = cluster_ids)
    pred <- gp_model$predict(y = y, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(0.75,1.25), predict_var = TRUE)
    expected_mu <- c(rep(0,3), 1.179557)
    expected_var <- c(rep(2,3), 0.8207547)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    # same thing when cluster_ids are strings
    group_data_pred = c(1,1,m,m)
    cluster_ids_pred_string = paste0(as.character(c(2,2,1,2)),"_s")
    gp_model <- GPModel(group_data = group, cluster_ids = cluster_ids_string)
    pred <- gp_model$predict(y = y, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred_string,
                             cov_pars = c(0.75,1.25), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    
    # cluster_ids and random coefficients
    y <- Z1%*%b1 + Z3%*%b3 + xi
    init_cov_pars <- c(var(y)/2,rep(var(y)/2,2))
    capture.output( gp_model <- fitGPModel(group_data = group,
                                           cluster_ids = cluster_ids,
                                           group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1,
                                           y = y, matrix_inversion_method = "cholesky",
                                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE, maxit = 1000, init_cov_pars=init_cov_pars,
                                                         convergence_criterion = "relative_change_in_parameters"))
                    , file='NUL')
    expected_values <- c(0.4927786, 1.2565102, 1.1333662)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 13)
    
  })
  
}
