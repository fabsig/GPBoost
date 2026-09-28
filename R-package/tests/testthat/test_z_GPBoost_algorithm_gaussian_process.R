if(Sys.getenv("NO_GPBOOST_ALGO_TESTS") != "NO_GPBOOST_ALGO_TESTS"){
  
  context("GPBoost_combined_boosting_GP_random_effects")
  
  TOLERANCE_STRICT <- 1e-6
  TOLERANCE <- 1E-3
  TOLERANCE2 <- 1E-2
  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov="fisher_scoring", delta_rel_conv=1E-6,
                               init_coef_aux_pars_from_iid_model = FALSE)
  DEFAULT_OPTIM_PARAMS_iterative <- list(maxit = 10,
                                         delta_rel_conv = 1e-2,
                                         optimizer_cov = "gradient_descent",
                                         cg_delta_conv = 1e-8,
                                         cg_preconditioner_type = "predictive_process_plus_diagonal",
                                         cg_max_num_it = 1000,
                                         cg_max_num_it_tridiag = 1000,
                                         num_rand_vec_trace = 1000,
                                         reuse_rand_vec_trace = TRUE, init_coef_aux_pars_from_iid_model = FALSE)
  OPTIM_PARAMS_GRAD_DESC <- list(optimizer_cov = "gradient_descent",
                                 lr_cov = 0.1, use_nesterov_acc = TRUE,
                                 acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                 optimizer_coef = "gradient_descent", lr_coef = 0.1,
                                 convergence_criterion = "relative_change_in_log_likelihood",
                                 init_coef_aux_pars_from_iid_model = FALSE)
  # Function that simulates uniform random variables
  sim_rand_unif <- function(n, init_c=0.1){
    mod_lcg <- 134456 # modulus for linear congruential generator (random0 used)
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for(i in 2:n) sim[i] <- (8121 * sim[i-1] + 28411) %% mod_lcg
    return(sim / mod_lcg)
  }
  # Function for non-linear mean
  sim_friedman3=function(n, n_irrelevant=5){
    X <- matrix(sim_rand_unif(4*n,init_c=0.24234),ncol=4)
    X[,1] <- 100*X[,1]
    X[,2] <- X[,2]*pi*(560-40)+40*pi
    X[,4] <- X[,4]*10+1
    f <- sqrt(10)*atan((X[,2]*X[,3]-1/(X[,2]*X[,4]))/X[,1])
    X <- cbind(rep(1,n),X)
    if(n_irrelevant>0) X <- cbind(X,matrix(sim_rand_unif(n_irrelevant*n,init_c=0.6543),ncol=n_irrelevant))
    return(list(X=X,f=f))
  }
  
  f1d <- function(x) 1.5*(1/(1+exp(-(x-0.5)*20))+0.75*x)
  sim_non_lin_f=function(n){
    X <- matrix(sim_rand_unif(2*n,init_c=0.96534),ncol=2)
    f <- f1d(X[,1])
    return(list(X=X,f=f))
  }
  
  # Make plot of fitted boosting ensemble ("manual test")
  n <- 1000
  m <- 100
  sim_data <- sim_non_lin_f(n=n)
  group <- rep(1,n) # grouping variable
  for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
  b1 <- qnorm(sim_rand_unif(n=m, init_c=0.943242))
  eps <- b1[group]
  eps <- eps - mean(eps)
  y <- sim_data$f + eps + 0.1^2*sim_rand_unif(n=n, init_c=0.32543)
  gp_model <- GPModel(group_data = group)
  bst <- gpboost(data = sim_data$X, label = y, gp_model = gp_model,
                 nrounds = 100, learning_rate = 0.05, max_depth = 6,
                 min_data_in_leaf = 5, objective = "regression_l2", verbose = 0,
                 leaves_newton_update = TRUE)
  nplot <- 200
  X_test_plot <- cbind(seq(from=0,to=1,length.out=nplot),rep(0.5,nplot))
  pred <- predict(bst, data = X_test_plot, group_data_pred = rep(-9999,nplot), 
                  pred_latent = TRUE)
  x <- seq(from=0,to=1,length.out=200)
  plot(x,f1d(x),type="l",lwd=3,col=2,main="True and fitted function")
  lines(X_test_plot[,1],pred$fixed_effect,col=4,lwd=3)
  
  # Avoid that long tests get executed on CRAN
  if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
    
    test_that("Combine tree-boosting and Gaussian process model ", {
      
      ntrain <- ntest <- 500
      n <- ntrain + ntest
      # Simulate fixed effects
      sim_data <- sim_friedman3(n=n, n_irrelevant=5)
      f <- sim_data$f
      X <- sim_data$X
      # Simulate grouped random effects
      sigma2_1 <- 1 # marginal variance of GP
      rho <- 0.1 # range parameter
      sigma2 <- 0.1 # error variance
      d <- 2 # dimension of GP locations
      coords <- matrix(sim_rand_unif(n=n*d, init_c=0.63), ncol=d)
      D <- as.matrix(dist(coords))
      Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
      C <- t(chol(Sigma))
      b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.864))
      eps <- as.vector(C %*% b_1)
      # Error term
      xi <- sqrt(sigma2) * qnorm(sim_rand_unif(n=n, init_c=0.36))
      # Observed data
      y <- f + eps + xi
      # Split in training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      coords_train <- coords[1:ntrain,]
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      coords_test <- coords[1:ntest+ntrain,]
      dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
      valids <- list(test = dtest)
      
      init_cov_pars <- c(var(y_train)/2,var(y_train)/2,mean(dist(coords_train))/3)
      
      # Train model
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring", init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain, gp_model = gp_model,
                       nrounds = 20, learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      cov_pars_est <- c(0.1358229, 0.9099908, 0.1115316)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                      predict_var=TRUE, pred_latent = TRUE)
      pred_re <- c(0.19200894, 0.08380017, 0.59402383, -0.75484438)
      pred_fe <- c(3.920440, 3.641091, 4.536346, 4.951052)
      pred_var <- c(0.3612252, 0.1596113, 0.1664702, 0.2577366)
      pred_var_resp <- pred_var + cov_pars_est[1]
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_var)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.5229658),TOLERANCE)
      expect_lt(abs(sqrt(mean((pred$fixed_effect - y_test)^2))-1.170505),TOLERANCE)
      # Response prediction
      expect_lt(abs(sqrt(mean((pred$fixed_effect + pred$random_effect_mean - y_test)^2))-0.8304062),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean, n=4)-(pred_re+pred_fe))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$response_var, n=4)-pred_var_resp)),TOLERANCE)
      # Predictive covariance
      cov_exp <- c(1.596106e-01, -2.333094e-09, -5.746473e-08, -2.333094e-09, 1.664696e-01, 
                   -1.303500e-05, -5.746473e-08, -1.303500e-05, 2.577362e-01)
      cov_exp_resp <- cov_exp
      cov_exp_resp[c(1,5,9)] <- cov_exp_resp[c(1,5,9)] + cov_pars_est[1]
      pred <- predict(bst, data =  tail(X_test,n=3), gp_coords_pred = tail(coords_test,n=3), 
                      predict_cov_mat=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=3)-tail(pred_re,n=3))),TOLERANCE)
      expect_lt(sum(abs(as.vector(pred$random_effect_cov)-cov_exp)),TOLERANCE)
      pred <- predict(bst, data =  tail(X_test,n=3), gp_coords_pred = tail(coords_test,n=3), 
                      predict_cov_mat=TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean, n=4)-tail(pred_re+pred_fe,n=3))),TOLERANCE)
      expect_lt(sum(abs(as.vector(pred$response_var)-cov_exp_resp)),TOLERANCE)
      # Sampling from the posterior
      pred <- predict(bst, data =  tail(X_test,n=3), gp_coords_pred = tail(coords_test,n=3), 
                      sample_posterior = TRUE, num_post_samples = 10000, pred_latent = TRUE)
      tol_mu <- 0.02
      expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-tail(pred_re+pred_fe,n=3))), tol_mu)
      expect_lt(sum(abs(cov(t(pred$posterior_samples))-cov_exp)), tol_mu)
      pred <- predict(bst, data =  tail(X_test,n=3), gp_coords_pred = tail(coords_test,n=3), 
                      sample_posterior = TRUE, num_post_samples = 10000, pred_latent = FALSE)
      expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-tail(pred_re+pred_fe,n=3))),tol_mu)
      expect_lt(sum(abs(cov(t(pred$posterior_samples))-cov_exp_resp)), 0.03)
      # Use other covariance parameters for prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                      predict_var=TRUE, pred_latent = TRUE, cov_pars = c(0.1358229, 0.9099908, 0.1115316))
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_var)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                      predict_var=TRUE, pred_latent = TRUE, cov_pars = c(0.2, 1.5, 0.2))
      pred_re2 <- c(0.2182825, 0.1131264, 0.5737999, -0.7441675)
      pred_var2 <- c(0.3540400, 0.1704857, 0.1720302, 0.2562620)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re2)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_var2)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      
      # Train model using Nelder-Mead
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain, gp_model = gp_model,
                       nrounds = 20, learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.1286928, 0.9140254, 0.1097192))),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(0.17291900, 0.09483055, 0.64271850, -0.78676614))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.3667703, 0.1596594, 0.1672984, 0.2607827))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(3.840684, 3.688580, 4.591930, 4.976685))),TOLERANCE)
      
      # Use validation set to determine number of boosting iteration
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring", init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model,
                       nrounds = 100, learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0,
                       valids = valids, early_stopping_rounds = 5, use_gp_model_for_validation = FALSE,
                       seed = 0, metric = "l2") , file='NUL')
      expect_equal(bst$best_iter, 27)
      expect_lt(abs(bst$best_score - 1.293498),TOLERANCE)
      
      # Also use GPModel for calculating validation error
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring", init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      gp_model$set_prediction_data(gp_coords_pred = coords_test)
      capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model,
                       nrounds = 100, learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0,
                       valids = valids, early_stopping_rounds = 5, use_gp_model_for_validation = TRUE,
                       seed = 0, metric = "l2") , file='NUL')
      expect_equal(bst$best_iter, 27)
      expect_lt(abs(bst$best_score - 0.5485127),TOLERANCE2)
    })
    
  }
  
}
