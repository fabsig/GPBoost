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
    
    test_that("GPBoost algorithm with Vecchia approximation and Wendland covariance", {
      
      ntrain <- ntest <- 100
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
      
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      params_gp <- list(maxit=100, optimizer_cov="gradient_descent", use_nesterov_acc = TRUE, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)
      gp_model$set_optim_params(params=params_gp)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2",
                       verbose = 0)
      cov_pars_est <- c(0.25092222818, 0.89280688318, 0.08302442786)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      pred_re <- c(-0.4977031114, -0.7868691089, -0.5953274636, -0.2458193940)
      pred_cov <- c(0.4779545982, 0.5962427309, 0.6227537278, 0.8390838534)
      pred_fe <- c(4.682603619, 4.534533783, 4.602049911, 4.457454183)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_cov)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      
      # Same thing with Vecchia approximation
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "vecchia", num_neighbors = ntrain-1, 
                                          vecchia_ordering = "none"), file='NUL')
      gp_model$set_optim_params(params=params_gp)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred = ntrain+ntest-1)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_cov)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      
      # Same thing with Vecchia approximation and random ordering
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "vecchia", num_neighbors = ntrain-1, 
                                          vecchia_ordering = "random"), file='NUL')
      gp_model$set_optim_params(params=params_gp)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred = ntrain+ntest-1)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_cov)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      
      # Same thing with Vecchia approximation and Nelder-Mead
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "vecchia", num_neighbors = ntrain-1,
                                          vecchia_ordering = "none"), file='NUL')
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.24097347, 0.88916662, 0.08253709))),TOLERANCE)
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred = ntrain+ntest-1)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(-0.4969191, -0.7867247, -0.5883281, -0.2374269))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.4761964, 0.5945182, 0.6208525, 0.8364343))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(4.679265, 4.562299, 4.570425, 4.392607))),TOLERANCE)
      
      # Vecchia approximation, less neighbors, and validation data: can call 'set_prediction_data' multiple times
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "vecchia", num_neighbors = 20, 
                                          vecchia_ordering = "random"), file='NUL')
      gp_model$set_prediction_data(gp_coords_pred = coords_test)
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred = 100)
      gp_model$set_optim_params(params=params_gp)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5, 
                       objective = "regression_l2", verbose = 0, valids = valids, metric="mse")
      iter <- 20
      score <- 1.54475
      cov_pars_estV <- c(0.26721270772, 0.89424739300, 0.08439964419)
      expect_equal(bst$best_iter, iter)
      expect_lt(abs(bst$best_score - score),TOLERANCE2)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_estV)),0.05)
      # Can also first set vecchia_pred_type and then gp_coords_pred
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "vecchia", num_neighbors = 20, 
                                          vecchia_ordering = "random"), file='NUL')
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred = 100)
      gp_model$set_prediction_data(gp_coords_pred = coords_test)
      gp_model$set_optim_params(params=params_gp)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5, 
                       objective = "regression_l2", verbose = 0, valids = valids, metric="mse")
      expect_equal(bst$best_iter, iter)
      expect_lt(abs(bst$best_score - score),TOLERANCE2)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_estV)),0.05)
      
      # Same thing with Wendland covariance function
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "wendland",
                                          cov_fct_taper_shape = 1, cov_fct_taper_range = 0.2), file='NUL')
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring", init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.3493528, 0.7810089))),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$fixed_effect)-c(4.569245, 4.833311, 4.565894, 4.644225, 4.616655, 4.409673))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_mean)-c(0.01965535, -0.01853082, -0.53218816, -0.98668655, -0.60581078, -0.03390602))),TOLERANCE)
      # Wendland covariance and Nelder-Mead
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "wendland",
                                          cov_fct_taper_shape = 1, cov_fct_taper_range = 0.2), file='NUL')
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.3489301, 0.7817690))),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$fixed_effect)-c(4.569268, 4.833340, 4.565855, 4.644194, 4.616647, 4.409668))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_mean)-c(0.01963911, -0.01852577, -0.53242988, -0.98747505, -0.60616534, -0.03392700))),TOLERANCE)
      
      # Tapering
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "tapering",
                                          cov_fct_taper_shape = 1, cov_fct_taper_range = 20), file='NUL')
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring", init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.24807538, 0.89147953, 0.08303885))),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(-0.4983809, -0.7873952, -0.5955610, -0.2461420))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.4767139, 0.5949467, 0.6214302, 0.8377825))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(4.683095, 4.534749, 4.602275, 4.457237))),TOLERANCE)
      # Tapering and Nelder-Mead
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "tapering",
                                          cov_fct_taper_shape = 1, cov_fct_taper_range = 10), file='NUL')
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.2386092, 0.9050819, 0.0835053 ))),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(-0.4893557, -0.7984212, -0.5994199, -0.2511335))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(4.650092, 4.574518, 4.618443, 4.409184))),TOLERANCE)
    })
    
    test_that("GPBoost algorithm with fitc", {
      ntrain <- ntest <- 100
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
      
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "matern", cov_fct_shape = 1.5,
                                          gp_approx = "fitc",num_ind_points = 50), file='NUL')
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="gradient_descent", init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.009778865, 1.142124739, 0.072746954))),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(-1.09009608, -1.02661256, -1.06180549, -0.04424235))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.3644896470, 0.6872674831, 0.5800297063, 1.1356006965))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(4.255524, 4.807404, 4.659824, 4.499290))),TOLERANCE)
    })
    
    test_that("GPBoost algorithm with FSA", {
      ntrain <- ntest <- 100
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
      
      vec_chol_or_iterative <- c("cholesky","iterative")
      for (i in vec_chol_or_iterative) {
        if(i == "iterative"){
          params <- DEFAULT_OPTIM_PARAMS_iterative
        } else{
          params <- list(maxit=10, optimizer_cov="gradient_descent", delta_rel_conv = 1e-2, init_coef_aux_pars_from_iid_model = FALSE)
        }
        params$init_cov_pars <- init_cov_pars
        
        capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "matern", cov_fct_shape = 1.5,
                                            gp_approx = "full_scale_tapering",num_ind_points = 50, cov_fct_taper_shape = 2, 
                                            cov_fct_taper_range = 0.5, matrix_inversion_method = i), file='NUL')
        gp_model$set_optim_params(params=params)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                         learning_rate = 0.05, max_depth = 6,
                         min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.49224227, 0.69948047, 0.08842094))),TOLERANCE2)
        if(i == "iterative"){
          gp_model$set_prediction_data(cg_delta_conv_pred = 1e-6, nsim_var_pred = 500)
        }
        pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(-0.4672591, -0.8086326, -0.6178553, -0.1621476))),1.5*TOLERANCE2)
        expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.2624237699, 0.3784147773, 0.3964287460, 0.6761869249))),1.5*TOLERANCE2)
        expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(4.683135, 4.608892, 4.571550, 4.406394))),TOLERANCE2)
      }
    })
    
    test_that("Saving and loading a booster with a gp_model from a file and from a string", {
      ntrain <- ntest <- 1000
      n <- ntrain + ntest
      # Simulate fixed effects
      sim_data <- sim_friedman3(n=n, n_irrelevant=5)
      f <- sim_data$f
      X <- sim_data$X
      # Simulate grouped random effects
      m <- 40 # number of categories / levels for grouping variable
      # first random effect
      group <- rep(1,ntrain) # grouping variable
      for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
      group <- c(group, group)
      n_new <- 3# number of new random effects in test data
      group[(length(group)-n_new+1):length(group)] <- rep(max(group)+1,n_new)
      b1 <- qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.542))
      eps <- b1[group]
      group_data <- group
      # Error term
      xi <- sqrt(0.01) * qnorm(sim_rand_unif(n=n, init_c=0.756))
      # Observed data
      y <- f + eps + xi
      # Split in training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      group_data_train <- group_data[1:ntrain]
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      group_data_test <- group_data[1:ntest+ntrain]
      params <- list(learning_rate = 0.01, max_depth = 6, min_data_in_leaf = 5,
                     objective = "regression_l2", feature_pre_filter = FALSE)
      # Train model and make predictions
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                     nrounds = 62, learning_rate = 0.01, max_depth = 6,
                     min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                      predict_var = TRUE, pred_latent = TRUE)
      num_iteration <- 50
      start_iteration <- 0# saving and loading with start_iteration!=0 currently does not work for the LightGBM part
      pred_num_it <- predict(bst, data = X_test, group_data_pred = group_data_test,
                             predict_var = TRUE, num_iteration = num_iteration, start_iteration = start_iteration, pred_latent = TRUE)
      pred_num_it2 <- predict(bst, data = X_test, group_data_pred = group_data_test,
                              predict_var = TRUE, num_iteration = 45, start_iteration = 10, pred_latent = TRUE)
      # Save to file
      filename <- tempfile(fileext = ".model")
      gpb.save(bst, filename=filename, save_raw_data = FALSE)
      filename_num_it <- tempfile(fileext = ".model")
      gpb.save(bst, filename=filename_num_it, save_raw_data = FALSE,
               num_iteration = num_iteration, start_iteration = start_iteration)
      filename_save_raw_data <- tempfile(fileext = ".model")
      gpb.save(bst, filename=filename_save_raw_data, save_raw_data = TRUE)
      # finalize and destroy models
      cov_pars_before_save <- as.vector(gp_model$get_cov_pars(std_err = TRUE))
      bst$.__enclos_env__$private$finalize()
      expect_null(bst$.__enclos_env__$private$handle)
      rm(bst)
      rm(gp_model)
      # Load from file and make predictions again with save_raw_data = FALSE option
      bst_loaded <- gpb.load(filename = filename)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test, predict_var= TRUE, pred_latent = TRUE)
      expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
      expect_lt(sum(abs(cov_pars_before_save - as.vector(bst_loaded$.__enclos_env__$private$gp_model$get_cov_pars(std_err = TRUE)))),TOLERANCE_STRICT)
      # Set num_iteration and start_iteration
      bst_loaded <- gpb.load(filename = filename_num_it)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test, predict_var= TRUE, pred_latent = TRUE)
      expect_equal(pred_num_it$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred_num_it$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred_num_it$random_effect_cov, pred_loaded$random_effect_cov)
      expect_error({
        pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                               predict_var= TRUE, start_iteration=5, pred_latent = TRUE)
      })
      # Load from file and make predictions again with save_raw_data = TRUE option
      capture.output( bst_loaded <- gpb.load(filename = filename_save_raw_data), file='NUL')
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var= TRUE, pred_latent = TRUE)
      expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
      expect_lt(sum(abs(cov_pars_before_save - as.vector(bst_loaded$.__enclos_env__$private$gp_model$get_cov_pars(std_err = TRUE)))),TOLERANCE_STRICT)
      # Set num_iteration and start_iteration
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var= TRUE, num_iteration = num_iteration, start_iteration = start_iteration, pred_latent = TRUE)
      expect_equal(pred_num_it$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred_num_it$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred_num_it$random_effect_cov, pred_loaded$random_effect_cov)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var= TRUE, num_iteration = 45, start_iteration = 10, pred_latent = TRUE)
      expect_equal(pred_num_it2$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred_num_it2$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred_num_it2$random_effect_cov, pred_loaded$random_effect_cov)
      
      # Saving also works with Nesterov-accelerated boosting
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                     nrounds = 62, learning_rate = 0.01, max_depth = 6,
                     min_data_in_leaf = 5, objective = "regression_l2", verbose = 0,
                     use_nesterov_acc = TRUE, momentum_offset = 10)
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                      predict_var = TRUE, pred_latent = TRUE)
      # Save to file
      filename <- tempfile(fileext = ".model")
      gpb.save(bst, filename=filename, save_raw_data = FALSE)
      # finalize and destroy models
      bst$.__enclos_env__$private$finalize()
      expect_null(bst$.__enclos_env__$private$handle)
      rm(bst)
      rm(gp_model)
      # Load from file and make predictions again with save_raw_data = FALSE option
      bst_loaded <- gpb.load(filename = filename)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test, 
                             predict_var= TRUE, pred_latent = TRUE)
      expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
      
      # Saving to string and loading
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                     nrounds = 62, learning_rate = 0.01, max_depth = 6,
                     min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                      predict_var = TRUE, pred_latent = TRUE)
      num_iteration <- 50
      start_iteration <- 0# saving and loading with start_iteration!=0 currently does not work for the LightGBM part
      pred_num_it <- predict(bst, data = X_test, group_data_pred = group_data_test,
                             predict_var = TRUE, num_iteration = num_iteration, start_iteration = start_iteration, pred_latent = TRUE)
      pred_num_it2 <- predict(bst, data = X_test, group_data_pred = group_data_test,
                              predict_var = TRUE, num_iteration = 45, start_iteration = 10, pred_latent = TRUE)
      # Save to string
      model_str <- bst$save_model_to_string(save_raw_data = FALSE)
      model_str_num_it <- bst$save_model_to_string(num_iteration = num_iteration, 
                                                   start_iteration = start_iteration)
      model_str_raw_data <- bst$save_model_to_string(save_raw_data = TRUE)
      # finalize and destroy models
      bst$.__enclos_env__$private$finalize()
      expect_null(bst$.__enclos_env__$private$handle)
      rm(bst)
      rm(gp_model)
      # Load from file and make predictions again with save_raw_data = FALSE option
      bst_loaded <- gpb.load(model_str = model_str)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test, 
                             predict_var= TRUE, pred_latent = TRUE)
      expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
      # Set num_iteration and start_iteration
      bst_loaded <- gpb.load(model_str = model_str_num_it)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test, 
                             predict_var= TRUE, pred_latent = TRUE)
      expect_equal(pred_num_it$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred_num_it$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred_num_it$random_effect_cov, pred_loaded$random_effect_cov)
      expect_error({
        pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                               predict_var= TRUE, start_iteration=5, pred_latent = TRUE)
      })
      # Load from file and make predictions again with save_raw_data = TRUE option
      capture.output( bst_loaded <- gpb.load(model_str = model_str_raw_data), file='NUL')
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var= TRUE, pred_latent = TRUE)
      expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
      # Set num_iteration and start_iteration
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var= TRUE, num_iteration = num_iteration, start_iteration = start_iteration, pred_latent = TRUE)
      expect_equal(pred_num_it$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred_num_it$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred_num_it$random_effect_cov, pred_loaded$random_effect_cov)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var= TRUE, num_iteration = 45, start_iteration = 10, pred_latent = TRUE)
      expect_equal(pred_num_it2$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred_num_it2$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred_num_it2$random_effect_cov, pred_loaded$random_effect_cov)
    })
  }
  
}
