if(Sys.getenv("NO_GPBOOST_ALGO_TESTS") != "NO_GPBOOST_ALGO_TESTS"){
  
  context("generalized_GPBoost_combined_boosting_GP_random_effects")
  
  TOLERANCE_STRICT <- 1e-6
  TOLERANCE <- 1E-3
  TOLERANCE_LOOSE <- 1E-2
  # The expected values are those of the reference platform. The slow test with the
  # 'gaussian_heteroscedastic_fixed_and_random' likelihood below fits its Gaussian process with a
  # stochastic (iterative) Vecchia Laplace approximation, where a different compiler reaches a
  # visibly different fit: the sums below differ by about 1 between the compilers, so their budgets
  # are widened off the reference platform.
  # See helper-tolerances.R, which defines this and reports it once per test run
  USE_STRICT_TOLERANCES <- gpb_use_strict_tolerances()
  relax_tolerance_stoch <- function(tol) if (USE_STRICT_TOLERANCES) tol else max(10 * tol, 4)
  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov="gradient_descent", use_nesterov_acc=TRUE,
                               delta_rel_conv=1E-6, lr_cov=0.1, lr_coef=0.1,
                               init_coef_aux_pars_from_iid_model = FALSE)
  DEFAULT_OPTIM_PARAMS_V2 <- list(optimizer_cov="gradient_descent", use_nesterov_acc=TRUE,
                                  delta_rel_conv=1E-6, lr_cov=0.01, lr_coef=0.1,
                                  init_coef_aux_pars_from_iid_model = FALSE)
  DEFAULT_OPTIM_PARAMS_NO_NESTEROV <- list(optimizer_cov="gradient_descent", use_nesterov_acc=FALSE,
                                           delta_rel_conv=1E-6, lr_cov=0.01, lr_coef=0.1,
                                           init_coef_aux_pars_from_iid_model = FALSE)
  DEFAULT_OPTIM_PARAMS_EARLY_STOP <- list(maxit=10, lr_cov=0.1, optimizer_cov="gradient_descent", lr_coef=0.1,
                                          init_coef_aux_pars_from_iid_model = FALSE)
  DEFAULT_OPTIM_PARAMS_EARLY_STOP_NO_NESTEROV <- list(maxit=20, lr_cov=0.01, use_nesterov_acc=FALSE,
                                                      optimizer_cov="gradient_descent", lr_coef=0.1,
                                                      init_coef_aux_pars_from_iid_model = FALSE)
  OPTIM_PARAMS_BFGS <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 1000,
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
  sim_friedman3=function(n, n_irrelevant=5, init_c=0.2644234){
    X <- matrix(sim_rand_unif(4*n,init_c=init_c),ncol=4)
    X[,1] <- 100*X[,1]
    X[,2] <- X[,2]*pi*(560-40)+40*pi
    X[,4] <- X[,4]*10+1
    f <- sqrt(10)*atan((X[,2]*X[,3]-1/(X[,2]*X[,4]))/X[,1])
    X <- cbind(rep(1,n),X)
    if(n_irrelevant>0) X <- cbind(X,matrix(sim_rand_unif(n_irrelevant*n,init_c=0.6543),ncol=n_irrelevant))
    return(list(X=X,f=f))
  }
  f1d <- function(x) 2*(1.5*(1/(1+exp(-(x-0.5)*20))+0.75*x)-1.3)
  sim_non_lin_f=function(n, init_c=0.4596534){
    X <- matrix(sim_rand_unif(2*n,init_c=init_c),ncol=2)
    f <- f1d(X[,1])
    return(list(X=X,f=f))
  }
  
  # Make plot of fitted boosting ensemble ("manual test")
  n <- 1000
  m <- 100
  sim_data <- sim_non_lin_f(n=n)
  group <- rep(1,n) # grouping variable
  for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
  b1 <- qnorm(sim_rand_unif(n=m, init_c=0.3242))
  eps <- b1[group]
  eps <- eps - mean(eps)
  probs <- pnorm(sim_data$f+eps)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.6352) < probs)
  
  nrounds <- 200
  learning_rate <- 0.2
  min_data_in_leaf <- 50
  gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
  bst <- gpboost(data = sim_data$X, label = y, gp_model = gp_model,
                 objective = "binary", nrounds=200, learning_rate=learning_rate,
                 train_gp_model_cov_pars=TRUE, min_data_in_leaf=min_data_in_leaf,verbose=0,
                 metric="approx_neg_marginal_log_likelihood")
  # summary(gp_model)
  nplot <- 200
  X_test_plot <- cbind(seq(from=0,to=1,length.out=nplot),rep(0.5,nplot))
  group_data_pred <- rep(-9999,dim(X_test_plot)[1])
  pred_prob <- predict(bst, data = X_test_plot, group_data_pred = group_data_pred, pred_latent = FALSE)$response_mean
  pred <- predict(bst, data = X_test_plot, group_data_pred = group_data_pred, pred_latent = TRUE)
  x <- seq(from=0,to=1,length.out=200)
  plot(x,f1d(x),type="l",lwd=3,col=2,main="Data, true and fitted function")
  points(sim_data$X[,1],y)
  lines(X_test_plot[,1],pred$fixed_effect,col=4,lwd=3)
  lines(X_test_plot[,1],pred_prob,col=3,lwd=3)
  legend(legend=c("True","Pred F","Pred p"),"bottomright",bty="n",lwd=3,col=c(2,4,3))
  
  # ## Compare to independent boosting
  # bst_std <- gpboost(data = sim_data$X, label = y,verbose=0,
  #                objective = "binary", nrounds=200, learning_rate=learning_rate,
  #                train_gp_model_cov_pars=FALSE, min_data_in_leaf=min_data_in_leaf)
  # pred <- predict(bst_std, data = X_test_plot, pred_latent=TRUE)
  # lines(X_test_plot[,1],pred,col=5,lwd=3, lty=2)
  
  
  # Avoid that long tests get executed on CRAN
  if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
    
    test_that("Saving and loading a booster with a gp_model for non-Gaussian data ", {
      
      ntrain <- ntest <- 1000
      n <- ntrain + ntest
      # Simulate fixed effects
      sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.2644234)
      f <- sim_data$f
      f <- f - mean(f)
      X <- sim_data$X
      # Simulate grouped random effects
      sigma2_1 <- 0.6 # variance of first random effect
      sigma2_2 <- 0.4 # variance of second random effect
      sigma2 <- 0.1^2 # error variance
      m <- 40 # number of categories / levels for grouping variable
      # first random effect
      group <- rep(1,ntrain) # grouping variable
      for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
      group <- c(group, group)
      n_new <- 3# number of new random effects in test data
      group[(length(group)-n_new+1):length(group)] <- rep(99999,n_new)
      Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
      b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.5542))
      # Second random effect
      n_obs_gr <- ntrain/m# number of sampels per group
      group2 <- rep(1,ntrain) # grouping variable
      for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
      group2 <- c(group2,group2)
      group2[(length(group2)-n_new+1):length(group2)] <- rep(99999,n_new)
      Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
      b2 <- sqrt(sigma2_2) * qnorm(sim_rand_unif(n=length(unique(group2)), init_c=0.82354))
      eps <- Z1 %*% b1 + Z2 %*% b2
      eps <- eps - mean(eps)
      group_data <- cbind(group,group2)
      # Observed data
      probs <- pnorm(f + eps)
      y <- as.numeric(sim_rand_unif(n=n, init_c=0.574) < probs)
      # Split in training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      group_data_train <- group_data[1:ntrain,]
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      group_data_test <- group_data[1:ntest+ntrain,]
      
      # Train model and make predictions
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_NO_NESTEROV)
      bst <- gpboost(data = X_train,
                     label = y_train,
                     gp_model = gp_model,
                     nrounds = 30,
                     learning_rate = 0.1,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
      # Predict raw score and response
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                      predict_var = TRUE, pred_latent = TRUE)
      pred_resp <- predict(bst, data = X_test, group_data_pred = group_data_test,
                           predict_var = TRUE, pred_latent = FALSE)
      pred2 <- predict(bst, data = X_test, group_data_pred = group_data_test,
                       predict_var = TRUE, pred_latent = TRUE,
                       num_iteration = 22, start_iteration = 0)
      pred_resp2 <- predict(bst, data = X_test, group_data_pred = group_data_test,
                            predict_var = TRUE, pred_latent = FALSE,
                            num_iteration = 22, start_iteration = 0)
      pred3 <- predict(bst, data = X_test, group_data_pred = group_data_test,
                       predict_var = TRUE, pred_latent = TRUE,
                       num_iteration = 20, start_iteration = 5)
      pred_resp3 <- predict(bst, data = X_test, group_data_pred = group_data_test,
                            predict_var = TRUE, pred_latent = FALSE,
                            num_iteration = 20, start_iteration = 5)
      # Save to file
      filename <- tempfile(fileext = ".model")
      gpb.save(bst, filename=filename, save_raw_data = FALSE)
      filename_num_it <- tempfile(fileext = ".model")
      gpb.save(bst, filename=filename_num_it, save_raw_data = FALSE, num_iteration = 22, start_iteration = 0)
      filename2 <- tempfile(fileext = ".model")
      gpb.save(bst, filename=filename2, save_raw_data = TRUE)
      # finalize and destroy models
      cov_pars_before_save <- as.vector(gp_model$get_cov_pars(std_err = FALSE))
      bst$.__enclos_env__$private$finalize()
      expect_null(bst$.__enclos_env__$private$handle)
      rm(bst)
      rm(gp_model)
      # Load from file and make predictions again with save_raw_data = FALSE option
      bst_loaded <- gpb.load(filename = filename)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var = TRUE, pred_latent = TRUE)
      pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                                  predict_var = TRUE, pred_latent = FALSE)
      expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
      expect_equal(pred_resp$response_mean, pred_resp_loaded$response_mean)
      expect_equal(pred_resp$response_var, pred_resp_loaded$response_var)
      expect_lt(sum(abs(cov_pars_before_save - as.vector(bst_loaded$.__enclos_env__$private$gp_model$get_cov_pars(std_err = FALSE)))),TOLERANCE_STRICT)
      # Different num_iteration when saving
      bst_loaded <- gpb.load(filename = filename_num_it)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var = TRUE, pred_latent = TRUE)
      pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                                  predict_var = TRUE, pred_latent = FALSE)
      expect_equal(pred2$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred2$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred2$random_effect_cov, pred_loaded$random_effect_cov)
      expect_equal(pred_resp2$response_mean, pred_resp_loaded$response_mean)
      expect_equal(pred_resp2$response_var, pred_resp_loaded$response_var)
      expect_error({
        pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                               predict_var= TRUE, start_iteration=5)
      })
      # Load from file and make predictions again with save_raw_data = TRUE option
      bst_loaded <- gpb.load(filename = filename2)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var= TRUE, pred_latent = TRUE)
      pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                                  predict_var= TRUE, pred_latent = FALSE)
      expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
      expect_equal(pred_resp$response_mean, pred_resp_loaded$response_mean)
      expect_equal(pred_resp$response_var, pred_resp_loaded$response_var)
      expect_lt(sum(abs(cov_pars_before_save - as.vector(bst_loaded$.__enclos_env__$private$gp_model$get_cov_pars(std_err = FALSE)))),TOLERANCE_STRICT)
      # A loaded model is also used in the GPBoost algorithm and thus has no standard errors either
      gp_model_loaded <- bst_loaded$.__enclos_env__$private$gp_model
      expect_false(gp_model_loaded$can_calculate_standard_errors_cov_pars())
      expect_equal(length(as.vector(gp_model_loaded$get_cov_pars(std_err = TRUE))),
                   length(as.vector(gp_model_loaded$get_cov_pars(std_err = FALSE))))
      # Same num_iteration when saving but different one for prediction
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var = TRUE, pred_latent = TRUE, num_iteration = 22, start_iteration = 0)
      pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                                  predict_var = TRUE, pred_latent = FALSE, num_iteration = 22, start_iteration = 0)
      expect_equal(pred2$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred2$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred2$random_effect_cov, pred_loaded$random_effect_cov)
      expect_equal(pred_resp2$response_mean, pred_resp_loaded$response_mean)
      expect_equal(pred_resp2$response_var, pred_resp_loaded$response_var)
      # Set num_iteration and start_iteration
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var= TRUE, pred_latent = TRUE,
                             num_iteration = 20, start_iteration = 5)
      pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                                  predict_var= TRUE, pred_latent = FALSE,
                                  num_iteration = 20, start_iteration = 5)
      expect_equal(pred3$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred3$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred3$random_effect_cov, pred_loaded$random_effect_cov)
      expect_equal(pred_resp3$response_mean, pred_resp_loaded$response_mean)
      expect_equal(pred_resp3$response_var, pred_resp_loaded$response_var)
      
    })
    
  }
}
