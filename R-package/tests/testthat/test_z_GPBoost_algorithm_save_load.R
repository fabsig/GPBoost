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
