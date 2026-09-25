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
    
    test_that("GPBoost algorithm: large data and 'reuse_learning_rates_gp_model' and 'line_search_step_length' options", {
      
      n <- 1e5
      X_train <- matrix(sim_rand_unif(n=2*n, init_c=0.135), ncol=2)
      # Simulate grouped random effects
      sigma2_1 <- 0.6 # variance of first random effect 
      sigma2 <- 0.1^2 # error variance
      m <- n / 100 # number of categories / levels for grouping variable
      group <- rep(1,n) # grouping variable
      for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
      b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.462))
      eps <- b1[group]
      xi <- sqrt(sigma2) * qnorm(sim_rand_unif(n=n, init_c=0.17556))
      xi[xi<(-100)] = 0
      y <- eps + xi
      params <- list(learning_rate = 0.01,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "regression_l2",
                     feature_pre_filter = FALSE,
                     seed = 1)
      set.seed(1)
      # For CV
      ycv <- y + X_train %*% c(1,1)
      params_cv <- params
      params_cv$learning_rate = 0.2
      dtrain <- gpb.Dataset(data = X_train, label = ycv)
      folds <- list()
      for(i in 1:4) folds[[i]] <- as.integer(1:(n/4) + (n/4) * (i-1))
      
      #################
      ### Tests for 'reuse_learning_rates_gp_model'
      #################
      # Check whether the option "reuse_learning_rates_gp_model" is used or not
      gp_model <- GPModel(group_data = group)
      params_loc <- OPTIM_PARAMS_GRAD_DESC
      params_loc$trace = TRUE
      set_optim_params(gp_model, params=params_loc)
      output <- capture.output( bst <- gpboost(data = X_train, label = y, gp_model = gp_model,
                                               nrounds = 2, params = params, verbose = 0, 
                                               reuse_learning_rates_gp_model = FALSE) )
      str <- output[length(output)-3]
      nb_ll_eval <- as.numeric(substr(str, nchar(str)-2, nchar(str)-2))
      expect_equal(nb_ll_eval, 6)
      # expect_gt(nb_ll_eval, 5)
      # expect_lt(nb_ll_eval, 8)
      # same thing with reuse_learning_rates_gp_model = TRUE
      gp_model <- GPModel(group_data = group)
      set_optim_params(gp_model, params=params_loc)
      output <- capture.output( bst <- gpboost(data = X_train, label = y, gp_model = gp_model,
                                               nrounds = 2, params = params, verbose = 0, 
                                               reuse_learning_rates_gp_model = TRUE) )
      str <- output[length(output)-3]
      nb_ll_eval <- as.numeric(substr(str, nchar(str)-2, nchar(str)-2))
      expect_equal(nb_ll_eval, 2)
      # CV: Check whether the option "reuse_learning_rates_gp_model" is used or not 
      gp_model <- GPModel(group_data = group)
      set_optim_params(gp_model, params=params_loc)
      output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                                nrounds = 2, nfold = 4, eval = "l2", early_stopping_rounds = 5,
                                                use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                                reuse_learning_rates_gp_model = FALSE) )
      str <- output[length(output)-3]
      nb_ll_eval <- as.numeric(substr(str, nchar(str)-3, nchar(str)-2))
      nb_opt <- as.numeric(substr(str, 64, 64))
      expect_equal(nb_ll_eval, 10)
      expect_equal(nb_opt, 5)
      # same thing with reuse_learning_rates_gp_model = TRUE
      gp_model <- GPModel(group_data = group)
      set_optim_params(gp_model, params=params_loc)
      output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                                nrounds = 2, nfold = 4, eval = "l2", early_stopping_rounds = 5,
                                                use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                                reuse_learning_rates_gp_model = TRUE) )
      str <- output[length(output)-3]
      nb_ll_eval <- as.numeric(substr(str, nchar(str)-2, nchar(str)-2))
      nb_opt <- as.numeric(substr(str, 64, 64))
      expect_equal(nb_ll_eval, 7)
      expect_equal(nb_opt, 4)
      
      # Create random effects model and train GPBoost model
      gp_model <- GPModel(group_data = group)
      set_optim_params(gp_model, params=OPTIM_PARAMS_GRAD_DESC)
      bst <- gpboost(data = X_train, label = y, gp_model = gp_model,
                     nrounds = 62, params = params, verbose = 0, 
                     reuse_learning_rates_gp_model = FALSE)
      cov_pars <- c(0.009426053798, 0.602785377299)
      nll <- -86930.9172156506
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
      expect_lt(abs((gp_model$get_current_neg_log_likelihood()-nll))/abs(nll),TOLERANCE)
      # With the option reuse_learning_rates_gp_model
      gp_model <- GPModel(group_data = group)
      set_optim_params(gp_model, params=OPTIM_PARAMS_GRAD_DESC)
      bst <- gpboost(data = X_train, label = y, gp_model = gp_model,
                     nrounds = 62, params = params, verbose = 0,
                     reuse_learning_rates_gp_model = TRUE)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
      expect_lt(abs((gp_model$get_current_neg_log_likelihood()-nll))/abs(nll),TOLERANCE)
      
      # CV
      gp_model <- GPModel(group_data = group)
      set_optim_params(gp_model, params=OPTIM_PARAMS_GRAD_DESC)
      best_iter_max <- 5
      best_iter_min <- 3
      score <- 0.624597895927245
      cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                      nrounds = 100, nfold = 4, eval = "l2", early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                      reuse_learning_rates_gp_model = FALSE)
      expect_lt(cvbst$best_iter, best_iter_max + 1)
      expect_gt(cvbst$best_iter, best_iter_min - 1)
      expect_lt(abs(cvbst$best_score-score), TOLERANCE)
      cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                      nrounds = 100, nfold = 4, eval = "l2", early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                      reuse_learning_rates_gp_model = TRUE)
      expect_lt(cvbst$best_iter, best_iter_max + 1)
      expect_gt(cvbst$best_iter, best_iter_min - 1)
      expect_lt(abs(cvbst$best_score-score), TOLERANCE)
      
      #################
      ### Tests for 'line_search_step_length'
      #################
      params_ls <- params
      params_ls$learning_rate <- 0.5
      # Create random effects model and train GPBoost model
      gp_model <- GPModel(group_data = group)
      set_optim_params(gp_model, params=OPTIM_PARAMS_GRAD_DESC)
      bst <- gpboost(data = X_train, label = ycv, gp_model = gp_model,
                     nrounds = 10, params = params_ls, verbose = 0, 
                     reuse_learning_rates_gp_model = TRUE,
                     line_search_step_length = FALSE)
      nll <- 162232.5638
      expect_lt(abs((gp_model$get_current_neg_log_likelihood()-nll))/abs(nll),TOLERANCE)
      # With the option line_search_step_length
      gp_model <- GPModel(group_data = group)
      set_optim_params(gp_model, params=OPTIM_PARAMS_GRAD_DESC)
      bst <- gpboost(data = X_train, label = ycv, gp_model = gp_model,
                     nrounds = 10, params = params_ls, verbose = 0,
                     reuse_learning_rates_gp_model = TRUE,
                     line_search_step_length = TRUE)
      nll <- -82056.84807
      expect_lt(abs((gp_model$get_current_neg_log_likelihood()-nll))/abs(nll),TOLERANCE)
      
      # CV
      gp_model <- GPModel(group_data = group)
      set_optim_params(gp_model, params=OPTIM_PARAMS_GRAD_DESC)
      best_iter_max <- 3
      best_iter_min <- 1
      score <- 0.631380111900653
      cvbst <- gpb.cv(params = params_ls, data = dtrain, gp_model = gp_model,
                      nrounds = 100, nfold = 4, eval = "l2", early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                      reuse_learning_rates_gp_model = TRUE,
                      line_search_step_length = FALSE)
      expect_lt(cvbst$best_iter, best_iter_max + 1)
      expect_gt(cvbst$best_iter, best_iter_min - 1)
      expect_lt(abs(cvbst$best_score-score), TOLERANCE)
      cvbst <- gpb.cv(params = params_ls, data = dtrain, gp_model = gp_model,
                      nrounds = 100, nfold = 4, eval = "l2", early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                      reuse_learning_rates_gp_model = TRUE,
                      line_search_step_length = TRUE)
      best_iter_max <- 31
      best_iter_min <- 30
      score <- 0.620687335204216
      expect_lt(cvbst$best_iter, best_iter_max + 1)
      expect_gt(cvbst$best_iter, best_iter_min - 1)
      expect_lt(abs(cvbst$best_score-score), TOLERANCE)
      
    })
    
  }
  
}
