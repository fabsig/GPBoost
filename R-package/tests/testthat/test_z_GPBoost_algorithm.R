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
    
    test_that("Combine tree-boosting and grouped random effects model ", {
      
      ntrain <- ntest <- 1000
      n <- ntrain + ntest
      # Simulate fixed effects
      sim_data <- sim_friedman3(n=n, n_irrelevant=5)
      f <- sim_data$f
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
      b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.542))
      # Second random effect
      n_obs_gr <- ntrain/m# number of sampels per group
      group2 <- rep(1,ntrain) # grouping variable
      for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
      group2 <- c(group2,group2)
      group2[(length(group2)-n_new+1):length(group2)] <- rep(99999,n_new)
      Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
      b2 <- sqrt(sigma2_2) * qnorm(sim_rand_unif(n=length(unique(group2)), init_c=0.2354))
      eps <- Z1 %*% b1 + Z2 %*% b2
      group_data <- cbind(group, group2)
      # Error term
      xi <- sqrt(sigma2) * qnorm(sim_rand_unif(n=n, init_c=0.756))
      # Observed data
      y <- f + eps + xi
      # Signal-to-noise ratio of approx. 1
      # var(f) / var(eps)
      # Split in training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      group_data_train <- group_data[1:ntrain,]
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      group_data_test <- group_data[1:ntest+ntrain,]
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
      valids <- list(test = dtest)
      params <- list(learning_rate = 0.01,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "regression_l2",
                     feature_pre_filter = FALSE)
      folds <- list()
      for(i in 1:4) folds[[i]] <- as.integer(1:(ntrain/4) + (ntrain/4) * (i-1))
      
      vec_chol_or_iterative <- c("iterative", "cholesky")
      for (inv_method in vec_chol_or_iterative) {
        PC <- "ssor"
        if(inv_method == "iterative") {
          tolerance_loc_1 <- 2*TOLERANCE2
          tolerance_loc_2 <- 0.1
          tolerance_loc_3 <- 1
          tolerance_loc_4 <- 10
        } else {
          tolerance_loc_1 <- TOLERANCE
          tolerance_loc_2 <- TOLERANCE
          tolerance_loc_3 <- TOLERANCE
          tolerance_loc_4 <- TOLERANCE
        }
        # Create random effects model and train GPBoost model
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        params_gp <- DEFAULT_OPTIM_PARAMS
        params_gp$init_cov_pars <- c(var(y)/2,rep(var(y)/2,2))
        params_gp$cg_preconditioner_type <- PC
        set_optim_params(gp_model, params=params_gp)
        bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                       nrounds = 62, learning_rate = 0.01, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
        cov_pars <- c(0.005087137, 0.590527753, 0.390570179)
        nll_opt <- -965.389782
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_1)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), tolerance_loc_4)
        # Prediction
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                        pred_latent = TRUE, predict_var = TRUE)
        expect_lt(sqrt(mean((pred$fixed_effect - f_test)^2)),0.262)
        if(inv_method == "iterative") {
          tol_y <- 1.025
        }else{
          tol_y <- 1.0241
        }
        expect_lt(sqrt(mean((pred$fixed_effect - y_test)^2)),tol_y)
        expect_lt(sqrt(mean((pred$fixed_effect + pred$random_effect_mean - y_test)^2)),0.235)
        re_mean <- c(0.3918770, -0.1655551, -1.2513672, rep(0,n_new))
        re_var <- c(0.0003254678, 0.0003254678, 0.0003254678, 0.9810979337, 0.9810979337, 0.9810979337)
        pred_fe <- c(4.392474, 4.294148, 3.561677, 5.072800, 5.048781, 3.864357)
        expect_lt(sum(abs(tail(pred$random_effect_mean) - re_mean)),tolerance_loc_3)
        expect_lt(sum(abs(tail(pred$random_effect_cov) - re_var)),tolerance_loc_1)
        expect_lt(sum(abs(tail(pred$fixed_effect) - pred_fe)),tolerance_loc_3)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                        pred_latent = FALSE, predict_var = TRUE)
        expect_lt(sum(abs(tail(pred$response_mean) - re_mean - pred_fe)),tolerance_loc_3)
        expect_lt(sum(abs(tail(pred$response_var) - (re_var + cov_pars[1]))),tolerance_loc_1)
        
        # objective does not need to be set
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=c(DEFAULT_OPTIM_PARAMS, cg_preconditioner_type=PC))
        bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                       nrounds = 62, learning_rate = 0.01, max_depth = 6,
                       min_data_in_leaf = 5, verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_1)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                        pred_latent = FALSE, predict_var = TRUE)
        expect_lt(sum(abs(tail(pred$response_mean) - re_mean - pred_fe)),tolerance_loc_3)
        expect_lt(sum(abs(tail(pred$response_var) - (re_var + cov_pars[1]))),tolerance_loc_1)
        
        # Training with alternative objective names
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=c(DEFAULT_OPTIM_PARAMS, cg_preconditioner_type=PC))
        bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                       nrounds = 62, learning_rate = 0.01, max_depth = 6,
                       min_data_in_leaf = 5, verbose = 0, objective = "gaussian")
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_1)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                        pred_latent = FALSE, predict_var = TRUE)
        expect_lt(sum(abs(tail(pred$response_mean) - re_mean - pred_fe)),tolerance_loc_3)
        expect_lt(sum(abs(tail(pred$response_var) - (re_var + cov_pars[1]))),tolerance_loc_1)
        
        # Training with "wrong" likelihood
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit",
                            matrix_inversion_method = inv_method)
        expect_error({ 
          bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                         nrounds = 62, learning_rate = 0.01, max_depth = 6,
                         min_data_in_leaf = 5, verbose = 0, objective = "gaussian")
        })
        # objective and likelihood do not match
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        capture.output( expect_error({ 
          bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                         nrounds = 62, learning_rate = 0.01, max_depth = 6,
                         min_data_in_leaf = 5, verbose = 0, objective = "binary")
        }) , file='NUL')
        
        # Validation metrics for training data
        # Default metric is "Negative log-likelihood" if there is only one training set
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                       objective = "regression_l2", train_gp_model_cov_pars=FALSE, nrounds=1), file='NUL')
        record_results <- gpb.get.eval.result(bst, "train", "Negative log-likelihood")
        expect_lt(abs(record_results[1]-1410.545), tolerance_loc_3)
        
        bst <- gpb.train(data = dtrain, gp_model = gp_model, verbose = 0, valids = list(train=dtrain),
                         objective = "regression_l2", train_gp_model_cov_pars=FALSE, nrounds=1)
        record_results <- gpb.get.eval.result(bst, "train", "Negative log-likelihood")
        expect_lt(abs(record_results[1]-1410.545), tolerance_loc_3)
        
        # CV for finding number of boosting iterations with use_gp_model_for_validation = FALSE
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        cvbst <- gpb.cv(params = params, data = dtrain, gp_model = gp_model,
                        nrounds = 100, nfold = 4, eval = "l2", early_stopping_rounds = 5,
                        use_gp_model_for_validation = FALSE, folds = folds, verbose = 0)
        expect_gte(cvbst$best_iter, 52)
        expect_lte(cvbst$best_iter, 63)
        expect_lt(abs(cvbst$best_score-1.027334), tolerance_loc_1)
        # CV for finding number of boosting iterations with use_gp_model_for_validation = TRUE
        cvbst <- gpb.cv(params = params, data = dtrain, gp_model = gp_model,
                        nrounds = 100, nfold = 4, eval = "l2", early_stopping_rounds = 5,
                        use_gp_model_for_validation = TRUE, folds = folds, verbose = 0)
        expect_gte(cvbst$best_iter, 52)
        expect_lte(cvbst$best_iter, 63)
        expect_lt(abs(cvbst$best_score-0.6526893), tolerance_loc_1)
        
        # Parameter tuning with 'gpb.grid.search.tune.parameters'
        param_grid = list("learning_rate" = c(1,0.1), 
                          "min_data_in_leaf" = c(10,100))
        other_params <- list(objective = "regression_l2", max_depth = 6, num_leaves = 2^10)
        opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                                      folds = folds, data = dtrain, gp_model = gp_model,
                                                      use_gp_model_for_validation=TRUE, verbose_eval = 0,
                                                      nrounds = 1000, early_stopping_rounds = 10,
                                                      metric = "l2")
        expect_equal(opt_params$best_params$learning_rate, 0.1)
        expect_equal(opt_params$best_params$min_data_in_leaf, 10)
        expect_equal(opt_params$best_iter, 7)
        expect_lt(abs(opt_params$best_score-0.6767217), TOLERANCE)
        # Parameter tuning: can catch errors
        param_grid_wrong = list("learning_rate" = c(-1,0.1), 
                                "min_data_in_leaf" = c(10,100))
        capture.output( capture_messages( capture_error(
          opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid_wrong, params = other_params,
                                                        folds = folds, data = dtrain, gp_model = gp_model,
                                                        use_gp_model_for_validation=TRUE, verbose_eval = 0,
                                                        nrounds = 1000, early_stopping_rounds = 10, metric = "l2") 
        ) ), file='NUL')
        expect_equal(opt_params$best_params$learning_rate, 0.1)
        expect_equal(opt_params$best_params$min_data_in_leaf, 10)
        expect_equal(opt_params$best_iter, 7)
        expect_lt(abs(opt_params$best_score-0.6767217), TOLERANCE)
        # Using 'test_neg_log_likelihood' as metric
        opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                                      folds = folds, data = dtrain, gp_model = gp_model,
                                                      use_gp_model_for_validation=TRUE, verbose_eval = 0,
                                                      nrounds = 1000, early_stopping_rounds = 10,
                                                      metric = "test_neg_log_likelihood")
        expect_equal(opt_params$best_params$learning_rate, 0.1)
        expect_equal(opt_params$best_params$min_data_in_leaf, 10)
        expect_equal(opt_params$best_iter, 7)
        expect_lt(abs(opt_params$best_score-1.224379), TOLERANCE)
        # Using 'crps_gaussian' as metric
        opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                                      folds = folds, data = dtrain, gp_model = gp_model,
                                                      use_gp_model_for_validation=TRUE, verbose_eval = 0,
                                                      nrounds = 1000, early_stopping_rounds = 10,
                                                      metric = "crps_gaussian")
        expect_equal(opt_params$best_params$learning_rate, 0.1)
        expect_equal(opt_params$best_params$min_data_in_leaf, 10)
        expect_equal(opt_params$best_iter, 7)
        expect_lt(abs(opt_params$best_score-0.4622949), TOLERANCE)
        
        if(inv_method=="cholesky"){
        
          ## Prediction when having only one grouped random effect
          group_1 <- rep(1,ntrain) # grouping variable
          for(i in 1:m) group_1[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
          y_1 <- f[1:ntrain] + b1[group_1] + xi[1:ntrain]
          gp_model <- GPModel(group_data = group_1)
          gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
          bst <- gpboost(data = X_train,
                         label = y_1,
                         gp_model = gp_model,
                         nrounds = 62,
                         learning_rate = 0.01,
                         max_depth = 6,
                         min_data_in_leaf = 5,
                         objective = "regression_l2",
                         verbose = 0,
                         leaves_newton_update = FALSE)
          pred <- predict(bst, data = X_test[1:length(unique(b1)),], 
                          group_data_pred = 1:length(unique(b1)), pred_latent = TRUE)
          # plot(pred$random_effect_mean,b1)
          expect_lt(abs(sqrt(sum((pred$random_effect_mean - b1)^2))-0.643814),TOLERANCE)
          expect_lt(abs(cor(pred$random_effect_mean,b1)-0.9914091),TOLERANCE)
        }
        
        # GPBoostOOS algorithm
        #   1. Run GPBoost algorithm separately on every fold and fit parameters on out-of-sample data
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        cvbst <- gpb.cv(params = params, data = dtrain, gp_model = gp_model,
                        nrounds = 100, nfold = 4, eval = "l2", early_stopping_rounds = 5,
                        use_gp_model_for_validation = FALSE, folds = folds, verbose = 0,
                        fit_GP_cov_pars_OOS = TRUE)
        expect_gte(cvbst$best_iter, 52)
        expect_lte(cvbst$best_iter, 63)
        cov_pars_OOS <- c(0.05103639, 0.60775408, 0.38378833)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_OOS)),tolerance_loc_1)
        #   2. Run GPBoost algorithm on entire data while holding covariance parameters fixed
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 59,
                         params = params, train_gp_model_cov_pars = FALSE, verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_OOS)),tolerance_loc_1)# no change in covariance parameters
        #   3. Prediction
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                        predict_var = TRUE, pred_latent = TRUE)
        if(inv_method == "iterative"){
          fe_tol <- 0.3
        } else{
          fe_tol <- 0.1
        }
        expect_lt(sum(abs(head(pred$fixed_effect, n=4)-c(4.891230, 4.121098, 3.140073, 4.236029))),fe_tol)
        expect_lt(sum(abs(tail(pred$random_effect_mean)-c(0.3953752, -0.1785115, -1.2413583,
                                                          rep(0,n_new)))),0.05)
        expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.003256045, 0.003256045, 0.003256045,
                                                         rep(0.991588837,n_new)))),tolerance_loc_1)
        
        # GPBoostOOS algorithm: fit parameters on out-of-sample data with random folds
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        cvbst <- gpb.cv(params = params, data = dtrain, gp_model = gp_model,
                        nrounds = 100, nfold = 4, eval = "l2", early_stopping_rounds = 5,
                        use_gp_model_for_validation = FALSE, fit_GP_cov_pars_OOS = TRUE,
                        verbose = 0)
        cov_pars_OOS <- c(0.055, 0.59, 0.39)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_OOS)),0.1)
        
        # Use Nelder-Mead for training
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params = list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6, init_cov_pars = params_gp$init_cov_pars,
                                                cg_preconditioner_type=PC, init_coef_aux_pars_from_iid_model = FALSE))
        bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                       nrounds = 62, learning_rate = 0.01, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
        cov_pars_NM <- c(0.004823767, 0.592422707, 0.394167937)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_NM)),tolerance_loc_1)
        # Prediction
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test, pred_latent = TRUE)
        expect_lt(sum(abs(tail(pred$random_effect_mean)-c(0.4157265, -0.1696440, -1.2674184,
                                                          rep(0,n_new)))),tolerance_loc_2)
        expect_lt(sum(abs(head(pred$fixed_effect)-c(4.818977, 4.174924, 3.269181, 4.222688, 4.997808, 4.947587))),tolerance_loc_3)
        
        # Use lbfgs for training
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params = list(optimizer_cov="lbfgs", optimizer_coef = "lbfgs", init_cov_pars = params_gp$init_cov_pars,
                                                cg_preconditioner_type=PC, init_coef_aux_pars_from_iid_model = FALSE))
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                                       nrounds = 62, learning_rate = 0.01, max_depth = 6,
                                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0) , file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
        nll_lbfgs <- -965.3904123
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_lbfgs), tolerance_loc_4)
        # same with optimizer_coef = "wls"
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params = list(optimizer_cov="lbfgs", optimizer_coef = "wls", init_cov_pars = params_gp$init_cov_pars,
                                                cg_preconditioner_type=PC, init_coef_aux_pars_from_iid_model = FALSE))
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                                       nrounds = 62, learning_rate = 0.01, max_depth = 6,
                                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0) , file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_lbfgs), tolerance_loc_4)
        
        # Newton updates for tree leaves
        if(inv_method=="cholesky"){
          params_loc <- list(learning_rate = 0.1,
                         max_depth = 6,
                         min_data_in_leaf = 5,
                         objective = "regression_l2",
                         leaves_newton_update = TRUE)
          gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
          gp_model$set_optim_params(params=params_gp)
          cvbst <- gpb.cv(params = params_loc,
                          data = dtrain,
                          gp_model = gp_model,
                          nrounds = 100,
                          nfold = 4,
                          eval = "l2",
                          early_stopping_rounds = 5,
                          use_gp_model_for_validation = FALSE,
                          fit_GP_cov_pars_OOS = TRUE,
                          folds = folds,
                          verbose = 0)
          expect_equal(cvbst$best_iter, 52)
          cov_pars_OOS <- c(0.04468342, 0.60930957, 0.38893938)
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_OOS)),TOLERANCE)
        }
        
        # Using validation set
        # Do not include random effect predictions for validation
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        bst <- gpb.train(data = dtrain,
                         gp_model = gp_model,
                         nrounds = 100,
                         learning_rate = 0.01,
                         max_depth = 6,
                         min_data_in_leaf = 5,
                         objective = "regression_l2",
                         verbose = 0,
                         valids = valids,
                         early_stopping_rounds = 5,
                         use_gp_model_for_validation = FALSE, metric = "l2")
        # 'cvbst' is from the gpb.cv above that uses random folds, i.e. 'best_iter' varies between runs (59 - 65 over 25 runs)
        expect_gte(cvbst$best_iter, 45)
        expect_lte(cvbst$best_iter, 85)
        expect_lt(abs(bst$best_score - 1.0326),10*TOLERANCE)
        # Include random effect predictions for validation 
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        gp_model$set_prediction_data(group_data_pred = group_data_test)
        bst <- gpb.train(data = dtrain,
                         gp_model = gp_model,
                         nrounds = 100,
                         learning_rate = 0.01,
                         max_depth = 6,
                         min_data_in_leaf = 5,
                         objective = "regression_l2",
                         verbose = 0,
                         valids = valids,
                         early_stopping_rounds = 5,
                         use_gp_model_for_validation = TRUE, metric = "l2")
        # 'cvbst' is from the gpb.cv above that uses random folds, i.e. 'best_iter' varies between runs (59 - 65 over 25 runs)
        expect_gte(cvbst$best_iter, 45)
        expect_lte(cvbst$best_iter, 85)
        expect_lt(abs(bst$best_score - 0.04753591),10*TOLERANCE)
        # Same thing using the set_prediction_data method 
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        set_prediction_data(gp_model, group_data_pred = group_data_test)
        bst <- gpb.train(data = dtrain,
                         gp_model = gp_model,
                         nrounds = 100,
                         learning_rate = 0.01,
                         max_depth = 6,
                         min_data_in_leaf = 5,
                         objective = "regression_l2",
                         verbose = 0,
                         valids = valids,
                         early_stopping_rounds = 5,
                         use_gp_model_for_validation = TRUE, metric = "l2")
        # 'cvbst' is from the gpb.cv above that uses random folds, i.e. 'best_iter' varies between runs (59 - 65 over 25 runs)
        expect_gte(cvbst$best_iter, 45)
        expect_lte(cvbst$best_iter, 85)
        expect_lt(abs(bst$best_score - 0.04753591),10*TOLERANCE)
        
        # Use of validation data and cross-validation with custom metric
        l4_loss <- function(preds, dtrain) {
          labels <- getinfo(dtrain, "label")
          return(list(name="l4",value=mean((preds-labels)^4),higher_better=FALSE))
        }
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        bst <- gpb.train(data = dtrain,
                         gp_model = gp_model,
                         nrounds = 100,
                         learning_rate = 0.01,
                         max_depth = 6,
                         min_data_in_leaf = 5,
                         objective = "regression_l2",
                         verbose = 0,
                         valids = valids,
                         early_stopping_rounds = 5,
                         use_gp_model_for_validation = FALSE,
                         eval = l4_loss, metric = "l4")
        expect_lt(abs(bst$best_score - 3.058637),0.15)
        # CV
        if(inv_method=="cholesky"){
          gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
          gp_model$set_optim_params(params=params_gp)
          cvbst <- gpb.cv(params = params,
                          data = dtrain,
                          gp_model = gp_model,
                          nrounds = 100,
                          nfold = 4,
                          early_stopping_rounds = 5,
                          use_gp_model_for_validation = FALSE,
                          fit_GP_cov_pars_OOS = FALSE,
                          folds = folds,
                          verbose = 0,
                          eval = l4_loss, metric = "l4")
          expect_gte(cvbst$best_iter, 52)
          expect_lte(cvbst$best_iter, 63)
          expect_lt(abs(cvbst$best_score - 2.983831),TOLERANCE2)
        }
        
        # Use of validation data and test_neg_log_likelihood as metric
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        set_prediction_data(gp_model, group_data_pred = group_data_test)
        set_optim_params(gp_model, params=params_gp)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 10,
                         learning_rate = 0.01, max_depth = 6, min_data_in_leaf = 5,
                         objective = "regression_l2", verbose = 0,
                         valids = valids, early_stopping_rounds = 5,
                         use_gp_model_for_validation = TRUE, metric = "test_neg_log_likelihood")
        expect_equal(bst$best_iter, 10)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                        pred_latent = FALSE, predict_var = TRUE)
        nll <- 0.5 * mean((y_test - pred[['response_mean']])^2 / 
                            pred[['response_var']] + log(pred[['response_var']] * 2 * pi))
        expect_lt(abs(bst$best_score - nll),TOLERANCE)
        # Use of validation data and test_neg_log_likelihood as metric but set use_gp_model_for_validation = FALSE
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 10,
                         learning_rate = 0.01, max_depth = 6, min_data_in_leaf = 5,
                         objective = "regression_l2", verbose = 0,
                         valids = valids, early_stopping_rounds = 5,
                         use_gp_model_for_validation = FALSE, metric = "test_neg_log_likelihood")
        expect_equal(bst$best_iter, 10)
        predtrain <- predict(bst, data = X_train, group_data_pred = group_data_train, pred_latent = TRUE)
        var_est <- var(y_train - predtrain$fixed_effect)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test, pred_latent = TRUE)
        nll <- 0.5 * mean((y_test - pred[['fixed_effect']])^2 / var_est + log(var_est * 2 * pi))
        expect_lt(abs(bst$best_score - nll),TOLERANCE)
        # Use of validation data and test_neg_log_likelihood as metric without a GPModel
        bst <- gpb.train(data = dtrain, nrounds = 10, learning_rate = 0.01, max_depth = 6, min_data_in_leaf = 5,
                         objective = "regression_l2", verbose = 0,
                         valids = valids, early_stopping_rounds = 5,
                         metric = "test_neg_log_likelihood")
        expect_equal(bst$best_iter, 10)
        predtrain <- predict(bst, data = X_train, pred_latent = TRUE)
        var_est <- var(y_train - predtrain)
        pred <- predict(bst, data = X_test, pred_latent = TRUE)
        nll <- 0.5 * mean((y_test - pred)^2 / var_est + log(var_est * 2 * pi))
        expect_lt(abs(bst$best_score - nll),TOLERANCE)
        
        # Use of validation data and crps_gaussian as metric
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        set_prediction_data(gp_model, group_data_pred = group_data_test)
        set_optim_params(gp_model, params=params_gp)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 10,
                         learning_rate = 0.01, max_depth = 6, min_data_in_leaf = 5,
                         objective = "regression_l2", verbose = 0,
                         valids = valids, early_stopping_rounds = 5,
                         use_gp_model_for_validation = TRUE, metric = "crps_gaussian")
        expect_equal(bst$best_iter, 10)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                        pred_latent = FALSE, predict_var = TRUE)
        resid <- (y_test - pred[['response_mean']]) / sqrt(pred[['response_var']])
        crps <- mean(sqrt(pred[['response_var']]) * (-1/sqrt(pi) + 2 * dnorm(resid) + resid * (2 * pnorm(resid) - 1)))
        expect_lt(abs(bst$best_score - crps),TOLERANCE)
        # Use of validation data and crps_gaussian as metric without a GPModel
        bst <- gpb.train(data = dtrain, nrounds = 10, learning_rate = 0.01, max_depth = 6, min_data_in_leaf = 5,
                         objective = "regression_l2", verbose = 0,
                         valids = valids, early_stopping_rounds = 5,
                         metric = "crps_gaussian")
        expect_equal(bst$best_iter, 10)
        predtrain <- predict(bst, data = X_train, pred_latent = TRUE)
        var_est <- var(y_train - predtrain)
        pred <- predict(bst, data = X_test, pred_latent = TRUE)
        resid <- (y_test - pred) / sqrt(var_est)
        crps <- mean(sqrt(var_est) * (-1/sqrt(pi) + 2 * dnorm(resid) + resid * (2 * pnorm(resid) - 1)))
        expect_lt(abs(bst$best_score - crps),TOLERANCE)
        
        ## Cannot have NA's in response variable
        expect_error({
          gp_model <- GPModel(group_data = group_data_train)
          y_train_NA <- y_train
          y_train_NA[2] <- NA
          bst <- gpboost(data = X_train,
                         label = y_train_NA,
                         gp_model = gp_model,
                         nrounds = 62,
                         learning_rate = 0.01,
                         max_depth = 6,
                         min_data_in_leaf = 5,
                         objective = "regression_l2",
                         verbose = 0,
                         leaves_newton_update = FALSE)
        })
      }
    })

    test_that("Early stopping sets the parameters of the gp_model to those of the best iteration", {

      ntrain <- ntest <- 500
      n <- ntrain + ntest
      sim_data <- sim_friedman3(n=n, n_irrelevant=5)
      m <- 50
      group <- rep(1,ntrain)
      for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
      group <- c(group, group)
      b1 <- sqrt(0.5) * qnorm(sim_rand_unif(n=m, init_c=0.542))
      eta <- sim_data$f / 4 + b1[group]
      X_train <- sim_data$X[1:ntrain,]
      X_test <- sim_data$X[1:ntest + ntrain,]
      group_train <- group[1:ntrain]
      group_test <- group[1:ntest + ntrain]
      params_bst <- list(learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5, verbose = 0)
      for (likelihood in c("gaussian", "gamma")) {
        params_lik <- params_bst
        if (likelihood == "gaussian") {
          y <- eta + 0.5 * qnorm(sim_rand_unif(n=n, init_c=0.756))
          params_lik$objective <- "regression_l2"
        } else {
          y <- qgamma(sim_rand_unif(n=n, init_c=0.756), scale = exp(eta) / 2, shape = 2)
        }
        y_train <- y[1:ntrain]
        y_test <- y[1:ntest + ntrain]
        dtrain <- gpb.Dataset(data = X_train, label = y_train)
        valids <- list(test = gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test))
        gp_model <- GPModel(group_data = group_train, likelihood = likelihood)
        set_prediction_data(gp_model, group_data_pred = group_test)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 100, params = params_lik,
                         valids = valids, early_stopping_rounds = 5, verbose = 0,
                         use_gp_model_for_validation = TRUE, metric = "test_neg_log_likelihood")
        expect_lt(bst$best_iter, bst$current_iter())
        # The same parameters and predictions as when training stops at the best iteration
        gp_model_best <- GPModel(group_data = group_train, likelihood = likelihood)
        bst_best <- gpb.train(data = dtrain, gp_model = gp_model_best, nrounds = bst$best_iter,
                              params = params_lik, verbose = 0)
        expect_lt(sum(abs(gp_model$get_cov_pars() - gp_model_best$get_cov_pars())), TOLERANCE_STRICT)
        if (likelihood == "gamma") {
          expect_lt(sum(abs(gp_model$get_aux_pars() - gp_model_best$get_aux_pars())), TOLERANCE_STRICT)
        }
        pred <- predict(bst, data = X_test, group_data_pred = group_test, pred_latent = FALSE, predict_var = TRUE)
        pred_best <- predict(bst_best, data = X_test, group_data_pred = group_test, pred_latent = FALSE, predict_var = TRUE)
        expect_lt(sum(abs(pred$response_mean - pred_best$response_mean)), TOLERANCE_STRICT)
        expect_lt(sum(abs(pred$response_var - pred_best$response_var)), TOLERANCE_STRICT)
        if (likelihood == "gaussian") {
          # The validation score of the best iteration is the score of the predictions of the returned model
          nll <- 0.5 * mean((y_test - pred$response_mean)^2 / pred$response_var + log(pred$response_var * 2 * pi))
          expect_lt(abs(bst$best_score - nll), TOLERANCE_STRICT)
        }
        # The models of the folds of a cross-validation use the best iteration as well
        folds <- list(seq(1, ntrain, by = 2), seq(2, ntrain, by = 2))
        cvbst <- gpb.cv(params = params_lik, data = gpb.Dataset(data = X_train, label = y_train),
                        gp_model = GPModel(group_data = group_train, likelihood = likelihood),
                        nrounds = 100, folds = folds, early_stopping_rounds = 5,
                        use_gp_model_for_validation = TRUE, metric = "test_neg_log_likelihood", verbose = 0)
        expect_lt(cvbst$best_iter, cvbst$boosters[[1]]$booster$current_iter())
        cvbst_best <- gpb.cv(params = params_lik, data = gpb.Dataset(data = X_train, label = y_train),
                             gp_model = GPModel(group_data = group_train, likelihood = likelihood),
                             nrounds = cvbst$best_iter, folds = folds,
                             use_gp_model_for_validation = TRUE, metric = "test_neg_log_likelihood", verbose = 0)
        for (k in seq_along(folds)) {
          fold_bst <- cvbst$boosters[[k]]$booster
          fold_bst_best <- cvbst_best$boosters[[k]]$booster
          expect_equal(fold_bst$best_iter, cvbst$best_iter)
          expect_equal(fold_bst_best$best_iter, -1L)
          expect_lt(sum(abs(fold_bst$.__enclos_env__$private$gp_model$get_cov_pars() -
                              fold_bst_best$.__enclos_env__$private$gp_model$get_cov_pars())), TOLERANCE_STRICT)
          pred <- predict(fold_bst, data = X_train[folds[[k]], ], group_data_pred = group_train[folds[[k]]],
                          pred_latent = FALSE, predict_var = TRUE)
          pred_best <- predict(fold_bst_best, data = X_train[folds[[k]], ], group_data_pred = group_train[folds[[k]]],
                               pred_latent = FALSE, predict_var = TRUE)
          expect_lt(sum(abs(pred$response_mean - pred_best$response_mean)), TOLERANCE_STRICT)
          expect_lt(sum(abs(pred$response_var - pred_best$response_var)), TOLERANCE_STRICT)
        }
      }
      # Without early stopping, all iterations are used, also when there are validation data
      gp_model <- GPModel(group_data = group_train, likelihood = likelihood)
      set_prediction_data(gp_model, group_data_pred = group_test)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20, params = params_lik, verbose = 0,
                       valids = valids, use_gp_model_for_validation = TRUE, metric = "test_neg_log_likelihood")
      expect_equal(bst$best_iter, -1L)
      test_scores <- unlist(bst$record_evals[["test"]][[1]][["eval"]])
      expect_equal(bst$best_score, test_scores[[20]])
    })

    test_that("The parameters of the gp_model are those of the iteration whose trees are used", {

      ntrain <- ntest <- 500
      n <- ntrain + ntest
      sim_data <- sim_friedman3(n=n, n_irrelevant=5)
      m <- 50
      group <- rep(1,ntrain)
      for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
      group <- c(group, group)
      b1 <- sqrt(0.5) * qnorm(sim_rand_unif(n=m, init_c=0.542))
      eta <- sim_data$f / 4 + b1[group]
      X_train <- sim_data$X[1:ntrain,]
      X_test <- sim_data$X[1:ntest + ntrain,]
      group_train <- group[1:ntrain]
      group_test <- group[1:ntest + ntrain]
      params_bst <- list(learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5, verbose = 0)
      folds <- list(seq(1, ntrain, by = 2), seq(2, ntrain, by = 2))
      for (likelihood in c("gaussian", "gamma")) {
        params_lik <- params_bst
        if (likelihood == "gaussian") {
          y <- eta + 0.5 * qnorm(sim_rand_unif(n=n, init_c=0.756))
          params_lik$objective <- "regression_l2"
        } else {
          y <- qgamma(sim_rand_unif(n=n, init_c=0.756), scale = exp(eta) / 2, shape = 2)
        }
        y_train <- y[1:ntrain]
        y_test <- y[1:ntest + ntrain]
        dtrain <- gpb.Dataset(data = X_train, label = y_train)
        valids <- list(test = gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test))
        gp_model <- GPModel(group_data = group_train, likelihood = likelihood)
        set_prediction_data(gp_model, group_data_pred = group_test)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 100, params = params_lik,
                         valids = valids, early_stopping_rounds = 5, verbose = 0,
                         use_gp_model_for_validation = TRUE, metric = "test_neg_log_likelihood")
        num_iter <- bst$current_iter()
        best_iter <- bst$best_iter
        best_score <- bst$best_score
        expect_lt(best_iter, num_iter)
        cov_pars_best <- gp_model$get_cov_pars()
        pred_best <- predict(bst, data = X_test, group_data_pred = group_test, pred_latent = FALSE, predict_var = TRUE)
        expect_same_as_best <- function() {
          expect_lt(sum(abs(gp_model$get_cov_pars() - cov_pars_best)), TOLERANCE_STRICT)
          pred <- predict(bst, data = X_test, group_data_pred = group_test, pred_latent = FALSE, predict_var = TRUE)
          expect_lt(sum(abs(pred$response_mean - pred_best$response_mean)), TOLERANCE_STRICT)
          expect_lt(sum(abs(pred$response_var - pred_best$response_var)), TOLERANCE_STRICT)
        }
        gp_model_all <- GPModel(group_data = group_train, likelihood = likelihood)
        bst_all <- gpb.train(data = dtrain, gp_model = gp_model_all, nrounds = num_iter, params = params_lik, verbose = 0)
        pred_all <- predict(bst_all, data = X_test, group_data_pred = group_test, pred_latent = FALSE, predict_var = TRUE)
        # Predictions with all iterations are those of the model trained for all iterations
        pred <- predict(bst, data = X_test, group_data_pred = group_test, num_iteration = num_iter,
                        pred_latent = FALSE, predict_var = TRUE)
        expect_lt(sum(abs(pred$response_mean - pred_all$response_mean)), TOLERANCE_STRICT)
        expect_lt(sum(abs(pred$response_var - pred_all$response_var)), TOLERANCE_STRICT)
        expect_same_as_best()
        if (likelihood != "gaussian") {
          # Training data random effects use the best iteration by default
          re_best <- predict_training_data_random_effects(bst, predict_var = TRUE, num_iteration = best_iter)
          re <- predict_training_data_random_effects(bst, predict_var = TRUE)
          expect_lt(sum(abs(as.matrix(re) - as.matrix(re_best))), TOLERANCE_STRICT)
          re_all <- predict_training_data_random_effects(bst_all, predict_var = TRUE)
          re <- predict_training_data_random_effects(bst, predict_var = TRUE, num_iteration = num_iter)
          expect_lt(sum(abs(as.matrix(re) - as.matrix(re_all))), TOLERANCE_STRICT)
          expect_same_as_best()
        }
        # A model saved with all iterations has the parameters of the last iteration
        bst_loaded <- gpb.load(model_str = bst$save_model_to_string(num_iteration = num_iter))
        gp_model_loaded <- bst_loaded$.__enclos_env__$private$gp_model
        expect_lt(sum(abs(gp_model_loaded$get_cov_pars() - gp_model_all$get_cov_pars())), TOLERANCE_STRICT)
        if (likelihood == "gamma") {
          expect_lt(sum(abs(gp_model_loaded$get_aux_pars() - gp_model_all$get_aux_pars())), TOLERANCE_STRICT)
        }
        pred <- predict(bst_loaded, data = X_test, group_data_pred = group_test, pred_latent = FALSE, predict_var = TRUE)
        # A loaded model finds the mode for non-Gaussian likelihoods anew, starting from zero, which changes the
        #   predictions in about the eighth digit (relative deviations of 1e-8 for the gamma likelihood)
        expect_equal(pred$response_mean, pred_all$response_mean, tolerance = 1e-6)
        expect_equal(pred$response_var, pred_all$response_var, tolerance = 1e-6)
        expect_same_as_best()
        # A model saved with the raw data has the parameters of all iterations, also when it is saved again after loading
        bst_loaded <- gpb.load(model_str = bst$save_model_to_string(num_iteration = num_iter, save_raw_data = TRUE))
        bst_loaded <- gpb.load(model_str = bst_loaded$save_model_to_string(save_raw_data = TRUE))
        for (num_iteration in c(best_iter, num_iter - 1L, num_iter)) {
          pred <- predict(bst, data = X_test, group_data_pred = group_test, num_iteration = num_iteration,
                          pred_latent = FALSE, predict_var = TRUE)
          pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_test, num_iteration = num_iteration,
                                 pred_latent = FALSE, predict_var = TRUE)
          expect_equal(pred_loaded$response_mean, pred$response_mean, tolerance = 1e-6)
          expect_equal(pred_loaded$response_var, pred$response_var, tolerance = 1e-6)
        }
        expect_lt(sum(abs(bst_loaded$.__enclos_env__$private$gp_model$get_cov_pars() - gp_model_all$get_cov_pars())),
                  TOLERANCE_STRICT)
        expect_same_as_best()
        # The evaluation after training is that of the last iteration
        test_scores <- unlist(bst$record_evals[["test"]][[1]][["eval"]])
        expect_lt(abs(bst$eval_valid()[[1L]]$value - test_scores[[num_iter]]), TOLERANCE_STRICT)
        expect_same_as_best()
        # Training continues from the last iteration
        bst$update()
        gp_model_all1 <- GPModel(group_data = group_train, likelihood = likelihood)
        bst_all1 <- gpb.train(data = dtrain, gp_model = gp_model_all1, nrounds = num_iter + 1, params = params_lik, verbose = 0)
        expect_lt(sum(abs(gp_model$get_cov_pars() - gp_model_all1$get_cov_pars())), TOLERANCE_STRICT)
        pred <- predict(bst, data = X_test, group_data_pred = group_test, num_iteration = num_iter + 1,
                        pred_latent = FALSE, predict_var = TRUE)
        pred_all1 <- predict(bst_all1, data = X_test, group_data_pred = group_test, pred_latent = FALSE, predict_var = TRUE)
        expect_lt(sum(abs(pred$response_mean - pred_all1$response_mean)), TOLERANCE_STRICT)
        expect_lt(sum(abs(pred$response_var - pred_all1$response_var)), TOLERANCE_STRICT)
        # Early stopping by a callback in 'callbacks'
        gp_model_cb <- GPModel(group_data = group_train, likelihood = likelihood)
        set_prediction_data(gp_model_cb, group_data_pred = group_test)
        cb_es <- gpboost:::cb.early.stop(stopping_rounds = 5, first_metric_only = TRUE, verbose = FALSE)
        bst_cb <- gpb.train(data = dtrain, gp_model = gp_model_cb, nrounds = 100, params = params_lik,
                            valids = valids, callbacks = list(cb_es), verbose = 0,
                            use_gp_model_for_validation = TRUE, metric = "test_neg_log_likelihood")
        expect_equal(bst_cb$best_iter, best_iter)
        expect_equal(bst_cb$best_score, best_score)
        expect_lt(sum(abs(gp_model_cb$get_cov_pars() - cov_pars_best)), TOLERANCE_STRICT)
        cvbst <- gpb.cv(params = params_lik, data = gpb.Dataset(data = X_train, label = y_train),
                        gp_model = GPModel(group_data = group_train, likelihood = likelihood),
                        nrounds = 100, folds = folds, early_stopping_rounds = 5,
                        use_gp_model_for_validation = TRUE, metric = "test_neg_log_likelihood", verbose = 0)
        cb_es <- gpboost:::cb.early.stop(stopping_rounds = 5, first_metric_only = TRUE, verbose = FALSE)
        cvbst_cb <- gpb.cv(params = params_lik, data = gpb.Dataset(data = X_train, label = y_train),
                           gp_model = GPModel(group_data = group_train, likelihood = likelihood),
                           nrounds = 100, folds = folds, callbacks = list(cb_es),
                           use_gp_model_for_validation = TRUE, metric = "test_neg_log_likelihood", verbose = 0)
        expect_equal(cvbst_cb$best_iter, cvbst$best_iter)
        for (k in seq_along(folds)) {
          fold_bst <- cvbst$boosters[[k]]$booster
          fold_bst_cb <- cvbst_cb$boosters[[k]]$booster
          expect_equal(fold_bst_cb$best_iter, cvbst$best_iter)
          expect_lt(sum(abs(fold_bst_cb$.__enclos_env__$private$gp_model$get_cov_pars() -
                              fold_bst$.__enclos_env__$private$gp_model$get_cov_pars())), TOLERANCE_STRICT)
        }
      }
    })

  }
  
}
