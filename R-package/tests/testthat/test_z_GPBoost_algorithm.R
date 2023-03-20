if(Sys.getenv("NO_GPBOOST_ALGO_TESTS") != "NO_GPBOOST_ALGO_TESTS"){
  
  context("GPBoost_combined_boosting_GP_random_effects")
  
  TOLERANCE <- 1E-3
  TOLERANCE2 <- 1E-2
  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov="fisher_scoring", delta_rel_conv=1E-6)
  
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
      
      # Validation metrics for training data
      # Default metric is "Negative log-likelihood" if there is only one training set
      gp_model <- GPModel(group_data = group_data_train)
      capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                     objective = "regression_l2", train_gp_model_cov_pars=FALSE, nrounds=1), file='NUL')
      record_results <- gpb.get.eval.result(bst, "train", "Negative log-likelihood")
      expect_lt(abs(record_results[1]-1573.9417522), TOLERANCE)
      
      bst <- gpb.train(data = dtrain, gp_model = gp_model, verbose = 0, valids = list(train=dtrain),
                       objective = "regression_l2", train_gp_model_cov_pars=FALSE, nrounds=1)
      record_results <- gpb.get.eval.result(bst, "train", "Negative log-likelihood")
      expect_lt(abs(record_results[1]-1573.9417522), TOLERANCE)
      
      # CV for finding number of boosting iterations with use_gp_model_for_validation = FALSE
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      cvbst <- gpb.cv(params = params,
                      data = dtrain,
                      gp_model = gp_model,
                      nrounds = 100,
                      nfold = 4,
                      eval = "l2",
                      early_stopping_rounds = 5,
                      use_gp_model_for_validation = FALSE,
                      fit_GP_cov_pars_OOS = FALSE,
                      folds = folds,
                      verbose = 0)
      expect_equal(cvbst$best_iter, 59)
      expect_lt(abs(cvbst$best_score-1.027334), TOLERANCE)
      # CV for finding number of boosting iterations with use_gp_model_for_validation = TRUE
      cvbst <- gpb.cv(params = params,
                      data = dtrain,
                      gp_model = gp_model,
                      nrounds = 100,
                      nfold = 4,
                      eval = "l2",
                      early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE,
                      fit_GP_cov_pars_OOS = FALSE,
                      folds = folds,
                      verbose = 0)
      expect_equal(cvbst$best_iter, 59)
      expect_lt(abs(cvbst$best_score-0.6526893), TOLERANCE)
      
      # Create random effects model and train GPBoost model
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      bst <- gpboost(data = X_train,
                     label = y_train,
                     gp_model = gp_model,
                     nrounds = 62,
                     learning_rate = 0.01,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "regression_l2",
                     verbose = 0,
                     leaves_newton_update = FALSE)
      cov_pars <- c(0.005087137, 0.590527753, 0.390570179)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                      pred_latent = TRUE)
      expect_lt(sqrt(mean((pred$fixed_effect - f_test)^2)),0.262)
      expect_lt(sqrt(mean((pred$fixed_effect - y_test)^2)),1.0241)
      expect_lt(sqrt(mean((pred$fixed_effect + pred$random_effect_mean - y_test)^2)),0.235)
      expect_lt(sum(abs(tail(pred$random_effect_mean)-c(0.3918770, -0.1655551, -1.2513672,
                                                        rep(0,n_new)))),TOLERANCE)
      expect_lt(sum(abs(head(pred$random_effect_mean)-c(-.5559122, 0.5031307, 0.5676980,
                                                        -0.9293673, -0.5188209, -0.2505326))),TOLERANCE)
      expect_lt(sum(abs(head(pred$fixed_effect)-c(4.894403, 3.957849, 3.281690,
                                                  4.162436, 5.101025, 4.889397))),TOLERANCE)
      
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
      
      # GPBoostOOS algorithm
      #   1. Run GPBoost algorithm separately on every fold and fit parameters on out-of-sample data
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      cvbst <- gpb.cv(params = params,
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
      expect_equal(cvbst$best_iter, 59)
      cov_pars_OOS <- c(0.05103639, 0.60775408, 0.38378833)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_OOS)),TOLERANCE)
      #   2. Run LaGaBoost algorithm on entire data while holding covariance parameters fixed
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 59,
                       params = params, train_gp_model_cov_pars = FALSE, verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_OOS)),TOLERANCE)# no change in covariance parameters
      #   3. Prediction
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                      predict_var = TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(head(pred$fixed_effect, n=4)-c(4.891230, 4.121098, 3.140073, 4.236029))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_mean)-c(0.3953752, -0.1785115, -1.2413583,
                                                        rep(0,n_new)))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.003256045, 0.003256045, 0.003256045,
                                                       rep(0.991588837,n_new)))),TOLERANCE)
      
      # GPBoostOOS algorithm: fit parameters on out-of-sample data with random folds
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      cvbst <- gpb.cv(params = params,
                      data = dtrain,
                      gp_model = gp_model,
                      nrounds = 100,
                      nfold = 4,
                      eval = "l2",
                      early_stopping_rounds = 5,
                      use_gp_model_for_validation = FALSE,
                      fit_GP_cov_pars_OOS = TRUE,
                      verbose = 0)
      cov_pars_OOS <- c(0.055, 0.59, 0.39)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_OOS)),0.1)
      
      # Newton updates for tree leaves
      params <- list(learning_rate = 0.1,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "regression_l2",
                     leaves_newton_update = TRUE)
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      cvbst <- gpb.cv(params = params,
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
      
      # Using validation set
      # Do not include random effect predictions for validation
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
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
                       use_gp_model_for_validation = FALSE)
      expect_equal(bst$best_iter, 57)
      expect_lt(abs(bst$best_score - 1.0326),TOLERANCE)
      # Include random effect predictions for validation 
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
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
                       use_gp_model_for_validation = TRUE)
      expect_equal(bst$best_iter, 59)
      expect_lt(abs(bst$best_score - 0.04753591),TOLERANCE)
      # Same thing using the set_prediction_data method 
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
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
                       use_gp_model_for_validation = TRUE)
      expect_equal(bst$best_iter, 59)
      expect_lt(abs(bst$best_score - 0.04753591),TOLERANCE)
      
      # Use of validation data and cross-validation with custom metric
      l4_loss <- function(preds, dtrain) {
        labels <- getinfo(dtrain, "label")
        return(list(name="l4",value=mean((preds-labels)^4),higher_better=FALSE))
      }
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
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
      expect_equal(bst$best_iter, 57)
      expect_lt(abs(bst$best_score - 3.058637),TOLERANCE)
      # CV
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
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
      expect_equal(cvbst$best_iter, 52)
      expect_lt(abs(cvbst$best_score - 2.932338),TOLERANCE)
      
      # Use of validation data and gaussian_neg_log_likelihood as metric
      gp_model <- GPModel(group_data = group_data_train)
      set_prediction_data(gp_model, group_data_pred = group_data_test)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 10,
                       learning_rate = 0.01, max_depth = 6, min_data_in_leaf = 5,
                       objective = "regression_l2", verbose = 0,
                       valids = valids, early_stopping_rounds = 5,
                       use_gp_model_for_validation = TRUE, metric = "gaussian_neg_log_likelihood")
      expect_equal(bst$best_iter, 10)
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test, 
                      pred_latent = FALSE, predict_var = TRUE)
      nll <- 0.5 * mean((y_test - pred[['response_mean']])^2 / 
                          pred[['response_var']] + log(pred[['response_var']] * 2 * pi))
      expect_lt(abs(bst$best_score - nll),TOLERANCE)
      # Use of validation data and gaussian_neg_log_likelihood as metric but set use_gp_model_for_validation = FALSE
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 10,
                       learning_rate = 0.01, max_depth = 6, min_data_in_leaf = 5,
                       objective = "regression_l2", verbose = 0,
                       valids = valids, early_stopping_rounds = 5,
                       use_gp_model_for_validation = FALSE, metric = "gaussian_neg_log_likelihood")
      expect_equal(bst$best_iter, 10)
      predtrain <- predict(bst, data = X_train, group_data_pred = group_data_train, pred_latent = TRUE)
      var_est <- var(y_train - predtrain$fixed_effect)
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test, pred_latent = TRUE)
      nll <- 0.5 * mean((y_test - pred[['fixed_effect']])^2 / var_est + log(var_est * 2 * pi))
      expect_lt(abs(bst$best_score - nll),TOLERANCE)
      # Use of validation data and gaussian_neg_log_likelihood as metric without a GPModel
      bst <- gpb.train(data = dtrain, nrounds = 10, learning_rate = 0.01, max_depth = 6, min_data_in_leaf = 5,
                       objective = "regression_l2", verbose = 0,
                       valids = valids, early_stopping_rounds = 5,
                       metric = "gaussian_neg_log_likelihood")
      expect_equal(bst$best_iter, 10)
      predtrain <- predict(bst, data = X_train, pred_latent = TRUE)
      var_est <- var(y_train - predtrain)
      pred <- predict(bst, data = X_test, pred_latent = TRUE)
      nll <- 0.5 * mean((y_test - pred)^2 / var_est + log(var_est * 2 * pi))
      expect_lt(abs(bst$best_score - nll),TOLERANCE)
      
      # Use Nelder-Mead for training
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params = list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6))
      bst <- gpboost(data = X_train,
                     label = y_train,
                     gp_model = gp_model,
                     nrounds = 62,
                     learning_rate = 0.01,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "regression_l2",
                     verbose = 0,
                     leaves_newton_update = FALSE)
      cov_pars <- c(0.004823767, 0.592422707, 0.394167937)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean)-c(0.4157265, -0.1696440, -1.2674184,
                                                        rep(0,n_new)))),TOLERANCE)
      expect_lt(sum(abs(head(pred$fixed_effect)-c(4.818977, 4.174924, 3.269181, 4.222688, 4.997808, 4.947587))),TOLERANCE)
      
      # Use BFGS for training
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params = list(optimizer_cov="bfgs"))
      bst <- gpboost(data = X_train,
                     label = y_train,
                     gp_model = gp_model,
                     nrounds = 62,
                     learning_rate = 0.01,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "regression_l2",
                     verbose = 0,
                     leaves_newton_update = FALSE)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
      
      # Use Adam for training
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params = list(optimizer_cov="adam"))
      bst <- gpboost(data = X_train,
                     label = y_train,
                     gp_model = gp_model,
                     nrounds = 62,
                     learning_rate = 0.01,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "regression_l2",
                     verbose = 0,
                     leaves_newton_update = FALSE)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
      
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
    })
    
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
      
      # Train model
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 20,
                       learning_rate = 0.05,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "regression_l2",
                       verbose = 0)
      cov_pars_est <- c(0.1358229, 0.9099908, 0.1115316)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                      predict_var=TRUE, pred_latent = TRUE)
      pred_re <- c(0.19200894, 0.08380017, 0.59402383, -0.75484438)
      pred_fe <- c(3.920440, 3.641091, 4.536346, 4.951052)
      pred_cov <- c(0.3612252, 0.1596113, 0.1664702, 0.2577366)
      pred_cov_no_nugget <- pred_cov + cov_pars_est[1]
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_cov)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.5229658),TOLERANCE)
      expect_lt(abs(sqrt(mean((pred$fixed_effect - y_test)^2))-1.170505),TOLERANCE)
      expect_lt(abs(sqrt(mean((pred$fixed_effect + pred$random_effect_mean - y_test)^2))-0.8304062),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean, n=4)-(pred_re+pred_fe))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$response_var, n=4)-pred_cov_no_nugget)),TOLERANCE)
      # Use other covariance parameters for prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                      predict_var=TRUE, pred_latent = TRUE, cov_pars = c(0.1358229, 0.9099908, 0.1115316))
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_cov)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                      predict_var=TRUE, pred_latent = TRUE, cov_pars = c(0.2, 1.5, 0.2))
      pred_re2 <- c(0.2182825, 0.1131264, 0.5737999, -0.7441675)
      pred_cov2 <- c(0.3540400, 0.1704857, 0.1720302, 0.2562620)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re2)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_cov2)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      
      # Train model using Nelder-Mead
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6))
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 20,
                       learning_rate = 0.05,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "regression_l2",
                       verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.1286928, 0.9140254, 0.1097192))),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(0.17291900, 0.09483055, 0.64271850, -0.78676614))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.3667703, 0.1596594, 0.1672984, 0.2607827))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(3.840684, 3.688580, 4.591930, 4.976685))),TOLERANCE)
      
      # Use validation set to determine number of boosting iteration
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 100,
                       learning_rate = 0.05,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "regression_l2",
                       verbose = 0,
                       valids = valids,
                       early_stopping_rounds = 5,
                       use_gp_model_for_validation = FALSE,
                       seed = 0)
      expect_equal(bst$best_iter, 27)
      expect_lt(abs(bst$best_score - 1.293498),TOLERANCE)
      
      # Also use GPModel for calculating validation error
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
      gp_model$set_prediction_data(gp_coords_pred = coords_test)
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 100,
                       learning_rate = 0.05,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "regression_l2",
                       verbose = 0,
                       valids = valids,
                       early_stopping_rounds = 5,
                       use_gp_model_for_validation = TRUE,
                       seed = 0)
      expect_equal(bst$best_iter, 27)
      expect_lt(abs(bst$best_score - 0.5485127),TOLERANCE)
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
      
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2",
                       verbose = 0)
      cov_pars_est <- c(0.24800160, 0.89155814, 0.08301144)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      pred_re <- c(-0.4983553, -0.7873598, -0.5955449, -0.2461463)
      pred_cov <- c(0.4768212, 0.5950547, 0.6215380, 0.8378539)
      pred_fe <- c(4.683095, 4.534746, 4.602277, 4.457230)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_cov)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      
      # Same thing with Vecchia approximation
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "vecchia", num_neighbors = ntrain-1, 
                                          num_neighbors_pred = ntrain+ntest-1, vecchia_ordering = "none",
                                          vecchia_pred_type = "order_obs_first_cond_all"), file='NUL')
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_cov)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      
      # Same thing with Vecchia approximation and random ordering
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "vecchia", num_neighbors = ntrain-1, 
                                          num_neighbors_pred = ntrain+ntest-1, vecchia_ordering = "random",
                                          vecchia_pred_type = "order_obs_first_cond_all"), file='NUL')
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-pred_cov)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-pred_fe)),TOLERANCE)
      
      # Same thing with Vecchia approximation and Nelder-Mead
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "vecchia", num_neighbors = ntrain-1,
                                          num_neighbors_pred = ntrain+ntest-1, vecchia_ordering = "none",
                                          vecchia_pred_type = "order_obs_first_cond_all"), file='NUL')
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.24097347, 0.88916662, 0.08253709))),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(-0.4969191, -0.7867247, -0.5883281, -0.2374269))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.4761964, 0.5945182, 0.6208525, 0.8364343))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(4.679265, 4.562299, 4.570425, 4.392607))),TOLERANCE)
      
      # Same thing with Wendland covariance function
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "wendland",
                                          cov_fct_taper_shape = 1, cov_fct_taper_range = 0.2), file='NUL')
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
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
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6))
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
      gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
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
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 20,
                       learning_rate = 0.05, max_depth = 6,
                       min_data_in_leaf = 5, objective = "regression_l2", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.2386092, 0.9050819, 0.0835053 ))),TOLERANCE)
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, predict_var=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(-0.4893557, -0.7984212, -0.5994199, -0.2511335))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(4.650092, 4.574518, 4.618443, 4.409184))),TOLERANCE)
    })
    
    test_that("GPBoost algorithm with Nesterov acceleration for grouped random effects model ", {
      
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
      group_data <- cbind(group,group2)
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
                     feature_pre_filter = FALSE,
                     use_nesterov_acc = TRUE)
      folds <- list()
      for(i in 1:4) folds[[i]] <- as.integer(1:(ntrain/4) + (ntrain/4) * (i-1))
      
      # CV for finding number of boosting iterations with use_gp_model_for_validation = FALSE
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      cvbst <- gpb.cv(params = params,
                      data = dtrain,
                      gp_model = gp_model,
                      nrounds = 100,
                      nfold = 4,
                      eval = "l2",
                      early_stopping_rounds = 5,
                      use_gp_model_for_validation = FALSE,
                      fit_GP_cov_pars_OOS = FALSE,
                      folds = folds,
                      verbose = 0)
      expect_equal(cvbst$best_iter, 19)
      expect_lt(abs(cvbst$best_score-1.040297), TOLERANCE)
      # CV for finding number of boosting iterations with use_gp_model_for_validation = TRUE
      cvbst <- gpb.cv(params = params,
                      data = dtrain,
                      gp_model = gp_model,
                      nrounds = 100,
                      nfold = 4,
                      eval = "l2",
                      early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE,
                      fit_GP_cov_pars_OOS = FALSE,
                      folds = folds,
                      verbose = 0)
      expect_equal(cvbst$best_iter, 19)
      expect_lt(abs(cvbst$best_score-0.6608819), TOLERANCE)
      
      # Create random effects model and train GPBoost model
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      bst <- gpboost(data = X_train,
                     label = y_train,
                     gp_model = gp_model,
                     nrounds = 20,
                     learning_rate = 0.01,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "regression_l2",
                     verbose = 0,
                     leaves_newton_update = FALSE,
                     use_nesterov_acc = TRUE)
      cov_pars <- c(0.01806612, 0.59318355, 0.39198746)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
      
      # Prediction
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test, pred_latent = TRUE)
      expect_lt(sqrt(mean((pred$fixed_effect - f_test)^2)),0.271)
      expect_lt(sqrt(mean((pred$fixed_effect - y_test)^2)),1.018)
      expect_lt(sqrt(mean((pred$fixed_effect + pred$random_effect_mean - y_test)^2)),0.238)
      expect_lt(sum(abs(tail(pred$random_effect_mean)-c(0.3737357, -0.1906376, -1.2750302,
                                                        rep(0,n_new)))),TOLERANCE)
      expect_lt(sum(abs(head(pred$fixed_effect)-c(4.921429, 4.176900, 2.743165,
                                                  4.141866, 5.018322, 4.935220))),TOLERANCE)
      
      # Using validation set
      # Do not include random effect predictions for validation
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
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
                       use_nesterov_acc = TRUE)
      expect_equal(bst$best_iter, 19)
      expect_lt(abs(bst$best_score - 1.035405),TOLERANCE)
      # Include random effect predictions for validation 
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
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
                       use_gp_model_for_validation = TRUE,
                       use_nesterov_acc = TRUE)
      expect_equal(bst$best_iter, 19)
      expect_lt(abs(bst$best_score - 0.05520368),TOLERANCE)
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
      bst$finalize()
      expect_null(bst$.__enclos_env__$private$handle)
      rm(bst)
      rm(gp_model)
      # Load from file and make predictions again with save_raw_data = FALSE option
      bst_loaded <- gpb.load(filename = filename)
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test, predict_var= TRUE, pred_latent = TRUE)
      expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
      expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
      expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
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
      bst_loaded <- gpb.load(filename = filename_save_raw_data)
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
      bst$finalize()
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
      bst$finalize()
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
      bst_loaded <- gpb.load(model_str = model_str_raw_data)
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