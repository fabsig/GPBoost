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
    
    test_that("GPBoost algorithm for binary classification with combined Gaussian process and grouped random effects model", {
      
      ntrain <- ntest <- 500
      n <- ntrain + ntest
      # Simulate fixed effects
      sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.6549)
      f <- sim_data$f
      f <- f - mean(f)
      X <- sim_data$X
      # Simulate spatial Gaussian process
      sigma2_1 <- 1 # marginal variance of GP
      rho <- 0.1 # range parameter
      d <- 2 # dimension of GP locations
      coords <- matrix(sim_rand_unif(n=n*d, init_c=0.633), ncol=d)
      D <- as.matrix(dist(coords))
      Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
      C <- t(chol(Sigma))
      b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.67))
      eps <- as.vector(C %*% b_1)
      # Simulate grouped random effects
      sigma2_grp <- 1 # variance of random effect
      m <- 50 # number of categories / levels for grouping variable
      # first random effect
      group <- rep(1,ntrain) # grouping variable
      for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
      group <- c(group, group)
      n_new <- 3# number of new random effects in test data
      group[(length(group)-n_new+1):length(group)] <- rep(99999,n_new)
      Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
      b_grp <- sqrt(sigma2_grp) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.52))
      eps <- C %*% b_1 + Z1 %*% b_grp
      group_data <- group
      eps <- eps - mean(eps)
      # Observed data
      probs <- pnorm(f + eps)
      y <- as.numeric(sim_rand_unif(n=n, init_c=0.234) < probs)
      # Split into training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      coords_train <- coords[1:ntrain,]
      group_data_train <- group_data[1:ntrain]
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      coords_test <- coords[1:ntest+ntrain,]
      group_data_test <- group_data[1:ntest+ntrain]
      eps_test <- eps[1:ntest+ntrain]
      
      init_cov_pars <- c(1,1,mean(dist(coords_train))/3)
      params = DEFAULT_OPTIM_PARAMS
      params$init_cov_pars <- init_cov_pars
      
      # Train model
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          group_data = group_data_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=params)
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 5,
                       learning_rate = 0.5,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "binary",
                       verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.2389226, 0.2944397, 0.3476084))),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      group_data_pred = group_data_test,
                      predict_var = TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.7599847557, 0.5543352568, 0.1063421898, 0.5439185071))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.18240965, 0.24704862, 0.09503084, 0.24807160))),TOLERANCE)
      
      # # The following test is very slow (not run anymore)
      # # Train model using Nelder-Mead
      # gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
      #                     group_data = group_data_train, likelihood = "bernoulli_probit")
      # gp_model$set_optim_params(params=list(optimizer_cov = "nelder_mead", delta_rel_conv=1E-8, init_coef_aux_pars_from_iid_model = FALSE))
      # bst <- gpb.train(data = dtrain,
      #                  gp_model = gp_model,
      #                  nrounds = 5,
      #                  learning_rate = 0.5,
      #                  max_depth = 6,
      #                  min_data_in_leaf = 5,
      #                  objective = "binary",
      #                  verbose = 0)
      # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.2390776, 0.2966670, 0.3499098))),TOLERANCE)
      # # Prediction
      # pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
      #                 group_data_pred = group_data_test,
      #                 predict_var = TRUE, pred_latent = FALSE)
      # expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.7600335, 0.5543040, 0.1062553, 0.5437832))),TOLERANCE)
      # expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.18238257, 0.24705107, 0.09496514, 0.24808303))),TOLERANCE)
      # 
      # # Use validation set to determine number of boosting iteration
      # dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
      # valids <- list(test = dtest)
      # gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
      #                     group_data = group_data_train, likelihood = "bernoulli_probit")
      # gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      # gp_model$set_prediction_data(gp_coords_pred = coords_test, group_data_pred = group_data_test)
      # bst <- gpb.train(data = dtrain,
      #                  gp_model = gp_model,
      #                  nrounds = 100,
      #                  learning_rate = 0.1,
      #                  max_depth = 6,
      #                  min_data_in_leaf = 5,
      #                  objective = "binary",
      #                  verbose = 0,
      #                  valids = valids,
      #                  early_stopping_rounds = 2,
      #                  use_gp_model_for_validation = TRUE)
      # expect_equal(bst$best_iter, 12)
      # expect_lt(abs(bst$best_score - 0.5826652),TOLERANCE)
      
    })
    
  }
}
