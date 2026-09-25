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
    
    test_that("GPBoost algorithm with grouped random effects for Poisson regression", {
      
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
      mu <- exp(f + eps)
      y <- qpois(sim_rand_unif(n=n, init_c=0.04532), lambda = mu)
      # Split in training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      group_data_train <- group_data[1:ntrain,]
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      group_data_test <- group_data[1:ntest+ntrain,]
      eps_test <- eps[1:ntest+ntrain]
      # Data for Booster
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      
      vec_chol_or_iterative <- c("iterative", "cholesky")
      for (inv_method in vec_chol_or_iterative) {
        PC <- "ssor"
        if(inv_method == "iterative") {
          tolerance_loc_1 <- TOLERANCE_LOOSE
          tolerance_loc_2 <- 0.1
          tolerance_loc_3 <- 2
          tolerance_loc_4 <- 15
        } else {
          tolerance_loc_1 <- TOLERANCE
          tolerance_loc_2 <- TOLERANCE
          tolerance_loc_3 <- TOLERANCE
          tolerance_loc_4 <- TOLERANCE
        }
        # Train model
        gp_model <- GPModel(group_data = group_data_train, likelihood = "poisson", matrix_inversion_method = inv_method)
        params_gp_v2 <- DEFAULT_OPTIM_PARAMS_V2
        params_gp_v2$init_cov_pars <- rep(1,2)
        params_gp_v2$cg_preconditioner_type <- PC
        gp_model$set_optim_params(params=params_gp_v2)
        bst <- gpboost(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 30,
                       learning_rate = 0.1,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "poisson",
                       verbose = 0)
        cov_pars_est <- c(0.5298689, 0.3680592)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),tolerance_loc_2)
        # Prediction
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-1.8259542, 0.9549629, -0.8691215, 0.4164422))),tolerance_loc_3)
        expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-0.9894769, -0.9276130, -1.0428837, rep(0,3)))),tolerance_loc_2)
        # Predict response
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.05882611, 4.07141506, 0.65698516, 2.37612226))),tolerance_loc_3)
        expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.05908199, 28.18720228, 1.28493044, 10.59000035))),tolerance_loc_4)
      }
    })
    
    test_that("GPBoost algorithm with grouped random effects for gamma regression", {
      
      OPTIM_PARAMS_GAMMA <- DEFAULT_OPTIM_PARAMS_V2
      OPTIM_PARAMS_GAMMA$estimate_aux_pars = FALSE
      OPTIM_PARAMS_GAMMA$init_aux_pars = 1.
      OPTIM_PARAMS_GAMMA$init_cov_pars <- rep(1,2)
      
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
      mu <- exp(f + eps)
      shape <- 1
      y <- qgamma(sim_rand_unif(n=n, init_c=0.652), scale = mu/shape, shape = shape)
      # Split in training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      group_data_train <- group_data[1:ntrain,]
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      group_data_test <- group_data[1:ntest+ntrain,]
      eps_test <- eps[1:ntest+ntrain]
      # Data for Booster
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      
      vec_chol_or_iterative <- c("iterative", "cholesky")
      for (inv_method in vec_chol_or_iterative) {
        OPTIM_PARAMS_GAMMA$cg_preconditioner_type <- "ssor"
        if(inv_method == "iterative") {
          tolerance_loc_1 <- 2*TOLERANCE_LOOSE
          tolerance_loc_2 <- 0.1
          tolerance_loc_3 <- 1
          tolerance_loc_4 <- 10
          tolerance_loc_5 <- 50
        } else {
          tolerance_loc_1 <- TOLERANCE
          tolerance_loc_2 <- TOLERANCE
          tolerance_loc_3 <- TOLERANCE
          tolerance_loc_4 <- TOLERANCE
          tolerance_loc_5 <- TOLERANCE
        }
        # Train model
        gp_model <- GPModel(group_data = group_data_train, likelihood = "gamma", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=OPTIM_PARAMS_GAMMA)
        bst <- gpboost(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 30,
                       learning_rate = 0.1,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "gamma",
                       verbose = 0)
        cov_pars_est <- c(0.5953036, 0.5056386)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),tolerance_loc_1)
        # Prediction
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-1.4076979, 0.8579932, -1.1317222, 0.5114238))),tolerance_loc_3)
        # Predict response
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.04968272, 4.08967031, 0.55919834, 2.89184563))),tolerance_loc_4)
        expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.002805733674, 83.861373015224, 1.567895626242, 41.931035647798))),tolerance_loc_5)
        
        # Also estimate shape parameter
        gp_model <- GPModel(group_data = group_data_train, likelihood = "gamma", matrix_inversion_method = inv_method)
        params_shape <- OPTIM_PARAMS_GAMMA
        params_shape$estimate_aux_pars <- TRUE
        gp_model$set_optim_params(params=params_shape)
        bst <- gpboost(data = dtrain,  gp_model = gp_model, nrounds = 30,
                       learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
                       objective = "gamma", verbose = 0)
        cov_pars_est <- c(0.6015308, 0.5169128)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-1.447807)),tolerance_loc_2)
        # Standard errors of the covariance and auxiliary parameters cannot be calculated for a model that is
        #   used in the GPBoost algorithm: they are obtained from the Hessian of the negative log-likelihood,
        #   which requires the fixed effects of the location parameter given by the tree ensemble. Requesting
        #   them returns the estimates alone and does not change the state of the model
        expect_false(gp_model$can_calculate_standard_errors_cov_pars())
        expect_false(gp_model$can_calculate_standard_errors_aux_pars())
        negll_before_std_err <- gp_model$get_current_neg_log_likelihood()
        expect_equal(length(as.vector(gp_model$get_cov_pars(std_err = TRUE))),
                     length(as.vector(gp_model$get_cov_pars(std_err = FALSE))))
        expect_equal(length(as.vector(gp_model$get_aux_pars(std_err = TRUE))),
                     length(as.vector(gp_model$get_aux_pars(std_err = FALSE))))
        capture.output( summary(gp_model) , file='NUL')
        expect_lt(abs(gp_model$get_current_neg_log_likelihood() - negll_before_std_err), TOLERANCE_STRICT)
      }
    })
    
    test_that("GPBoost algorithm with grouped random effects for negative binomial regression", {
      
      OPTIM_PARAMS_GAMMA <- DEFAULT_OPTIM_PARAMS_V2
      OPTIM_PARAMS_GAMMA$estimate_aux_pars = FALSE
      OPTIM_PARAMS_GAMMA$init_aux_pars = 1.
      OPTIM_PARAMS_GAMMA$init_cov_pars <- rep(1,2)
      
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
      mu <- exp(f + eps)
      shape <- 0.9
      y <- qnbinom(sim_rand_unif(n=n, init_c=0.134686), mu = mu, size = shape)
      # Split in training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      group_data_train <- group_data[1:ntrain,]
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      group_data_test <- group_data[1:ntest+ntrain,]
      eps_test <- eps[1:ntest+ntrain]
      # Data for Booster
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      
      vec_chol_or_iterative <- c("iterative", "cholesky")
      for (inv_method in vec_chol_or_iterative) {
        OPTIM_PARAMS_GAMMA$cg_preconditioner_type <- "ssor"
        if(inv_method == "iterative") {
          tolerance_loc_1 <- 0.1
          tolerance_loc_2 <- 0.1
          tolerance_loc_3 <- 1
          tolerance_loc_4 <- 15
        } else {
          tolerance_loc_1 <- TOLERANCE_LOOSE
          tolerance_loc_2 <- TOLERANCE
          tolerance_loc_3 <- TOLERANCE
          tolerance_loc_4 <- TOLERANCE_LOOSE
        }
        # Train model
        gp_model <- GPModel(group_data = group_data_train, likelihood = "negative_binomial", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=OPTIM_PARAMS_GAMMA)
        bst <- gpboost(data = dtrain, gp_model = gp_model, nrounds = 30,
                       learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
                       verbose = 0)
        cov_pars_est <- c(0.5539764, 0.4821519 )
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),tolerance_loc_2)
        # Prediction
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-0.0005228073, 0.5865594605, -0.5128394937, 0.6025058992))),tolerance_loc_3)
        # Predict response
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        expect_lt(sum(abs(tail(pred$response_mean, n=4)-c( 0.2521111, 3.0180883, 1.0052383, 3.0666018))),tolerance_loc_3)
        expect_lt(sum(abs(tail(pred$response_var, n=4)-c( 0.338194, 45.251929, 5.690510, 46.669110))), tolerance_loc_4)
        
        # Also estimate shape parameter
        gp_model <- GPModel(group_data = group_data_train, likelihood = "negative_binomial", matrix_inversion_method = inv_method)
        params_shape <- OPTIM_PARAMS_GAMMA
        params_shape$estimate_aux_pars <- TRUE
        gp_model$set_optim_params(params=params_shape)
        bst <- gpboost(data = dtrain,  gp_model = gp_model, nrounds = 30,
                       learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5, verbose = 0)
        cov_pars_est <- c(0.5701868, 0.4876992)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),tolerance_loc_3)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-3.105610326 )),tolerance_loc_3)
      }
    })
    
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
    
    test_that("Parameter tuning for GPBoost algorithm ", {
      
      ntrain <- 1000
      # Simulate fixed effects
      sim_data <- sim_friedman3(n=ntrain, n_irrelevant=5, init_c=0.12644234)
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
      n_new <- 3# number of new random effects in test data
      group[(length(group)-n_new+1):length(group)] <- rep(99999,n_new)
      Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
      b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.53542))
      # Second random effect
      n_obs_gr <- ntrain/m# number of sampels per group
      group2 <- rep(1,ntrain) # grouping variable
      for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
      group2[(length(group2)-n_new+1):length(group2)] <- rep(99999,n_new)
      Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
      b2 <- sqrt(sigma2_2) * qnorm(sim_rand_unif(n=length(unique(group2)), init_c=0.282354))
      eps <- Z1 %*% b1 + Z2 %*% b2
      eps <- eps - mean(eps)
      group_data <- cbind(group,group2)
      
      vec_chol_or_iterative <- c("iterative", "cholesky")
      for (inv_method in vec_chol_or_iterative) {
        PC <- "ssor"
        if(inv_method == "iterative") {
          tolerance_loc_1 <- TOLERANCE_LOOSE
        } else {
          tolerance_loc_1 <- TOLERANCE
        }
        
        # Observed data
        probs <- pnorm(f + eps)
        y <- as.numeric(sim_rand_unif(n=ntrain, init_c=0.6574) < probs)
        # Folds for CV
        group_aux <- rep(1,ntrain) # grouping variable
        for(i in 1:(ntrain/4)) group_aux[(1:4)+4*(i-1)] <- 1:4
        folds <- list()
        for(i in 1:4) folds[[i]] <- as.integer(which(group_aux==i))
        
        #Parameter tuning using cross-validation: deterministic and random grid search
        gp_model <- GPModel(group_data = group_data, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        params_gp <- DEFAULT_OPTIM_PARAMS
        params_gp$init_cov_pars <- rep(1,2)
        params_gp$cg_preconditioner_type <- PC
        gp_model$set_optim_params(params=params_gp)
        dtrain <- gpb.Dataset(data = X, label = y)
        params <- list(objective = "binary", verbose = 0)
        param_grid = list("learning_rate" = c(0.5,0.11), "min_data_in_leaf" = c(20),
                          "max_depth" = c(2), "num_leaves" = 2^17, "max_bin" = c(10,255))
        opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = params,
                                                      data = dtrain, gp_model = gp_model, verbose_eval = 1,
                                                      nrounds = 100, early_stopping_rounds = 5,
                                                      eval = "binary_logloss", folds = folds)
        expect_lt(abs(opt_params$best_score-0.51101812),tolerance_loc_1)
        expect_lte(opt_params$best_iter,68)
        expect_gte(opt_params$best_iter,59)
        expect_equal(opt_params$best_params$learning_rate,0.11)
        expect_equal(opt_params$best_params$max_bin,10)
        expect_equal(opt_params$best_params$max_depth,2)
        
        if (inv_method == "iterative") {
          opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = params,
                                                        data = dtrain, gp_model = gp_model, verbose_eval = 1,
                                                        nrounds = 100, early_stopping_rounds = 5,
                                                        eval = "test_neg_log_likelihood", folds = folds)
          expect_lt(abs(opt_params$best_score-0.51101812),tolerance_loc_1)
          expect_lte(opt_params$best_iter,68)
          expect_gte(opt_params$best_iter,59)
          expect_equal(opt_params$best_params$learning_rate,0.11)
          expect_equal(opt_params$best_params$max_bin,10)
          expect_equal(opt_params$best_params$max_depth,2)
          opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = params,
                                                        data = dtrain, gp_model = gp_model, verbose_eval = 1,
                                                        nrounds = 100, early_stopping_rounds = 5,
                                                        eval = "auc", folds = folds)
          expect_lt(abs(opt_params$best_score-0.65502697),tolerance_loc_1)
          if(inv_method=="cholesky") {
            tol_iter <- 52
            expect_equal(opt_params$best_iter,tol_iter)
            tol_lr <- 0.11
            expect_equal(opt_params$best_params$learning_rate,tol_lr)
          }
          expect_equal(opt_params$best_params$max_bin,10)
          expect_equal(opt_params$best_params$max_depth,2)
          
          # Gamma distribution
          mu <- exp(f + eps)
          shape <- 1
          y <- qgamma(sim_rand_unif(n=n, init_c=0.1864), scale = mu/shape, shape = shape)
          gp_model <- GPModel(group_data = group_data, likelihood = "gamma", matrix_inversion_method = inv_method)
          gp_model$set_optim_params(params=params_gp)
          dtrain <- gpb.Dataset(data = X, label = y)
          params <- list(objective = "gamma", verbose = 0)
          param_grid = list("learning_rate" = c(0.5,0.11), "min_data_in_leaf" = c(20),
                            "max_depth" = c(5), "num_leaves" = 2^17, "max_bin" = c(10,255))
          opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = params,
                                                        data = dtrain, gp_model = gp_model, verbose_eval = 1,
                                                        nrounds = 100, early_stopping_rounds = 5,
                                                        eval = "test_neg_log_likelihood", folds = folds)
          expect_lt(abs(opt_params$best_score-1.177383),tolerance_loc_1)
          if(inv_method=="iterative") tol_iter <- 26 else tol_iter <- 25
          expect_equal(opt_params$best_iter,tol_iter)
          expect_equal(opt_params$best_params$learning_rate,0.11)
          expect_equal(opt_params$best_params$max_bin,10)
          
          # Poisson distribution
          mu <- exp(f + eps)
          y <- qpois(sim_rand_unif(n=n, init_c=0.879), lambda = mu)
          gp_model <- GPModel(group_data = group_data, likelihood = "poisson", matrix_inversion_method = inv_method)
          gp_model$set_optim_params(params=params_gp)
          dtrain <- gpb.Dataset(data = X, label = y)
          params <- list(objective = "poisson", verbose = 0)
          param_grid = list("learning_rate" = c(0.5,0.11), "min_data_in_leaf" = c(20),
                            "max_depth" = c(5), "num_leaves" = 2^17, "max_bin" = c(10,255))
          opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = params,
                                                        data = dtrain, gp_model = gp_model, verbose_eval = 1,
                                                        nrounds = 100, early_stopping_rounds = 5,
                                                        eval = "test_neg_log_likelihood", folds = folds)
          expect_lt(abs(opt_params$best_score-1.560792764),tolerance_loc_1)
          expect_lte(opt_params$best_iter,20)
          expect_gte(opt_params$best_iter,14)
          expect_equal(opt_params$best_params$learning_rate,0.11)
          if(inv_method=="cholesky") {
            tol_bin <- 255
            expect_equal(opt_params$best_params$max_bin,tol_bin)
          }
        }
      }
    })

    # The GPBoost algorithm reuses the approximate Hessian of the lbfgs optimizer of the covariance
    # parameters between boosting iterations. It cannot be reused when the optimization of the previous
    # iteration has not stored a single correction pair, which happens when its line search is already
    # unsuccessful in the first iteration, and the matrix of the solver has to be initialized instead.
    # 'maxit' is deliberately small: what matters here is the second call of the optimizer. The fit is
    # only checked for finiteness, the point of the test is that it runs at all
    test_that("GPBoost algorithm with a Gaussian process model and an lbfgs optimization without a correction pair", {
      
      ntrain <- ntest <- 500
      n <- ntrain + ntest
      sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.69)
      f <- sim_data$f
      f <- f - mean(f)
      X <- sim_data$X
      coords <- matrix(sim_rand_unif(n=n*2, init_c=0.63), ncol=2)
      D <- as.matrix(dist(coords))
      Sigma <- exp(-D/0.1) + diag(1E-20,n)
      C <- t(chol(Sigma))
      eps <- as.vector(C %*% qnorm(sim_rand_unif(n=n, init_c=0.987864)))
      probs <- 1/(1+exp(-(f+eps)))
      y <- as.numeric(sim_rand_unif(n=n, init_c=0.52574) < probs)
      dtrain <- gpb.Dataset(data = X[1:ntrain,], label = y[1:ntrain])
      gp_model <- GPModel(gp_coords = coords[1:ntrain,], cov_function = "exponential",
                          likelihood = "gaussian_heteroscedastic_fixed_and_random", gp_approx = "vecchia",
                          matrix_inversion_method = "cholesky")
      gp_model$set_optim_params(params = list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 2,
                                             init_coef_aux_pars_from_iid_model = FALSE))
      capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 1,
                                       learning_rate = 0.5, max_depth = 6, min_data_in_leaf = 5,
                                       verbose = 0, deterministic = TRUE), file='NUL')
      cov_pars <- as.vector(gp_model$get_cov_pars())
      expect_equal(length(cov_pars), 4L)
      expect_true(all(is.finite(cov_pars)))
      expect_true(all(cov_pars > 0))
      expect_true(is.finite(gp_model$get_current_neg_log_likelihood()))
    })
    
  }
}
