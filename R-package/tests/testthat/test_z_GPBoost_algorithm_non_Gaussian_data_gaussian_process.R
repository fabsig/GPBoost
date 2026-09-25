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
    
    test_that("GPBoost algorithm with Gaussian process model for binary classification ", {
      
      ntrain <- ntest <- 500
      n <- ntrain + ntest
      # Simulate fixed effects
      sim_data <- sim_non_lin_f(n=n, init_c=0.78345)
      f <- sim_data$f/2
      f <- f - mean(f)
      X <- sim_data$X
      # Simulate spatial Gaussian process
      sigma2_1 <- 1 # marginal variance of GP
      rho <- 0.1 # range parameter
      d <- 2 # dimension of GP locations
      coords <- matrix(sim_rand_unif(n=n*d, init_c=0.63), ncol=d)
      D <- as.matrix(dist(coords))
      Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
      C <- t(chol(Sigma))
      b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.987864))
      eps <- as.vector(C %*% b_1)
      eps <- eps - mean(eps)
      # Observed data
      probs <- pnorm(f + eps)
      y <- as.numeric(sim_rand_unif(n=n, init_c=0.52574) < probs)
      # Split into training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      coords_train <- coords[1:ntrain,]
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      coords_test <- coords[1:ntest+ntrain,]
      eps_test <- eps[1:ntest+ntrain]
      
      init_cov_pars <- c(1,mean(dist(coords))/3)
      params = DEFAULT_OPTIM_PARAMS
      params$init_cov_pars <- init_cov_pars
      
      # Train model
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=params)
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 9,
                       learning_rate = 0.2,
                       max_depth = 10,
                       min_data_in_leaf = 5,
                       objective = "binary",
                       verbose = 0)
      cov_pars_est <- c(0.1776908, 0.1887078)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = TRUE)
      pred_re <- c(-0.25248234, 0.07336944, 0.19282985, 0.04100225)
      pred_fe <- c(0.4087100, -0.5570364, -0.7904685, 0.5055812)
      expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-pred_re)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.09672839, 0.10432856, 0.09164587, 0.09215657))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-pred_fe)),TOLERANCE)
      # Predict response
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean,n=4)-c(0.5592939, 0.3226671, 0.2836602, 0.6995181))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$response_var,n=4)-c(0.2464842, 0.2185530, 0.2031971, 0.2101925))),TOLERANCE)
      # Predictive covariance
      cov_exp <- c(1.043281e-01, -6.034087e-05, -8.979587e-05, -6.034087e-05, 9.164516e-02, 
                   4.336540e-03, -8.979587e-05, 4.336540e-03, 9.215582e-02)
      cov_exp_resp <- cov_exp
      pred <- predict(bst, data =  tail(X_test,n=3), gp_coords_pred = tail(coords_test,n=3), 
                      predict_cov_mat=TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=3)-tail(pred_re,n=3))),TOLERANCE)
      expect_lt(sum(abs(as.vector(pred$random_effect_cov)-cov_exp)),TOLERANCE)
      # pred <- predict(bst, data =  tail(X_test,n=3), gp_coords_pred = tail(coords_test,n=3), 
      #                 predict_var=TRUE, pred_latent = FALSE)
      # expect_lt(sum(abs(tail(pred$response_mean, n=4)-tail(pred_re+pred_fe,n=3))),TOLERANCE)
      # expect_lt(sum(abs(as.vector(pred$response_var)-cov_exp_resp)),TOLERANCE)
      
      # Sampling from the posterior
      pred <- predict(bst, data =  tail(X_test,n=3), gp_coords_pred = tail(coords_test,n=3), 
                      sample_posterior = TRUE, num_post_samples = 10000, pred_latent = TRUE)
      tol_mu <- 0.02
      expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-tail(pred_re+pred_fe,n=3))), tol_mu)
      expect_lt(sum(abs(cov(t(pred$posterior_samples))-cov_exp)), tol_mu)
      # not yet implemented
      # pred <- predict(bst, data =  tail(X_test,n=3), gp_coords_pred = tail(coords_test,n=3), 
      #                 sample_posterior = TRUE, num_post_samples = 10000, pred_latent = FALSE)
      # expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-tail(pred_re+pred_fe,n=3))),tol_mu)
      # expect_lt(sum(abs(cov(t(pred$posterior_samples))-cov_exp_resp)), 0.03)
      
      # Use validation set to determine number of boosting iteration with use_gp_model_for_validation = TRUE
      dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
      valids <- list(test = dtest)
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=params)
      gp_model$set_prediction_data(gp_coords_pred = coords_test)
      bst <- gpb.train(data = dtrain, gp_model = gp_model,
                       nrounds = 20, learning_rate = 0.2, max_depth = 10,
                       min_data_in_leaf = 5, objective = "binary",
                       verbose = 0, valids = valids, early_stopping_rounds = 2,
                       use_gp_model_for_validation = TRUE)
      expect_equal(bst$best_iter, 9)
      expect_lt(abs(bst$best_score - 0.5785662),TOLERANCE)
      
      # Train tree-boosting model while holding the GPModel fix
      init_cov_pars = c(2.4,1.1)
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params = list(init_cov_pars = init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain, gp_model = gp_model, train_gp_model_cov_pars = FALSE,
                       nrounds = 2, objective = "binary", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-init_cov_pars)),TOLERANCE)
      
      # Training with Vecchia approximation
      for(inv_method in c("cholesky", "iterative")){
        if(inv_method == "iterative"){
          tolerance_loc <- 0.1
        } else{
          tolerance_loc <- TOLERANCE
        }
        capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                            likelihood = "bernoulli_probit", gp_approx = "vecchia", 
                                            num_neighbors = 30, vecchia_ordering = "none", matrix_inversion_method = inv_method),
                        file='NUL')
        if(inv_method == "iterative"){
          params$num_rand_vec_trace = 500 
          params$cg_delta_conv = sqrt(1e-6)
          params$cg_preconditioner_type = "piv_chol_on_Sigma"
        }
        gp_model$set_optim_params(params=params)
        bst <- gpb.train(data = dtrain, gp_model = gp_model,
                         nrounds = 9, learning_rate = 0.2, max_depth = 10,
                         min_data_in_leaf = 5, objective = "binary", verbose = 0)
        cov_pars_est <- c(0.1786872, 0.1902082)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),tolerance_loc)
        # Prediction
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", num_neighbors_pred = 30)
        pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                        predict_var = TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-c(-0.25123649, 0.07750260, 0.19457371, 0.04771122))),tolerance_loc)
        expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.09503200, 0.10440602, 0.09169082, 0.09131758))),tolerance_loc)
        if(inv_method == "iterative") tolerance_loc <- 0.3
        expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(0.4060860, -0.5598213, -0.7936279, 0.5029883))),tolerance_loc)
        
        # Train tree-boosting model while holding the GPModel fix
        capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                            likelihood = "bernoulli_probit", gp_approx = "vecchia", 
                                            num_neighbors = 30, vecchia_ordering = "none", matrix_inversion_method = inv_method),
                        file='NUL')
        gp_model$set_optim_params(params = list(init_cov_pars = init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
        bst <- gpb.train(data = dtrain, gp_model = gp_model, train_gp_model_cov_pars = FALSE,
                         nrounds = 2, objective = "binary", verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-init_cov_pars)),TOLERANCE)
      }
      
      # Training with Wendland covariance
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "wendland",
                                          cov_fct_taper_shape = 1, cov_fct_taper_range = 0.2,
                                          likelihood = "bernoulli_probit"), file='NUL')
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 9,
                       learning_rate = 0.2,
                       max_depth = 10,
                       min_data_in_leaf = 5,
                       objective = "binary",
                       verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.1632674)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-c(-0.26087248, -0.04472871, 0.19212327, 0.15252393))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.1364254, 0.1208446, 0.1170245, 0.1250811))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(0.4514654, -0.6156319, -0.5838128, 0.4800570))),TOLERANCE)
      
      # Wendland covariance and Nelder-Mead
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "wendland",
                                          cov_fct_taper_shape = 1, cov_fct_taper_range = 0.2,
                                          likelihood = "bernoulli_probit"), file='NUL')
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 9,
                       learning_rate = 0.2,
                       max_depth = 10,
                       min_data_in_leaf = 5,
                       objective = "binary",
                       verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.1626625 )),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-c(-0.25745441, -0.04200966, 0.19468910, 0.15492142))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.1359487, 0.1204699, 0.1166453, 0.1246189))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(0.4443580, -0.6230536, -0.5912199, 0.4729334))),TOLERANCE)
      
      # Tapering
      capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                          gp_approx = "tapering", likelihood = "bernoulli_probit",
                                          cov_fct_taper_shape = 1, cov_fct_taper_range = 10), file='NUL')
      gp_model$set_optim_params(params=params)
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 9,
                       learning_rate = 0.2,
                       max_depth = 10,
                       min_data_in_leaf = 5,
                       objective = "binary",
                       verbose = 0)
      cov_pars_est <- c(0.1777562, 0.1898083)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-c(-0.25264933, 0.07306853, 0.19296519, 0.04058235))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.09654161, 0.10422011, 0.09149145, 0.09198057))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(0.4089283, -0.5569100, -0.7903136, 0.5057746))),TOLERANCE)
      
      ## CV function
      # Folds for CV
      group_aux <- rep(1,ntrain) # grouping variable
      for(i in 1:(ntrain/4)) group_aux[(1:4)+4*(i-1)] <- 1:4
      folds <- list()
      for(i in 1:4) folds[[i]] <- as.integer(which(group_aux==i))
      
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
      cvbst <- gpb.cv(data = dtrain, gp_model = gp_model, nrounds = 10,
                      learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
                      objective = "binary", eval = "binary_error",
                      early_stopping_rounds = 5, use_gp_model_for_validation = TRUE,
                      folds = folds, verbose = 0)
      expcet_iter <- 8
      expcet_score <- 0.288
      expect_equal(cvbst$best_iter, expcet_iter)
      expect_lt(abs(cvbst$best_score-expcet_score), TOLERANCE)
      
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "gaussian")
      gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
      capture.output( cvbst <- gpb.cv(data = dtrain, gp_model = gp_model, nrounds = 10,
                                      learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
                                      objective = "binary", eval = "binary_error",
                                      early_stopping_rounds = 5, use_gp_model_for_validation = TRUE,
                                      folds = folds, verbose = 0) 
                      , file='NUL')
      expcet_iter <- 10
      expcet_score <- 0.322
      expect_equal(cvbst$best_iter, expcet_iter)
      expect_lt(abs(cvbst$best_score-expcet_score), TOLERANCE)
      
      # With Vecchia approx
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit", gp_approx="vecchia")
      gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
      cvbst <- gpb.cv(data = dtrain, gp_model = gp_model, nrounds = 10,
                      learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
                      objective = "binary", eval = "binary_error",
                      early_stopping_rounds = 5, use_gp_model_for_validation = TRUE,
                      folds = folds, verbose = 0)
      expcet_score <- 0.282
      expect_gte(cvbst$best_iter, 8)
      expect_lte(cvbst$best_iter, 10)
      expect_lt(abs(cvbst$best_score-expcet_score), 3*TOLERANCE)
      
      # likelihood and objective do not match
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "gaussian", gp_approx="vecchia")
      gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
      capture.output( cvbst <- gpb.cv(data = dtrain, gp_model = gp_model, nrounds = 10,
                                      learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
                                      objective = "binary", eval = "binary_error",
                                      early_stopping_rounds = 5, use_gp_model_for_validation = TRUE,
                                      folds = folds, verbose = 0) , file='NUL')
      expcet_iter <- 10
      expcet_score <- 0.32
      expect_equal(cvbst$best_iter, expcet_iter)
      expect_lt(abs(cvbst$best_score-expcet_score), 15*TOLERANCE)
    })
    
    test_that("GPBoost algorithm with GP model for binary classification with multiple observations at the same location", {
      
      ntrain <- ntest <- 400
      n <- ntrain + ntest
      # Simulate fixed effects
      sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.69)
      f <- sim_data$f
      f <- f - mean(f)
      X <- sim_data$X
      # Simulate spatial Gaussian process
      sigma2_1 <- 1 # marginal variance of GP
      rho <- 0.1 # range parameter
      d <- 2 # dimension of GP locations
      coords <- matrix(sim_rand_unif(n=n*d/8, init_c=0.12), ncol=d)
      coords <- rbind(coords,coords,coords,coords,coords,coords,coords,coords)
      D <- as.matrix(dist(coords))
      Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-15,n)
      C <- t(chol(Sigma))
      b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.987864))
      eps <- as.vector(C %*% b_1)
      # Observed data
      probs <- pnorm(f + eps)
      y <- as.numeric(sim_rand_unif(n=n, init_c=0.52574) < probs)
      # Split into training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      coords_train <- coords[1:ntrain,]
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      coords_test <- coords[1:ntest+ntrain,]
      eps_test <- eps[1:ntest+ntrain]
      # Folds for CV
      group_aux <- rep(1,ntrain) # grouping variable
      for(i in 1:(ntrain/4)) group_aux[(1:4)+4*(i-1)] <- 1:4
      folds <- list()
      for(i in 1:4) folds[[i]] <- as.integer(which(group_aux==i))
      
      init_cov_pars <- c(1,mean(dist(unique(coords_train)))/3)
      
      # Train model
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=list(maxit=10, lr_cov=0.01, optimizer_cov="gradient_descent",
                                            lr_coef=0.1, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 2,
                       learning_rate = 0.5,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "binary",
                       verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.6094175, 0.1137471))),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(0.4466759, 0.5293270, 0.5031217, 0.5293270))),TOLERANCE)
      # Predict response
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.4775558, 0.5465922, 0.2294873, 0.3157580))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.2494963, 0.2478292, 0.1768229, 0.2160549))),TOLERANCE)
      
      # Use validation set to determine number of boosting iteration
      dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
      valids <- list(test = dtest)
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit")
      params_ES = DEFAULT_OPTIM_PARAMS_EARLY_STOP
      params_ES$init_cov_pars <- init_cov_pars
      gp_model$set_optim_params(params=params_ES)
      gp_model$set_prediction_data(gp_coords_pred = coords_test)
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 10,
                       learning_rate = 0.1,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "binary",
                       verbose = 0,
                       valids = valids,
                       early_stopping_rounds = 2,
                       use_gp_model_for_validation = TRUE, metric = "binary_logloss")
      expect_equal(bst$best_iter, 10)
      expect_lt(abs(bst$best_score - 0.6129572),TOLERANCE)
      
      # CV for finding number of boosting iterations when use_gp_model_for_validation = TRUE
      gp_model <- GPModel(gp_coords = coords_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
      cvbst <- gpb.cv(data = dtrain, gp_model = gp_model, nrounds = 10,
                      learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
                      objective = "binary", eval = "binary_error",
                      early_stopping_rounds = 5, use_gp_model_for_validation = TRUE,
                      folds = folds, verbose = 0)
      expcet_iter <- 6
      expcet_score <- 0.315
      expect_equal(cvbst$best_iter, expcet_iter)
      expect_lt(abs(cvbst$best_score-expcet_score), TOLERANCE)
    })
    
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
    
    test_that("GPBoost algorithm for binary classification: equivalence of Vecchia approximation", {
      
      ntrain <- ntest <- 100
      n <- ntrain + ntest
      # Simulate fixed effects
      sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.69)
      f <- sim_data$f
      f <- f - mean(f)
      X <- sim_data$X
      # Simulate grouped random effects
      sigma2_1 <- 1 # marginal variance of GP
      rho <- 0.1 # range parameter
      d <- 2 # dimension of GP locations
      coords <- matrix(sim_rand_unif(n=n*d, init_c=0.63), ncol=d)
      D <- as.matrix(dist(coords))
      Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
      C <- t(chol(Sigma))
      b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.987864))
      eps <- as.vector(C %*% b_1)
      # Observed data
      probs <- pnorm(f + eps)
      y <- as.numeric(sim_rand_unif(n=n, init_c=0.52574) < probs)
      # Split in training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      coords_train <- coords[1:ntrain,]
      make_dtrain <- function() gpb.Dataset(data = X_train, label = y_train)
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      coords_test <- coords[1:ntest+ntrain,]
      eps_test <- eps[1:ntest+ntrain]
      
      init_cov_pars <- c(1,mean(dist(coords_train))/3)
      params = DEFAULT_OPTIM_PARAMS_EARLY_STOP_NO_NESTEROV
      params$init_cov_pars <- init_cov_pars
      
      # Train model
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=params)
      bst <- gpb.train(data = make_dtrain(), gp_model = gp_model,
                       nrounds = 5, learning_rate = 0.5, max_depth = 6,
                       min_data_in_leaf = 5, objective = "binary", verbose = 0,
                       deterministic = TRUE, force_col_wise = TRUE, num_threads = 1,
                       seed = 1, data_random_seed = 1, feature_fraction_seed = 1, bagging_seed = 1, drop_seed = 1)
      cov_pars_est <- c(0.1195943, 0.1479688) # numerical instability on Mac can cause two outcomes
      cov_pars_est_alt <- c(0.1093850, 0.1625686)
      cov_pars_diff <- c(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),
                         sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est_alt)))
      expect_lt(min(cov_pars_diff),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = TRUE)
      P_RE_mean <- c(-0.03827765, -0.15611348, 0.04603207, -0.03903325)
      P_RE_cov <- c(0.1013040, 0.1029115, 0.1098251, 0.1142902)
      P_F <- c(0.2807203, 0.9713023, -0.2379479, 1.1268341)
      expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-P_RE_mean)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-P_RE_cov)),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-P_F)),TOLERANCE)
      
      # Same thing with Vecchia approximation
      for(inv_method in c("cholesky", "iterative")){
        if(inv_method == "iterative"){
          tolerance_loc <- 0.02
        } else{
          tolerance_loc <- TOLERANCE
        }
        capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                            likelihood = "bernoulli_probit", gp_approx = "vecchia", 
                                            num_neighbors = ntrain-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        params_loc <- params
        if(inv_method == "iterative"){
          params_loc$num_rand_vec_trace = 1000 
          params_loc$cg_delta_conv = sqrt(1e-6)
          params_loc$cg_preconditioner_type = "piv_chol_on_Sigma"
        }
        gp_model$set_optim_params(params=params_loc)
        bst <- gpb.train(data = make_dtrain(), gp_model = gp_model,
                         nrounds = 5, learning_rate = 0.5, max_depth = 6,
                         min_data_in_leaf = 5, objective = "binary", verbose = 0, deterministic = TRUE)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),tolerance_loc)
        # Prediction
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", 
                                     nsim_var_pred=2000,
                                     num_neighbors_pred = ntest+ntrain-1)
        pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                        predict_var = TRUE, pred_latent = TRUE)
        adjust_tol <- 1
        if (inv_method == "iterative") adjust_tol <- 1.5
        expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-P_RE_mean)),adjust_tol*tolerance_loc)
        expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-P_RE_cov)),tolerance_loc)
        expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-P_F)),tolerance_loc)
        
        # Same thing with Vecchia approximation and random ordering
        capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                            likelihood = "bernoulli_probit", gp_approx = "vecchia", 
                                            num_neighbors = ntrain-1, vecchia_ordering = "random",
                                            matrix_inversion_method = inv_method), file='NUL')
        gp_model$set_optim_params(params=params_loc)
        bst <- gpb.train(data = make_dtrain(), gp_model = gp_model,
                         nrounds = 5, learning_rate = 0.5, max_depth = 6, deterministic = TRUE,
                         min_data_in_leaf = 5, objective = "binary", verbose = 0)
        adjust_tol <- 1
        if (inv_method == "iterative") adjust_tol <- 10
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),adjust_tol*tolerance_loc)
        
        # Prediction
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", num_neighbors_pred = ntest+ntrain-1)
        pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                        predict_var = TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-P_RE_mean)),adjust_tol*tolerance_loc)
        expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-P_RE_cov)),adjust_tol*tolerance_loc)
        expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-P_F)),adjust_tol*tolerance_loc)
      }
    })
    
    test_that("GPBoost algorithm with Gaussian process model for binary classification with logit link", {
      
      ntrain <- ntest <- 500
      n <- ntrain + ntest
      # Simulate fixed effects
      sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.69)
      f <- sim_data$f
      f <- f - mean(f)
      X <- sim_data$X
      # Simulate spatial Gaussian process
      sigma2_1 <- 1 # marginal variance of GP
      rho <- 0.1 # range parameter
      d <- 2 # dimension of GP locations
      coords <- matrix(sim_rand_unif(n=n*d, init_c=0.63), ncol=d)
      D <- as.matrix(dist(coords))
      Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
      C <- t(chol(Sigma))
      b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.987864))
      eps <- as.vector(C %*% b_1)
      # Observed data
      probs <- 1/(1+exp(-(f+eps)))
      y <- as.numeric(sim_rand_unif(n=n, init_c=0.52574) < probs)
      # Split into training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      coords_train <- coords[1:ntrain,]
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      coords_test <- coords[1:ntest+ntrain,]
      eps_test <- eps[1:ntest+ntrain]
      
      init_cov_pars <- c(1,mean(dist(coords_train))/3)
      
      # Train model
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_logit")
      gp_model$set_optim_params(params=list(maxit=10, lr_cov=0.01, optimizer_cov="gradient_descent",
                                            lr_coef=0.1, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 2,
                       learning_rate = 0.5,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "binary",
                       verbose = 0)
      cov_pars_est <- c(0.41398781, 0.07678912)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = TRUE)
      expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.8197184),TOLERANCE)
      expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-0.9186907),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.3368866, 0.3202246, 0.3128022, 0.3221874))),TOLERANCE)
      # Predict response
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                      predict_var = TRUE, pred_latent = FALSE)
      expect_equal(mean(as.numeric(pred$response_mean>0.5) != y_test),0.362)
      expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.2365583, 0.2499360, 0.2041193, 0.2496736))),TOLERANCE)
    })
    
    if (Sys.getenv("GPBOOST_ADDITIONAL_SLOW_TESTS") == "GPBOOST_ADDITIONAL_SLOW_TESTS") {
      # slow test 
      test_that("GPBoost algorithm with Gaussian process model and 'gaussian_heteroscedastic_fixed_and_random' likelihood", {
        
        ntrain <- ntest <- 500
        n <- ntrain + ntest
        # Simulate fixed effects
        sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.69)
        f <- sim_data$f
        f <- f - mean(f)
        X <- sim_data$X
        # Simulate spatial Gaussian process
        sigma2_1 <- 1 # marginal variance of GP
        rho <- 0.1 # range parameter
        d <- 2 # dimension of GP locations
        coords <- matrix(sim_rand_unif(n=n*d, init_c=0.63), ncol=d)
        D <- as.matrix(dist(coords))
        Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
        C <- t(chol(Sigma))
        b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.987864))
        eps <- as.vector(C %*% b_1)
        # Observed data
        probs <- 1/(1+exp(-(f+eps)))
        y <- as.numeric(sim_rand_unif(n=n, init_c=0.52574) < probs)
        # Split into training and test data
        y_train <- y[1:ntrain]
        X_train <- X[1:ntrain,]
        coords_train <- coords[1:ntrain,]
        dtrain <- gpb.Dataset(data = X_train, label = y_train)
        y_test <- y[1:ntest+ntrain]
        X_test <- X[1:ntest+ntrain,]
        f_test <- f[1:ntest+ntrain]
        coords_test <- coords[1:ntest+ntrain,]
        eps_test <- eps[1:ntest+ntrain]
        
        init_cov_pars <- c(1,mean(dist(coords_train))/3)
        
        # Train model
        gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                            likelihood = "gaussian_heteroscedastic_fixed_and_random", gp_approx = "vecchia",
                            matrix_inversion_method = "iterative")
        gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
        bst <- gpb.train(data = dtrain,
                         gp_model = gp_model,
                         nrounds = 2,
                         learning_rate = 0.5,
                         max_depth = 6,
                         min_data_in_leaf = 5,
                         verbose = 0, deterministic = TRUE)
        # the response is binary while the likelihood is a heteroscedastic Gaussian one, so the model is
        #	misspecified and the optimum of both range parameters lies at the boundary: the mean process
        #	becomes uncorrelated with a marginal variance close to the variance of the observations and the
        #	process of the log-error variance becomes constant. The negative log-likelihood is 361.6 there
        #	and 723.2 at the values that this test expected before, so the fit is well separated from them
        cov_pars_est <- c(2.489125e-01, 1.621601e-06, 5.241877e-07, 1.796105e-04)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),relax_tolerance_stoch(0.4))

        # Prediction
        pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                        predict_var = TRUE, pred_latent = TRUE)
        npred <- dim(X_test)[1]
        expect_lt(sum(abs(pred$fixed_effect[1:4]-c(0.5871531, 0.5670663, 0.6189220, 0.5871531))),relax_tolerance_stoch(2))
        # the mean process is uncorrelated, so its posterior mean at new locations is zero and its
        #	predictive variance is the marginal variance
        expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(0, 0, 0, 0))),relax_tolerance_stoch(0.4))
        expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.2489125, 0.2489125, 0.2489125, 0.2489125))),relax_tolerance_stoch(0.4))
        # Predict response
        pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                        predict_var = TRUE, pred_latent = FALSE)
        expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.6192911, 0.5871531, 0.5967787, -0.7748928))),relax_tolerance_stoch(1))
        expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.2489135, 0.2489135, 0.2489135, 0.2489135))),relax_tolerance_stoch(0.3))
        
        # Parameter tuning
        if (!identical(Sys.info()[["sysname"]], "Darwin")) {# these tests fail on Mac OS
          group_aux <- rep(1,ntrain) # grouping variable
          nfold <- 2
          for(i in 1:(ntrain/nfold)) group_aux[(1:nfold)+nfold*(i-1)] <- 1:nfold
          folds <- list()
          for(i in 1:nfold) folds[[i]] <- as.integer(which(group_aux==i))
          
          params <- list(verbose = 0)
          metric = "crps_gaussian"
          param_grid = list("learning_rate" = c(0.5,0.11), "min_data_in_leaf" = c(20),
                            "max_depth" = c(2), "num_leaves" = 2^17, "max_bin" = c(10,255))
          opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = params,
                                                        data = dtrain, gp_model = gp_model, verbose_eval = 1,
                                                        nrounds = 100, early_stopping_rounds = 5,
                                                        metric = metric, folds = folds)
          expect_lt(abs(opt_params$best_score-0.2826264),0.01)
          # the number of boosting iterations that the tuning selects can differ between builds, so it is
          # only bracketed here (3 with MSVC and with gcc on Linux)
          expect_gte(opt_params$best_iter,2)
          expect_lte(opt_params$best_iter,24)
          expect_equal(opt_params$best_params$learning_rate,0.11)
          expect_gte(opt_params$best_params$max_bin,10)
          expect_lte(opt_params$best_params$max_bin,255)
          expect_equal(opt_params$best_params$max_depth,2)
        }
        
      })## end gaussian_heteroscedastic_fixed_and_random
    }

  }
}
