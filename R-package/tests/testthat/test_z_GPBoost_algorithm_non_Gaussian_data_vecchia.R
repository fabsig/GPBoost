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
    
  }
}
