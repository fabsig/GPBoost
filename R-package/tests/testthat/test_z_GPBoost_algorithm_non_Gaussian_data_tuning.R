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
  }
}
