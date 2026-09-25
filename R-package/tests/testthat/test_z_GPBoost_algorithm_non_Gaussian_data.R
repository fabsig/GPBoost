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
    
    test_that("GPBoost algorithm with grouped random effects model for binary classification ", {
      
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
      probs <- pnorm(f + eps)
      y <- as.numeric(sim_rand_unif(n=n, init_c=0.574) < probs)
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
      # Data for Booster
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
      valids <- list(test = dtest)
      params <- list(learning_rate = 0.1, objective = "binary")
      # Folds for CV
      group_aux <- rep(1,ntrain) # grouping variable
      for(i in 1:(ntrain/4)) group_aux[(1:4)+4*(i-1)] <- 1:4
      folds <- list()
      for(i in 1:4) folds[[i]] <- as.integer(which(group_aux==i))
      
      vec_chol_or_iterative <- c("iterative", "cholesky")
      for (inv_method in vec_chol_or_iterative) {
        PC <- "ssor"
        if(inv_method == "iterative") {
          tolerance_loc_1 <- 2*TOLERANCE_LOOSE
          tolerance_loc_2 <- 0.1
          tolerance_loc_3 <- 1
          tolerance_loc_4 <- 10
        } else {
          tolerance_loc_1 <- TOLERANCE
          tolerance_loc_2 <- TOLERANCE
          tolerance_loc_3 <- TOLERANCE
          tolerance_loc_4 <- TOLERANCE
        }
        # Label needs to have correct format
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=list(maxit=2, optimizer_cov="gradient_descent", cg_preconditioner_type=PC, init_coef_aux_pars_from_iid_model = FALSE))
        expect_error(gpboost(data = X_train, label = probs[1:ntrain], gp_model = gp_model,
                             objective = "binary", nrounds=1))
        # fisher_scoring cannot be used
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=list(maxit=2, optimizer_cov="fisher_scoring", cg_preconditioner_type=PC, init_coef_aux_pars_from_iid_model = FALSE))
        expect_error(gpboost(data = X_train, label = y_train, gp_model = gp_model,
                             objective = "binary", verbose=0, nrounds=1))
        # Prediction data needs to be set when use_gp_model_for_validation=TRUE
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        capture.output( expect_error(gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                             objective = "binary", train_gp_model_cov_pars=FALSE, nrounds=1, valids=valids)), file='NUL')
        
        # Create random effects model and train GPBoost model
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        params_gp <- DEFAULT_OPTIM_PARAMS_NO_NESTEROV
        params_gp$init_cov_pars <- rep(1,2)
        params_gp$cg_preconditioner_type=PC
        set_optim_params(gp_model, params=params_gp)
        bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                       min_data_in_leaf = 5, objective = "binary", verbose = 0)
        cov_pars <- c(0.4578282, 0.3456973)
        nll_opt <- 372.1352713
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_1)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), tolerance_loc_4)
        # Prediction
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(head(pred$fixed_effect, n=4)-c(0.51189335, -0.05534681, 1.01832308, 0.82839003))),tolerance_loc_3)
        expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.122524, -1.070761, -1.239508,
                                                          rep(0,n_new)))),tolerance_loc_2)
        expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.1291345, 0.1285406, 0.1291397,
                                                         rep(0.8035255,n_new)))),tolerance_loc_2)
        # Predict response
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        resp_mean <- c(0.01602001, 0.63412570, 0.20171037, 0.62036433)
        resp_var <- c(0.01576337, 0.23201030, 0.16102330, 0.23551243)
        expect_lt(sum(abs(tail(pred$response_mean, n=4) - resp_mean)),tolerance_loc_2)
        expect_lt(sum(abs(tail(pred$response_var, n=4) - resp_var)),tolerance_loc_2)
        
        # objective does not need to be set
        bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                       min_data_in_leaf = 5, verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_1)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        expect_lt(sum(abs(tail(pred$response_mean, n=4) - resp_mean)),tolerance_loc_2)
        expect_lt(sum(abs(tail(pred$response_var, n=4) - resp_var)),tolerance_loc_2)
        bst <- gpb.train(data = dtrain, gp_model = gp_model,
                         nrounds = 30, learning_rate = 0.1, max_depth = 6,
                         min_data_in_leaf = 5, verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_1)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        expect_lt(sum(abs(tail(pred$response_mean, n=4) - resp_mean)),tolerance_loc_2)
        expect_lt(sum(abs(tail(pred$response_var, n=4) - resp_var)),tolerance_loc_2)
        
        # Training with alternative likelihood names
        gp_model <- GPModel(group_data = group_data_train, likelihood = "binary_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                       min_data_in_leaf = 5, verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_1)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        expect_lt(sum(abs(tail(pred$response_mean, n=4) - resp_mean)),tolerance_loc_2)
        expect_lt(sum(abs(tail(pred$response_var, n=4) - resp_var)),tolerance_loc_2)
        # Training with alternative objective names
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                                       min_data_in_leaf = 5, objective = "bernoulli_probit", verbose = 0), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),tolerance_loc_1)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        expect_lt(sum(abs(tail(pred$response_mean, n=4) - resp_mean)),tolerance_loc_2)
        expect_lt(sum(abs(tail(pred$response_var, n=4) - resp_var)),tolerance_loc_2)
        # Training with "wrong" default likelihood
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=list(cg_preconditioner_type=PC, init_coef_aux_pars_from_iid_model = FALSE))
        params_gp_gaus <- params_gp
        params_gp_gaus$init_cov_pars <- NULL
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                                       min_data_in_leaf = 5, objective = "binary", verbose = 0), file='NUL')
        cov_pars_logit <- c(1.0573056321, 0.7713219552)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_logit)),0.02)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        resp_mean_logit <- c(0.06530459538, 0.60594941290, 0.34418503952, 0.55480188856)
        resp_var_logit <- c(0.0610399052, 0.2387747219, 0.2257216981, 0.2469967530)
        if(inv_method=="iterative") l_tol <- 0.06 else l_tol <- 0.05
        expect_lt(sum(abs(tail(pred$response_mean, n=4) - resp_mean_logit)),l_tol)
        if(inv_method=="iterative") l_tol <- 0.03 else l_tol <- 0.02
        expect_lt(sum(abs(tail(pred$response_var, n=4) - resp_var_logit)),l_tol)
        # Training with "wrong" default likelihood
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp_gaus)
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                                       min_data_in_leaf = 5, objective = "binary_probit", verbose = 0), file='NUL')
        if(inv_method=="iterative") l_tol <- 0.04 else l_tol <- 0.002
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),l_tol)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        expect_lt(sum(abs(tail(pred$response_mean, n=4) - resp_mean)),0.05)
        expect_lt(sum(abs(tail(pred$response_var, n=4) - resp_var)),0.02)
        # objective and likelihood do not match
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        expect_error({ 
          bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                         nrounds = 30, learning_rate = 0.1, max_depth = 6,
                         min_data_in_leaf = 5, objective = "bernoulli_logit", verbose = 0)
        })
        expect_error({ 
          bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                         nrounds = 30, learning_rate = 0.1, max_depth = 6,
                         min_data_in_leaf = 5, objective = "gamma", verbose = 0)
        })
        expect_error({ 
          bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                         nrounds = 30, learning_rate = 0.1, max_depth = 6,
                         min_data_in_leaf = 5, objective = "regression", verbose = 0)
        })
        
        if(inv_method=="cholesky"){
          # Prediction when having only one grouped random effect
          group_1 <- rep(1,ntrain) # grouping variable
          for(i in 1:m) group_1[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
          probs_1 <- pnorm(f[1:ntrain] + b1[group_1])
          y_1 <- as.numeric(sim_rand_unif(n=ntrain, init_c=0.574) < probs_1)
          gp_model <- GPModel(group_data = group_1, likelihood = "bernoulli_probit")
          gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_NO_NESTEROV)
          bst <- gpboost(data = X_train,
                         label = y_1,
                         gp_model = gp_model,
                         nrounds = 30,
                         learning_rate = 0.1,
                         max_depth = 6,
                         min_data_in_leaf = 5,
                         objective = "binary",
                         verbose = 0,
                         leaves_newton_update = FALSE)
          pred <- predict(bst, data = X_test[1:length(unique(b1)),], group_data_pred = 1:length(unique(b1)), pred_latent = TRUE)
          expect_lt(abs(sqrt(sum((pred$random_effect_mean - b1)^2))-1.667952),TOLERANCE)
          # Prediction for only new groups
          group_test <- c(-1,-1,-2,-2)
          pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test, pred_latent = TRUE)
          fix_eff <- c(0.2292592, 0.3296304, 0.6725046, 0.5069731)
          expect_lt(sum(abs(pred$fixed_effect-fix_eff)),TOLERANCE)
          expect_lt(sum(abs(pred$random_effect_mean-rep(0,4))),TOLERANCE)
          pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test, pred_latent = FALSE)
          resp <- c(0.5739159, 0.6056269, 0.7076881, 0.6598638)
          expect_lt(sum(abs(pred$response_mean-resp)),TOLERANCE)
          # Prediction for only new cluster_ids
          cluster_ids_pred <- c(-1L,-1L,-2L,-2L)
          group_test <- c(1,3,3,9999)
          pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test,
                          cluster_ids_pred = cluster_ids_pred, pred_latent = TRUE)
          expect_lt(sum(abs(pred$random_effect_mean-rep(0,4))),TOLERANCE)
          expect_lt(sum(abs(pred$fixed_effect-fix_eff)),TOLERANCE)
          pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test,
                          cluster_ids_pred = cluster_ids_pred, pred_latent = FALSE)
          expect_lt(sum(abs(pred$response_mean-resp)),TOLERANCE)  
        }
        
        # Train tree-boosting model while holding the GPModel fix
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        bst <- gpboost(data = X_train,
                       label = y_train,
                       gp_model = gp_model,
                       nrounds = 30,
                       learning_rate = 0.1,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       objective = "binary",
                       train_gp_model_cov_pars = FALSE,
                       verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1, 1))),TOLERANCE)
        # LaGaBoostOOS algorithm
        #   1. Run LaGaBoost algorithm separately on every fold and fit parameters on out-of-sample data
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        set.seed(1)
        cvbst <- gpb.cv(params = params, data = dtrain, gp_model = gp_model,
                        nrounds = 100, nfold = 4, eval = "binary_error",
                        early_stopping_rounds = 5, use_gp_model_for_validation = TRUE,
                        fit_GP_cov_pars_OOS = TRUE, folds = folds, verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4255016, 0.3026152))),20*tolerance_loc_1)
        expect_lte(cvbst$best_iter, 16)
        expect_gte(cvbst$best_iter, 12)
        expect_lt(abs(cvbst$best_score-0.242), 2.5*tolerance_loc_1)
        #   2. Run LaGaBoost algorithm on entire data while holding covariance parameters fixed
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 15,
                         params = params, train_gp_model_cov_pars = FALSE, verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4255016, 0.3026152))),20*tolerance_loc_1)
        #   3. Prediction
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(head(pred$fixed_effect, n=4)-c(0.4456027, -0.2227075, 0.8109699, 0.6144861))),300*tolerance_loc_2)
        expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.050475, -1.025386, -1.187071,
                                                          rep(0,n_new)))),50*tolerance_loc_2)
        if(inv_method=="iterative") l_tol <- 0.08 else l_tol <- 50*TOLERANCE
        expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.1165832, 0.1175566, 0.1174304,
                                                         rep(0.7282295,n_new)))),l_tol)
        
        # Training using Nelder-Mead
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6,
                                              init_cov_pars = c(1,1), cg_preconditioner_type=PC, init_coef_aux_pars_from_iid_model = FALSE))
        bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                       nrounds = 30, learning_rate = 0.1,  max_depth = 6,
                       min_data_in_leaf = 5, objective = "binary", verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4682746, 0.3544995))),tolerance_loc_1)
        # Prediction
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(head(pred$fixed_effect,n=4)-c(0.53963543, -0.09143685, 0.97199209, 0.82756999))),tolerance_loc_3)
        expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.121577, -1.057764, -1.243746,
                                                          rep(0,n_new)))),tolerance_loc_2)
        expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.1294601, 0.1286418, 0.1289668,
                                                         rep(0.8227741,n_new)))),tolerance_loc_1)
        
        # Training using lbfgs
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=list(optimizer_cov="lbfgs", cg_preconditioner_type=PC, init_coef_aux_pars_from_iid_model = FALSE))
        bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                       min_data_in_leaf = 5, objective = "binary", verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4653201461, 0.3556254916))),tolerance_loc_1)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-375.4033342), tolerance_loc_4)
        
        # Validation metrics for training data
        # Default metric is "Approx. negative marginal log-likelihood" if there is only one training set
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                       objective = "binary", train_gp_model_cov_pars=FALSE, nrounds=1), file='NUL')
        record_results <- gpb.get.eval.result(bst, "train", "Approx. negative marginal log-likelihood")
        expect_value <- 599.7875
        expect_lt(abs(record_results[1]-expect_value), 10*tolerance_loc_2)
        # do not specify objective
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                       train_gp_model_cov_pars=FALSE, nrounds=1), file='NUL')
        record_results <- gpb.get.eval.result(bst, "train", "Approx. negative marginal log-likelihood")
        expect_lt(abs(record_results[1]-expect_value), 10*tolerance_loc_2)
        # Can also use other metrics
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                       objective = "binary", train_gp_model_cov_pars=FALSE, nrounds=1,
                                       eval=list("binary_logloss","binary_error"), use_gp_model_for_validation = FALSE), file='NUL')
        record_results <- gpb.get.eval.result(bst, "train", "binary_logloss")
        expect_lt(abs(record_results[1]-0.6749475), TOLERANCE)
        record_results <- gpb.get.eval.result(bst, "train", "binary_error")
        expect_lt(abs(record_results[1]-0.466), TOLERANCE)
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                       train_gp_model_cov_pars=FALSE, nrounds=1,
                                       eval=list("l2","binary_error"), use_gp_model_for_validation = FALSE), file='NUL')
        record_results <- gpb.get.eval.result(bst, "train", "l2")
        expect_lt(abs(record_results[1]-0.2409613), TOLERANCE)
        record_results <- gpb.get.eval.result(bst, "train", "binary_error")
        expect_lt(abs(record_results[1]-0.466), TOLERANCE)
        
        # Find number of iterations using validation data with use_gp_model_for_validation=FALSE
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        params_gp_v2 <- DEFAULT_OPTIM_PARAMS_V2
        params_gp_v2$init_cov_pars <- rep(1,2)
        params_gp_v2$cg_preconditioner_type=PC
        gp_model$set_optim_params(params=params_gp_v2)
        capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                                         learning_rate=0.1, objective = "binary", verbose = 0,
                                         use_gp_model_for_validation=FALSE, eval = "binary_error",
                                         early_stopping_rounds=10), file='NUL')
        record_results <- gpb.get.eval.result(bst, "test", "binary_error")
        expect_lt(abs(min(record_results)-0.323), 3*TOLERANCE)
        if(inv_method=="iterative") expect_iter <- 10 else expect_iter <- 11
        expect_equal(which.min(record_results), expect_iter)
        
        # Find number of iterations using validation data with use_gp_model_for_validation=TRUE
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp_v2)
        gp_model$set_prediction_data(group_data_pred = group_data_test)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                         learning_rate=0.1, objective = "binary", verbose = 0,
                         use_gp_model_for_validation=TRUE, eval = "binary_error",
                         early_stopping_rounds=10)
        record_results <- gpb.get.eval.result(bst, "test", "binary_error")
        expect_lt(abs(min(record_results)-0.241), tolerance_loc_1)
        if(inv_method=="iterative") expect_iter <- 18 else expect_iter <- 16
        expect_equal(which.min(record_results), expect_iter)
        # Compare to when ignoring random effects part
        bst <- gpb.train(data = dtrain, nrounds=100, valids=valids,
                         learning_rate=0.1, objective = "binary", verbose = 0,
                         use_gp_model_for_validation=TRUE, eval = "binary_error", early_stopping_rounds=10)
        expect_lt(abs(bst$best_score-.345), TOLERANCE)
        
        # Other metrics / losses
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp_v2)
        gp_model$set_prediction_data(group_data_pred = group_data_test)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                         learning_rate=0.5, objective = "binary", verbose = 0,
                         use_gp_model_for_validation=TRUE, eval = "binary_logloss",
                         early_stopping_rounds=10)
        record_results <- gpb.get.eval.result(bst, "test", "binary_logloss")
        expect_lt(abs(min(record_results)-0.4917727), tolerance_loc_1)
        capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                                         learning_rate=0.5, verbose = 0,
                                         use_gp_model_for_validation=TRUE, eval = "l2", early_stopping_rounds=10), file='NUL')
        record_results <- gpb.get.eval.result(bst, "test", "l2")
        expect_lt(abs(min(record_results)-0.1643671), tolerance_loc_1)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                         learning_rate=0.5, objective = "binary", verbose = 0,
                         use_gp_model_for_validation=TRUE, eval = "l2", early_stopping_rounds=10)
        record_results <- gpb.get.eval.result(bst, "test", "l2")
        expect_lt(abs(min(record_results)-0.1643671), tolerance_loc_1)
        
        # CV for finding number of boosting iterations when use_gp_model_for_validation = FALSE
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp_v2)
        cvbst <- gpb.cv(params = params, data = dtrain, gp_model=gp_model,
                        nrounds = 100, nfold = 4, eval = "binary_error",
                        early_stopping_rounds = 5, use_gp_model_for_validation = FALSE,
                        fit_GP_cov_pars_OOS = FALSE, folds = folds, verbose = 0)
        expect_score <- 0.352
        expect_gte(cvbst$best_iter, 7)
        expect_lte(cvbst$best_iter, 23)
        expect_lt(abs(cvbst$best_score-expect_score), 2*TOLERANCE_LOOSE)
        # same thing but "wrong" likelihood given in gp_model
        gp_model <- GPModel(group_data = group_data_train, likelihood="gaussian", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=c(DEFAULT_OPTIM_PARAMS_V2, cg_preconditioner_type=PC))
        capture.output( cvbst <- gpb.cv(params = params, data = dtrain, gp_model=gp_model,
                                        nrounds = 100, nfold = 4, eval = "binary_error",
                                        early_stopping_rounds = 5, use_gp_model_for_validation = FALSE,
                                        fit_GP_cov_pars_OOS = FALSE, folds = folds, verbose = 0), file='NUL')
        expect_score_logit <- 0.35
        expect_lt(abs(cvbst$best_score-expect_score_logit), 0.02)
        # CV for finding number of boosting iterations when use_gp_model_for_validation = TRUE
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        cvbst <- gpb.cv(params = params, data = dtrain, gp_model = gp_model,
                        nrounds = 100, nfold = 4, eval = "binary_error",
                        early_stopping_rounds = 5, use_gp_model_for_validation = TRUE,
                        fit_GP_cov_pars_OOS = FALSE, folds = folds, verbose = 0)
        expect_score <- 0.242
        expect_lte(cvbst$best_iter, 17)
        expect_gte(cvbst$best_iter, 11)
        expect_lt(abs(cvbst$best_score-expect_score), 2*tolerance_loc_1)
        
        # Use of validation data and cross-validation with custom metric
        bin_cust_error <- function(preds, dtrain) {
          labels <- getinfo(dtrain, "label")
          predsbin <- preds > 0.55
          error <- mean(predsbin!=labels)#mean((preds-labels)^4)
          return(list(name="bin_cust_error",value=error,higher_better=FALSE))
        }
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp_v2)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                         learning_rate=0.1, objective = "binary", verbose = 0,
                         use_gp_model_for_validation=FALSE,
                         early_stopping_rounds=10, eval = bin_cust_error, metric = "bin_cust_error")
        expect_lt(abs(bst$best_score - 0.359),tolerance_loc_1)
        # CV
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp_v2)
        cvbst <- gpb.cv(params = params, data = dtrain, gp_model=gp_model,
                        nrounds = 100, nfold = 4, early_stopping_rounds = 5,
                        use_gp_model_for_validation = FALSE, fit_GP_cov_pars_OOS = FALSE,
                        folds = folds, verbose = 0, eval = bin_cust_error, metric = "bin_cust_error")
        expect_lt(abs(cvbst$best_score-0.364), tolerance_loc_1)
      }
      
      # Using offsets for gaussian likelihood
      gp_model <- GPModel(group_data = group_data_train, likelihood = "gaussian", matrix_inversion_method = "cholesky")
      params_gp <- OPTIM_PARAMS_BFGS
      params_gp$init_cov_pars <- rep(1,3)
      set_optim_params(gp_model, params=params_gp)
      dtrain <- gpb.Dataset(data = X_train, label = y_train, init_score = -y_train)
      bst <- gpb.train(data = dtrain, gp_model = gp_model,
                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                       min_data_in_leaf = 5, verbose = 0)
      # Prediction
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                      predict_var = TRUE, pred_latent = TRUE)
      exp_pred <- c(0.8070121, 0.3620259, -0.1525967, 0.9387820)
      expect_lt(sum(abs(head(pred$fixed_effect, n=4)-exp_pred)),TOLERANCE_STRICT)
      offset_pred = rep(0.5,dim(X_test)[1])
      pred_offset <- predict(bst, data = X_test, group_data_pred = group_data_test,
                             predict_var = TRUE, pred_latent = TRUE, offset_pred = offset_pred)
      expect_lt(sum(abs(head(pred_offset$fixed_effect, n=4)-offset_pred-exp_pred)),TOLERANCE)
      # Predict response
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                      predict_var = TRUE, pred_latent = FALSE)
      exp_pred <- c(-0.3654355, 0.1826726, -0.2455984, 1.1575480)
      expect_lt(sum(abs(tail(pred$response_mean, n=4) - exp_pred)),TOLERANCE_STRICT)
      pred_offset <- predict(bst, data = X_test, group_data_pred = group_data_test,
                             predict_var = TRUE, pred_latent = FALSE, offset_pred = offset_pred)
      expect_lt(sum(abs(tail(pred_offset$response_mean, n=4) -offset_pred - exp_pred)),TOLERANCE)
      
      # Using offsets for binary likelihood
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
      params_gp <- OPTIM_PARAMS_BFGS
      params_gp$init_cov_pars <- rep(1,2)
      set_optim_params(gp_model, params=params_gp)
      dtrain <- gpb.Dataset(data = X_train, label = y_train, init_score = -y_train)
      bst <- gpb.train(data = dtrain, gp_model = gp_model,
                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                       min_data_in_leaf = 5, objective = "binary", verbose = 0)
      # Prediction
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                      predict_var = TRUE, pred_latent = TRUE)
      exp_pred <- c(1.1360952, 0.3089050, 1.7356259, 1.4984141)
      expect_lt(sum(abs(head(pred$fixed_effect, n=4)-exp_pred)),TOLERANCE_STRICT)
      offset_pred = rep(0.2,dim(X_test)[1])
      pred_offset <- predict(bst, data = X_test, group_data_pred = group_data_test,
                             predict_var = TRUE, pred_latent = TRUE, offset_pred = offset_pred)
      expect_lt(sum(abs(head(pred_offset$fixed_effect, n=4)-offset_pred-exp_pred)),TOLERANCE)
      # Predict response
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                      predict_var = TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean, n=4) - c(0.0076108, 0.7559669, 0.2865552, 0.7461261))),TOLERANCE_STRICT)
      pred_offset <- predict(bst, data = X_test, group_data_pred = group_data_test,
                             predict_var = TRUE, pred_latent = FALSE, offset_pred = offset_pred)
      expect_lt(sum(abs(tail(pred_offset$response_mean, n=4) - c(0.0125890, 0.7941900, 0.3314843, 0.7852381))),TOLERANCE)
    })
    
    test_that("GPBoost algorithm for binary classification when having only one grouping variable", {
      
      ntrain <- ntest <- 1000
      n <- ntrain + ntest
      # Simulate fixed effects
      sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.2644234)
      f <- sim_data$f
      f <- f - mean(f)
      X <- sim_data$X
      # Simulate grouped random effects
      sigma2_1 <- 1 # variance of random effect
      m <- 40 # number of categories / levels for grouping variable
      # first random effect
      group <- rep(1,ntrain) # grouping variable
      for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
      group <- c(group, group)
      n_new <- 3# number of new random effects in test data
      group[(length(group)-n_new+1):length(group)] <- rep(99999,n_new)
      Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
      b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.5542))
      eps <- Z1 %*% b1
      eps <- eps - mean(eps)
      group_data <- group
      # Observed data
      probs <- pnorm(f + eps)
      y <- as.numeric(sim_rand_unif(n=n, init_c=0.574) < probs)
      # Signal-to-noise ratio of approx. 1
      # var(f) / var(eps)
      # Split in training and test data
      y_train <- y[1:ntrain]
      X_train <- X[1:ntrain,]
      group_data_train <- group_data[1:ntrain]
      y_test <- y[1:ntest+ntrain]
      X_test <- X[1:ntest+ntrain,]
      f_test <- f[1:ntest+ntrain]
      group_data_test <- group_data[1:ntest+ntrain]
      # Data for Booster
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
      dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
      valids <- list(test = dtest)
      params <- list(learning_rate = 0.1, objective = "binary")
      # Folds for CV
      group_aux <- rep(1,ntrain) # grouping variable
      for(i in 1:(ntrain/4)) group_aux[(1:4)+4*(i-1)] <- 1:4
      folds <- list()
      for(i in 1:4) folds[[i]] <- as.integer(which(group_aux==i))
      
      # Find number of iterations using validation data with use_gp_model_for_validation=FALSE
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                       learning_rate=0.1, objective = "binary", verbose = 0,
                       use_gp_model_for_validation=FALSE, eval = "binary_error",
                       early_stopping_rounds=10)
      record_results <- gpb.get.eval.result(bst, "test", "binary_error")
      expect_lt(abs(min(record_results)-0.356), TOLERANCE)
      expect_equal(which.min(record_results), 17)
      # Find number of iterations using validation data with use_gp_model_for_validation=TRUE
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      gp_model$set_prediction_data(group_data_pred = group_data_test)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                       learning_rate=0.1, objective = "binary", verbose = 0,
                       use_gp_model_for_validation=TRUE, eval = "binary_error",
                       early_stopping_rounds=10)
      record_results <- gpb.get.eval.result(bst, "test", "binary_error")
      expect_lt(abs(min(record_results)-0.263), TOLERANCE)
      expect_equal(which.min(record_results), 31)
      # Find number of iterations using validation when specifying "wrong" default likelihood in gp_model
      gp_model <- GPModel(group_data = group_data_train)
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      gp_model$set_prediction_data(group_data_pred = group_data_test)
      capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                                       learning_rate=0.1, objective = "binary", verbose = 0,
                                       use_gp_model_for_validation=TRUE, eval = "binary_error",
                                       early_stopping_rounds=10), file='NUL')
      record_results <- gpb.get.eval.result(bst, "test", "binary_error")
      expect_lt(abs(min(record_results)-0.262), TOLERANCE)
      expect_equal(which.min(record_results), 44)
      # Find number of iterations using validation when not specifying objective in gpb.train
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      gp_model$set_prediction_data(group_data_pred = group_data_test)
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                       learning_rate=0.1, verbose = 0,
                       use_gp_model_for_validation=TRUE, eval = "binary_error",
                       early_stopping_rounds=10)
      record_results <- gpb.get.eval.result(bst, "test", "binary_error")
      expect_lt(abs(min(record_results)-0.263), TOLERANCE)
      expect_equal(which.min(record_results), 31)
      
      # CV for finding number of boosting iterations when use_gp_model_for_validation = FALSE
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      cvbst <- gpb.cv(params = params,
                      data = dtrain,
                      gp_model=gp_model,
                      nrounds = 100,
                      nfold = 4,
                      eval = "binary_error",
                      early_stopping_rounds = 5,
                      use_gp_model_for_validation = FALSE,
                      fit_GP_cov_pars_OOS = FALSE,
                      folds = folds,
                      verbose = 0)
      expect_equal(cvbst$best_iter, 6)
      expect_lt(abs(cvbst$best_score-0.387), TOLERANCE)
      # CV for finding number of boosting iterations when use_gp_model_for_validation = TRUE
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      cvbst <- gpb.cv(params = params,
                      data = dtrain,
                      gp_model = gp_model,
                      nrounds = 100,
                      nfold = 4,
                      eval = "binary_error",
                      early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE,
                      folds = folds,
                      verbose = 0)
      expect_equal(cvbst$best_iter, 5)
      expect_lt(abs(cvbst$best_score-0.259), TOLERANCE)
      # same thing but "wrong" likelihood in gp_model
      gp_model <- GPModel(group_data = group_data_train, likelihood = "gaussian")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      capture.output( cvbst <- gpb.cv(params = params,
                                      data = dtrain,
                                      gp_model = gp_model,
                                      nrounds = 100,
                                      nfold = 4,
                                      eval = "binary_error",
                                      early_stopping_rounds = 5,
                                      use_gp_model_for_validation = TRUE,
                                      folds = folds,
                                      verbose = 0), file='NUL')
      expect_equal(cvbst$best_iter, 14)
      expect_lt(abs(cvbst$best_score-0.255), TOLERANCE)
      # same thing but no objective in gpb.cv
      params_w <- params
      params_w[["objective"]] <- NULL
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      cvbst <- gpb.cv(params = params_w,
                      data = dtrain,
                      gp_model = gp_model,
                      nrounds = 100,
                      nfold = 4,
                      eval = "binary_error",
                      early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE,
                      folds = folds,
                      verbose = 0)
      expect_equal(cvbst$best_iter, 5)
      expect_lt(abs(cvbst$best_score-0.259), TOLERANCE)
      
      # Create random effects model and train GPBoost model
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
      bst <- gpboost(data = X_train,
                     label = y_train,
                     gp_model = gp_model,
                     nrounds = 30,
                     learning_rate = 0.1,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.9865279)),TOLERANCE)
      
      # Prediction
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                      predict_var = TRUE, pred_latent = TRUE)
      expect_lt(sum(abs(head(pred$fixed_effect,n=4)-c(0.3650635, 0.5201485, 0.6266364, 0.5428810))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-2.003974, -2.003974, -2.003974,
                                                        rep(0,n_new)))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.2156478, 0.2156478, 0.2156478,
                                                       rep(0.9865279,n_new)))),TOLERANCE)
      # Predict response
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                      predict_var = TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean,n=4)-c(0.003515544, 0.589497590, 0.261914849, 0.409295302))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$response_var,n=4)-c(0.003503185, 0.241990181, 0.193315461, 0.241772658))),TOLERANCE)
      
      # Training using Nelder-Mead
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6,
                                            init_cov_pars = 1, init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                     nrounds = 30, learning_rate = 0.1, max_depth = 6,
                     min_data_in_leaf = 5, objective = "binary", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.9823336838)),TOLERANCE)
      # Prediction
      pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                      predict_var = TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean,n=4)-c(0.003529128402, 0.590128529164, 0.262148832429, 0.409728732652))),TOLERANCE)
      
      # Training using BFGS
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=list(optimizer_cov="lbfgs", init_coef_aux_pars_from_iid_model = FALSE))
      bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                     nrounds = 30, learning_rate = 0.1, max_depth = 6,
                     min_data_in_leaf = 5, objective = "binary", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.984982603)),TOLERANCE)
      
    })
    
    # This is a slow test
  }
}
