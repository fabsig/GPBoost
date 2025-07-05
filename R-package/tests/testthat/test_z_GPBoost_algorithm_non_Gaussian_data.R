if(Sys.getenv("NO_GPBOOST_ALGO_TESTS") != "NO_GPBOOST_ALGO_TESTS"){
  
  context("generalized_GPBoost_combined_boosting_GP_random_effects")
  
  TOLERANCE <- 1E-3
  TOLERANCE_LOOSE <- 1E-2
  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov="gradient_descent", use_nesterov_acc=TRUE,
                               delta_rel_conv=1E-6, lr_cov=0.1, lr_coef=0.1)
  DEFAULT_OPTIM_PARAMS_V2 <- list(optimizer_cov="gradient_descent", use_nesterov_acc=TRUE,
                                  delta_rel_conv=1E-6, lr_cov=0.01, lr_coef=0.1)
  DEFAULT_OPTIM_PARAMS_NO_NESTEROV <- list(optimizer_cov="gradient_descent", use_nesterov_acc=FALSE,
                                           delta_rel_conv=1E-6, lr_cov=0.01, lr_coef=0.1)
  DEFAULT_OPTIM_PARAMS_EARLY_STOP <- list(maxit=10, lr_cov=0.1, optimizer_cov="gradient_descent", lr_coef=0.1)
  DEFAULT_OPTIM_PARAMS_EARLY_STOP_NO_NESTEROV <- list(maxit=20, lr_cov=0.01, use_nesterov_acc=FALSE,
                                                      optimizer_cov="gradient_descent", lr_coef=0.1)
  OPTIM_PARAMS_BFGS <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 1000)
  
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
          tolerance_loc_1 <- TOLERANCE_LOOSE
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
        gp_model$set_optim_params(params=list(maxit=2, optimizer_cov="gradient_descent", cg_preconditioner_type=PC))
        expect_error(gpboost(data = X_train, label = probs[1:ntrain], gp_model = gp_model,
                             objective = "binary", nrounds=1))
        # fisher_scoring cannot be used
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=list(maxit=2, optimizer_cov="fisher_scoring", cg_preconditioner_type=PC))
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
        gp_model$set_optim_params(params=list(cg_preconditioner_type=PC))
        params_gp_gaus <- params_gp
        params_gp_gaus$init_cov_pars <- NULL
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                                       min_data_in_leaf = 5, objective = "binary", verbose = 0), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),0.02)
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = FALSE)
        if(inv_method=="iterative") l_tol <- 0.06 else l_tol <- 0.05
        expect_lt(sum(abs(tail(pred$response_mean, n=4) - resp_mean)),l_tol)
        if(inv_method=="iterative") l_tol <- 0.03 else l_tol <- 0.02
        expect_lt(sum(abs(tail(pred$response_var, n=4) - resp_var)),l_tol)
        # Training with "wrong" default likelihood
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp_gaus)
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                                       nrounds = 30, learning_rate = 0.1, max_depth = 6,
                                       min_data_in_leaf = 5, objective = "binary_probit", verbose = 0), file='NUL')
        if(inv_method=="iterative") l_tol <- 0.008 else l_tol <- 0.002
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
        cvbst <- gpb.cv(params = params,
                        data = dtrain,
                        gp_model = gp_model,
                        nrounds = 100,
                        nfold = 4,
                        eval = "binary_error",
                        early_stopping_rounds = 5,
                        use_gp_model_for_validation = TRUE,
                        fit_GP_cov_pars_OOS = TRUE,
                        folds = folds,
                        verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4255016, 0.3026152))),2*tolerance_loc_1)
        expect_lt(cvbst$best_iter, 16)
        expect_gt(cvbst$best_iter, 12)
        expect_lt(abs(cvbst$best_score-0.242), tolerance_loc_1)
        #   2. Run LaGaBoost algorithm on entire data while holding covariance parameters fixed
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 15,
                         params = params, train_gp_model_cov_pars = FALSE, verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4255016, 0.3026152))),2*tolerance_loc_1)
        #   3. Prediction
        pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                        predict_var = TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(head(pred$fixed_effect, n=4)-c(0.4456027, -0.2227075, 0.8109699, 0.6144861))),2*tolerance_loc_2)
        expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.050475, -1.025386, -1.187071,
                                                          rep(0,n_new)))),2*tolerance_loc_2)
        if(inv_method=="iterative") l_tol <- 0.08 else l_tol <- 2*TOLERANCE
        expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.1165832, 0.1175566, 0.1174304,
                                                         rep(0.7282295,n_new)))),l_tol)
        
        # Training using Nelder-Mead
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6,
                                              init_cov_pars = c(1,1), cg_preconditioner_type=PC))
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
        gp_model$set_optim_params(params=list(optimizer_cov="lbfgs", cg_preconditioner_type=PC))
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
        expect_lt(abs(record_results[1]-expect_value), tolerance_loc_2)
        # do not specify objective
        capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                       train_gp_model_cov_pars=FALSE, nrounds=1), file='NUL')
        record_results <- gpb.get.eval.result(bst, "train", "Approx. negative marginal log-likelihood")
        expect_lt(abs(record_results[1]-expect_value), tolerance_loc_2)
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
        expect_lt(abs(min(record_results)-0.323), TOLERANCE)
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
        if(inv_method=="iterative") expect_iter <- 4 else expect_iter <- 6
        expect_equal(which.min(record_results), expect_iter)
        capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                                         learning_rate=0.5, verbose = 0,
                                         use_gp_model_for_validation=TRUE, eval = "l2", early_stopping_rounds=10), file='NUL')
        record_results <- gpb.get.eval.result(bst, "test", "l2")
        expect_lt(abs(min(record_results)-0.1643671), tolerance_loc_1)
        expect_equal(which.min(record_results), expect_iter)
        bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                         learning_rate=0.5, objective = "binary", verbose = 0,
                         use_gp_model_for_validation=TRUE, eval = "l2", early_stopping_rounds=10)
        record_results <- gpb.get.eval.result(bst, "test", "l2")
        expect_lt(abs(min(record_results)-0.1643671), tolerance_loc_1)
        expect_equal(which.min(record_results), expect_iter)
        
        # CV for finding number of boosting iterations when use_gp_model_for_validation = FALSE
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp_v2)
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
        if(inv_method=="iterative") expect_iter <- 7 else expect_iter <- 9
        expect_score <- 0.352
        expect_equal(cvbst$best_iter, expect_iter)
        expect_lt(abs(cvbst$best_score-expect_score), TOLERANCE)
        # same thing but "wrong" likelihood given in gp_model
        gp_model <- GPModel(group_data = group_data_train, likelihood="gaussian", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=c(DEFAULT_OPTIM_PARAMS_V2, cg_preconditioner_type=PC))
        capture.output( cvbst <- gpb.cv(params = params,
                                        data = dtrain,
                                        gp_model=gp_model,
                                        nrounds = 100,
                                        nfold = 4,
                                        eval = "binary_error",
                                        early_stopping_rounds = 5,
                                        use_gp_model_for_validation = FALSE,
                                        fit_GP_cov_pars_OOS = FALSE,
                                        folds = folds,
                                        verbose = 0), file='NUL')
        if(inv_method=="iterative") expect_iter <- 9 else expect_iter <- 8
        expect_equal(cvbst$best_iter, expect_iter)
        expect_lt(abs(cvbst$best_score-expect_score), 0.002)
        # CV for finding number of boosting iterations when use_gp_model_for_validation = TRUE
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp)
        cvbst <- gpb.cv(params = params,
                        data = dtrain,
                        gp_model = gp_model,
                        nrounds = 100,
                        nfold = 4,
                        eval = "binary_error",
                        early_stopping_rounds = 5,
                        use_gp_model_for_validation = TRUE,
                        fit_GP_cov_pars_OOS = FALSE,
                        folds = folds,
                        verbose = 0)
        expect_iter <- 15
        expect_score <- 0.242
        expect_equal(cvbst$best_iter, expect_iter)
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
        if(inv_method=="iterative") expect_iter <- 24 else expect_iter <- 17
        expect_equal(bst$best_iter, expect_iter)
        expect_lt(abs(bst$best_score - 0.359),tolerance_loc_1)
        # CV
        gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=params_gp_v2)
        cvbst <- gpb.cv(params = params,
                        data = dtrain,
                        gp_model=gp_model,
                        nrounds = 100,
                        nfold = 4,
                        early_stopping_rounds = 5,
                        use_gp_model_for_validation = FALSE,
                        fit_GP_cov_pars_OOS = FALSE,
                        folds = folds,
                        verbose = 0,
                        eval = bin_cust_error, metric = "bin_cust_error")
        expect_equal(cvbst$best_iter, 7)
        expect_lt(abs(cvbst$best_score-0.364), tolerance_loc_1)
      }
    })
    
    test_that("GPBoost algorithm: large data and 'reuse_learning_rates_gp_model' and 'line_search_step_length' options", {
      
      n <- 1e4
      X_train <- matrix(sim_rand_unif(n=2*n, init_c=0.9135), ncol=2)
      # Simulate grouped random effects
      sigma2_1 <- 0.6 # variance of first random effect 
      sigma2 <- 0.1^2 # error variance
      m <- n / 100 # number of categories / levels for grouping variable
      group <- rep(1,n) # grouping variable
      for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
      b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.3462))
      eps <- b1[group]
      lp <- X_train %*% c(1,1)
      lp <- lp - mean(lp)
      probs <- pnorm(lp + eps)
      y <- as.numeric(sim_rand_unif(n=n, init_c=0.5574) < probs)
      params <- list(learning_rate = 0.1, max_depth = 6,
                     min_data_in_leaf = 5, seed = 1)
      set.seed(1)
      dtrain <- gpb.Dataset(data = X_train, label = y)
      folds <- list()
      for(i in 1:4) folds[[i]] <- as.integer(1:(n/4) + (n/4) * (i-1))
      
      # Create random effects model and train GPBoost model
      gp_model <- GPModel(group_data = group, likelihood = "binary_logit")
      set_optim_params(gp_model, params=DEFAULT_OPTIM_PARAMS)
      bst <- gpboost(data = X_train, label = y, gp_model = gp_model,
                     nrounds = 10, params = params, verbose = 0, 
                     reuse_learning_rates_gp_model = TRUE,
                     line_search_step_length = FALSE)
      nll <- 5644.699232
      expect_lt(abs((gp_model$get_current_neg_log_likelihood()-nll))/abs(nll),TOLERANCE)
      # With the option line_search_step_length
      gp_model <- GPModel(group_data = group, likelihood = "binary_logit")
      set_optim_params(gp_model, params=DEFAULT_OPTIM_PARAMS)
      bst <- gpboost(data = X_train, label = y, gp_model = gp_model,
                     nrounds = 10, params = params, verbose = 0,
                     reuse_learning_rates_gp_model = TRUE,
                     line_search_step_length = TRUE)
      nll <- 5317.368493
      expect_lt(abs((gp_model$get_current_neg_log_likelihood()-nll))/abs(nll),TOLERANCE)
      
      # CV
      gp_model <- GPModel(group_data = group, likelihood = "binary_logit")
      set_optim_params(gp_model, params=DEFAULT_OPTIM_PARAMS)
      score <- 0.683929491027179
      cvbst <- gpb.cv(params = params, data = dtrain, gp_model = gp_model,
                      nrounds = 20, nfold = 4, eval = "binary_logloss", early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                      reuse_learning_rates_gp_model = TRUE, line_search_step_length = FALSE)
      expect_lt(abs(cvbst$best_score-score), TOLERANCE)
      cvbst <- gpb.cv(params = params, data = dtrain, gp_model = gp_model,
                      nrounds = 20, nfold = 4, eval = "binary_logloss", early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                      reuse_learning_rates_gp_model = TRUE,
                      line_search_step_length = TRUE)
      score <- 0.672572152021631
      expect_lt(abs(cvbst$best_score-score), TOLERANCE)
      
      # Check whether the option "reuse_learning_rates_gp_model" is used or not
      gp_model <- GPModel(group_data = group, likelihood = "binary_logit")
      params_loc <- DEFAULT_OPTIM_PARAMS
      params_loc$trace = TRUE
      set_optim_params(gp_model, params=params_loc)
      output <- capture.output( bst <- gpboost(data = X_train, label = y, gp_model = gp_model,
                                               nrounds = 2, params = params, verbose = 0, 
                                               reuse_learning_rates_gp_model = FALSE,
                                               line_search_step_length = TRUE) )
      str <- output[length(output)-10]
      nb_ll_eval <- as.numeric(substr(str, nchar(str)-3, nchar(str)-2))
      expect_equal(nb_ll_eval, 10)
      # same thing with reuse_learning_rates_gp_model = TRUE
      gp_model <- GPModel(group_data = group, likelihood = "binary_logit")
      set_optim_params(gp_model, params=params_loc)
      output <- capture.output( bst <- gpboost(data = X_train, label = y, gp_model = gp_model,
                                               nrounds = 2, params = params, verbose = 0, 
                                               reuse_learning_rates_gp_model = TRUE,
                                               line_search_step_length = TRUE) )
      str <- output[length(output)-9]
      nb_ll_eval <- as.numeric(substr(str, nchar(str)-2, nchar(str)-2))
      expect_equal(nb_ll_eval, 4)
      
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
      expect_lt(abs(min(record_results)-0.263), TOLERANCE)
      expect_equal(which.min(record_results), 31)
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
      expect_equal(cvbst$best_iter, 5)
      expect_lt(abs(cvbst$best_score-0.259), TOLERANCE)
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
                                            init_cov_pars = 1))
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
      gp_model$set_optim_params(params=list(optimizer_cov="lbfgs"))
      bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model,
                     nrounds = 30, learning_rate = 0.1, max_depth = 6,
                     min_data_in_leaf = 5, objective = "binary", verbose = 0)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.984982603)),TOLERANCE)
      
    })
    
    # This is a slow test
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
      expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-c(-0.25248234, 0.07336944, 0.19282985, 0.04100225))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.09672839, 0.10432856, 0.09164587, 0.09215657))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(0.4087100, -0.5570364, -0.7904685, 0.5055812))),TOLERANCE)
      # Predict response
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean,n=4)-c(0.5592939, 0.3226671, 0.2836602, 0.6995181))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$response_var,n=4)-c(0.2464842, 0.2185530, 0.2031971, 0.2101925))),TOLERANCE)
    
      # Use validation set to determine number of boosting iteration with use_gp_model_for_validation = TRUE
      dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
      valids <- list(test = dtest)
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=params)
      gp_model$set_prediction_data(gp_coords_pred = coords_test)
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 20,
                       learning_rate = 0.2,
                       max_depth = 10,
                       min_data_in_leaf = 5,
                       objective = "binary",
                       verbose = 0,
                       valids = valids,
                       early_stopping_rounds = 2,
                       use_gp_model_for_validation = TRUE)
      expect_equal(bst$best_iter, 9)
      expect_lt(abs(bst$best_score - 0.5785662),TOLERANCE)
      
      # Train tree-boosting model while holding the GPModel fix
      init_cov_pars = c(2.4,1.1)
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params = list(init_cov_pars = init_cov_pars))
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
        gp_model$set_optim_params(params = list(init_cov_pars = init_cov_pars))
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
      gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead", delta_rel_conv=1e-6))
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
                                            lr_coef=0.1, init_cov_pars=init_cov_pars))
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
      # same thing but "wrong" default likelihood in gp_model
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
      gp_model$set_prediction_data(gp_coords_pred = coords_test)
      capture.output( bst <- gpb.train(data = dtrain,
                                       gp_model = gp_model,
                                       nrounds = 10,
                                       learning_rate = 0.1,
                                       max_depth = 6,
                                       min_data_in_leaf = 5,
                                       objective = "binary",
                                       verbose = 0,
                                       valids = valids,
                                       early_stopping_rounds = 2,
                                       use_gp_model_for_validation = TRUE), file='NUL')
      expect_equal(bst$best_iter, 10)
      expect_lt(abs(bst$best_score - 0.6129572),TOLERANCE)
      # same thing without objective in gpb.train
      gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                          likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=params_ES)
      gp_model$set_prediction_data(gp_coords_pred = coords_test)
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 10,
                       learning_rate = 0.1,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       verbose = 0,
                       valids = valids,
                       early_stopping_rounds = 2,
                       use_gp_model_for_validation = TRUE,
                       metric = "binary_logloss")
      expect_equal(bst$best_iter, 10)
      expect_lt(abs(bst$best_score - 0.6129572),TOLERANCE)
      
      # CV for finding number of boosting iterations when use_gp_model_for_validation = TRUE
      gp_model <- GPModel(gp_coords = coords_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
      cvbst <- gpb.cv(data = dtrain,
                      gp_model = gp_model,
                      nrounds = 10,
                      learning_rate = 0.1,
                      max_depth = 6,
                      min_data_in_leaf = 5,
                      objective = "binary",
                      eval = "binary_error",
                      early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE,
                      folds = folds,
                      verbose = 0)
      expcet_iter <- 6
      expcet_score <- 0.315
      expect_equal(cvbst$best_iter, expcet_iter)
      expect_lt(abs(cvbst$best_score-expcet_score), TOLERANCE)
      # same thing but "wrong" default likelihood in gp_model
      gp_model <- GPModel(gp_coords = coords_train, likelihood = "gaussian")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
      capture.output( cvbst <- gpb.cv(data = dtrain,
                                      gp_model = gp_model,
                                      nrounds = 10,
                                      learning_rate = 0.1,
                                      max_depth = 6,
                                      min_data_in_leaf = 5,
                                      objective = "binary",
                                      eval = "binary_error",
                                      early_stopping_rounds = 5,
                                      use_gp_model_for_validation = TRUE,
                                      folds = folds,
                                      verbose = 0), file='NUL')
      expect_equal(cvbst$best_iter, expcet_iter)
      expect_lt(abs(cvbst$best_score-expcet_score), TOLERANCE)
      # same thing but no objective in gpb.cv
      gp_model <- GPModel(gp_coords = coords_train, likelihood = "bernoulli_probit")
      gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
      cvbst <- gpb.cv(data = dtrain,
                      gp_model = gp_model,
                      nrounds = 10,
                      learning_rate = 0.1,
                      max_depth = 6,
                      min_data_in_leaf = 5,
                      eval = "binary_error",
                      early_stopping_rounds = 5,
                      use_gp_model_for_validation = TRUE,
                      folds = folds,
                      verbose = 0)
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
      # gp_model$set_optim_params(params=list(optimizer_cov = "nelder_mead", delta_rel_conv=1E-8))
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
      dtrain <- gpb.Dataset(data = X_train, label = y_train)
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
      bst <- gpb.train(data = dtrain, gp_model = gp_model,
                       nrounds = 5, learning_rate = 0.5, max_depth = 6,
                       min_data_in_leaf = 5, objective = "binary", verbose = 0)
      cov_pars_est <- c(0.1195943, 0.1479688)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
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
          tolerance_loc <- 0.01
        } else{
          tolerance_loc <- TOLERANCE
        }
        capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                            likelihood = "bernoulli_probit", gp_approx = "vecchia", 
                                            num_neighbors = ntrain-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        if(inv_method == "iterative"){
          params$num_rand_vec_trace = 1000 
          params$cg_delta_conv = sqrt(1e-6)
          params$cg_preconditioner_type = "piv_chol_on_Sigma"
        }
        gp_model$set_optim_params(params=params)
        bst <- gpb.train(data = dtrain, gp_model = gp_model,
                         nrounds = 5, learning_rate = 0.5, max_depth = 6,
                         min_data_in_leaf = 5, objective = "binary", verbose = 0)
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
        gp_model$set_optim_params(params=params)
        bst <- gpb.train(data = dtrain, gp_model = gp_model,
                         nrounds = 5, learning_rate = 0.5, max_depth = 6,
                         min_data_in_leaf = 5, objective = "binary", verbose = 0)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),tolerance_loc)
        
        # Prediction
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", num_neighbors_pred = ntest+ntrain-1)
        pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                        predict_var = TRUE, pred_latent = TRUE)
        expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-P_RE_mean)),tolerance_loc)
        adjust_tol <- 1
        if (inv_method == "iterative") adjust_tol <- 1.5
        expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-P_RE_cov)),adjust_tol*tolerance_loc)
        expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-P_F)),tolerance_loc)
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
                                            lr_coef=0.1, init_cov_pars=init_cov_pars))
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
          tolerance_loc_3 <- 1
          tolerance_loc_4 <- 10
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
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),tolerance_loc_1)
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
          tolerance_loc_1 <- TOLERANCE_LOOSE
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
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-1.447807)),TOLERANCE)
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
          tolerance_loc_4 <- 10
        } else {
          tolerance_loc_1 <- TOLERANCE_LOOSE
          tolerance_loc_2 <- TOLERANCE
          tolerance_loc_3 <- TOLERANCE
          tolerance_loc_4 <- TOLERANCE_LOOSE
        }
        # Train model
        gp_model <- GPModel(group_data = group_data_train, likelihood = "negative_binomial", matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=OPTIM_PARAMS_GAMMA)
        bst <- gpboost(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 30,
                       learning_rate = 0.1,
                       max_depth = 6,
                       min_data_in_leaf = 5,
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
        cov_pars_est <- c(0.5693555853, 0.4920194242)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE_LOOSE)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-2.768791332  )),tolerance_loc_1)
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
      gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
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
        if(inv_method=="iterative") tol_iter <- 75 else tol_iter <- 59
        expect_equal(opt_params$best_iter,tol_iter)
        expect_equal(opt_params$best_params$learning_rate,0.11)
        expect_equal(opt_params$best_params$max_bin,10)
        expect_equal(opt_params$best_params$max_depth,2)
        opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = params,
                                                      data = dtrain, gp_model = gp_model, verbose_eval = 1,
                                                      nrounds = 100, early_stopping_rounds = 5,
                                                      metric = "binary_logloss", folds = folds)
        expect_lt(abs(opt_params$best_score-0.51101812),tolerance_loc_1)
        expect_equal(opt_params$best_iter,tol_iter)
        expect_equal(opt_params$best_params$learning_rate,0.11)
        expect_equal(opt_params$best_params$max_bin,10)
        expect_equal(opt_params$best_params$max_depth,2)
        opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = params,
                                                      data = dtrain, gp_model = gp_model, verbose_eval = 1,
                                                      nrounds = 100, early_stopping_rounds = 5,
                                                      eval = "test_neg_log_likelihood", folds = folds)
        expect_lt(abs(opt_params$best_score-0.51101812),tolerance_loc_1)
        expect_equal(opt_params$best_iter,tol_iter)
        expect_equal(opt_params$best_params$learning_rate,0.11)
        expect_equal(opt_params$best_params$max_bin,10)
        expect_equal(opt_params$best_params$max_depth,2)
        opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = params,
                                                      data = dtrain, gp_model = gp_model, verbose_eval = 1,
                                                      nrounds = 100, early_stopping_rounds = 5,
                                                      eval = "auc", folds = folds)
        expect_lt(abs(opt_params$best_score-0.65502697),tolerance_loc_1)
        if(inv_method=="iterative") tol_iter <- 13 else tol_iter <- 52
        expect_equal(opt_params$best_iter,tol_iter)
        if(inv_method=="iterative") tol_lr <- 0.5 else tol_lr <- 0.11
        expect_equal(opt_params$best_params$learning_rate,tol_lr)
        expect_equal(opt_params$best_params$max_bin,10)
        expect_equal(opt_params$best_params$max_depth,2)
        opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = params,
                                                      data = dtrain, gp_model = gp_model, verbose_eval = 1,
                                                      nrounds = 100, early_stopping_rounds = 5,
                                                      metric = "auc", folds = folds)
        expect_lt(abs(opt_params$best_score-0.65502697),tolerance_loc_1)
        expect_equal(opt_params$best_iter,tol_iter)
        expect_equal(opt_params$best_params$learning_rate,tol_lr)
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
        if(inv_method=="iterative") tol_iter <- 13 else tol_iter <- 17
        expect_equal(opt_params$best_iter,tol_iter)
        expect_equal(opt_params$best_params$learning_rate,0.11)
        if(inv_method=="iterative") tol_bin <- 10 else tol_bin <- 255
        expect_equal(opt_params$best_params$max_bin,tol_bin)
      }
    })
    
    test_that("GPBoost algorithm with Gaussian process model and 'gaussian_heteroscedastic' likelihood", {
      
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
                          likelihood = "gaussian_heteroscedastic", gp_approx = "vecchia",
                          matrix_inversion_method = "iterative")
      gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
      bst <- gpb.train(data = dtrain,
                       gp_model = gp_model,
                       nrounds = 2,
                       learning_rate = 0.5,
                       max_depth = 6,
                       min_data_in_leaf = 5,
                       verbose = 0, deterministic = TRUE)
      cov_pars_est <- c(1.127432e-01, 2.989325e-02, 1.064309e-06, 1.970296e-01)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
      
      # Prediction
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                      predict_var = TRUE, pred_latent = TRUE)
      npred <- dim(X_test)[1]
      expect_lt(sum(abs(pred$fixed_effect[1:4]-c(0.9287969, 0.9392324, 0.6386508, 0.6837547))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(0.013631968, 0.001888464, 0.072146693, 0.118746547))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.10732077, 0.07915076, 0.07670887, 0.09757507))),0.01)
      # Predict response
      pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                      predict_var = TRUE, pred_latent = FALSE)
      expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(1.2609604, 0.4655176, 0.8336034, 0.6664328))),TOLERANCE)
      expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.2599629, 0.2383203, 0.2248717, 0.3464849))),TOLERANCE)
      
      # Parameter tuning
      # Folds for CV
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
      expect_lt(abs(opt_params$best_score-0.2723836),0.01)
      expect_equal(opt_params$best_iter,7)
      expect_equal(opt_params$best_params$learning_rate,0.11)
      expect_equal(opt_params$best_params$max_bin,255)
      expect_equal(opt_params$best_params$max_depth,2)
      
    })
  }
}
