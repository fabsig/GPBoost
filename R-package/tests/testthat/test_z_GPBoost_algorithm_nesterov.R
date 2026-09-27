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
      
      # vec_chol_or_iterative <- c("iterative", "cholesky")
      vec_chol_or_iterative <- c("cholesky")
      for (inv_method in vec_chol_or_iterative) {
        PC <- "ssor"
        if(inv_method == "iterative") {
          tolerance_loc_1 <- TOLERANCE2
          tolerance_loc_2 <- 0.1
          tolerance_loc_3 <- 1
        } else {
          tolerance_loc_1 <- TOLERANCE
          tolerance_loc_2 <- TOLERANCE
          tolerance_loc_3 <- TOLERANCE
        }
        # CV for finding number of boosting iterations with use_gp_model_for_validation = FALSE
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=c(DEFAULT_OPTIM_PARAMS, cg_preconditioner_type=PC))
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
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=c(DEFAULT_OPTIM_PARAMS, cg_preconditioner_type=PC))
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
        if(inv_method=="iterative") l_tol <- 0.28 else l_tol <- 0.271
        expect_lt(sqrt(mean((pred$fixed_effect - f_test)^2)),l_tol)
        if(inv_method=="iterative") l_tol <- 1.03 else l_tol <- 1.018
        expect_lt(sqrt(mean((pred$fixed_effect - y_test)^2)),l_tol)
        if(inv_method=="iterative") l_tol <- 0.25 else l_tol <- 0.238
        expect_lt(sqrt(mean((pred$fixed_effect + pred$random_effect_mean - y_test)^2)),l_tol)
        expect_lt(sum(abs(tail(pred$random_effect_mean)-c(0.3737357, -0.1906376, -1.2750302,
                                                          rep(0,n_new)))),tolerance_loc_2)
        expect_lt(sum(abs(head(pred$fixed_effect)-c(4.921429, 4.176900, 2.743165,
                                                    4.141866, 5.018322, 4.935220))),tolerance_loc_3)
        
        # Using validation set
        # Do not include random effect predictions for validation
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=c(DEFAULT_OPTIM_PARAMS, cg_preconditioner_type=PC))
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
                         use_nesterov_acc = TRUE, metric = "l2")
        expect_equal(bst$best_iter, 19)
        expect_lt(abs(bst$best_score - 1.035405),tolerance_loc_1)
        # Include random effect predictions for validation 
        gp_model <- GPModel(group_data = group_data_train, matrix_inversion_method = inv_method)
        gp_model$set_optim_params(params=c(DEFAULT_OPTIM_PARAMS, cg_preconditioner_type=PC))
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
                         use_nesterov_acc = TRUE, metric = "l2")
        expect_equal(bst$best_iter, 19)
        expect_lt(abs(bst$best_score - 0.05520368),tolerance_loc_1)
      }
    })
    
  }
  
}
