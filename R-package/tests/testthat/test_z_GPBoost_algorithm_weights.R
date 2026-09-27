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
    
    test_that("GPBoost algorithm supports Gaussian sample weights ", {

      group_w <- c(1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5)
      X_w <- matrix(c(-1.0, -0.6, -0.2, 0.1, 0.4, 0.7, 1.0, 1.3, -0.8, -0.1, 0.5, 1.1,
                      0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9, 0.1, 0.45, 0.65, 0.85),
                    ncol = 2)
      y_w <- c(0.20, -0.35, 0.95, 0.70, -0.10, 1.25, 0.15, -0.55, 0.35, 0.05, 1.05, -0.20)
      weights_no <- rep(1.000000001, length(y_w))
      weights_w <- c(1.0, 2.0, 0.8, 1.5, 0.7, 2.2, 1.3, 0.9, 1.8, 0.6, 1.1, 0.5)
      
      params <- list(objective = "regression_l2", learning_rate = 0.05,
                     max_depth = 2,  min_data_in_leaf = 1,
                     feature_pre_filter = FALSE, optimizer_cov = "lbfgs", trace = FALSE, init_coef_aux_pars_from_iid_model = FALSE)
      capture.output( gp_model_w <- GPModel(group_data = group_w,
                                            weights = weights_no) , file='NUL')
      capture.output( bst_w <- gpboost(data = X_w, label = y_w, gp_model = gp_model_w,
                                       nrounds = 5, params = params, verbose = 0) , file='NUL')
      capture.output( gp_model <- GPModel(group_data = group_w) , file='NUL')
      capture.output( bst <- gpboost(data = X_w, label = y_w, gp_model = gp_model,
                                       nrounds = 5, params = params, verbose = 0) , file='NUL')
      cov_pars <- c(2.028712e-01, 1.053762e-07 )
      nll <- 7.456163
      expect_lt(sum(abs(as.vector(gp_model_w$get_cov_pars()) - cov_pars)), TOLERANCE_STRICT)
      expect_lt(abs(gp_model_w$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars()) - cov_pars)), TOLERANCE_STRICT)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
      
      pred_w <- predict(bst_w, data = X_w, group_data_pred = group_w,
                        pred_latent = TRUE, predict_var = TRUE)
      pred <- predict(bst, data = X_w, group_data_pred = group_w,
                        pred_latent = TRUE, predict_var = TRUE)
      pred_fe <- c(0.1552112, 0.3873440, 0.4667916, 0.2930946)
      pred_re <- c(-7.404650e-08, -7.404650e-08, 4.680724e-08, 4.680724e-08)
      pred_re_var <- c(1.053761e-07, 1.053761e-07, 1.053761e-07, 1.053761e-07)
      expect_lt(sum(abs(tail(pred_w$fixed_effect, n = 4) - pred_fe)), TOLERANCE_STRICT)
      expect_lt(sum(abs(tail(pred_w$random_effect_mean, n = 4) - pred_re)), TOLERANCE_STRICT)
      expect_lt(sum(abs(tail(pred_w$random_effect_cov, n = 4) - pred_re_var)), TOLERANCE_STRICT)
      expect_lt(sum(abs(tail(pred$fixed_effect, n = 4) - pred_fe)), TOLERANCE_STRICT)
      expect_lt(sum(abs(tail(pred$random_effect_mean, n = 4) - pred_re)), TOLERANCE_STRICT)
      expect_lt(sum(abs(tail(pred$random_effect_cov, n = 4) - pred_re_var)), TOLERANCE_STRICT)
      
      capture.output( gp_model_w <- GPModel(group_data = group_w,
                                            weights = weights_w) , file='NUL')
      capture.output( bst_w <- gpboost(data = X_w, label = y_w, gp_model = gp_model_w,
                                       nrounds = 5, params = params, verbose = 0) , file='NUL')
      cov_pars <- c(2.341871e-01, 1.424805e-07)
      nll <- 7.845767
      expect_lt(sum(abs(as.vector(gp_model_w$get_cov_pars()) - cov_pars)), TOLERANCE_STRICT)
      expect_lt(abs(gp_model_w$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)

      pred_w <- predict(bst_w, data = X_w, group_data_pred = group_w,
                        pred_latent = TRUE, predict_var = TRUE)
      pred_fe <- c(0.2142461, 0.4736939, 0.5318590, 0.5318590)
      pred_re <- c(-5.998477e-09, -5.998477e-09, 1.241301e-07, 1.241301e-07)
      pred_re_var <- c(1.424803e-07, 1.424803e-07, 1.424804e-07, 1.424804e-07)
      expect_lt(sum(abs(tail(pred_w$fixed_effect, n = 4) - pred_fe)), TOLERANCE_STRICT)
      expect_lt(sum(abs(tail(pred_w$random_effect_mean, n = 4) - pred_re)), TOLERANCE_STRICT)
      expect_lt(sum(abs(tail(pred_w$random_effect_cov, n = 4) - pred_re_var)), TOLERANCE_STRICT)
    })
    
  }
  
}
