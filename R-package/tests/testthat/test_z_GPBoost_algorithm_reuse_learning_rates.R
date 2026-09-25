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
    
    # The same options as in 'test_z_GPBoost_algorithm_large_data.R', with a sample size that is
    # small enough for the checks under valgrind, which skip the large data file. Whether reusing
    # the learning rates saves likelihood evaluations depends on the data - at n = 1e5 it does, at
    # the size used here it does not - so the number of evaluations is not asserted here; the large
    # data test covers that. What is asserted is the property that has to hold for every data set:
    # the option changes how the optimum is reached, not the optimum itself.
    test_that("GPBoost algorithm: 'reuse_learning_rates_gp_model' does not change the result", {

      n <- 2000
      X_train <- matrix(sim_rand_unif(n=2*n, init_c=0.135), ncol=2)
      sigma2_1 <- 0.6 # variance of the random effect
      sigma2 <- 0.1^2 # error variance
      m <- n / 100 # number of categories / levels for grouping variable
      group <- rep(1,n)
      for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
      b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.462))
      eps <- b1[group]
      xi <- sqrt(sigma2) * qnorm(sim_rand_unif(n=n, init_c=0.17556))
      xi[xi<(-100)] = 0
      y <- eps + xi
      params <- list(learning_rate = 0.01,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "regression_l2",
                     feature_pre_filter = FALSE,
                     seed = 1)

      fit_with <- function(reuse) {
        gp_model <- GPModel(group_data = group)
        params_loc <- OPTIM_PARAMS_GRAD_DESC
        params_loc$trace <- TRUE
        set_optim_params(gp_model, params = params_loc)
        output <- capture.output( bst <- gpboost(data = X_train, label = y, gp_model = gp_model,
                                                 nrounds = 2, params = params, verbose = 0,
                                                 reuse_learning_rates_gp_model = reuse) )
        trace <- grep("nb. likelihood evaluations", output, value = TRUE)
        list(nll = gp_model$get_current_neg_log_likelihood(),
             cov_pars = as.vector(gp_model$get_cov_pars()),
             num_ll_eval = as.integer(sub(".*evaluations = ([0-9]+).*", "\\1",
                                          trace[length(trace)])))
      }

      without <- fit_with(FALSE)
      with <- fit_with(TRUE)
      # the option is live: the trace reports the evaluations that the optimizer needed
      expect_gt(without$num_ll_eval, 0)
      expect_gt(with$num_ll_eval, 0)
      # and it leads to the same optimum
      expect_lt(abs(with$nll - without$nll) / abs(without$nll), TOLERANCE)
      expect_lt(sum(abs(with$cov_pars - without$cov_pars)), TOLERANCE2)
      expect_true(all(is.finite(with$cov_pars)))

    })

  }
  
}
