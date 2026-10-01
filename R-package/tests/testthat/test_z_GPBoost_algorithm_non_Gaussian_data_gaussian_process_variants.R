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
    
    test_that("GPBoost algorithm with 'gaussian_heteroscedastic_fixed_and_random': initial values from a homoscedastic model", {
      
      # The covariance parameters are not estimated and keep their initial values: those of a homoscedastic
      #   Gaussian process model with an intercept for the Gaussian process of the mean, and its range and a
      #   variance of 0.01 for the Gaussian process of the log-error variance
      ntrain <- 200
      sim_data <- sim_friedman3(n=ntrain, n_irrelevant=5, init_c=0.69)
      X <- sim_data$X
      coords <- matrix(sim_rand_unif(n=ntrain*2, init_c=0.63), ncol=2)
      C <- t(chol(exp(-as.matrix(dist(coords))/0.1) + diag(1E-20,ntrain)))
      y <- sim_data$f + as.vector(C %*% qnorm(sim_rand_unif(n=ntrain, init_c=0.987864))) +
        0.1 * qnorm(sim_rand_unif(n=ntrain, init_c=0.52574))
      capture.output( gp_model_hom <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "vecchia",
                                                 num_neighbors = 10, vecchia_ordering = "none",
                                                 y = y, X = matrix(1, nrow = ntrain, ncol = 1)), file='NUL')
      cov_pars_hom <- as.vector(gp_model_hom$get_cov_pars())
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                          likelihood = "gaussian_heteroscedastic_fixed_and_random", gp_approx = "vecchia",
                          num_neighbors = 10, vecchia_ordering = "none")
      capture.output( bst <- gpb.train(data = gpb.Dataset(data = X, label = y), gp_model = gp_model, nrounds = 1,
                                       train_gp_model_cov_pars = FALSE, verbose = 0), file='NUL')
      expect_lt(max(abs(as.vector(gp_model$get_cov_pars()) / c(cov_pars_hom[2:3], 0.01, cov_pars_hom[3]) - 1)), TOLERANCE)
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
        #	misspecified. From the initial values of a homoscedastic Gaussian process, the fit ends at a
        #	spatially correlated mean process (negative log-likelihood 369.4). Another local optimum lies at the
        #	boundary, where the mean process is uncorrelated and the process of the log-error variance constant
        #	(361.8); its predictions are worse (test NLPD 1.155 instead of 1.009), and the tolerances of the
        #	covariance parameters and of the predicted random effects below separate the two
        cov_pars_est <- c(1.021396e-01, 1.046500e-02, 1.599117e-05, 7.435874e-03)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),relax_tolerance_stoch(0.05))

        # Prediction
        pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                        predict_var = TRUE, pred_latent = TRUE)
        npred <- dim(X_test)[1]
        expect_lt(sum(abs(pred$fixed_effect[1:4]-c(0.6035652, 0.4646622, 0.4646622, 0.6035652))),relax_tolerance_stoch(2))
        expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(-0.001173176, -0.086163283, 0.054093018, 0.018289771))),relax_tolerance_stoch(0.05))
        # the predictive variances are simulation-based and vary with the number of threads by about 0.001
        expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.10213093, 0.09556879, 0.09591550, 0.10173293))),relax_tolerance_stoch(0.05))
        # Predict response
        pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                        predict_var = TRUE, pred_latent = FALSE)
        expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.9276366, 0.5174019, 0.7283017, 0.6616905))),relax_tolerance_stoch(1))
        expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.2857612, 0.2487550, 0.3031877, 0.2444298))),relax_tolerance_stoch(0.3))
        
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
