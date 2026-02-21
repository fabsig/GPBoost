context("GPModel_combined_GP_grouped_random_effects")

# Avoid being tested on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  TOLERANCE_ITERATIVE <- 1E-1
  TOLERANCE_LOOSE <- 1E-2
  TOLERANCE_MEDIUM <- 1e-3
  TOLERANCE_STRICT <- 1E-5
  
  OPTIM_PARAMS_BFGS <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 1000)
  
  # Function that simulates uniform random variables
  sim_rand_unif <- function(n, init_c=0.1){
    mod_lcg <- 2^32 # modulus for linear congruential generator (random0 used)
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for(i in 2:n) sim[i] <- (22695477 * sim[i-1] + 1) %% mod_lcg
    return(sim / mod_lcg)
  }
  
  # Create data
  n <- 100 # number of samples
  
  # Simulate locations / features of GP
  d <- 2 # dimension of GP locations
  coords <- matrix(sim_rand_unif(n=n*d, init_c=0.1), ncol=d)
  D <- as.matrix(dist(coords))
  # Simulate GP
  sigma2_1 <- 1^2 # marginal variance of GP
  rho <- 0.1 # range parameter
  Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
  L <- t(chol(Sigma))
  b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.8))
  
  # GP random coefficients
  Z_SVC <- matrix(sim_rand_unif(n=n*2, init_c=0.6), ncol=2) # covariate data for random coeffients
  colnames(Z_SVC) <- c("var1","var2")
  b_2 <- qnorm(sim_rand_unif(n=n, init_c=0.17))
  b_3 <- qnorm(sim_rand_unif(n=n, init_c=0.42))
  
  # First grouped random effects model
  m <- 10 # number of categories / levels for grouping variable
  group <- rep(1,n) # grouping variable
  for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
  Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
  b_gr_1 <- qnorm(sim_rand_unif(n=m, init_c=0.56))
  # Second grouped random effect
  n_obs_gr <- n/m # number of sampels per group
  group2 <- rep(1,n) # grouping variable
  for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
  Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
  b_gr_2 <- qnorm(sim_rand_unif(n=n_obs_gr, init_c=0.36))
  # Grouped random slope / coefficient
  x <- cos((1:n-n/2)^2*5.5*pi/n) # covariate data for random slope
  Z3 <- diag(x) %*% Z1
  b_gr_3 <- qnorm(sim_rand_unif(n=m, init_c=0.5678))
  # Error term
  xi <- qnorm(sim_rand_unif(n=n, init_c=0.1)) / 5
  # Data for linear mixed effects model
  X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # desing matrix / covariate data for fixed effect
  beta <- c(2,2) # regression coefficents
  # cluster_ids 
  cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))
  # Sum up random effects
  eps <- as.vector(L %*% b_1) + as.vector(Z1 %*% b_gr_1)
  eps_svc <- as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3) + 
    Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3
  
  test_that("Combined Gaussian process and grouped random effects model ", {
    
    y <- eps + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,var(y)/2,mean(dist(coords))/3)
    
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group)
    nll_exp <- 134.3491913
    cov_pars_eval <- c(0.1,0.9,1.6,0.2)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),1E-6)
    # Estimation 
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                           params = OPTIM_PARAMS_BFGS), file='NUL')
    cov_pars_exp <- c(0.02289067637, 0.09244934369, 0.61508804662, 0.30607202462, 1.02397535406, 0.25670906899, 0.11180921688, 0.04165092072)
    num_it_exp <- 12
    nll_fit_exp <- 132.4136173
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_exp)),1E-6)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[2], 4)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), num_it_exp)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),1E-6)
    # Prediction from fitted model
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    group_test <- c(1,2,9999)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test, predict_cov_mat = TRUE)
    expected_mu <- c(0.3769074, 0.6779193, 0.1803276)
    expected_cov <- c(0.619329940, 0.007893047, 0.001356784, 0.007893047, 0.402082274,
                      -0.014950019, 0.001356784, -0.014950019, 1.046082243)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    # Predict training data random effects
    cov_pars <- gp_model$get_cov_pars(std_err = FALSE)
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]),
                       predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,2] - pred_GP$mu)),1E-6)
    expect_lt(sum(abs(training_data_random_effects[,4] - (pred_GP$var - cov_pars[2]))),1E-6)
    # Grouped REs
    preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords + 1e6,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-6)
    expect_lt(sum(abs(training_data_random_effects[,3] - (preds$var - cov_pars[3]))),1E-6)
    # Prediction using given parameters
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test,
                    cov_pars = c(0.02,1,1.2,0.9), predict_cov_mat = TRUE)
    expected_mu_given <- c(0.3995192, 0.6775987, 0.3710522)
    expected_cov_given <- c(0.1257410304, 0.0017195802, 0.0007660953, 0.0017195802,
                            0.0905110441, -0.0028869470, 0.0007660953, -0.0028869470, 1.1680614026)
    expect_lt(sum(abs(pred$mu-expected_mu_given)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_given)),1E-6)
    
    # with Vecchia
    for (matrix_inversion_method in c("cholesky","iterative")) {
      if (matrix_inversion_method == "cholesky") {
        tol_loc <- 1E-6
        tol_loc2 <- 0.2
        tol_loc3 <- 0.002
        tol_loc4 <- 1e-4
      } else{
        tol_loc <- 0.5
        tol_loc2 <- 0.5
        tol_loc3 <- 0.002
        tol_loc4 <- 1e-4
      }
      # Evaluate negative log-likelihood
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval[2:4], y=y, aux_pars=cov_pars_eval[1])
      expect_lt(abs(nll-nll_exp),tol_loc)
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = matrix_inversion_method)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval[2:4], y=y, aux_pars=cov_pars_eval[1])
      expect_lt(abs(nll-nll_exp),tol_loc2)
      # Estimation 
      if (matrix_inversion_method == "cholesky") {
        ## numerically instable for iterative methods due to small sample size for gaussian likelihood
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                               gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method,
                                               params = OPTIM_PARAMS_BFGS), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp[c(3,5,7)])),tol_loc3)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars_exp[1])),tol_loc3)
        expect_equal(gp_model$get_num_optim_iter(), 14)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),tol_loc4)
      }
    }
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                           params = list(optimizer_cov = "fisher_scoring", 
                                                         init_cov_pars=init_cov_pars)), file='NUL')
    cov_pars <- c(0.02262645, 0.61471473, 1.02446559, 0.11177327)
    cov_pars_est <- as.vector(gp_model$get_cov_pars(std_err = FALSE))
    expect_lt(sum(abs(cov_pars_est-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(class(cov_pars_est), "numeric")
    expect_equal(length(cov_pars_est), 4)
    expect_equal(gp_model$get_num_optim_iter(), 7)
    
    # Do optimization using optim
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group)
    opt <- optim(par=c(0.1,1.5,2,0.2), fn=gp_model$neg_log_likelihood, 
                 y=y, method="L-BFGS-B", lower=1E-10)
    cov_pars_exp_opt <- c(0.02260170497, 0.61475162304, 1.02448807571, 0.11177069792)
    expect_lt(sum(abs(opt$par-cov_pars_exp_opt)),1E-5)
    expect_lt(abs(opt$value-(132.4136164)),1E-5)
    expect_equal(as.integer(opt$counts[1]), 30)
    
    ## Duplicate coordinates
    coords_dupl <- coords
    for(i in 2:10) coords_dupl[i,] <- coords_dupl[1,]
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_dupl, cov_function = "exponential", group_data = group)
    nll_exp <- 158.5590203
    cov_pars_eval <- c(0.1,0.9,1.6,0.2)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),1E-6)
    # with Vecchia
    gp_model <- GPModel(gp_coords = coords_dupl, cov_function = "exponential", group_data = group, 
                        gp_approx = "vecchia", num_neighbors = 90, matrix_inversion_method = "cholesky")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval[2:4], y=y, aux_pars=cov_pars_eval[1])
    expect_lt(abs(nll-nll_exp),1E-6)
    gp_model <- GPModel(gp_coords = coords_dupl, cov_function = "exponential", group_data = group, 
                        gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = "cholesky")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval[2:4], y=y, aux_pars=cov_pars_eval[1])
    expect_lt(abs(nll-nll_exp),0.1)
  })
  
  test_that("Combined Gaussian process and grouped random effects model with 't' likelihood ", {
    
    y <- eps + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    init_aux_pars <- c(1,3)
    # params = OPTIM_PARAMS_BFGS
    # params$init_cov_pars <- init_cov_pars
    likelihood <- "t"
    
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, likelihood=likelihood)
    nll_exp <- 223.618399
    cov_pars_eval <- c(0.9,1.6,0.2)
    aux_pars_eval <- c(3,3)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y,aux_pars=aux_pars_eval)
    expect_lt(abs(nll-nll_exp),1E-6)
    # Estimation 
    params_no_aux <- OPTIM_PARAMS_BFGS
    params_no_aux$estimate_aux_pars <- FALSE
    params_no_aux$init_cov_pars <- init_cov_pars
    params_no_aux$init_aux_pars <- init_aux_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                           params = params_no_aux, likelihood=likelihood), file='NUL')
    cov_pars_exp_no_aux <- c(0.5537338, 0.6624236, 0.1827465)
    num_it_exp_no_aux <- 4
    nll_fit_exp_no_aux <- 151.6831043
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp_no_aux)),1E-6)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-init_aux_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), num_it_exp_no_aux)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp_no_aux)),1E-6)
    # also estimating aux pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                           params = OPTIM_PARAMS_BFGS, likelihood=likelihood), file='NUL')
    cov_pars_exp <- c(0.6226558721, 1.0472243490, 0.1063655318)
    aux_pars_exp <- c(0.004424372434, 1.635889324257 )
    num_it_exp <- 17
    nll_fit_exp <- 118.7810787
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),1E-6)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_exp)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), num_it_exp)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),1E-6)
    # Prediction from fitted model
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    group_test <- c(1,2,9999)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(13.017788419, -2.738854979,  8.583333782)
    expected_var <- c(0.6252113051, 0.3888090839, 1.0504367097)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)
    # Predict training data random effects
    cov_pars <- gp_model$get_cov_pars(std_err = FALSE)
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = FALSE)
    pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]),
                       predict_var = FALSE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,2] - pred_GP$mu)),1E-6)
    # Grouped REs
    preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords + 1e6,
                     predict_var = FALSE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-6)
    
    # with Vecchia
    for (matrix_inversion_method in c("cholesky","iterative")) {
      if (matrix_inversion_method == "cholesky") {
        tol_loc <- 1E-6
        tol_loc2 <- 0.2
        tol_loc3 <- 0.002
        tol_loc4 <- 0.03
      } else{
        tol_loc <- 0.2
        tol_loc2 <- 0.5
        tol_loc3 <- 0.002
        tol_loc4 <- 1e-4
      }
      # Evaluate negative log-likelihood
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method, 
                          likelihood=likelihood)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval, y=y, aux_pars=aux_pars_eval)
      expect_lt(abs(nll-nll_exp),tol_loc)
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = matrix_inversion_method,
                          likelihood=likelihood)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval, y=y, aux_pars=aux_pars_eval)
      expect_lt(abs(nll-nll_exp),tol_loc2)
      # Estimation 
      if (matrix_inversion_method == "cholesky") {
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                               gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method,
                                               params = OPTIM_PARAMS_BFGS, likelihood=likelihood), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_loc3)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_exp)),tol_loc3)
        expect_equal(gp_model$get_num_optim_iter(), num_it_exp)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),tol_loc3)
        
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                               gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = matrix_inversion_method,
                                               params = OPTIM_PARAMS_BFGS, likelihood=likelihood), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_loc4)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_exp)),tol_loc4)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),tol_loc2)
      }

      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                             gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method,
                                             params = params_no_aux, likelihood=likelihood), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp_no_aux)),tol_loc2)
      expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-init_aux_pars)),1E-6)
      expect_equal(gp_model$get_num_optim_iter(), num_it_exp_no_aux)
      expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp_no_aux)),tol_loc2)
      
    }
    
  })
  
  test_that("Combined GP and grouped random effects model with linear regression term ", {
    
    y <- eps + X%*%beta + xi
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls"))
    cov_pars <- c(0.02258493, 0.09172947, 0.61704845, 0.30681934, 1.01910740, 0.25561489, 0.11202133, 0.04174140)
    coef <- c(2.06686646, 0.34643130, 1.92847425, 0.09983966)
    nll_opt <- 132.1449371
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    
    # Prediction 
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    group_test <- c(1,2,9999)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    pred <- predict(gp_model, gp_coords_pred = coord_test, group_data_pred = group_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expected_mu <- c(1.442617, 3.129006, 2.946252)
    expected_cov <- c(0.615200495, 0.007850776, 0.001344528, 0.007850776, 0.399458031,
                      -0.014866034, 0.001344528, -0.014866034, 1.045700453)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
  })
  
  test_that("Combined GP and grouped random effects model with random coefficients ", {
    
    y <- eps_svc + xi
    init_cov_pars <- c(var(y)/2,var(y)/2,var(y)/2,var(y)/2,var(y)/2,mean(dist(coords))/3,var(y)/2,mean(dist(coords))/3,var(y)/2,mean(dist(coords))/3)
    # Fit model
    gp_model <- fitGPModel(y = y, gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                           group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                           params = list(optimizer_cov = "gradient_descent",
                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                         acc_rate_cov = 0.5, maxit=10, init_cov_pars=init_cov_pars))
    expected_values <- c(0.4005820, 0.3111155, 0.4564903, 0.2693683, 1.3819153, 0.7034572,
                         1.0378165, 0.5916405, 1.3684672, 0.6861339, 0.1854759, 0.1430030,
                         0.5790945, 0.9748316, 0.2103132, 0.4453663, 0.2639379, 0.8772996, 0.2210313, 0.9282390)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-expected_values)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Prediction
    gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential",
                        group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1)
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    group_data_pred = cbind(c(1,1,7),c(2,1,3))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.1,0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
    expected_mu <- c(0.8657964, 1.5419953, -2.5645509)
    expected_cov <- c(1.177484599, 0.073515374, 0.030303784, 0.073515374,
                      0.841043737, 0.004484463, 0.030303784, 0.004484463, 1.011570695)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    
    # Fisher scoring
    gp_model <- fitGPModel(y = y, gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                           group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                           params = list(optimizer_cov = "fisher_scoring", 
                                         use_nesterov_acc= FALSE, maxit=2, init_cov_pars=init_cov_pars))
    expected_values <- c(0.3522488799, 0.5692314997, 1.4557330868, 1.0711929149, 1.5665274019, 0.1601443490, 0.9923054860, 0.1095828593, 0.2211923864, 0.3846536135)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 2)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-182.3674191),1E-5)
  })
  
  test_that("Combined GP and grouped random effects model with cluster_id's not constant ", {
    
    y <- eps + xi
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                                           y = y, cluster_ids = cluster_ids,
                                           params = list(optimizer_cov = "fisher_scoring")), file='NUL')
    cov_pars <- c(0.005306836, 0.087915468, 0.615012714, 0.315022228,
                  1.043024690, 0.228236254, 0.113716679, 0.039839629)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_MEDIUM)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    group_data_pred = c(1,1,9999)
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                        cluster_ids = cluster_ids)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(0.1,1.5,1,0.15), predict_cov_mat = TRUE)
    expected_mu <- c(0.1275193, 0.0000000, 0.5948827)
    expected_cov <- c(0.76147286, 0.00000000, -0.01260688, 0.00000000, 2.60000000,
                      0.00000000, -0.01260688, 0.00000000, 2.15607110)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  })
  
  test_that("Saving a GPModel and loading from file works ", {
    
    y <- eps + X%*%beta + xi
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                           y = y, X=X,
                           params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls"))
    
    # Prediction 
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    group_test <- c(1,2,9999)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    pred <- predict(gp_model, gp_coords_pred = coord_test, group_data_pred = group_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    # Delete model
    rm(gp_model)
    # Load from file and make predictions again
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, group_data_pred = group_test,
                           X_pred = X_test, predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    
  })
  
}
