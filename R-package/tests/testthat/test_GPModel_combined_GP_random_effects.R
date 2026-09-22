context("GPModel_combined_GP_grouped_random_effects")

# Non-convex / stochastic optimization: a different compiler or standard library (e.g. clang + libc++ or
# gcc + libstdc++ on Linux, used by the sanitizer containers of R-hub and CRAN) can converge to a different
# stationary point with practically the same likelihood. Only require the tight tolerances on the reference
# platform on which the expected values were calculated.
# See helper-tolerances.R, which defines this and reports it once per test run
USE_STRICT_TOLERANCES <- gpb_use_strict_tolerances()
relax_tolerance <- function(tol) if (USE_STRICT_TOLERANCES) tol else max(2 * tol, 0.5)
# Separate helper for ABSOLUTE differences of negative log-likelihoods: these are on the scale of the
# log-likelihood itself (typically 100-1000 here), so a larger absolute tolerance is still a small relative one
relax_tolerance_nll <- function(tol) if (USE_STRICT_TOLERANCES) tol else max(3 * tol, 3)

# Avoid being tested on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  TOLERANCE_ITERATIVE <- 1E-1
  TOLERANCE_LOOSE <- 1E-2
  TOLERANCE_MEDIUM <- 1e-3
  TOLERANCE_STRICT <- 1E-5
  
  OPTIM_PARAMS_BFGS <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 1000,
                            init_coef_aux_pars_from_iid_model = FALSE)
  
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
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                           group_data = group, y = y,
                                           params = OPTIM_PARAMS_BFGS), file='NUL')
    cov_pars_exp <- c(0.02289067637, 0.09244934369, 0.61508804662, 0.30607202462, 
                      1.02397535406, 0.25670906899, 0.11180921688, 0.04165092072)
    num_it_exp <- 12
    nll_fit_exp <- 132.4136173
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_exp)),1E-6)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[2], 4)
    expect_equal(dim(gp_model$get_cov_pars(std_err = TRUE))[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), num_it_exp)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),1E-6)
    # Prediction
    cov_pars_pred <- c(0.2,1.6,0.8,0.1)
    gp_model$set_optim_params(params=list(init_cov_pars=cov_pars_pred, init_coef_aux_pars_from_iid_model = FALSE))
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    group_test <- c(1,2,9999)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.3721319527, 0.5343947787, 0.2443116107)
    expected_cov <- c(0.5645577689509, 0.0045582767527, 0.0004141028041, 0.0045582767527, 0.4240101561305,
                      -0.0192167346651, 0.0004141028041, -0.0192167346651, 1.9971330900008)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    # Predict only GP
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test + 1e6, predict_var = TRUE, predict_response = FALSE)
    expected_mu_gp <- c(-0.1426408669, 1.2066955813, 0.2443116107)
    expected_var_gp <- c(2.146560921, 1.977119721, 1.997133090)
    expect_lt(sum(abs(pred$mu-expected_mu_gp)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_gp)),TOLERANCE_MEDIUM)
    # Predict only grouped RE
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test + 1e6,
                    group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expected_mu_group <- c(0.5147728196, -0.6723008026, 0.0000000000)
    expected_var_group <- c(0.8873613202, 0.8945594521, 2.4000000000)
    expect_lt(sum(abs(pred$mu-expected_mu_group)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_group)),TOLERANCE_MEDIUM)
    # Predict training data random effects
    cov_pars <- gp_model$get_cov_pars(std_err = FALSE)
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]),
                       predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,2] - pred_GP$mu)),1E-6)
    expect_lt(sum(abs(training_data_random_effects[,4] - (pred_GP$var - cov_pars[2]))),1E-6)
    preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords + 1e6,
                     predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-6)
    expect_lt(sum(abs(training_data_random_effects[,3] - (preds$var - cov_pars[3]))),1E-6)
    # Sampling from posterior
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE,
                    sample_posterior = TRUE, num_post_samples=100000)
    expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-pred$mu)), 0.01)
    expect_lt(sum(abs(cov(t(pred$posterior_samples))-pred$cov)), 0.2)
    # with weights being 1
    weights <- rep(1,length(y))
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, weights=weights)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),1E-6)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                                           group_data = group, y = y, weights=weights,
                                           params = OPTIM_PARAMS_BFGS), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars_exp)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), num_it_exp)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),1E-6)
    gp_model$set_optim_params(params=list(init_cov_pars=cov_pars_pred, init_coef_aux_pars_from_iid_model = FALSE))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    
    # matrix_inversion_method = "cholesky"
    # with Vecchia
    for (matrix_inversion_method in c("cholesky","iterative")) {
      if (matrix_inversion_method == "cholesky") {
        tol_loc <- 1E-6
        tol_loc2 <- 0.2
        tol_loc3 <- 0.002
        tol_loc4 <- 1e-4
        tol_loc_5 <- 0.05
      } else{
        tol_loc <- 0.5
        tol_loc2 <- 0.5
        tol_loc3 <- 0.002
        tol_loc4 <- 1e-4
        tol_loc_5 <- 0.15
      }
      # Evaluate negative log-likelihood
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000, init_coef_aux_pars_from_iid_model = FALSE))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval[2:4], y=y, aux_pars=cov_pars_eval[1])
      expect_lt(abs(nll-nll_exp),tol_loc)
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = 20, vecchia_ordering = "none", 
                          matrix_inversion_method = matrix_inversion_method)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000, init_coef_aux_pars_from_iid_model = FALSE))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval[2:4], y=y, aux_pars=cov_pars_eval[1])
      expect_lt(abs(nll-nll_exp),tol_loc2)
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = 20, vecchia_ordering = "none", 
                          matrix_inversion_method = matrix_inversion_method, weights=weights)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000, init_coef_aux_pars_from_iid_model = FALSE))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval[2:4], y=y, aux_pars=cov_pars_eval[1])
      expect_lt(abs(nll-nll_exp),tol_loc2)
      # Estimation 
      params = OPTIM_PARAMS_BFGS
      if (matrix_inversion_method == "iterative") params$maxit <- 5 ## very slow for iterative methods likely due to small sample size
      ## numerically unstable for iterative methods due to small sample size (it works well for larger data sets)
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                             gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method,
                                             params = params), file='NUL')
      if (matrix_inversion_method == "iterative") {
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-132.2011049)),1.1)
      } else {
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp[c(3,5,7)])),tol_loc3)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-cov_pars_exp[1])),tol_loc3)
        expect_equal(gp_model$get_num_optim_iter(), 14)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),tol_loc4)
      }
      # Prediction
      predict_var <- TRUE
      gp_model$set_optim_params(params=list(init_cov_pars=cov_pars_pred[-1], init_aux_pars = cov_pars_pred[1], init_coef_aux_pars_from_iid_model = FALSE))
      gp_model$set_prediction_data(nsim_var_pred = 1000)
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                      group_data_pred = group_test, predict_var = predict_var, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
      if(predict_var) expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tol_loc_5)
      # Predict only GP
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                      group_data_pred = group_test + 1e6, predict_var = predict_var, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu_gp)),TOLERANCE_MEDIUM)
      if(predict_var) expect_lt(sum(abs(as.vector(pred$var)-expected_var_gp)),0.02)
      # Predict only grouped RE
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test + 1e6,
                                      group_data_pred = group_test, predict_var = predict_var, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu_group)),TOLERANCE_MEDIUM)
      if(predict_var) expect_lt(sum(abs(as.vector(pred$var)-expected_var_group)),0.2)
      # Predict training data random effects
      cov_pars <- gp_model$get_cov_pars(std_err = FALSE)
      training_data_random_effects_vecchia <- predict_training_data_random_effects(gp_model, predict_var = FALSE)
      capture.output( pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]),
                                         predict_var = FALSE, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(training_data_random_effects_vecchia[,2] - pred_GP$mu)),1E-6)
      capture.output( preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords + 1e6,
                                       predict_var = FALSE, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(training_data_random_effects_vecchia[,1] - preds$mu)),1E-6)
      expect_lt(sum(abs(training_data_random_effects_vecchia - training_data_random_effects[,c(1,2)])),0.05)
    }
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                           params = list(optimizer_cov = "fisher_scoring", 
                                                         init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
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
    expect_lt(abs(nll-nll_exp), 0.3)
    
    # with weights
    coords_w <- cbind(c(0.05, 0.18, 0.31, 0.52, 0.74, 0.91),
                      c(0.12, 0.44, 0.27, 0.83, 0.35, 0.66))
    group_w <- c(1, 1, 2, 2, 3, 3)
    y_w <- c(0.25, -0.40, 1.20, 0.75, -0.15, 1.45)
    weights_w <- c(1.0, 2.0, 3.0, 1.5, 0.7, 2.2)
    cov_pars_w <- c(0.45, 0.80, 1.20, 0.35)
    capture.output( gp_model_w <- GPModel(gp_coords = coords_w, cov_function = "exponential",
                                          group_data = group_w, weights = weights_w) , file='NUL')
    nll_w <- gp_model_w$neg_log_likelihood(cov_pars = cov_pars_w, y = y_w)
    Z_w <- model.matrix(rep(1, length(group_w)) ~ factor(group_w) - 1)
    D_w <- as.matrix(dist(coords_w))
    Sigma_w <- cov_pars_w[2] * tcrossprod(Z_w) +
      cov_pars_w[3] * exp(-D_w / cov_pars_w[4]) +
      cov_pars_w[1] * diag(1 / weights_w)
    chol_Sigma_w <- chol(Sigma_w)
    nll_w_manual <- 0.5 * drop(crossprod(y_w, solve(Sigma_w, y_w))) +
      sum(log(diag(chol_Sigma_w))) + length(y_w) / 2 * log(2 * pi)
    expect_lt(abs(nll_w - nll_w_manual), TOLERANCE_STRICT)
    
    coords_pred_w <- cbind(c(0.16, 0.60, 0.88), c(0.20, 0.70, 0.40))
    group_pred_w <- c(1, 3, 4)
    pred_w <- predict(gp_model_w, y = y_w, gp_coords_pred = coords_pred_w,
                      group_data_pred = group_pred_w, cov_pars = cov_pars_w,
                      predict_response = TRUE, predict_cov_mat = TRUE)
    D_pred_obs_w <- as.matrix(dist(rbind(coords_pred_w, coords_w)))[1:nrow(coords_pred_w),
                                                                    -(1:nrow(coords_pred_w))]
    D_pred_w <- as.matrix(dist(coords_pred_w))
    Z_pred_w <- model.matrix(rep(1, length(group_pred_w)) ~
                               factor(group_pred_w, levels = c(sort(unique(group_w)), 4)) - 1)
    Z_obs_w <- model.matrix(rep(1, length(group_w)) ~
                              factor(group_w, levels = c(sort(unique(group_w)), 4)) - 1)
    cross_cov_w <- cov_pars_w[2] * Z_pred_w %*% t(Z_obs_w) +
      cov_pars_w[3] * exp(-D_pred_obs_w / cov_pars_w[4])
    pred_cov_prior_w <- cov_pars_w[2] * tcrossprod(Z_pred_w) +
      cov_pars_w[3] * exp(-D_pred_w / cov_pars_w[4]) +
      cov_pars_w[1] * diag(length(group_pred_w))
    pred_mean_manual_w <- as.vector(cross_cov_w %*% solve(Sigma_w, y_w))
    pred_cov_manual_w <- pred_cov_prior_w - cross_cov_w %*% solve(Sigma_w, t(cross_cov_w))
    expect_lt(sum(abs(pred_w$mu - pred_mean_manual_w)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred_w$cov) - as.vector(pred_cov_manual_w))), TOLERANCE_STRICT)
    
    X_w <- cbind(1, c(-1.0, -0.5, 0.2, 0.7, 1.1, -0.2))
    capture.output( gp_model_w_fit_X <- fitGPModel(gp_coords = coords_w,
                                                   cov_function = "exponential",
                                                   group_data = group_w,
                                                   y = y_w, X = X_w,
                                                   weights = weights_w,
                                                   params = list(optimizer_cov = "lbfgs",
                                                                 optimizer_coef = "wls", init_coef_aux_pars_from_iid_model = FALSE)) , file='NUL')
    cov_pars_fit_X <- as.vector(gp_model_w_fit_X$get_cov_pars())
    beta_fit_X <- as.vector(gp_model_w_fit_X$get_coef())
    cov_pars <- c(2.49299439302e-08, 1.21361827019e+00, 4.44355584411e-03, 5.62394305334e-12)
    coef <- c(0.576402422316, -1.194715383519)
    expect_lt(sum(abs(cov_pars_fit_X - cov_pars)), TOLERANCE_STRICT)
    expect_lt(sum(abs(beta_fit_X - coef)), TOLERANCE_STRICT)
  })
  
  test_that("Combined Gaussian process and grouped random effects model with 'gamma' likelihood ", {
    
    mu <- exp(eps)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.234), scale = mu, shape = 0.5)
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    init_aux_pars <- c(1)
    # params = OPTIM_PARAMS_BFGS
    # params$init_cov_pars <- init_cov_pars
    likelihood <- "gamma"
    
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, likelihood=likelihood)
    nll_exp <- 86.20875547
    cov_pars_eval <- c(0.9,1.6,0.2)
    aux_pars_eval <- c(1.25)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y,aux_pars=aux_pars_eval)
    expect_lt(abs(nll-nll_exp),1E-6)
    # Estimation 
    params_no_aux <- OPTIM_PARAMS_BFGS
    params_no_aux$estimate_aux_pars <- FALSE
    params_no_aux$init_cov_pars <- init_cov_pars
    params_no_aux$init_aux_pars <- init_aux_pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                           params = params_no_aux, likelihood=likelihood), file='NUL')
    cov_pars_exp_no_aux <- c(1.8453330924, 3.1754659533, 0.0518091748)
    num_it_exp_no_aux <- 8
    nll_fit_exp_no_aux <- 66.03473498
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp_no_aux)),1E-6)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-init_aux_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), num_it_exp_no_aux)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp_no_aux)),1E-6)
    # also estimating aux pars
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                           params = OPTIM_PARAMS_BFGS, likelihood=likelihood), file='NUL')
    cov_pars_exp <- c(0.80774348439, 0.74706773675, 0.09791784209)
    aux_pars_exp <- c(0.4396852858)
    num_it_exp <- 9
    nll_fit_exp <- 55.15492325
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),1E-6)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_exp)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), num_it_exp)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),1E-6)
    # Prediction
    aux_pars_pred <- c(0.6)
    cov_pars_pred <- c(0.8,1.6,0.1)
    gp_model$set_optim_params(params=list(init_aux_pars=aux_pars_pred, init_cov_pars=cov_pars_pred, init_coef_aux_pars_from_iid_model = FALSE))
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    group_test <- c(1,2,9999)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.3867816583, -0.4921888663, -0.4173773440)
    expected_var <- c(1.571479626, 1.178924600, 1.805437351)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)
    # Predict only GP
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test + 1e6, predict_var = TRUE, predict_response = FALSE)
    expected_mu_gp <- c(-0.6644972327, 0.9465254470, -0.4173773440)
    expected_var_gp <- c(2.138215372, 1.677297126, 1.805437351)
    expect_lt(sum(abs(pred$mu-expected_mu_gp)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_gp)),TOLERANCE_MEDIUM)
    # Predict only grouped RE
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test + 1e6,
                    group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expected_mu_group <- c(0.2777155744, -1.4387143133, 0.0000000000)
    expected_var_group <- c(1.862818611, 1.938540348, 2.400000000)
    expect_lt(sum(abs(pred$mu-expected_mu_group)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_group)),TOLERANCE_MEDIUM)
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = FALSE)
    pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]),
                       predict_var = FALSE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,2] - pred_GP$mu)),1E-6)
    # Grouped REs
    preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords + 1e6,
                     predict_var = FALSE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-6)
    # Sampling from posterior
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE,
                    sample_posterior = TRUE, num_post_samples=100000)
    expect_lt(sum(abs(apply(pred$posterior_samples,1,mean)-pred$mu)), 0.01)
    expect_lt(sum(abs(cov(t(pred$posterior_samples))-pred$cov)), 0.3)
    
    # with Vecchia
    # matrix_inversion_method = "cholesky"
    for (matrix_inversion_method in c("cholesky","iterative")) {
      if (matrix_inversion_method == "cholesky") {
        tol_loc <- 1E-6
        tol_loc2 <- 0.1
        tol_loc3 <- 0.2
        tol_loc4 <- 0.03
      } else{
        tol_loc <- 0.1
        tol_loc2 <- 0.15
        tol_loc3 <- 0.4
        tol_loc4 <- 0.03
      }
      # Evaluate negative log-likelihood
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method, 
                          likelihood=likelihood)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000, init_coef_aux_pars_from_iid_model = FALSE))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval, y=y, aux_pars=aux_pars_eval)
      expect_lt(abs(nll-nll_exp),tol_loc)
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = matrix_inversion_method,
                          likelihood=likelihood)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000, init_coef_aux_pars_from_iid_model = FALSE))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval, y=y, aux_pars=aux_pars_eval)
      expect_lt(abs(nll-nll_exp),tol_loc2)
      # Estimation 
      params <- OPTIM_PARAMS_BFGS
      params$num_rand_vec_trace <- 1000
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                             gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method,
                                             params = params, likelihood=likelihood), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_loc3)
      expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_exp)),tol_loc3)
      # if (matrix_inversion_method == "cholesky") expect_equal(gp_model$get_num_optim_iter(), num_it_exp)
      expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),relax_tolerance_nll(tol_loc2))
      # Prediction
      predict_var <- TRUE
      gp_model$set_optim_params(params=list(init_aux_pars=aux_pars_pred, init_cov_pars=cov_pars_pred, init_coef_aux_pars_from_iid_model = FALSE))
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                      group_data_pred = group_test, predict_var = predict_var, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu)),tol_loc4)
      if (predict_var) expect_lt(sum(abs(as.vector(pred$var)-expected_var)),2.5*tol_loc2)
      # Predict only GP
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                      group_data_pred = group_test + 1e6, predict_var = predict_var, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu_gp)),tol_loc4)
      if (predict_var) expect_lt(sum(abs(as.vector(pred$var)-expected_var_gp)),tol_loc2)
      # Predict only grouped RE
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test + 1e6,
                                      group_data_pred = group_test, predict_var = predict_var, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu_group)),tol_loc4)
      if (predict_var) expect_lt(sum(abs(as.vector(pred$var)-expected_var_group)),tol_loc2*3)
      # Predict training data random effects
      training_data_random_effects_vecchia <- predict_training_data_random_effects(gp_model, predict_var = FALSE)
      capture.output( pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]),
                                         predict_var = FALSE, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(training_data_random_effects_vecchia[,2] - pred_GP$mu)),1E-6)
      # Grouped REs
      capture.output(  preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords + 1e6,
                                        predict_var = FALSE, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(training_data_random_effects_vecchia[,1] - preds$mu)),1E-6)
      expect_lt(sum(abs(training_data_random_effects_vecchia - training_data_random_effects)),0.7)
      
      if (matrix_inversion_method == "cholesky") {
        # Less neighbors
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                               gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = matrix_inversion_method,
                                               params = OPTIM_PARAMS_BFGS, likelihood=likelihood), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_loc3)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_exp)),tol_loc3)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),relax_tolerance_nll(tol_loc2))
        
        # Do not estimate aux_pars
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                               gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method,
                                               params = params_no_aux, likelihood=likelihood), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp_no_aux)),2)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-init_aux_pars)),1E-6)
        # expect_equal(gp_model$get_num_optim_iter(), num_it_exp_no_aux)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp_no_aux)),2)
      }
      
    }
    
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
    # Prediction
    aux_pars_pred <- c(0.1,1.5)
    cov_pars_pred <- c(0.8,1.6,0.1)
    gp_model$set_optim_params(params=list(init_aux_pars=aux_pars_pred, init_cov_pars=cov_pars_pred, init_coef_aux_pars_from_iid_model = FALSE))
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    group_test <- c(1,2,9999)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(0.3686265299, 0.6870758253, 0.1594843254)
    expected_var <- c(1.0067689176, 0.6368106842, 1.4868522924)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)
    # Predict only GP
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    group_data_pred = group_test + 1e6, predict_var = TRUE, predict_response = FALSE)
    expected_mu_gp <- c(-0.1362060119, 1.3524178596, 0.1594843254)
    expected_var_gp <- c(1.796037327, 1.380933498, 1.486852292)
    expect_lt(sum(abs(pred$mu-expected_mu_gp)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_gp)),TOLERANCE_MEDIUM)
    # Predict only grouped RE
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test + 1e6,
                    group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expected_mu_group <- c(0.5048325418, -0.6653420343, 0.00000000)
    expected_var_group <- c(1.705078075, 1.713909773, 2.400000000)
    expect_lt(sum(abs(pred$mu-expected_mu_group)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_group)),TOLERANCE_MEDIUM)
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = FALSE)
    pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]),
                       predict_var = FALSE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,2] - pred_GP$mu)),1E-6)
    # Grouped REs
    preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords + 1e6,
                     predict_var = FALSE, predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-6)
    
    # with Vecchia
    # matrix_inversion_method = "iterative"
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
        tol_loc4 <- 0.03
      }
      # Evaluate negative log-likelihood
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method, 
                          likelihood=likelihood)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000, init_coef_aux_pars_from_iid_model = FALSE))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval, y=y, aux_pars=aux_pars_eval)
      expect_lt(abs(nll-nll_exp),tol_loc)
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group, 
                          gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = matrix_inversion_method,
                          likelihood=likelihood)
      gp_model$set_optim_params(params = list(num_rand_vec_trace=1000, init_coef_aux_pars_from_iid_model = FALSE))
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval, y=y, aux_pars=aux_pars_eval)
      expect_lt(abs(nll-nll_exp),tol_loc2)
      # Estimation 
      params = OPTIM_PARAMS_BFGS
      if (matrix_inversion_method == "iterative") params$maxit <- 5 ## very slow for iterative methods likely due to small sample size
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                             gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method,
                                             params = params, likelihood=likelihood), file='NUL')
      if (matrix_inversion_method == "iterative") {
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-119.3419885)), 2.5)
      } else {
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_loc3)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_exp)),tol_loc3)
        expect_equal(gp_model$get_num_optim_iter(), num_it_exp)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),tol_loc3)
      }
      # Prediction
      predict_var <- TRUE
      gp_model$set_optim_params(params=list(init_aux_pars=aux_pars_pred, init_cov_pars=cov_pars_pred, init_coef_aux_pars_from_iid_model = FALSE))
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                      group_data_pred = group_test, predict_var = predict_var, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu)),tol_loc4)
      if (predict_var) expect_lt(sum(abs(as.vector(pred$var)-expected_var)),tol_loc2)
      # Predict only GP
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                      group_data_pred = group_test + 1e6, predict_var = predict_var, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu_gp)),tol_loc4)
      if (predict_var) expect_lt(sum(abs(as.vector(pred$var)-expected_var_gp)),tol_loc4)
      # Predict only grouped RE
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test + 1e6,
                                      group_data_pred = group_test, predict_var = predict_var, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu_group)),tol_loc4)
      if (predict_var) expect_lt(sum(abs(as.vector(pred$var)-expected_var_group)),tol_loc2*1.25)
      # Predict training data random effects
      training_data_random_effects_vecchia <- predict_training_data_random_effects(gp_model, predict_var = FALSE)
      capture.output( pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]),
                                         predict_var = FALSE, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(training_data_random_effects_vecchia[,2] - pred_GP$mu)),1E-6)
      # Grouped REs
      capture.output(  preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords + 1e6,
                                        predict_var = FALSE, predict_response = FALSE), file='NUL')
      expect_lt(sum(abs(training_data_random_effects_vecchia[,1] - preds$mu)),1E-6)
      expect_lt(sum(abs(training_data_random_effects_vecchia - training_data_random_effects)),0.7)
      
      if (matrix_inversion_method == "cholesky") {
        # Less neighbors
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                               gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = matrix_inversion_method,
                                               params = OPTIM_PARAMS_BFGS, likelihood=likelihood), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_loc4)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_exp)),tol_loc4)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp)),relax_tolerance_nll(tol_loc2))
        
        # Do not estimate aux_pars
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                                               gp_approx = "vecchia", num_neighbors = n-1, matrix_inversion_method = matrix_inversion_method,
                                               params = params_no_aux, likelihood=likelihood), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp_no_aux)),tol_loc2)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-init_aux_pars)),1E-6)
        expect_equal(gp_model$get_num_optim_iter(), num_it_exp_no_aux)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_fit_exp_no_aux)),tol_loc2)
      }
      
    }
    
  })
  
  test_that("Combined GP and grouped random effects model with linear regression term ", {
    
    y <- eps + X%*%beta + xi
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls", init_coef_aux_pars_from_iid_model = FALSE))
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
    invisible(capture.output( gp_model <- fitGPModel(y = y, gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                                     group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                                     params = list(optimizer_cov = "gradient_descent",
                                                                   lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                                   acc_rate_cov = 0.5, maxit=10, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)) ))
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
    invisible(capture.output( gp_model <- fitGPModel(y = y, gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                                     group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                                     params = list(optimizer_cov = "fisher_scoring",
                                                                   use_nesterov_acc= FALSE, maxit=2, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)) ))
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
                                           params = list(optimizer_cov = "fisher_scoring", init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
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
                           params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls", init_coef_aux_pars_from_iid_model = FALSE))
    
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
  

  test_that("Predictive variances of grouped random effects and a Vecchia GP include their posterior covariance ", {

    # With all neighbors the Vecchia approximation is exact, so the predictive variances of this Gaussian
    # model have to equal the ones of exact Gaussian conditioning. The grouped random effects and the GP are
    # correlated a posteriori; leaving that covariance out (which the implementations did) changes the result
    # by a factor of about 2.7 (cholesky) and 4.4 (iterative)
    n_jc <- 60
    m_jc <- 6
    group_jc <- rep(1:m_jc, each = n_jc / m_jc)
    coords_jc <- cbind(sim_rand_unif(n = n_jc, init_c = 0.312), sim_rand_unif(n = n_jc, init_c = 0.573))
    D_jc <- as.matrix(dist(coords_jc))
    sigma2_gp_jc <- 1.0
    rho_jc <- 0.2
    sigma2_gr_jc <- 0.7
    nugget_jc <- 0.4
    Z_jc <- model.matrix(~ factor(group_jc) - 1)
    Sigma_jc <- sigma2_gp_jc * exp(-D_jc / rho_jc) + sigma2_gr_jc * (Z_jc %*% t(Z_jc))
    y_jc <- as.vector(t(chol(Sigma_jc + diag(nugget_jc, n_jc))) %*%
                        qnorm(sim_rand_unif(n = n_jc, init_c = 0.751)))
    np_jc <- 4
    coords_pred_jc <- cbind(c(0.12, 0.44, 0.71, 0.93), c(0.22, 0.51, 0.83, 0.07))
    group_pred_jc <- c(1, 2, 3, 4)
    # Exact Gaussian conditioning of the latent process
    Dpo_jc <- as.matrix(dist(rbind(coords_pred_jc, coords_jc)))[1:np_jc, (np_jc + 1):(np_jc + n_jc)]
    Dpp_jc <- as.matrix(dist(coords_pred_jc))
    Zp_jc <- matrix(0, np_jc, m_jc)
    for (i in 1:np_jc) Zp_jc[i, group_pred_jc[i]] <- 1
    Cpo_jc <- sigma2_gp_jc * exp(-Dpo_jc / rho_jc) + sigma2_gr_jc * (Zp_jc %*% t(Z_jc))
    Cpp_jc <- sigma2_gp_jc * exp(-Dpp_jc / rho_jc) + sigma2_gr_jc * (Zp_jc %*% t(Zp_jc))
    Syy_jc <- Sigma_jc + diag(nugget_jc, n_jc)
    var_exact_jc <- diag(Cpp_jc - Cpo_jc %*% solve(Syy_jc, t(Cpo_jc)))
    mu_exact_jc <- as.vector(Cpo_jc %*% solve(Syy_jc, y_jc))

    for (inv_method_jc in c("cholesky", "iterative")) {
      gp_model_jc <- GPModel(gp_coords = coords_jc, cov_function = "exponential", group_data = group_jc,
                             gp_approx = "vecchia", num_neighbors = n_jc - 1, vecchia_ordering = "none",
                             likelihood = "gaussian", matrix_inversion_method = inv_method_jc)
      gp_model_jc$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all",
                                      num_neighbors_pred = n_jc + np_jc, nsim_var_pred = 2000)
      gp_model_jc$set_optim_params(params = list(init_aux_pars = nugget_jc, seed_rand_vec_trace = 1,
                                                 init_cov_pars = c(sigma2_gr_jc, sigma2_gp_jc, rho_jc),
                                                 init_coef_aux_pars_from_iid_model = FALSE))
      capture.output( pred_jc <- gp_model_jc$predict(y = y_jc, gp_coords_pred = coords_pred_jc,
                                                     group_data_pred = group_pred_jc,
                                                     cov_pars = c(sigma2_gr_jc, sigma2_gp_jc, rho_jc),
                                                     predict_var = TRUE, predict_response = FALSE), file = 'NUL')
      tol_jc <- if (inv_method_jc == "iterative") 1E-2 else 1E-6
      expect_lt(sum(abs(pred_jc$mu - mu_exact_jc)), tol_jc)
      expect_lt(sum(abs(as.vector(pred_jc$var) - var_exact_jc)), tol_jc)
    }

    # The same with an additional random slope, and with a group that does not occur in the training data.
    # The unconditional variance of a random slope must only be added for such a new group, its posterior
    # variance is already contained in the joint calculation above for the groups that do occur
    x_jc <- 0.5 + sim_rand_unif(n = n_jc, init_c = 0.234)
    Zx_jc <- diag(x_jc) %*% Z_jc
    s2_slp_jc <- 0.5
    Sigma_s_jc <- Sigma_jc + s2_slp_jc * (Zx_jc %*% t(Zx_jc))
    Syy_s_jc <- Sigma_s_jc + diag(nugget_jc, n_jc)
    y_s_jc <- as.vector(t(chol(Syy_s_jc)) %*% qnorm(sim_rand_unif(n = n_jc, init_c = 0.337)))
    group_pred_s_jc <- c(1, 2, 3, 99999)# the last group does not occur in the training data
    x_pred_jc <- c(1.2, 0.7, 1.1, 0.9)
    Zp_s_jc <- matrix(0, np_jc, m_jc)
    for (i in 1:np_jc) if (group_pred_s_jc[i] <= m_jc) Zp_s_jc[i, group_pred_s_jc[i]] <- 1
    Cpo_s_jc <- sigma2_gr_jc * (Zp_s_jc %*% t(Z_jc)) +
      s2_slp_jc * (diag(x_pred_jc) %*% Zp_s_jc %*% t(Z_jc) %*% diag(x_jc)) + sigma2_gp_jc * exp(-Dpo_jc / rho_jc)
    Cpp_s_jc <- sigma2_gr_jc * (Zp_s_jc %*% t(Zp_s_jc)) +
      s2_slp_jc * (diag(x_pred_jc) %*% Zp_s_jc %*% t(Zp_s_jc) %*% diag(x_pred_jc)) + sigma2_gp_jc * exp(-Dpp_jc / rho_jc)
    for (i in 1:np_jc) {
      if (group_pred_s_jc[i] > m_jc) {# a new group still has its prior random effects
        Cpp_s_jc[i, i] <- Cpp_s_jc[i, i] + sigma2_gr_jc + s2_slp_jc * x_pred_jc[i]^2
      }
    }
    var_exact_s_jc <- diag(Cpp_s_jc - Cpo_s_jc %*% solve(Syy_s_jc, t(Cpo_s_jc)))
    for (inv_method_jc in c("cholesky", "iterative")) {
      gp_model_s_jc <- GPModel(group_data = group_jc, group_rand_coef_data = x_jc,
                               ind_effect_group_rand_coef = 1, gp_coords = coords_jc,
                               cov_function = "exponential", gp_approx = "vecchia",
                               num_neighbors = n_jc - 1, vecchia_ordering = "none",
                               likelihood = "gaussian", matrix_inversion_method = inv_method_jc)
      gp_model_s_jc$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all",
                                        num_neighbors_pred = n_jc + np_jc, nsim_var_pred = 2000)
      gp_model_s_jc$set_optim_params(params = list(init_aux_pars = nugget_jc, seed_rand_vec_trace = 1,
                                                   init_cov_pars = c(sigma2_gr_jc, s2_slp_jc, sigma2_gp_jc, rho_jc),
                                                   init_coef_aux_pars_from_iid_model = FALSE))
      capture.output( pred_s_jc <- gp_model_s_jc$predict(y = y_s_jc, gp_coords_pred = coords_pred_jc,
                                                         group_data_pred = group_pred_s_jc,
                                                         group_rand_coef_data_pred = x_pred_jc,
                                                         cov_pars = c(sigma2_gr_jc, s2_slp_jc, sigma2_gp_jc, rho_jc),
                                                         predict_var = TRUE, predict_response = FALSE), file = 'NUL')
      tol_s_jc <- if (inv_method_jc == "iterative") 1E-2 else 1E-6
      expect_lt(sum(abs(as.vector(pred_s_jc$var) - var_exact_s_jc)), tol_s_jc)
    }
  })


  test_that("Predictive covariances and posterior sampling of grouped random effects and a Vecchia GP ", {

    # Neither is implemented for this model class: the predictive covariance would have to add the
    # unconditional covariance only for the groups that do not occur in the training data, and the
    # posterior sampling does not cover several random effect components. The guards are checked here so
    # that they cannot be dropped without the results being compared to exact Gaussian conditioning
    n_ni <- 40
    m_ni <- 5
    group_ni <- rep(1:m_ni, each = n_ni / m_ni)
    coords_ni <- cbind(sim_rand_unif(n = n_ni, init_c = 0.312), sim_rand_unif(n = n_ni, init_c = 0.573))
    y_ni <- qnorm(sim_rand_unif(n = n_ni, init_c = 0.751))
    cov_pars_ni <- c(0.7, 1.0, 0.2)
    coords_pred_ni <- cbind(c(0.12, 0.44), c(0.22, 0.51))
    group_pred_ni <- c(1, 2)
    for (inv_method_ni in c("cholesky", "iterative")) {
      gp_model_ni <- GPModel(gp_coords = coords_ni, cov_function = "exponential", group_data = group_ni,
                             gp_approx = "vecchia", num_neighbors = n_ni - 1, vecchia_ordering = "none",
                             likelihood = "gaussian", matrix_inversion_method = inv_method_ni)
      gp_model_ni$set_optim_params(params = list(init_aux_pars = 0.4, seed_rand_vec_trace = 1,
                                                 init_cov_pars = cov_pars_ni,
                                                 init_coef_aux_pars_from_iid_model = FALSE))
      expect_error(gp_model_ni$predict(y = y_ni, gp_coords_pred = coords_pred_ni,
                                       group_data_pred = group_pred_ni, cov_pars = cov_pars_ni,
                                       predict_cov_mat = TRUE, predict_response = FALSE),
                   "Predictive covariances are not implemented")
      expect_error(gp_model_ni$predict(y = y_ni, gp_coords_pred = coords_pred_ni,
                                       group_data_pred = group_pred_ni, cov_pars = cov_pars_ni,
                                       predict_var = TRUE, predict_response = FALSE,
                                       sample_posterior = TRUE, num_post_samples = 10),
                   "Posterior sampling is not implemented")
    }
  })


  test_that("Predictions of a non-Gaussian model with a grouped random slope and a Vecchia GP ", {

    # With all neighbors the Vecchia approximation is exact, so the predictions have to equal the ones of
    # the same model without an approximation. The groups that occur in the training data and one that
    # does not are predicted together: the unconditional variance of the intercept random effect and of
    # the random slope is added only for the new group
    n_ng <- 60
    m_ng <- 6
    group_ng <- rep(1:m_ng, each = n_ng / m_ng)
    coords_ng <- cbind(sim_rand_unif(n = n_ng, init_c = 0.312), sim_rand_unif(n = n_ng, init_c = 0.573))
    x_ng <- 0.5 + sim_rand_unif(n = n_ng, init_c = 0.234)
    D_ng <- as.matrix(dist(coords_ng))
    Z_ng <- model.matrix(~ factor(group_ng) - 1)
    Zx_ng <- diag(x_ng) %*% Z_ng
    cov_pars_ng <- c(0.7, 0.5, 1.0, 0.2)# group variance, random slope variance, GP variance, GP range
    Sigma_ng <- cov_pars_ng[1] * (Z_ng %*% t(Z_ng)) + cov_pars_ng[2] * (Zx_ng %*% t(Zx_ng)) +
      cov_pars_ng[3] * exp(-D_ng / cov_pars_ng[4])
    b_ng <- as.vector(t(chol(Sigma_ng + diag(1E-10, n_ng))) %*% qnorm(sim_rand_unif(n = n_ng, init_c = 0.751)))
    y_ng <- qgamma(sim_rand_unif(n = n_ng, init_c = 0.418), shape = 1, rate = exp(-b_ng))
    coords_pred_ng <- cbind(c(0.12, 0.44, 0.71, 0.93), c(0.22, 0.51, 0.83, 0.07))
    group_pred_ng <- c(1, 2, 3, 99999)# the last group does not occur in the training data
    x_pred_ng <- c(1.2, 0.7, 1.1, 0.9)

    args_ng <- list(group_data = group_ng, group_rand_coef_data = x_ng, ind_effect_group_rand_coef = 1,
                    gp_coords = coords_ng, cov_function = "exponential", likelihood = "gamma")
    predict_ng <- function(args) {
      gp_model_ng <- do.call(GPModel, args)
      gp_model_ng$set_optim_params(params = list(seed_rand_vec_trace = 1, num_rand_vec_trace = 1000,
                                                 init_coef_aux_pars_from_iid_model = FALSE))
      if (!is.null(args$gp_approx)) gp_model_ng$set_prediction_data(nsim_var_pred = 2000)
      capture.output( pred_ng <- gp_model_ng$predict(y = y_ng, gp_coords_pred = coords_pred_ng,
                                                     group_data_pred = group_pred_ng,
                                                     group_rand_coef_data_pred = x_pred_ng,
                                                     cov_pars = cov_pars_ng, predict_var = TRUE,
                                                     predict_response = FALSE), file = 'NUL')
      pred_ng
    }
    pred_exact_ng <- predict_ng(args_ng)
    for (inv_method_ng in c("cholesky", "iterative")) {
      pred_vecchia_ng <- predict_ng(c(args_ng, list(gp_approx = "vecchia", num_neighbors = n_ng - 1,
                                                    vecchia_ordering = "none",
                                                    matrix_inversion_method = inv_method_ng)))
      tol_ng <- if (inv_method_ng == "iterative") 1E-2 else 1E-6
      expect_lt(sum(abs(pred_vecchia_ng$mu - pred_exact_ng$mu)), tol_ng)
      expect_lt(sum(abs(as.vector(pred_vecchia_ng$var) - as.vector(pred_exact_ng$var))), tol_ng)
    }
  })


  test_that("Predictions of a weighted model with grouped random effects and a Vecchia GP ", {

    # The error variance of observation i is 'cov_pars[1] / weights[i]'. With all neighbors the Vecchia
    # approximation is exact, so the predictions have to equal exact Gaussian conditioning with that
    # error covariance. Only the predictions are compared: the weighted negative log-likelihood of this
    # model class currently uses frequency weights instead of the precision weights that the Gaussian
    # likelihood uses everywhere else, see the test file of the weights
    n_w <- 60
    m_w <- 6
    group_w <- rep(1:m_w, each = n_w / m_w)
    coords_w <- cbind(sim_rand_unif(n = n_w, init_c = 0.312), sim_rand_unif(n = n_w, init_c = 0.573))
    D_w <- as.matrix(dist(coords_w))
    Z_w <- model.matrix(~ factor(group_w) - 1)
    cov_pars_w <- c(0.4, 0.7, 1.0, 0.2)# error variance, group variance, GP variance, GP range
    weights_w <- 0.5 + sim_rand_unif(n = n_w, init_c = 0.828)
    Sigma_w <- cov_pars_w[2] * (Z_w %*% t(Z_w)) + cov_pars_w[3] * exp(-D_w / cov_pars_w[4])
    Syy_w <- Sigma_w + diag(cov_pars_w[1] / weights_w)
    y_w <- as.vector(t(chol(Syy_w)) %*% qnorm(sim_rand_unif(n = n_w, init_c = 0.751)))
    np_w <- 4
    coords_pred_w <- cbind(c(0.12, 0.44, 0.71, 0.93), c(0.22, 0.51, 0.83, 0.07))
    group_pred_w <- c(1, 2, 3, 4)
    Dpo_w <- as.matrix(dist(rbind(coords_pred_w, coords_w)))[1:np_w, (np_w + 1):(np_w + n_w)]
    Dpp_w <- as.matrix(dist(coords_pred_w))
    Zp_w <- matrix(0, np_w, m_w)
    for (i in 1:np_w) Zp_w[i, group_pred_w[i]] <- 1
    Cpo_w <- cov_pars_w[3] * exp(-Dpo_w / cov_pars_w[4]) + cov_pars_w[2] * (Zp_w %*% t(Z_w))
    Cpp_w <- cov_pars_w[3] * exp(-Dpp_w / cov_pars_w[4]) + cov_pars_w[2] * (Zp_w %*% t(Zp_w))
    mu_exact_w <- as.vector(Cpo_w %*% solve(Syy_w, y_w))
    var_exact_w <- diag(Cpp_w - Cpo_w %*% solve(Syy_w, t(Cpo_w)))

    for (inv_method_w in c("cholesky", "iterative")) {
      capture.output( gp_model_w <- GPModel(gp_coords = coords_w, cov_function = "exponential",
                                            group_data = group_w, weights = weights_w,
                                            gp_approx = "vecchia", num_neighbors = n_w - 1,
                                            vecchia_ordering = "none", likelihood = "gaussian",
                                            matrix_inversion_method = inv_method_w), file = 'NUL')
      gp_model_w$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all",
                                     num_neighbors_pred = n_w + np_w, nsim_var_pred = 2000)
      gp_model_w$set_optim_params(params = list(init_aux_pars = cov_pars_w[1], seed_rand_vec_trace = 1,
                                                init_cov_pars = cov_pars_w[2:4],
                                                init_coef_aux_pars_from_iid_model = FALSE))
      capture.output( pred_w <- gp_model_w$predict(y = y_w, gp_coords_pred = coords_pred_w,
                                                   group_data_pred = group_pred_w,
                                                   cov_pars = cov_pars_w[2:4], predict_var = TRUE,
                                                   predict_response = FALSE), file = 'NUL')
      tol_w <- if (inv_method_w == "iterative") 1E-2 else 1E-6
      expect_lt(sum(abs(pred_w$mu - mu_exact_w)), tol_w)
      expect_lt(sum(abs(as.vector(pred_w$var) - var_exact_w)), tol_w)
    }
  })


  test_that("Grouped random coefficient without an intercept random effect and a Vecchia GP ", {

    # Dropping an intercept random effect changes the number of components and thus the index of the
    # Gaussian process component, which the Vecchia approximation uses to separate the grouped random
    # effects from the GP. With all neighbors the approximation is exact, so both the likelihood and the
    # predictions have to equal exact Gaussian conditioning
    n_di <- 60
    m_di <- 6
    group_di <- rep(1:m_di, each = n_di / m_di)
    coords_di <- cbind(sim_rand_unif(n = n_di, init_c = 0.312), sim_rand_unif(n = n_di, init_c = 0.573))
    x_di <- 0.5 + sim_rand_unif(n = n_di, init_c = 0.234)
    D_di <- as.matrix(dist(coords_di))
    Z_di <- model.matrix(~ factor(group_di) - 1)
    Zx_di <- diag(x_di) %*% Z_di
    cov_pars_di <- c(0.4, 0.5, 1.0, 0.2)# error variance, random slope variance, GP variance, GP range
    np_di <- 4
    coords_pred_di <- cbind(c(0.12, 0.44, 0.71, 0.93), c(0.22, 0.51, 0.83, 0.07))
    group_pred_di <- c(1, 2, 3, 4)
    x_pred_di <- c(1.2, 0.7, 1.1, 0.9)
    Dpo_di <- as.matrix(dist(rbind(coords_pred_di, coords_di)))[1:np_di, (np_di + 1):(np_di + n_di)]
    Dpp_di <- as.matrix(dist(coords_pred_di))
    Zp_di <- matrix(0, np_di, m_di)
    for (i in 1:np_di) Zp_di[i, group_pred_di[i]] <- 1
    Zxp_di <- diag(x_pred_di) %*% Zp_di

    # A single cluster, and several clusters with independent realizations of all components
    for (several_clusters_di in c(FALSE, TRUE)) {
      cluster_ids_di <- if (several_clusters_di) c(rep(1, 0.4 * n_di), rep(2, 0.6 * n_di)) else rep(1, n_di)
      cluster_pred_di <- if (several_clusters_di) c(1, 1, 2, 2) else rep(1, np_di)
      same_cl_di <- outer(cluster_ids_di, cluster_ids_di, "==") * 1
      same_cl_po_di <- outer(cluster_pred_di, cluster_ids_di, "==") * 1
      same_cl_pp_di <- outer(cluster_pred_di, cluster_pred_di, "==") * 1
      Sigma_di <- (cov_pars_di[2] * (Zx_di %*% t(Zx_di)) +
                     cov_pars_di[3] * exp(-D_di / cov_pars_di[4])) * same_cl_di
      Syy_di <- Sigma_di + diag(cov_pars_di[1], n_di)
      y_di <- as.vector(t(chol(Syy_di)) %*% qnorm(sim_rand_unif(n = n_di, init_c = 0.751)))
      nll_exact_di <- as.numeric(0.5 * (determinant(Syy_di, logarithm = TRUE)$modulus +
                                          t(y_di) %*% solve(Syy_di, y_di) + n_di * log(2 * pi)))
      Cpo_di <- (cov_pars_di[2] * (Zxp_di %*% t(Zx_di)) +
                   cov_pars_di[3] * exp(-Dpo_di / cov_pars_di[4])) * same_cl_po_di
      Cpp_di <- (cov_pars_di[2] * (Zxp_di %*% t(Zxp_di)) +
                   cov_pars_di[3] * exp(-Dpp_di / cov_pars_di[4])) * same_cl_pp_di
      mu_exact_di <- as.vector(Cpo_di %*% solve(Syy_di, y_di))
      var_exact_di <- diag(Cpp_di - Cpo_di %*% solve(Syy_di, t(Cpo_di)))

      for (inv_method_di in c("cholesky", "iterative")) {
        gp_model_di <- GPModel(group_data = group_di, group_rand_coef_data = x_di,
                               ind_effect_group_rand_coef = 1, drop_intercept_group_rand_effect = TRUE,
                               gp_coords = coords_di, cov_function = "exponential",
                               cluster_ids = cluster_ids_di, gp_approx = "vecchia",
                               num_neighbors = n_di - 1, vecchia_ordering = "none",
                               likelihood = "gaussian", matrix_inversion_method = inv_method_di)
        gp_model_di$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all",
                                        nsim_var_pred = 2000)
        gp_model_di$set_optim_params(params = list(init_aux_pars = cov_pars_di[1], seed_rand_vec_trace = 1,
                                                   num_rand_vec_trace = 1000,
                                                   init_cov_pars = cov_pars_di[2:4],
                                                   init_coef_aux_pars_from_iid_model = FALSE))
        capture.output( nll_di <- gp_model_di$neg_log_likelihood(cov_pars = cov_pars_di[2:4], y = y_di,
                                                                 aux_pars = cov_pars_di[1]), file = 'NUL')
        # The log determinant of the iterative methods is a stochastic estimate, its tolerance is far
        #   above the observed deviation of about 0.08 but far below what a misplaced component gives
        tol_di <- if (inv_method_di == "iterative") 0.5 else 1E-6
        expect_lt(abs(nll_di - nll_exact_di), tol_di)
        capture.output( pred_di <- gp_model_di$predict(y = y_di, gp_coords_pred = coords_pred_di,
                                                       group_data_pred = group_pred_di,
                                                       group_rand_coef_data_pred = x_pred_di,
                                                       cluster_ids_pred = cluster_pred_di,
                                                       cov_pars = cov_pars_di[2:4], predict_var = TRUE,
                                                       predict_response = FALSE), file = 'NUL')
        tol_pred_di <- if (inv_method_di == "iterative") 1E-2 else 1E-6
        expect_lt(sum(abs(pred_di$mu - mu_exact_di)), tol_pred_di)
        expect_lt(sum(abs(as.vector(pred_di$var) - var_exact_di)), tol_pred_di)
      }
    }
  })


  test_that("Predictive variances of training data grouped random coefficients with a Gaussian process ", {

    # The posterior covariance of the coefficients of a grouped random slope is mapped back to the
    # observations with the unweighted group incidence matrix. Using the slope-weighted one (which the
    # implementation did) multiplies the variance reduction with the squared covariate of the observation
    # and makes most of the variances negative
    y_rs <- as.vector(L %*% b_1) + as.vector(Z1 %*% b_gr_1) + as.vector(Z3 %*% b_gr_3) + xi
    init_cov_pars_rs <- c(rep(var(y_rs) / 4, 4), mean(dist(coords)) / 3)
    for (use_weights_rs in c(FALSE, TRUE)) {
      weights_rs <- if (use_weights_rs) 0.5 + sim_rand_unif(n = n, init_c = 0.828) else NULL
      error_var_scale_rs <- if (use_weights_rs) 1 / weights_rs else rep(1, n)
      capture.output( gp_model_rs <- fitGPModel(group_data = group, group_rand_coef_data = x,
                                                ind_effect_group_rand_coef = 1, gp_coords = coords,
                                                cov_function = "exponential", y = y_rs, weights = weights_rs,
                                                params = c(OPTIM_PARAMS_BFGS,
                                                           list(init_cov_pars = init_cov_pars_rs))), file = 'NUL')
      cov_pars_rs <- as.numeric(gp_model_rs$get_cov_pars())
      # Exact Gaussian conditioning: the training data random effect of an observation is the random
      #   coefficient of its group, not the random coefficient times the covariate
      Sigma_gp_rs <- cov_pars_rs[4] * exp(-D / cov_pars_rs[5])
      psi_rs <- cov_pars_rs[2] * (Z1 %*% t(Z1)) + cov_pars_rs[3] * (Z3 %*% t(Z3)) + Sigma_gp_rs +
        diag(cov_pars_rs[1] * error_var_scale_rs, n)
      psi_inv_y_rs <- solve(psi_rs, y_rs)
      post_var_int_rs <- cov_pars_rs[2] * diag(m) - cov_pars_rs[2]^2 * (t(Z1) %*% solve(psi_rs, Z1))
      post_var_slope_rs <- cov_pars_rs[3] * diag(m) - cov_pars_rs[3]^2 * (t(Z3) %*% solve(psi_rs, Z3))
      expected_rs <- cbind(as.vector(cov_pars_rs[2] * Z1 %*% (t(Z1) %*% psi_inv_y_rs)),
                           as.vector(cov_pars_rs[3] * Z1 %*% (t(Z3) %*% psi_inv_y_rs)),
                           as.vector(Sigma_gp_rs %*% psi_inv_y_rs),
                           diag(post_var_int_rs)[group],
                           diag(post_var_slope_rs)[group],
                           diag(Sigma_gp_rs - Sigma_gp_rs %*% solve(psi_rs, Sigma_gp_rs)))
      re_rs <- predict_training_data_random_effects(gp_model_rs, predict_var = TRUE)
      expect_equal(dim(re_rs), c(n, 6))
      expect_true(all(re_rs[, 4:6] > 0))
      expect_lt(max(abs(as.vector(re_rs) - as.vector(expected_rs))), TOLERANCE_MEDIUM)
    }
  })

  test_that("Grouped random coefficient without an intercept random effect and a Gaussian process ", {

    # Dropping an intercept random effect changes the number of components and thus the index of the
    # Gaussian process component. Both have to be determined once and not while constructing the
    # components of a cluster, which left the model in an inconsistent state
    y_di <- as.vector(L %*% b_1) + as.vector(Z3 %*% b_gr_3) + xi
    cov_pars_di <- c(0.2, 0.8, 1.1, 0.15)# error variance, random slope variance, GP variance, GP range
    Sigma_gp_di <- cov_pars_di[3] * exp(-D / cov_pars_di[4])
    psi_di <- cov_pars_di[2] * (Z3 %*% t(Z3)) + Sigma_gp_di + diag(cov_pars_di[1], n)
    nll_di_manual <- as.numeric(0.5 * (determinant(psi_di, logarithm = TRUE)$modulus +
                                         t(y_di) %*% solve(psi_di, y_di) + n * log(2 * pi)))
    gp_model_di <- GPModel(group_data = group, group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                           drop_intercept_group_rand_effect = TRUE, gp_coords = coords,
                           cov_function = "exponential")
    expect_lt(abs(gp_model_di$neg_log_likelihood(cov_pars = cov_pars_di, y = y_di) - nll_di_manual),
              TOLERANCE_MEDIUM)
    # Training data random effects: the random coefficient and the Gaussian process
    capture.output( gp_model_di <- fitGPModel(group_data = group, group_rand_coef_data = x,
                                              ind_effect_group_rand_coef = 1,
                                              drop_intercept_group_rand_effect = TRUE, gp_coords = coords,
                                              cov_function = "exponential", y = y_di,
                                              params = c(OPTIM_PARAMS_BFGS,
                                                         list(init_cov_pars = c(rep(var(y_di) / 3, 3),
                                                                                mean(dist(coords)) / 3)))), file = 'NUL')
    cov_pars_fit_di <- as.numeric(gp_model_di$get_cov_pars())
    expect_equal(length(cov_pars_fit_di), 4)
    expect_equal(names(gp_model_di$get_cov_pars()),
                 c("Error_var", "Group_1_rand_coef_nb_1", "GP_var", "GP_range"))
    Sigma_gp_fit_di <- cov_pars_fit_di[3] * exp(-D / cov_pars_fit_di[4])
    psi_fit_di <- cov_pars_fit_di[2] * (Z3 %*% t(Z3)) + Sigma_gp_fit_di + diag(cov_pars_fit_di[1], n)
    psi_inv_y_di <- solve(psi_fit_di, y_di)
    post_var_slope_di <- cov_pars_fit_di[2] * diag(m) - cov_pars_fit_di[2]^2 * (t(Z3) %*% solve(psi_fit_di, Z3))
    expected_di <- cbind(as.vector(cov_pars_fit_di[2] * Z1 %*% (t(Z3) %*% psi_inv_y_di)),
                         as.vector(Sigma_gp_fit_di %*% psi_inv_y_di),
                         diag(post_var_slope_di)[group],
                         diag(Sigma_gp_fit_di - Sigma_gp_fit_di %*% solve(psi_fit_di, Sigma_gp_fit_di)))
    re_di <- predict_training_data_random_effects(gp_model_di, predict_var = TRUE)
    expect_equal(dim(re_di), c(n, 4))
    expect_lt(max(abs(as.vector(re_di) - as.vector(expected_di))), TOLERANCE_MEDIUM)
    # Prediction for the observed locations and groups
    pred_di <- predict(gp_model_di, group_data_pred = group, group_rand_coef_data_pred = x,
                       gp_coords_pred = coords, predict_var = TRUE, predict_response = FALSE)
    prior_cov_di <- cov_pars_fit_di[2] * (Z3 %*% t(Z3)) + Sigma_gp_fit_di
    expect_lt(max(abs(pred_di$mu - as.vector(prior_cov_di %*% psi_inv_y_di))), TOLERANCE_MEDIUM)
    expect_lt(max(abs(as.vector(pred_di$var) -
                        diag(prior_cov_di - prior_cov_di %*% solve(psi_fit_di, prior_cov_di)))),
              TOLERANCE_MEDIUM)
  })

}
