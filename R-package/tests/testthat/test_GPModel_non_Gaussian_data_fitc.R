context("GPModel_non_Gaussian_data")

# Avoid being tested on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

  TOLERANCE_ITERATIVE <- 1e-1
  TOLERANCE_LOOSE <- 1E-2
  TOLERANCE_MEDIUM <- 1e-3
  TOLERANCE_STRICT_LOWER <- 1E-5
  TOLERANCE_STRICT <- 1E-6
  # Some of the optimization problems below are non-convex and/or use stochastic (iterative) methods.
  # A different compiler / standard library (e.g. clang + libc++ on Linux, which is used by the sanitizer
  # containers of R-hub and CRAN) can then converge to a DIFFERENT stationary point with practically the
  # same likelihood value (the negative log-likelihoods agree to ~0.1%, the coefficients differ by ~0.1).
  # The tight tolerances are therefore only required on the reference platform on which the expected
  # values below have been calculated.
  # See helper-tolerances.R, which defines this and reports it once per test run
  USE_STRICT_TOLERANCES <- gpb_use_strict_tolerances()
  TOLERANCE_NON_CONVEX <- if (USE_STRICT_TOLERANCES) TOLERANCE_MEDIUM else 0.5
  # Separate helper for the very strict tolerances (1e-6) of comparisons whose expected values cannot be
  # handed to 'relax_tolerance', so that its lower bound cannot be tied to their magnitude. Deviations of
  # a few 1e-6 occur under valgrind in particular, which does not reproduce floating point arithmetic
  # bit-wise (it rounds the 80 bit intermediate results of x87 to 64 bit and its libm differs)
  relax_tolerance_strict <- function(tol) if (USE_STRICT_TOLERANCES) tol else 100 * tol
  # Covariance functions with a general (non-fixed) smoothness need 'std::cyl_bessel_k', which is a C++17
  # feature that is not provided by every standard library (in particular not by libc++, which is used by
  # clang on macOS and in the clang sanitizer containers of R-hub / CRAN)
  SKIP_BESSEL_COV_TESTS <- !gpboost:::has_std_cyl_bessel_k() &&
    Sys.getenv("GPBOOST_RUN_BESSEL_COV_TESTS") != "true"

  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov = "gradient_descent", optimizer_coef = "gradient_descent",
                               use_nesterov_acc = TRUE, lr_cov=0.1, lr_coef = 0.1, maxit = 1000,
                               acc_rate_cov = 0.5, init_coef_aux_pars_from_iid_model = FALSE)
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

  # Simulate data
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
  Z_SVC <- matrix(sim_rand_unif(n=n*2, init_c=0.6), ncol=2) # covariate data for random coefficients
  colnames(Z_SVC) <- c("var1","var2")
  b_2 <- qnorm(sim_rand_unif(n=n, init_c=0.17))
  b_3 <- qnorm(sim_rand_unif(n=n, init_c=0.42))
  # First grouped random effects model
  m <- 10 # number of categories / levels for grouping variable
  group <- rep(1,n) # grouping variable
  for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
  Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
  b_gr_1 <- qnorm(sim_rand_unif(n=m, init_c=0.565))
  # Second grouped random effect
  n_obs_gr <- n/m # number of samples per group
  group2 <- rep(1,n) # grouping variable
  for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
  Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
  b_gr_2 <- qnorm(sim_rand_unif(n=n_obs_gr, init_c=0.36))
  # Grouped random slope / coefficient
  x <- cos((1:n-n/2)^2*5.5*pi/n) # covariate data for random slope
  Z3 <- diag(x) %*% Z1
  b_gr_3 <- qnorm(sim_rand_unif(n=m, init_c=0.5678))
  # Data for linear mixed effects model
  X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
  beta <- c(0.1,2) # regression coefficients
  # cluster_ids
  cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))
  # GP with multiple observations at the same locations
  coords_multiple <- matrix(sim_rand_unif(n=n*d/4, init_c=0.1), ncol=d)
  coords_multiple <- rbind(coords_multiple,coords_multiple,coords_multiple,coords_multiple)
  D_multiple <- as.matrix(dist(coords_multiple))
  Sigma_multiple <- sigma2_1*exp(-D_multiple/rho)+diag(1E-10,n)
  L_multiple <- t(chol(Sigma_multiple))
  b_multiple <- qnorm(sim_rand_unif(n=n, init_c=0.8))
  # Space-time GP
  time <- (1:n)/n
  rho_time <- 0.1
  coords_ST_scaled <- cbind(time/rho_time, coords/rho)
  D_ST <- as.matrix(dist(coords_ST_scaled))
  Sigma_ST <- sigma2_1 * exp(-D_ST) + diag(1E-20,n)
  C_ST <- t(chol(Sigma_ST))
  b_ST <- qnorm(sim_rand_unif(n=n, init_c=0.86574))
  eps_ST <- as.vector(C_ST %*% b_ST)
  # For CV
  params_cv <- list(learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
                    feature_pre_filter = FALSE, seed = 1, deterministic = TRUE)
  folds <- list()
  nf <- 2
  for(i in 1:nf) folds[[i]] <- as.integer(((1:(n/nf)) -1) * nf + i)

  test_that("FITC for binary classification", {

    probs <- pnorm(L %*% b_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.199) < probs)
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    coord_test_v1 <- rbind(c(0.11,0.45),coords[1:2,])
    coord_test_multiple <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    cov_pars_ll <- c(1, 0.2)
    cov_pars_pred = c(1, 0.2)
    y_multiple <- as.numeric(sim_rand_unif(n=n, init_c=0.6779) < pnorm(L_multiple %*% b_multiple + X%*%beta))
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- c(1,mean(dist(coords))/3)
    params_mult = DEFAULT_OPTIM_PARAMS
    params_mult$init_cov_pars <- c(1,mean(dist(unique(coords_multiple)))/3)
    cluster_ids_ip <- c(rep(1,n/2),rep(2,n/2))
    cluster_ids_pred <- c(1,2,2)
    cluster_ids_pred_new <- c(1,2,99)
    X_test_clus <- cbind(rep(0,3),rep(0,3))

    # Cannot have more inducing points than samples
    expect_error( fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                             y = y, X=X, params = params, gp_approx = "fitc",
                             num_ind_points = n+1, ind_points_selection = "random") )

    ## Evaluate log-likelihood
    gp_model_no_approx <- GPModel(gp_coords = coords, cov_function = "exponential",
                                  likelihood = "bernoulli_probit")
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        likelihood = "bernoulli_probit", gp_approx = "fitc",
                        num_ind_points = n, ind_points_selection = "random")
    expect_lt(abs(gp_model$neg_log_likelihood(y = y, cov_pars = cov_pars_ll) -
                    gp_model_no_approx$neg_log_likelihood(y = y, cov_pars = cov_pars_ll)),TOLERANCE_STRICT)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        likelihood = "bernoulli_probit", gp_approx = "fitc",
                                        num_ind_points  = 50, ind_points_selection = "kmeans++") , file='NUL')
    nll2 <- 63.19375632
    expect_lt(abs(gp_model$neg_log_likelihood(y = y, cov_pars = cov_pars_ll) - nll2),TOLERANCE_STRICT)

    # Estimation without approximation
    gp_model_no_approx <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                     y = y, X=X, params = params)
    nll_exp <- gp_model_no_approx$get_current_neg_log_likelihood() + 0.
    # Nelder-Mead
    params_NM <- params
    params_NM$optimizer_coef <- params_NM$optimizer_cov <- "nelder_mead"
    gp_model_NM_no_approx <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                        y = y, X=X, params = params_NM)
    nll_NM_exp <- gp_model_NM_no_approx$get_current_neg_log_likelihood() + 0.
    # Prediction
    pred_var_no_approx <- predict(gp_model_no_approx, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                                  predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    pred_cov_no_approx <- predict(gp_model_no_approx, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                                  predict_cov = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    pred_resp_no_approx <- predict(gp_model_no_approx, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                                   predict_var = FALSE, predict_response = TRUE, cov_pars = cov_pars_pred)
    X0 <- matrix(0, nrow=nrow(X), ncol=ncol(X))
    pred_train_no_approx <- predict(gp_model_no_approx, y=y, gp_coords_pred = coords, X_pred = X0,
                                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    # duplicate coordinates
    gp_model_mult_no_approx <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", likelihood = "bernoulli_probit",
                                          y = y_multiple, X=X, params = params_mult)
    nll_mult_exp <- gp_model_mult_no_approx$get_current_neg_log_likelihood() + 0.
    pred_mult_no_approx <- predict(gp_model_mult_no_approx, gp_coords_pred = coord_test_multiple, X_pred = X_test,
                                   predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    # cluster_ids
    gp_model_clus_no_approx <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                          cluster_ids = cluster_ids_ip, y = y, X=X, params = params)
    nll_clus_exp <- gp_model_clus_no_approx$get_current_neg_log_likelihood() + 0.
    pred_clus_no_approx <- predict(gp_model_clus_no_approx, y=y, gp_coords_pred = coord_test_v1,
                                   X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred,
                                   predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    pred_clus_no_approx_new <- predict(gp_model_clus_no_approx, y=y, gp_coords_pred = coord_test_v1,
                                       X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred_new,
                                       predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)

    # Fitc and large num_ind_points
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = params, gp_approx = "fitc",
                           num_ind_points = n, ind_points_selection = "random")
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_no_approx$get_cov_pars(std_err = FALSE)))),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_no_approx$get_coef(std_err = FALSE)))),TOLERANCE_STRICT_LOWER)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),TOLERANCE_STRICT_LOWER)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_var_no_approx$mu)),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_approx$var))),TOLERANCE_STRICT_LOWER)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                    predict_cov = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_cov_no_approx$mu)),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_cov_no_approx$cov))),TOLERANCE_STRICT_LOWER)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                    predict_var = FALSE, predict_response = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_resp_no_approx$mu)),TOLERANCE_STRICT)
    # Predict training data
    pred_train_fitc <- predict(gp_model, y=y, gp_coords_pred = coords, X_pred = X0,
                               predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred_train_no_approx$mu - pred_train_fitc$mu)), TOLERANCE_LOOSE)
    expect_lt(sum(abs(pred_train_no_approx$var - pred_train_fitc$var)), TOLERANCE_LOOSE)
    # With duplicate locations
    gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y_multiple, X=X, params = params_mult, gp_approx = "fitc",
                           num_ind_points = dim(unique(coords_multiple))[1], ind_points_selection = "random")
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_mult_no_approx$get_cov_pars(std_err = FALSE)))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_mult_no_approx$get_coef(std_err = FALSE)))),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), gp_model_mult_no_approx$get_num_optim_iter())
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_mult_exp),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test_multiple, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_mult_no_approx$mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_mult_no_approx$var))),TOLERANCE_STRICT_LOWER)
    # cluster_ids
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = params, gp_approx = "fitc", cluster_ids = cluster_ids_ip,
                           num_ind_points = n/2, ind_points_selection = "random")
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_clus_no_approx$get_cov_pars(std_err = FALSE)))),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_clus_no_approx$get_coef(std_err = FALSE)))),TOLERANCE_STRICT_LOWER)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_clus_exp),TOLERANCE_STRICT_LOWER)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1,
                    X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_clus_no_approx$mu)),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_clus_no_approx$var))),TOLERANCE_STRICT_LOWER)
    # Prediction for a new cluster: there is no observed data, hence the prior is used, and the inducing
    # points are determined from the prediction locations of this cluster. The predictive variances differ
    # from the exact ones by the jitter that is added to the diagonal of the inducing point matrix
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1,
                    X_pred = X_test_clus, cluster_ids_pred = cluster_ids_pred_new,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_clus_no_approx_new$mu)),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_clus_no_approx_new$var))),TOLERANCE_STRICT_LOWER)

    # Fitc and smaller num_ind_points
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           y = y, X=X, params = params, gp_approx = "fitc",
                                           num_ind_points = 50, ind_points_selection = "kmeans++") , file='NUL')
    cov_pars_2 <- c(1.67443064, 0.23625418)
    coefs_2 <- c(0.28845306, 1.62607540)
    nll_2 <- 48.11931906
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_2)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_2)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_2), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_no_approx$get_cov_pars(std_err = FALSE)))),1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_no_approx$get_coef(std_err = FALSE)))),0.5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_exp),0.5)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    mu_exp <- c(0.16002215, 1.49343290, 1.37457340)
    cov_exp <- c(0.80926655, 0.28066810, -0.00027551, 0.28066810, 0.65993096, -0.00015671, -0.00027551, -0.00015671, 0.39237780)
    expect_lt(sum(abs(pred$mu - mu_exp)),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(as.vector(pred$var) - cov_exp[c(1,5,9)])),TOLERANCE_STRICT_LOWER)
    expect_lt(sum(abs(pred$mu - pred_var_no_approx$mu)),0.5)
    expect_lt(sum(abs(as.vector(pred$var) - as.vector(pred_var_no_approx$var))),0.5)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test_v1, X_pred = X_test,
                    predict_cov = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(as.vector(pred$cov) - as.vector(pred_cov_no_approx$cov))),0.5)
    expect_lt(sum(abs(as.vector(pred$cov) - cov_exp)),TOLERANCE_STRICT_LOWER)
    # Predict training data
    pred_train_fitc <- predict(gp_model, y=y, gp_coords_pred = coords, X_pred = X0,
                               predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred_train_no_approx$mu - pred_train_fitc$mu)), 7)
    expect_lt(sum(abs(pred_train_no_approx$var - pred_train_fitc$var)), 5)
    # With duplicate locations
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           y = y_multiple, X=X, params = params_mult, gp_approx = "fitc",
                                           num_ind_points = 12, ind_points_selection = "kmeans++")  , file='NUL')
    cov_pars_2 <- c(4.58741750, 0.07401628)
    coefs_2 <- c(1.47775348, 4.19935782)
    nll_2 <- 31.49300336
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_2)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_2)),TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_2), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_mult_no_approx$get_cov_pars(std_err = FALSE)))),1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_mult_no_approx$get_coef(std_err = FALSE)))),0.5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_mult_exp),0.2)

    # Nelder-Mead for fitc and large num_ind_points
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = params_NM, gp_approx = "fitc",
                           num_ind_points = n, ind_points_selection = "random")
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_NM_no_approx$get_cov_pars(std_err = FALSE)))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - as.vector(gp_model_NM_no_approx$get_coef(std_err = FALSE)))),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_NM_exp),TOLERANCE_STRICT_LOWER)
    # Nelder-Mead for fitc and smaller num_ind_points
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           y = y, X=X, params = params_NM, gp_approx = "fitc",
                                           num_ind_points = 50, ind_points_selection = "kmeans++") , file='NUL')
    cov_pars_NM2 <- c(1.6426189413, 0.2444053821)
    coefs_NM2 <- c(0.249596402, 1.609043132)
    nll_NM2 <- 48.11741695
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_NM2)), TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_NM2)), TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll_NM2), TOLERANCE_STRICT_LOWER)

  })

}
