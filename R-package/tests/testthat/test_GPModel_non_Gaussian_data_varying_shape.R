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

  test_that("gamma_varying_shape likelihood for linear, GP and GPBoost models ", {

    n_vs <- 100
    group_vs <- rep(1:10, each = 10)
    X_vs <- cbind(rep(1, n_vs), sim_rand_unif(n = n_vs, init_c = 0.415))
    beta_mean_vs <- c(0.4, 0.8)
    beta_shape_vs <- c(0.9, -0.7)
    gr_var_vs <- 0.5
    b_gr_vs <- qnorm(sim_rand_unif(n = 10, init_c = 0.628))
    eta_true_vs <- as.vector(X_vs %*% beta_mean_vs) + sqrt(gr_var_vs) * b_gr_vs[group_vs]
    log_shape_true_vs <- as.vector(X_vs %*% beta_shape_vs)
    # Gamma draws via the inverse cdf of the uniform LCG stream (no R RNG in the tests)
    y_vs <- qgamma(sim_rand_unif(n = n_vs, init_c = 0.537), shape = exp(log_shape_true_vs),
                   rate = exp(log_shape_true_vs) / exp(eta_true_vs))
    X_test_vs <- cbind(rep(1, 3), c(0.1, 0.4, 0.8))
    group_test_vs <- c(1, 3, 11)
    X_zero_vs <- matrix(0, nrow = n_vs, ncol = ncol(X_vs))

    # Likelihood evaluated at given (not estimated) parameters: a pure formula check, independent of any optimizer
    fixed_effects_given_vs <- as.vector(cbind(X_vs %*% c(0.2, 0.5), X_vs %*% c(0.6, -0.4)))
    nll_given_vs <- GPModel(group_data = group_vs, likelihood = "gamma_varying_shape")$neg_log_likelihood(
      cov_pars = 0.3, y = y_vs, fixed_effects = fixed_effects_given_vs)
    expect_lt(abs(nll_given_vs - 206.80227775), TOLERANCE_MEDIUM)

    # A fixed-effects-only shape requires a fixed effects term (covariates and / or GPBoost boosting):
    # without any covariates and without the GPBoost algorithm, fitting should raise an informative error
    expect_error(capture.output(fitGPModel(group_data = group_vs, likelihood = "gamma_varying_shape", y = y_vs,
                                           params = list(maxit = 2, init_coef_aux_pars_from_iid_model = FALSE)),
                                file = "NUL"))

    ###################
    ## Linear regression model (mean has a grouped random effect, shape is fixed-effects only)
    ###################
    capture.output(gp_model_vs <- fitGPModel(group_data = group_vs, likelihood = "gamma_varying_shape",
                                             y = y_vs, X = X_vs, params = OPTIM_PARAMS_BFGS), file = "NUL")
    coef_vs <- as.vector(gp_model_vs$get_coef(std_err = FALSE))
    expect_equal(length(coef_vs), 4L)
    coef_vs_std_err <- gp_model_vs$get_coef(std_err = TRUE)
    expect_equal(dim(coef_vs_std_err), c(2L, 4L))
    # The coefficients of the log-shape block are named with the suffix "_shape"
    expect_equal(colnames(coef_vs_std_err), c("Covariate_1", "Covariate_2", "Covariate_1_shape", "Covariate_2_shape"))
    # Note: std. errs. must be strictly positive; a plain is.finite() check would not catch a regression where the
    # shape block's std. errs. are silently left at their R-side zero-initialized default (0 is finite)
    expect_true(all(coef_vs_std_err["Std. err.", ] > 0))
    expected_coef_vs <- c(0.77924412, 0.46142710, 1.23460587, -1.05625429)
    expect_lt(sum(abs(coef_vs - expected_coef_vs)), TOLERANCE_MEDIUM)
    expected_coef_vs_std_err <- c(0.26941702, 0.27624782, 0.31016743, 0.56209832)
    expect_lt(sum(abs(as.vector(coef_vs_std_err["Std. err.", ]) - expected_coef_vs_std_err)), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_vs$get_cov_pars(std_err = FALSE)) - 0.55387036), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_vs$get_current_neg_log_likelihood() - 198.77618711), TOLERANCE_MEDIUM)
    expect_equal(gp_model_vs$get_num_aux_pars(), 0L)
    # Prediction: response mean and variance
    pred_vs <- predict(gp_model_vs, y = y_vs, group_data_pred = group_test_vs, X_pred = X_test_vs,
                       predict_var = TRUE, predict_response = TRUE)
    expected_mu_vs <- c(1.79642869, 2.48366142, 4.15919264)
    expected_var_vs <- c(1.25330792, 3.09395219, 33.18798313)
    expect_lt(sum(abs(pred_vs$mu - expected_mu_vs)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_vs$var - expected_var_vs)), TOLERANCE_LOOSE)
    re_pred_train_vs <- predict_training_data_random_effects(gp_model_vs)
    expected_re_pred_train_vs <- c(-0.26356084, -0.02855334, -0.07365381, 0.16825595, -0.49747275,
                                   0.95312724, 0.52931772, 0.56368985, 0.24315553, -1.79218732)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_vs[, 1])) - expected_re_pred_train_vs)), TOLERANCE_MEDIUM)
    re_pred_train_vs_var <- predict_training_data_random_effects(gp_model_vs, predict_var = TRUE)
    expected_re_pred_train_vs_var <- c(0.04794928, 0.04015937, 0.03914541, 0.04358211, 0.04611738,
                                       0.04431024, 0.03587677, 0.04670429, 0.03685070, 0.04917412)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_vs_var[, 2])) - expected_re_pred_train_vs_var)), TOLERANCE_MEDIUM)
    pred_train_re_vs <- predict(gp_model_vs, y = y_vs, group_data_pred = group_vs, X_pred = X_zero_vs,
                                predict_response = FALSE, predict_var = FALSE)
    expect_lt(sum(abs(as.vector(re_pred_train_vs[, 1]) - pred_train_re_vs$mu)), TOLERANCE_STRICT)
    # Predicting requires covariate data for the model's linear predictors (mean and log-shape)
    expect_error(predict(gp_model_vs, y = y_vs, group_data_pred = group_test_vs,
                         predict_var = TRUE, predict_response = TRUE))

    ###################
    ## No random effects at all (iid model, pure linear regression for the mean and the log-shape)
    ###################
    capture.output(gp_model_vs_iid <- fitGPModel(likelihood = "gamma_varying_shape", y = y_vs, X = X_vs,
                                                 params = OPTIM_PARAMS_BFGS), file = "NUL")
    expected_coef_vs_iid <- c(0.96393460, 0.48554127, 0.42234138, -0.47047123)
    expect_lt(sum(abs(as.vector(gp_model_vs_iid$get_coef(std_err = FALSE)) - expected_coef_vs_iid)), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_vs_iid$get_current_neg_log_likelihood() - 217.80011791), TOLERANCE_MEDIUM)

    ###################
    ## Equivalence with the constant-shape "gamma" likelihood for an intercept-only design matrix
    ## With X = intercept the log-shape predictor is constant, so the model is exactly "gamma" with an estimated shape
    ###################
    X_int_vs <- X_vs[, 1, drop = FALSE]
    capture.output(gp_model_const <- fitGPModel(group_data = group_vs, likelihood = "gamma", y = y_vs,
                                                X = X_int_vs, params = OPTIM_PARAMS_BFGS), file = "NUL")
    capture.output(gp_model_vary <- fitGPModel(group_data = group_vs, likelihood = "gamma_varying_shape", y = y_vs,
                                               X = X_int_vs, params = OPTIM_PARAMS_BFGS), file = "NUL")
    expect_lt(abs(gp_model_const$get_current_neg_log_likelihood() - 201.56477918), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_vary$get_current_neg_log_likelihood() -
                    gp_model_const$get_current_neg_log_likelihood()), TOLERANCE_STRICT_LOWER)
    coef_vary <- as.vector(gp_model_vary$get_coef(std_err = FALSE))
    expect_lt(abs(coef_vary[1] - as.vector(gp_model_const$get_coef(std_err = FALSE))[1]), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(coef_vary[2] - log(as.vector(gp_model_const$get_aux_pars()))), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(as.vector(gp_model_vary$get_cov_pars(std_err = FALSE)) -
                    as.vector(gp_model_const$get_cov_pars(std_err = FALSE))), TOLERANCE_STRICT_LOWER)

    ###################
    ## GPBoost algorithm (tree-boosting): mean via a grouped random effect + trees, log-shape via a second tree ensemble
    ###################
    gp_model_vs_boost <- GPModel(group_data = group_vs, likelihood = "gamma_varying_shape")
    gp_model_vs_boost$set_optim_params(params = OPTIM_PARAMS_BFGS)
    dtrain_vs <- gpb.Dataset(data = X_vs[, 2, drop = FALSE], label = y_vs)
    bst_vs <- gpb.train(data = dtrain_vs, gp_model = gp_model_vs_boost, nrounds = 20, learning_rate = 0.05,
                        max_depth = 2, min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    pred_vs_boost <- predict(bst_vs, data = X_vs[1:3, 2, drop = FALSE], group_data_pred = group_test_vs,
                             predict_var = TRUE, pred_latent = FALSE)
    expect_lt(abs(as.vector(gp_model_vs_boost$get_cov_pars(std_err = FALSE)) - 0.50240259), TOLERANCE_MEDIUM)
    expected_response_mean_vs_boost <- c(2.14032953, 2.89035370, 3.88305497)
    expected_response_var_vs_boost <- c(2.51382547, 5.94426437, 21.68702320)
    expect_lt(sum(abs(pred_vs_boost$response_mean - expected_response_mean_vs_boost)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_vs_boost$response_var - expected_response_var_vs_boost)), TOLERANCE_LOOSE)
    re_pred_train_vs_boost <- predict_training_data_random_effects(bst_vs)
    expected_re_pred_train_vs_boost <- c(-0.33267371, -0.06968062, -0.09469346, 0.06752114, -0.42291020,
                                         0.92782963, 0.41615823, 0.48271165, 0.24984357, -1.72663065)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_vs_boost[, 1])) - expected_re_pred_train_vs_boost)), TOLERANCE_MEDIUM)

    ###################
    ## GPs
    ###################
    n_vs2 <- 100
    X_vs2 <- cbind(rep(1, n_vs2), sim_rand_unif(n = n_vs2, init_c = 0.193))
    coords_vs2 <- matrix(sim_rand_unif(n = n_vs2 * 2, init_c = 0.749), ncol = 2)
    D_vs2 <- as.matrix(dist(coords_vs2))
    Sigma_vs2 <- 0.5 * exp(-D_vs2 / 0.15) + diag(1E-10, n_vs2)
    b_gp_vs2 <- as.vector(t(chol(Sigma_vs2)) %*% qnorm(sim_rand_unif(n = n_vs2, init_c = 0.836)))
    eta_true_vs2 <- as.vector(X_vs2 %*% beta_mean_vs) + b_gp_vs2
    log_shape_true_vs2 <- as.vector(X_vs2 %*% beta_shape_vs)
    y_vs2 <- qgamma(sim_rand_unif(n = n_vs2, init_c = 0.582), shape = exp(log_shape_true_vs2),
                    rate = exp(log_shape_true_vs2) / exp(eta_true_vs2))
    optim_params_vs2 <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 300,
                             init_coef_aux_pars_from_iid_model = FALSE)
    optim_params_vs2_iter <- c(optim_params_vs2, list(seed_rand_vec_trace = 1))

    # Dense GP ("Stable")
    nll_given_gp_vs <- GPModel(gp_coords = coords_vs2, cov_function = "exponential",
                               likelihood = "gamma_varying_shape")$neg_log_likelihood(
      cov_pars = c(1, mean(dist(coords_vs2)) / 3), y = y_vs2, fixed_effects = rep(0, 2 * n_vs2))
    expect_lt(abs(nll_given_gp_vs - 213.11732908), TOLERANCE_MEDIUM)
    capture.output(gp_model_gp_vs <- fitGPModel(gp_coords = coords_vs2, cov_function = "exponential",
                                                likelihood = "gamma_varying_shape", y = y_vs2, X = X_vs2,
                                                params = optim_params_vs2), file = "NUL")
    expected_coef_gp_vs <- c(0.55506594, 0.63362836, 1.53277451, -1.44199544)
    expect_lt(sum(abs(as.vector(gp_model_gp_vs$get_coef(std_err = FALSE)) - expected_coef_gp_vs)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_gp_vs$get_cov_pars(std_err = FALSE)) - c(0.31334850, 0.11748570))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_gp_vs$get_current_neg_log_likelihood() - 191.32567865), TOLERANCE_MEDIUM)
    coord_test_gp_vs <- coords_vs2[1:3, , drop = FALSE] + 1e-3
    pred_gp_vs <- predict(gp_model_gp_vs, y = y_vs2, gp_coords_pred = coord_test_gp_vs,
                          X_pred = X_vs2[1:3, , drop = FALSE], predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_gp_vs$mu - c(2.77793063, 2.08568765, 1.92724362))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_gp_vs$var - c(3.46163484, 1.59458276, 2.23585839))), TOLERANCE_LOOSE)

    # GP with a Vecchia approximation: with num_neighbors = n - 1 this is exact and must match the dense GP fit
    capture.output(gp_model_vecchia_vs <- fitGPModel(gp_coords = coords_vs2, cov_function = "exponential",
                                                     likelihood = "gamma_varying_shape", gp_approx = "vecchia",
                                                     num_neighbors = n_vs2 - 1, vecchia_ordering = "none",
                                                     matrix_inversion_method = "cholesky",
                                                     y = y_vs2, X = X_vs2, params = optim_params_vs2), file = "NUL")
    expect_lt(sum(abs(as.vector(gp_model_vecchia_vs$get_coef(std_err = FALSE)) - expected_coef_gp_vs)), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_vecchia_vs$get_current_neg_log_likelihood() - 191.32567865), TOLERANCE_MEDIUM)
    # matrix_inversion_method = "iterative"
    capture.output(gp_model_vecchia_vs_iter <- fitGPModel(gp_coords = coords_vs2, cov_function = "exponential",
                                                          likelihood = "gamma_varying_shape", gp_approx = "vecchia",
                                                          num_neighbors = n_vs2 - 1, vecchia_ordering = "none",
                                                          matrix_inversion_method = "iterative",
                                                          y = y_vs2, X = X_vs2, params = optim_params_vs2_iter), file = "NUL")
    expected_coef_vecchia_vs_iter <- c(0.54332885, 0.65141197, 1.60868319, -1.54241987)
    expect_lt(sum(abs(as.vector(gp_model_vecchia_vs_iter$get_coef(std_err = FALSE)) - expected_coef_vecchia_vs_iter)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(as.vector(gp_model_vecchia_vs_iter$get_cov_pars(std_err = FALSE)) - c(0.32828906, 0.11355682))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_vecchia_vs_iter$get_current_neg_log_likelihood() - 191.12958279), TOLERANCE_NON_CONVEX)

    # GP with an FITC approximation
    capture.output(gp_model_fitc_vs <- fitGPModel(gp_coords = coords_vs2, cov_function = "exponential",
                                                  likelihood = "gamma_varying_shape", gp_approx = "fitc",
                                                  num_ind_points = 50, y = y_vs2, X = X_vs2,
                                                  params = optim_params_vs2), file = "NUL")
    expected_coef_fitc_vs <- c(0.55047297, 0.63864813, 1.53279989, -1.45984724)
    expect_lt(sum(abs(as.vector(gp_model_fitc_vs$get_coef(std_err = FALSE)) - expected_coef_fitc_vs)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_fitc_vs$get_cov_pars(std_err = FALSE)) - c(0.30514950, 0.11908394))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_fitc_vs$get_current_neg_log_likelihood() - 191.59666396), TOLERANCE_MEDIUM)
  })

  test_that("prediction for new clusters for likelihoods with several location parameter blocks ", {

    # For a cluster without any observed data, the predictive distribution of the latent random effects / GPs
    # is the prior. The expected values below are thus obtained analytically from the prior and the offsets
    n_nc <- 40
    cluster_ids_nc <- rep(c(1, 2), each = n_nc / 2)
    group_nc <- rep(1:8, each = 5)
    y_nc <- qgamma(sim_rand_unif(n = n_nc, init_c = 0.213), shape = 2, rate = 2)
    group_pred_nc <- c(1, 6, 20)
    cluster_ids_pred_nc <- c(1, 2, 3)# cluster 3 has not been observed
    var_nc <- 0.5# marginal variance of the grouped random effect
    eta_nc <- c(0.3, -0.2, 0.7)# offset for the first location parameter block (the mean)
    zeta_nc <- c(0.4, 0.1, -0.3)# offset for the second block

    ###################
    ## Two fixed effects blocks and one set of random effects ('gamma_varying_shape': the log-shape is a second,
    ## fixed-effects-only location parameter block)
    ###################
    gp_model_nc <- GPModel(group_data = group_nc, cluster_ids = cluster_ids_nc, likelihood = "gamma_varying_shape")
    pred_nc <- predict(gp_model_nc, y = y_nc, cov_pars = var_nc, offset = rep(0, 2 * n_nc),
                       group_data_pred = group_pred_nc, cluster_ids_pred = cluster_ids_pred_nc,
                       offset_pred = c(eta_nc, zeta_nc), predict_var = TRUE, predict_response = FALSE)
    expect_lt(abs(pred_nc$mu[3] - eta_nc[3]), TOLERANCE_STRICT)
    expect_lt(abs(pred_nc$var[3] - var_nc), TOLERANCE_STRICT)
    # Predictions for the observed clusters must not be affected by the presence of a new cluster
    pred_nc_obs <- predict(gp_model_nc, y = y_nc, cov_pars = var_nc, offset = rep(0, 2 * n_nc),
                           group_data_pred = group_pred_nc[1:2], cluster_ids_pred = cluster_ids_pred_nc[1:2],
                           offset_pred = c(eta_nc[1:2], zeta_nc[1:2]), predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_nc$mu[1:2] - pred_nc_obs$mu)), TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_nc$var[1:2] - pred_nc_obs$var)), TOLERANCE_STRICT)
    # Response prediction: E(Y) = exp(m + v / 2) and Var(Y) = exp(-zeta) * exp(2m + 2v) + exp(2m + v) * (exp(v) - 1)
    pred_nc_resp <- predict(gp_model_nc, y = y_nc, cov_pars = var_nc, offset = rep(0, 2 * n_nc),
                            group_data_pred = group_pred_nc, cluster_ids_pred = cluster_ids_pred_nc,
                            offset_pred = c(eta_nc, zeta_nc), predict_var = TRUE, predict_response = TRUE)
    m_nc <- eta_nc[3]
    expect_lt(abs(pred_nc_resp$mu[3] - exp(m_nc + var_nc / 2)), TOLERANCE_STRICT)
    expect_lt(abs(pred_nc_resp$var[3] - (exp(-zeta_nc[3]) * exp(2 * m_nc + 2 * var_nc) +
                                           exp(2 * m_nc + var_nc) * expm1(var_nc))), TOLERANCE_STRICT)
    # Equivalence with the constant-shape "gamma" likelihood: with a constant log-shape offset, the two models
    # have the same response prediction for the new cluster ("gamma" has only one location parameter block)
    shape_nc <- 1.7
    gp_model_nc_const <- GPModel(group_data = group_nc, cluster_ids = cluster_ids_nc, likelihood = "gamma")
    gp_model_nc_const$set_optim_params(params = list(init_aux_pars = shape_nc))
    pred_const_nc <- predict(gp_model_nc_const, y = y_nc, cov_pars = var_nc, offset = rep(0, n_nc),
                             group_data_pred = group_pred_nc, cluster_ids_pred = cluster_ids_pred_nc,
                             offset_pred = eta_nc, predict_var = TRUE, predict_response = TRUE)
    pred_vary_nc <- predict(gp_model_nc, y = y_nc, cov_pars = var_nc, offset = rep(0, 2 * n_nc),
                            group_data_pred = group_pred_nc, cluster_ids_pred = cluster_ids_pred_nc,
                            offset_pred = c(eta_nc, rep(log(shape_nc), 3)), predict_var = TRUE, predict_response = TRUE)
    expect_lt(abs(pred_vary_nc$mu[3] - pred_const_nc$mu[3]), TOLERANCE_STRICT)
    expect_lt(abs(pred_vary_nc$var[3] - pred_const_nc$var[3]), TOLERANCE_STRICT)

    ###################
    ## Three fixed effects blocks ('hurdle_regression_gamma_varying_shape': mean, structural-zero predictor, log-shape)
    ###################
    y_nc_hurdle <- y_nc
    y_nc_hurdle[c(2, 7, 13, 24, 33)] <- 0
    xi_nc <- c(-0.5, 0.2, 0.6)# offset for the third block (the log-shape)
    gp_model_nc3 <- GPModel(group_data = group_nc, cluster_ids = cluster_ids_nc,
                            likelihood = "hurdle_regression_gamma_varying_shape")
    pred_nc3 <- predict(gp_model_nc3, y = y_nc_hurdle, cov_pars = var_nc, offset = rep(0, 3 * n_nc),
                        group_data_pred = group_pred_nc, cluster_ids_pred = cluster_ids_pred_nc,
                        offset_pred = c(eta_nc, zeta_nc, xi_nc), predict_var = TRUE, predict_response = FALSE)
    expect_lt(abs(pred_nc3$mu[3] - eta_nc[3]), TOLERANCE_STRICT)
    expect_lt(abs(pred_nc3$var[3] - var_nc), TOLERANCE_STRICT)
    pred_nc3_resp <- predict(gp_model_nc3, y = y_nc_hurdle, cov_pars = var_nc, offset = rep(0, 3 * n_nc),
                             group_data_pred = group_pred_nc, cluster_ids_pred = cluster_ids_pred_nc,
                             offset_pred = c(eta_nc, zeta_nc, xi_nc), predict_var = TRUE, predict_response = TRUE)
    q_nc <- 1 / (1 + exp(zeta_nc[3]))# probability of a non-zero response
    expect_lt(abs(pred_nc3_resp$mu[3] - q_nc * exp(m_nc + var_nc / 2)), TOLERANCE_STRICT)
    expect_lt(abs(pred_nc3_resp$var[3] - (q_nc * (exp(-xi_nc[3]) + 1 - q_nc) * exp(2 * m_nc + 2 * var_nc) +
                                            q_nc^2 * exp(2 * m_nc + var_nc) * expm1(var_nc))), TOLERANCE_STRICT)

    ###################
    ## Two sets of random effects ('gaussian_heteroscedastic_fixed_and_random', which requires a Vecchia approximation).
    ## Both the mean and the log-error variance have their own GP, and the prior of the second one must be used
    ###################
    y_nc_norm <- qnorm(sim_rand_unif(n = n_nc, init_c = 0.417))
    coords_nc <- cbind(sim_rand_unif(n = n_nc, init_c = 0.51), sim_rand_unif(n = n_nc, init_c = 0.62))
    coords_pred_nc <- cbind(c(0.1, 0.4, 0.7), c(0.2, 0.5, 0.8))
    cov_pars_nc <- c(1.3, 0.2, 0.4, 0.3)# (marginal variance, range) for the mean and for the log-error variance
    gp_model_nc_het <- GPModel(gp_coords = coords_nc, cov_function = "exponential", gp_approx = "vecchia",
                               num_neighbors = 10, cluster_ids = cluster_ids_nc,
                               likelihood = "gaussian_heteroscedastic_fixed_and_random")
    # Note: the new cluster has only one prediction point, for which the number of neighbors of the
    #   Vecchia approximation is reduced (this is reported by an information message)
    capture.output(pred_nc_het <- predict(gp_model_nc_het, y = y_nc_norm, cov_pars = cov_pars_nc, offset = rep(0, 2 * n_nc),
                                          gp_coords_pred = coords_pred_nc, cluster_ids_pred = cluster_ids_pred_nc,
                                          offset_pred = c(eta_nc, zeta_nc), predict_var = TRUE, predict_response = FALSE),
                   file = "NUL")
    expect_lt(abs(pred_nc_het$mu[3] - eta_nc[3]), TOLERANCE_STRICT)
    expect_lt(abs(pred_nc_het$var[3] - cov_pars_nc[1]), TOLERANCE_STRICT)
    # Response variance = prior variance of the mean + E(error variance) = v1 + exp(zeta + v2 / 2).
    # It thus depends on the prior of the second set of GPs, which is calculated with its own covariance parameters
    capture.output(pred_nc_het_resp <- predict(gp_model_nc_het, y = y_nc_norm, cov_pars = cov_pars_nc, offset = rep(0, 2 * n_nc),
                                               gp_coords_pred = coords_pred_nc, cluster_ids_pred = cluster_ids_pred_nc,
                                               offset_pred = c(eta_nc, zeta_nc), predict_var = TRUE, predict_response = TRUE),
                   file = "NUL")
    expect_lt(abs(pred_nc_het_resp$mu[3] - eta_nc[3]), TOLERANCE_STRICT)
    expect_lt(abs(pred_nc_het_resp$var[3] - (cov_pars_nc[1] + exp(zeta_nc[3] + cov_pars_nc[3] / 2))), TOLERANCE_STRICT)
  })

  test_that("hurdle_gamma_varying_shape likelihood for linear models ", {

    n_vs <- 100
    group_vs <- rep(1:10, each = 10)
    X_vs <- cbind(rep(1, n_vs), sim_rand_unif(n = n_vs, init_c = 0.415))
    beta_mean_vs <- c(0.4, 0.8)
    beta_shape_vs <- c(0.9, -0.7)
    b_gr_vs <- qnorm(sim_rand_unif(n = 10, init_c = 0.628))
    eta_true_vs <- as.vector(X_vs %*% beta_mean_vs) + sqrt(0.5) * b_gr_vs[group_vs]
    log_shape_true_vs <- as.vector(X_vs %*% beta_shape_vs)
    y_vs <- qgamma(sim_rand_unif(n = n_vs, init_c = 0.537), shape = exp(log_shape_true_vs),
                   rate = exp(log_shape_true_vs) / exp(eta_true_vs))
    y_hurdle_vs <- y_vs
    y_hurdle_vs[sim_rand_unif(n = n_vs, init_c = 0.264) < 0.25] <- 0
    expect_equal(sum(y_hurdle_vs == 0), 22L)

    # Likelihood evaluated at given (not estimated) parameters: a pure formula check, independent of any optimizer
    fixed_effects_given_vs <- as.vector(cbind(X_vs %*% c(0.2, 0.5), X_vs %*% c(0.6, -0.4)))
    nll_given_h <- GPModel(group_data = group_vs, likelihood = "hurdle_gamma_varying_shape")$neg_log_likelihood(
      cov_pars = 0.3, y = y_hurdle_vs, fixed_effects = fixed_effects_given_vs, aux_pars = 0.3)
    expect_lt(abs(nll_given_h - 217.78954815), TOLERANCE_MEDIUM)

    capture.output(gp_model_h <- fitGPModel(group_data = group_vs, likelihood = "hurdle_gamma_varying_shape",
                                            y = y_hurdle_vs, X = X_vs, params = OPTIM_PARAMS_BFGS), file = "NUL")
    coef_h <- as.vector(gp_model_h$get_coef(std_err = FALSE))
    expected_coef_h <- c(0.72724776, 0.52321528, 1.20120448, -0.96100261)
    expect_lt(sum(abs(coef_h - expected_coef_h)), TOLERANCE_MEDIUM)
    coef_h_std_err <- gp_model_h$get_coef(std_err = TRUE)
    expect_equal(colnames(coef_h_std_err), c("Covariate_1", "Covariate_2", "Covariate_1_shape", "Covariate_2_shape"))
    expect_true(all(coef_h_std_err["Std. err.", ] > 0))
    expected_coef_h_std_err <- c(0.28167051, 0.30412134, 0.35264012, 0.61410078)
    expect_lt(sum(abs(as.vector(coef_h_std_err["Std. err.", ]) - expected_coef_h_std_err)), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_h$get_cov_pars(std_err = FALSE)) - 0.56347247), TOLERANCE_MEDIUM)
    # p0 is the only auxiliary parameter and the structural zero decouples from both location parameter
    # blocks, so its maximum likelihood estimate is exactly the observed zero fraction
    expect_equal(gp_model_h$get_num_aux_pars(), 1L)
    expect_lt(abs(as.vector(gp_model_h$get_aux_pars()) - mean(y_hurdle_vs == 0)), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(gp_model_h$get_current_neg_log_likelihood() - 209.40746414), TOLERANCE_MEDIUM)
    # Prediction: E(y) = (1 - p0) * mu, Var(y) = (1 - p0) * (1 / shape + p0) * E(mu^2) + (1 - p0)^2 * Var(mu)
    X_test_vs <- cbind(rep(1, 3), c(0.1, 0.4, 0.8))
    group_test_vs <- c(1, 3, 11)
    pred_h <- predict(gp_model_h, y = y_hurdle_vs, group_data_pred = group_test_vs, X_pred = X_test_vs,
                      predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_h$mu - c(1.47334282, 1.89515945, 3.25142989))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_h$var - c(1.73274210, 3.43132816, 28.69036300))), TOLERANCE_LOOSE)
    re_pred_train_h <- predict_training_data_random_effects(gp_model_h)
    expected_re_pred_train_h <- c(-0.16971243, -0.22678700, -0.07686744, -0.01780991, -0.71284661,
                                  0.97960075, 0.62734841, 0.59290798, 0.38627911, -1.63260735)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_h[, 1])) - expected_re_pred_train_h)), TOLERANCE_MEDIUM)

    ###################
    ## Equivalence with the constant-shape "hurdle_gamma" likelihood for an intercept-only design matrix
    ###################
    X_int_vs <- X_vs[, 1, drop = FALSE]
    capture.output(gp_model_h_const <- fitGPModel(group_data = group_vs, likelihood = "hurdle_gamma", y = y_hurdle_vs,
                                                  X = X_int_vs, params = OPTIM_PARAMS_BFGS), file = "NUL")
    capture.output(gp_model_h_vary <- fitGPModel(group_data = group_vs, likelihood = "hurdle_gamma_varying_shape",
                                                 y = y_hurdle_vs, X = X_int_vs, params = OPTIM_PARAMS_BFGS), file = "NUL")
    expect_lt(abs(gp_model_h_const$get_current_neg_log_likelihood() - 211.77652907), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_h_vary$get_current_neg_log_likelihood() -
                    gp_model_h_const$get_current_neg_log_likelihood()), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(as.vector(gp_model_h_vary$get_coef(std_err = FALSE))[2] -
                    log(as.vector(gp_model_h_const$get_aux_pars())[1])), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(as.vector(gp_model_h_vary$get_aux_pars())[1] -
                    as.vector(gp_model_h_const$get_aux_pars())[2]), TOLERANCE_STRICT_LOWER)
  })

  test_that("hurdle_regression_gamma_varying_shape likelihood (three predictors) for linear and GPBoost models ", {

    n_vs <- 100
    group_vs <- rep(1:10, each = 10)
    X_vs <- cbind(rep(1, n_vs), sim_rand_unif(n = n_vs, init_c = 0.415))
    beta_mean_vs <- c(0.4, 0.8)
    beta_shape_vs <- c(0.9, -0.7)
    b_gr_vs <- qnorm(sim_rand_unif(n = 10, init_c = 0.628))
    eta_true_vs <- as.vector(X_vs %*% beta_mean_vs) + sqrt(0.5) * b_gr_vs[group_vs]
    log_shape_true_vs <- as.vector(X_vs %*% beta_shape_vs)
    y_vs <- qgamma(sim_rand_unif(n = n_vs, init_c = 0.537), shape = exp(log_shape_true_vs),
                   rate = exp(log_shape_true_vs) / exp(eta_true_vs))
    zeta_zero_true_vs <- as.vector(X_vs %*% c(-0.8, 1.0))
    y_hr_vs <- y_vs
    y_hr_vs[sim_rand_unif(n = n_vs, init_c = 0.264) < 1 / (1 + exp(-zeta_zero_true_vs))] <- 0
    expect_equal(sum(y_hr_vs == 0), 40L)
    X_test_vs <- cbind(rep(1, 3), c(0.1, 0.4, 0.8))
    group_test_vs <- c(1, 3, 11)

    # Likelihood evaluated at given (not estimated) parameters: a pure formula check, independent of any optimizer.
    # The three blocks are the response mean, the structural-zero logit, and log(shape), in this order
    fixed_effects_given_hr <- as.vector(cbind(X_vs %*% c(0.2, 0.5), X_vs %*% c(-0.5, 0.7), X_vs %*% c(0.6, -0.4)))
    nll_given_hr <- GPModel(group_data = group_vs, likelihood = "hurdle_regression_gamma_varying_shape")$neg_log_likelihood(
      cov_pars = 0.3, y = y_hr_vs, fixed_effects = fixed_effects_given_hr)
    expect_lt(abs(nll_given_hr - 200.73666702), TOLERANCE_MEDIUM)

    capture.output(gp_model_hr <- fitGPModel(group_data = group_vs, likelihood = "hurdle_regression_gamma_varying_shape",
                                             y = y_hr_vs, X = X_vs, params = OPTIM_PARAMS_BFGS), file = "NUL")
    coef_hr <- as.vector(gp_model_hr$get_coef(std_err = FALSE))
    expect_equal(length(coef_hr), 6L)
    expected_coef_hr <- c(0.77235050, 0.46757541, -0.37334779, -0.06796162, 1.26023275, -1.30887498)
    expect_lt(sum(abs(coef_hr - expected_coef_hr)), TOLERANCE_MEDIUM)
    coef_hr_std_err <- gp_model_hr$get_coef(std_err = TRUE)
    expect_equal(dim(coef_hr_std_err), c(2L, 6L))
    expect_equal(colnames(coef_hr_std_err), c("Covariate_1", "Covariate_2", "Covariate_1_zero", "Covariate_2_zero",
                                              "Covariate_1_shape", "Covariate_2_shape"))
    expect_true(all(coef_hr_std_err["Std. err.", ] > 0))
    expected_coef_hr_std_err <- c(0.28298350, 0.37030350, 0.40243850, 0.73474785, 0.41319946, 0.72677140)
    expect_lt(sum(abs(as.vector(coef_hr_std_err["Std. err.", ]) - expected_coef_hr_std_err)), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_hr$get_cov_pars(std_err = FALSE)) - 0.48861192), TOLERANCE_MEDIUM)
    expect_equal(gp_model_hr$get_num_aux_pars(), 0L)
    expect_lt(abs(gp_model_hr$get_current_neg_log_likelihood() - 193.91157916), TOLERANCE_MEDIUM)
    pred_hr <- predict(gp_model_hr, y = y_hr_vs, group_data_pred = group_test_vs, X_pred = X_test_vs,
                       predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_hr$mu - c(1.34955443, 1.45064588, 2.43204538))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_hr$var - c(2.57046994, 3.40478305, 22.88326100))), TOLERANCE_LOOSE)
    re_pred_train_hr <- predict_training_data_random_effects(gp_model_hr)
    expected_re_pred_train_hr <- c(-0.03783471, -0.14727182, -0.10378481, -0.01071364, -0.79564801,
                                   0.86784269, 0.61021241, 0.55427350, 0.17198142, -1.44187877)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_hr[, 1])) - expected_re_pred_train_hr)), TOLERANCE_MEDIUM)

    ###################
    ## GPBoost algorithm with three tree ensembles (mean, structural-zero logit, log-shape)
    ###################
    gp_model_hr_boost <- GPModel(group_data = group_vs, likelihood = "hurdle_regression_gamma_varying_shape")
    gp_model_hr_boost$set_optim_params(params = OPTIM_PARAMS_BFGS)
    dtrain_hr <- gpb.Dataset(data = X_vs[, 2, drop = FALSE], label = y_hr_vs)
    bst_hr <- gpb.train(data = dtrain_hr, gp_model = gp_model_hr_boost, nrounds = 20, learning_rate = 0.05,
                        max_depth = 2, min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    pred_hr_boost <- predict(bst_hr, data = X_vs[1:3, 2, drop = FALSE], group_data_pred = group_test_vs,
                             predict_var = TRUE, pred_latent = FALSE)
    expect_lt(abs(as.vector(gp_model_hr_boost$get_cov_pars(std_err = FALSE)) - 0.28607370), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_hr_boost$response_mean - c(1.71093425, 1.55926607, 2.19092195))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_hr_boost$response_var - c(5.19566634, 3.93816976, 11.59660949))), TOLERANCE_LOOSE)
  })

  test_that("varying-shape gamma likelihoods: density and derivatives against independent formulas ", {

    # Reference implementations of the log-densities, independent of the GPBoost C++ code
    # (eta = log(mean), zeta = log(shape), so shape = exp(zeta) and rate = exp(zeta) / exp(eta))
    ll_gamma_vs <- function(y, eta, zeta) dgamma(y, shape = exp(zeta), rate = exp(zeta) / exp(eta), log = TRUE)
    ll_hurdle_vs <- function(y, eta, zeta, p0) ifelse(y > 0, log1p(-p0) + ll_gamma_vs(pmax(y, 1e-300), eta, zeta), log(p0))
    ll_hurdle_regr_vs <- function(y, eta, zeta_zero, zeta_shape) {
      pi_i <- 1 / (1 + exp(-zeta_zero))
      ifelse(y > 0, log1p(-pi_i) + ll_gamma_vs(pmax(y, 1e-300), eta, zeta_shape), log(pi_i))
    }

    n_d <- 200
    X_d <- cbind(rep(1, n_d), sim_rand_unif(n = n_d, init_c = 0.311))
    beta_eta_d <- c(0.3, 0.9)
    beta_shape_d <- c(0.6, -0.8)
    beta_zero_d <- c(-0.7, 1.1)
    eta_d <- as.vector(X_d %*% beta_eta_d)
    zeta_shape_d <- as.vector(X_d %*% beta_shape_d)
    zeta_zero_d <- as.vector(X_d %*% beta_zero_d)
    y_d <- qgamma(sim_rand_unif(n = n_d, init_c = 0.428), shape = exp(zeta_shape_d), rate = exp(zeta_shape_d) / exp(eta_d))
    u_zero_d <- sim_rand_unif(n = n_d, init_c = 0.173)
    y_hurdle_d <- y_d
    y_hurdle_d[u_zero_d < 0.3] <- 0
    y_hr_d <- y_d
    y_hr_d[u_zero_d < 1 / (1 + exp(-zeta_zero_d))] <- 0
    p0_d <- 0.35
    # An iid model (no random effects at all) has no Laplace approximation: its negative log-likelihood is
    # exactly the (weighted) sum of the per-observation log-densities and its gradient wrt the fixed effects
    # is exactly the negative score. This makes the C++ formulas directly comparable to the R references above
    # (the 'cov_pars' argument is required by the interface but is not used for an iid model)
    gp_iid_vs <- GPModel(num_data = n_d, likelihood = "gamma_varying_shape")
    gp_iid_h <- GPModel(num_data = n_d, likelihood = "hurdle_gamma_varying_shape")
    gp_iid_hr <- GPModel(num_data = n_d, likelihood = "hurdle_regression_gamma_varying_shape")

    ###################
    ## 1) The C++ log-likelihood against R's 'dgamma', at parameters that are not the maximizer
    ###################
    nll_cpp_vs <- gp_iid_vs$neg_log_likelihood(cov_pars = 1, y = y_d, fixed_effects = c(eta_d, zeta_shape_d))
    expect_lt(abs(nll_cpp_vs + sum(ll_gamma_vs(y_d, eta_d, zeta_shape_d))), relax_tolerance_strict(TOLERANCE_STRICT))
    nll_cpp_h <- gp_iid_h$neg_log_likelihood(cov_pars = 1, y = y_hurdle_d, fixed_effects = c(eta_d, zeta_shape_d), aux_pars = p0_d)
    expect_lt(abs(nll_cpp_h + sum(ll_hurdle_vs(y_hurdle_d, eta_d, zeta_shape_d, p0_d))), relax_tolerance_strict(TOLERANCE_STRICT))
    nll_cpp_hr <- gp_iid_hr$neg_log_likelihood(cov_pars = 1, y = y_hr_d, fixed_effects = c(eta_d, zeta_zero_d, zeta_shape_d))
    expect_lt(abs(nll_cpp_hr + sum(ll_hurdle_regr_vs(y_hr_d, eta_d, zeta_zero_d, zeta_shape_d))), relax_tolerance_strict(TOLERANCE_STRICT))

    ###################
    ## 2) The analytical per-observation derivatives against central finite differences of the reference density.
    ## These are the six quantities the C++ code implements: the eta score, the eta information W = -l_etaeta,
    ## its derivative wrt eta, the zeta score, the cross derivative l_eta_zeta, and dW/dzeta
    ###################
    # Central stencils, each of order h^2, with step sizes balancing truncation against roundoff per derivative order
    fd1 <- function(f, x, h = 1e-5) (f(x + h) - f(x - h)) / (2 * h)
    fd2 <- function(f, x, h = 1e-3) (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    fd3 <- function(f, x, h = 3e-3) (-f(x - 2 * h) + 2 * f(x - h) - 2 * f(x + h) + f(x + 2 * h)) / (2 * h^3)
    fd_dxdy <- function(f, x, y, h = 1e-3) (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h)
    fd_dx2dy <- function(f, x, y, h = 3e-3) (f(x + h, y + h) - 2 * f(x, y + h) + f(x - h, y + h) -
                                               f(x + h, y - h) + 2 * f(x, y - h) - f(x - h, y - h)) / (2 * h^3)
    # The derivatives reach ~200 on this grid, so they are compared on a relative scale
    rel_err <- function(analytical, numerical) abs(analytical - numerical) / max(abs(analytical), 1)
    err <- c(l_eta = 0, J_eta = 0, dJ_deta = 0, l_zeta = 0, l_eta_zeta = 0, dJ_dzeta = 0)
    # Small, ordinary and large shape; small and large mean; small and large y / mu
    for (y in c(0.05, 0.3, 1, 2.5, 9)) for (eta in c(-1.2, -0.2, 0.5, 1.7)) for (zeta in c(-1, 0, 0.8, 2)) {
      k <- exp(zeta)
      y_exp_neg_eta <- y * exp(-eta)
      f_eta <- function(e) ll_gamma_vs(y, e, zeta)
      f_zeta <- function(z) ll_gamma_vs(y, eta, z)
      f_both <- function(e, z) ll_gamma_vs(y, e, z)
      err["l_eta"] <- max(err["l_eta"], rel_err(k * (y_exp_neg_eta - 1), fd1(f_eta, eta)))
      err["J_eta"] <- max(err["J_eta"], rel_err(k * y_exp_neg_eta, -fd2(f_eta, eta)))
      err["dJ_deta"] <- max(err["dJ_deta"], rel_err(-k * y_exp_neg_eta, -fd3(f_eta, eta)))
      err["l_zeta"] <- max(err["l_zeta"], rel_err(k * (zeta + 1 - eta - digamma(k) + log(y) - y_exp_neg_eta), fd1(f_zeta, zeta)))
      err["l_eta_zeta"] <- max(err["l_eta_zeta"], rel_err(k * (y_exp_neg_eta - 1), fd_dxdy(f_both, eta, zeta)))
      err["dJ_dzeta"] <- max(err["dJ_dzeta"], rel_err(k * y_exp_neg_eta, -fd_dx2dy(f_both, eta, zeta)))
    }
    expect_lt(max(err[c("l_eta", "l_zeta")]), 1e-6)
    expect_lt(max(err[c("J_eta", "l_eta_zeta")]), 1e-5)
    expect_lt(max(err[c("dJ_deta", "dJ_dzeta")]), 1e-4)

    ###################
    ## 3) The C++ score against the analytical formulas, via finite differences of the iid negative
    ## log-likelihood wrt the regression coefficients of every location parameter block
    ###################
    fd_coef_grad <- function(model, y, coefs, num_blocks, aux_pars = NULL) {
      eval_nll <- function(cf) {
        fixed_effects <- as.vector(sapply(1:num_blocks, function(k) X_d %*% cf[(k - 1) * 2 + 1:2]))
        model$neg_log_likelihood(cov_pars = 1, y = y, fixed_effects = fixed_effects, aux_pars = aux_pars)
      }
      h <- 1e-5
      sapply(seq_along(coefs), function(j) {
        cp <- cm <- coefs; cp[j] <- cp[j] + h; cm[j] <- cm[j] - h
        (eval_nll(cp) - eval_nll(cm)) / (2 * h)
      })
    }
    # Analytical scores: l_eta = k*(y/mu - 1), l_zeta = k*(zeta + 1 - eta - digamma(k) + log(y) - y/mu), both 0 at y = 0,
    # and the structural-zero score l_zeta_zero = 1{y=0} - pi. The gradient of the negative log-likelihood wrt the
    # coefficients of a block is -X^T (weights * score of that block)
    analytical_grad <- function(scores, weights_used = NULL) {
      w <- if (is.null(weights_used)) rep(1, n_d) else weights_used
      as.vector(sapply(scores, function(s) -as.vector(t(X_d) %*% (w * s))))
    }
    score_eta <- function(y, eta, zeta) ifelse(y > 0, exp(zeta) * (y * exp(-eta) - 1), 0)
    score_zeta <- function(y, eta, zeta) {
      k <- exp(zeta)
      ifelse(y > 0, k * (zeta + 1 - eta - digamma(k) + log(pmax(y, 1e-300)) - y * exp(-eta)), 0)
    }
    coefs_vs <- c(beta_eta_d, beta_shape_d)
    grad_fd <- fd_coef_grad(gp_iid_vs, y_d, coefs_vs, 2)
    grad_an <- analytical_grad(list(score_eta(y_d, eta_d, zeta_shape_d), score_zeta(y_d, eta_d, zeta_shape_d)))
    expect_lt(max(abs(grad_fd - grad_an)) / max(abs(grad_an)), TOLERANCE_STRICT_LOWER)
    # The same with non-unit sample weights, which must multiply every block's score
    w_d <- 0.5 + 2 * sim_rand_unif(n = n_d, init_c = 0.652)
    w_d <- w_d * (n_d / sum(w_d))# scale to sum to the number of data points, which avoids an informational message
    gp_iid_w <- GPModel(num_data = n_d, likelihood = "gamma_varying_shape", weights = w_d)
    grad_fd_w <- fd_coef_grad(gp_iid_w, y_d, coefs_vs, 2)
    grad_an_w <- analytical_grad(list(score_eta(y_d, eta_d, zeta_shape_d), score_zeta(y_d, eta_d, zeta_shape_d)), w_d)
    expect_lt(max(abs(grad_fd_w - grad_an_w)) / max(abs(grad_an_w)), TOLERANCE_STRICT_LOWER)
    # Constant-p0 hurdle: the zeros contribute nothing to either block
    grad_fd_h <- fd_coef_grad(gp_iid_h, y_hurdle_d, coefs_vs, 2, aux_pars = p0_d)
    grad_an_h <- analytical_grad(list(score_eta(y_hurdle_d, eta_d, zeta_shape_d), score_zeta(y_hurdle_d, eta_d, zeta_shape_d)))
    expect_lt(max(abs(grad_fd_h - grad_an_h)) / max(abs(grad_an_h)), TOLERANCE_STRICT_LOWER)
    # Regression hurdle: three blocks, the middle one being the structural-zero logit
    coefs_hr <- c(beta_eta_d, beta_zero_d, beta_shape_d)
    grad_fd_hr <- fd_coef_grad(gp_iid_hr, y_hr_d, coefs_hr, 3)
    grad_an_hr <- analytical_grad(list(score_eta(y_hr_d, eta_d, zeta_shape_d),
                                       as.numeric(y_hr_d <= 0) - 1 / (1 + exp(-zeta_zero_d)),
                                       score_zeta(y_hr_d, eta_d, zeta_shape_d)))
    expect_lt(max(abs(grad_fd_hr - grad_an_hr)) / max(abs(grad_an_hr)), TOLERANCE_STRICT_LOWER)
  })

  test_that("zero-censored shifted gamma likelihoods: density and derivatives against independent formulas ", {

    # Reference implementation of the log-density, independent of the GPBoost C++ code:
    # Y = max(Z - xi, 0) with Z ~ Gamma(shape = k, scale = mu / k) and mu = exp(eta)
    ll_zc <- function(y, eta, k, xi) ifelse(y > 0, dgamma(y + xi, shape = k, scale = exp(eta) / k, log = TRUE),
                                            pgamma(xi, shape = k, scale = exp(eta) / k, log.p = TRUE))

    n_zc <- 200
    X_zc <- cbind(rep(1, n_zc), sim_rand_unif(n = n_zc, init_c = 0.317))
    beta_eta_zc <- c(0.3, 0.9)
    beta_shape_zc <- c(0.4, -0.8)
    eta_zc <- as.vector(X_zc %*% beta_eta_zc)
    log_shape_zc <- as.vector(X_zc %*% beta_shape_zc)
    k_zc <- 1.7
    xi_zc <- 0.6
    y_zc <- pmax(qgamma(sim_rand_unif(n = n_zc, init_c = 0.431), shape = k_zc, scale = exp(eta_zc) / k_zc) - xi_zc, 0)
    y_vs_zc <- pmax(qgamma(sim_rand_unif(n = n_zc, init_c = 0.752), shape = exp(log_shape_zc), scale = exp(eta_zc) / exp(log_shape_zc)) - xi_zc, 0)
    expect_equal(sum(y_zc == 0), 25L)
    expect_equal(sum(y_vs_zc == 0), 49L)
    # An iid model (no random effects at all) has no Laplace approximation: its negative log-likelihood is exactly the
    # (weighted) sum of the per-observation log-densities and its gradient wrt the fixed effects is exactly the negative
    # score. This makes the C++ formulas directly comparable to the R reference above ('cov_pars' is unused for an iid model)
    gp_iid_zc <- GPModel(num_data = n_zc, likelihood = "zero_censored_shifted_gamma")
    gp_iid_vs_zc <- GPModel(num_data = n_zc, likelihood = "zero_censored_shifted_gamma_varying_shape")

    ###################
    ## 1) The C++ log-likelihood against R's 'dgamma' / 'pgamma', at parameters that are not the maximizer
    ###################
    nll_cpp_zc <- gp_iid_zc$neg_log_likelihood(cov_pars = 1, y = y_zc, fixed_effects = eta_zc, aux_pars = c(k_zc, xi_zc))
    expect_lt(abs(nll_cpp_zc + sum(ll_zc(y_zc, eta_zc, k_zc, xi_zc))), relax_tolerance_strict(TOLERANCE_STRICT))
    nll_cpp_vs_zc <- gp_iid_vs_zc$neg_log_likelihood(cov_pars = 1, y = y_vs_zc, fixed_effects = c(eta_zc, log_shape_zc), aux_pars = xi_zc)
    expect_lt(abs(nll_cpp_vs_zc + sum(ll_zc(y_vs_zc, eta_zc, exp(log_shape_zc), xi_zc))), relax_tolerance_strict(TOLERANCE_STRICT))
    # With an intercept-only log-shape block, the varying-shape density equals the constant-shape one
    nll_cpp_const_shape <- gp_iid_vs_zc$neg_log_likelihood(cov_pars = 1, y = y_zc, fixed_effects = c(eta_zc, rep(log(k_zc), n_zc)), aux_pars = xi_zc)
    expect_lt(abs(nll_cpp_const_shape - nll_cpp_zc), relax_tolerance_strict(TOLERANCE_STRICT))

    ###################
    ## 2) The analytical per-observation derivatives against central finite differences of the reference density.
    ## These are the quantities the C++ code implements: the eta score, the eta information W = -l_etaeta and its
    ## derivative wrt eta, and, for log(shape) and log(xi), the score, the cross derivative and dW/d(par)
    ###################
    fd1 <- function(f, x, h = 1e-5) (f(x + h) - f(x - h)) / (2 * h)
    fd2 <- function(f, x, h = 1e-3) (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    fd3 <- function(f, x, h = 3e-3) (-f(x - 2 * h) + 2 * f(x - h) - 2 * f(x + h) + f(x + 2 * h)) / (2 * h^3)
    fd_dxdy <- function(f, x, y, h = 1e-3) (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h)
    fd_dx2dy <- function(f, x, y, h = 3e-3) (f(x + h, y + h) - 2 * f(x, y + h) + f(x - h, y + h) -
                                               f(x + h, y - h) + 2 * f(x, y - h) - f(x - h, y - h)) / (2 * h^3)
    rel_err <- function(analytical, numerical) abs(analytical - numerical) / max(abs(analytical), 1)
    err_zc <- c(l_eta = 0, J_eta = 0, dJ_deta = 0, l_logk = 0, l_eta_logk = 0, dJ_dlogk = 0, l_logxi = 0, l_eta_logxi = 0, dJ_dlogxi = 0)
    # y = 0 exercises the point mass, the positive values the continuous part; small / ordinary / large shape and mean
    for (y in c(0, 0.05, 0.4, 1.5, 6)) for (eta in c(-0.8, 0.2, 1.3)) for (k in c(0.6, 1.5, 4)) for (xi in c(0.1, 0.7)) {
      f_eta <- function(e) ll_zc(y, e, k, xi)
      f_logxi <- function(lx) ll_zc(y, eta, k, exp(lx))
      f_eta_logk <- function(e, lk) ll_zc(y, e, exp(lk), xi)
      f_eta_logxi <- function(e, lx) ll_zc(y, e, k, exp(lx))
      if (y > 0) {
        z <- y + xi
        mu <- exp(eta)
        err_zc["l_eta"] <- max(err_zc["l_eta"], rel_err(k * z / mu - k, fd1(f_eta, eta)))
        err_zc["J_eta"] <- max(err_zc["J_eta"], rel_err(k * z / mu, -fd2(f_eta, eta)))
        err_zc["dJ_deta"] <- max(err_zc["dJ_deta"], rel_err(-k * z / mu, -fd3(f_eta, eta)))
        err_zc["l_logk"] <- max(err_zc["l_logk"], rel_err(k * (log(k) + 1 - eta - z / mu + log(z) - digamma(k)), fd1(function(lk) ll_zc(y, eta, exp(lk), xi), log(k))))
        err_zc["l_eta_logk"] <- max(err_zc["l_eta_logk"], rel_err(k * (z / mu - 1), fd_dxdy(f_eta_logk, eta, log(k))))
        err_zc["dJ_dlogk"] <- max(err_zc["dJ_dlogk"], rel_err(k * z / mu, -fd_dx2dy(f_eta_logk, eta, log(k))))
        err_zc["l_logxi"] <- max(err_zc["l_logxi"], rel_err(xi * ((k - 1) / z - k / mu), fd1(f_logxi, log(xi))))
        err_zc["l_eta_logxi"] <- max(err_zc["l_eta_logxi"], rel_err(xi * k / mu, fd_dxdy(f_eta_logxi, eta, log(xi))))
        err_zc["dJ_dlogxi"] <- max(err_zc["dJ_dlogxi"], rel_err(xi * k / mu, -fd_dx2dy(f_eta_logxi, eta, log(xi))))
      } else {# point mass: the analytical formulas use t = k * xi / mu and Q = g(t; k, 1) / G(k, t)
        t <- k * xi / exp(eta)
        Q <- exp(dgamma(t, shape = k, log = TRUE) - pgamma(t, shape = k, log.p = TRUE))
        Qp <- Q * ((k - 1) / t - 1) - Q^2
        dJdt <- (2 * t - k) * Q + 2 * t * Q^2 + Qp * (t * (t - k) + 2 * t^2 * Q)
        err_zc["l_eta"] <- max(err_zc["l_eta"], rel_err(-t * Q, fd1(f_eta, eta)))
        err_zc["J_eta"] <- max(err_zc["J_eta"], rel_err(t * (t - k) * Q + t^2 * Q^2, -fd2(f_eta, eta)))
        err_zc["dJ_deta"] <- max(err_zc["dJ_deta"], rel_err(-t * dJdt, -fd3(f_eta, eta)))
        err_zc["l_logxi"] <- max(err_zc["l_logxi"], rel_err(t * Q, fd1(f_logxi, log(xi))))
        err_zc["l_eta_logxi"] <- max(err_zc["l_eta_logxi"], rel_err(-t * (Q + t * Qp), fd_dxdy(f_eta_logxi, eta, log(xi))))
        err_zc["dJ_dlogxi"] <- max(err_zc["dJ_dlogxi"], rel_err(t * dJdt, -fd_dx2dy(f_eta_logxi, eta, log(xi))))
      }
    }
    expect_lt(max(err_zc[c("l_eta", "l_logk", "l_logxi")]), 1e-5)
    expect_lt(max(err_zc[c("J_eta", "l_eta_logk", "l_eta_logxi")]), 1e-4)
    expect_lt(max(err_zc[c("dJ_deta", "dJ_dlogk", "dJ_dlogxi")]), 1e-3)

    ###################
    ## 3) The C++ score against the analytical formulas, via finite differences of the iid negative log-likelihood
    ## wrt the regression coefficients of every location parameter block and wrt the log auxiliary parameters
    ###################
    fd_grad <- function(f, par, h = 1e-5) sapply(seq_along(par), function(j) {
      pp <- pm <- par; pp[j] <- pp[j] + h; pm[j] <- pm[j] - h; (f(pp) - f(pm)) / (2 * h) })
    tail_ratio <- function(eta, k, xi) exp(dgamma(k * xi / exp(eta), shape = k, log = TRUE) - pgamma(k * xi / exp(eta), shape = k, log.p = TRUE))
    score_eta_zc <- function(y, eta, k, xi) ifelse(y > 0, k * (y + xi) / exp(eta) - k, -(k * xi / exp(eta)) * tail_ratio(eta, k, xi))
    # At the point mass, d/d log(k) of the incomplete gamma function has no elementary closed form, so the reference
    # itself is a central difference of the R density there (which is what the C++ code does as well)
    score_logk_zc <- function(y, eta, k, xi) ifelse(y > 0, k * (log(k) + 1 - eta - (y + xi) / exp(eta) + log(y + xi) - digamma(k)),
                                                    (ll_zc(y, eta, k * exp(1e-4), xi) - ll_zc(y, eta, k * exp(-1e-4), xi)) / 2e-4)
    score_logxi_zc <- function(y, eta, k, xi) ifelse(y > 0, xi * ((k - 1) / (y + xi) - k / exp(eta)), (k * xi / exp(eta)) * tail_ratio(eta, k, xi))
    # Constant shape: the two coefficients of the single block plus log(shape) and log(xi)
    nll_zc_par <- function(par) gp_iid_zc$neg_log_likelihood(cov_pars = 1, y = y_zc, fixed_effects = as.vector(X_zc %*% par[1:2]), aux_pars = exp(par[3:4]))
    grad_fd_zc <- fd_grad(nll_zc_par, c(beta_eta_zc, log(k_zc), log(xi_zc)))
    grad_an_zc <- c(-as.vector(t(X_zc) %*% score_eta_zc(y_zc, eta_zc, k_zc, xi_zc)),
                    -sum(score_logk_zc(y_zc, eta_zc, k_zc, xi_zc)), -sum(score_logxi_zc(y_zc, eta_zc, k_zc, xi_zc)))
    expect_lt(max(abs(grad_fd_zc - grad_an_zc)) / max(abs(grad_an_zc)), TOLERANCE_STRICT_LOWER)
    # Varying shape: the coefficients of both blocks plus log(xi). The log(shape) block score is the log(shape) auxiliary
    # parameter score of the constant-shape variant, evaluated at the per-observation shape exp(zeta_i)
    nll_vs_zc_par <- function(par) gp_iid_vs_zc$neg_log_likelihood(cov_pars = 1, y = y_vs_zc, fixed_effects = c(X_zc %*% par[1:2], X_zc %*% par[3:4]), aux_pars = exp(par[5]))
    grad_fd_vs_zc <- fd_grad(nll_vs_zc_par, c(beta_eta_zc, beta_shape_zc, log(xi_zc)))
    shape_i_zc <- exp(log_shape_zc)
    grad_an_vs_zc <- c(-as.vector(t(X_zc) %*% score_eta_zc(y_vs_zc, eta_zc, shape_i_zc, xi_zc)),
                       -as.vector(t(X_zc) %*% score_logk_zc(y_vs_zc, eta_zc, shape_i_zc, xi_zc)),
                       -sum(score_logxi_zc(y_vs_zc, eta_zc, shape_i_zc, xi_zc)))
    expect_lt(max(abs(grad_fd_vs_zc - grad_an_vs_zc)) / max(abs(grad_an_vs_zc)), TOLERANCE_STRICT_LOWER)
    # The same with non-unit sample weights, which must multiply every block's score
    w_zc <- 0.5 + 2 * sim_rand_unif(n = n_zc, init_c = 0.658)
    w_zc <- w_zc * (n_zc / sum(w_zc))# scale to sum to the number of data points, which avoids an informational message
    gp_iid_w_zc <- GPModel(num_data = n_zc, likelihood = "zero_censored_shifted_gamma", weights = w_zc)
    nll_w_zc_par <- function(par) gp_iid_w_zc$neg_log_likelihood(cov_pars = 1, y = y_zc, fixed_effects = as.vector(X_zc %*% par[1:2]), aux_pars = exp(par[3:4]))
    grad_fd_w_zc <- fd_grad(nll_w_zc_par, c(beta_eta_zc, log(k_zc), log(xi_zc)))
    grad_an_w_zc <- c(-as.vector(t(X_zc) %*% (w_zc * score_eta_zc(y_zc, eta_zc, k_zc, xi_zc))),
                      -sum(w_zc * score_logk_zc(y_zc, eta_zc, k_zc, xi_zc)), -sum(w_zc * score_logxi_zc(y_zc, eta_zc, k_zc, xi_zc)))
    expect_lt(max(abs(grad_fd_w_zc - grad_an_w_zc)) / max(abs(grad_an_w_zc)), TOLERANCE_STRICT_LOWER)

    ###################
    ## 4) The predictive response mean and variance: predict() integrates the closed-form conditional moments of
    ## Y given eta over the (Gaussian) predictive distribution of eta. This is compared to a Gauss-Hermite
    ## integration of the same reference moments in R
    ###################
    # E(Y | eta) = k * theta * S_1 - xi * S_0 and E(Y^2 | eta) = k(k+1) theta^2 S_2 - 2 xi k theta S_1 + xi^2 S_0,
    # with theta = exp(eta) / k and S_j = P(Gamma(k + j, 1) > xi / theta)
    moments_zc <- function(eta, k, xi, second) {
      th <- exp(eta) / k
      t0 <- xi / th
      S0 <- pgamma(t0, shape = k, lower.tail = FALSE)
      S1 <- pgamma(t0, shape = k + 1, lower.tail = FALSE)
      m1 <- k * th * S1 - xi * S0
      if (!second) return(m1)
      k * (k + 1) * th^2 * pgamma(t0, shape = k + 2, lower.tail = FALSE) - 2 * xi * k * th * S1 + xi^2 * S0
    }
    capture.output(gp_pred_zc <- fitGPModel(group_data = rep(1:20, each = 10), likelihood = "zero_censored_shifted_gamma",
                                            y = y_zc, X = X_zc, params = OPTIM_PARAMS_BFGS), file = "NUL")
    X_test_zc <- cbind(rep(1, 3), c(0.1, 0.4, 0.8))
    pred_lat_zc <- predict(gp_pred_zc, group_data_pred = c(1, 3, 21), X_pred = X_test_zc, predict_var = TRUE, predict_response = FALSE)
    pred_resp_zc <- predict(gp_pred_zc, group_data_pred = c(1, 3, 21), X_pred = X_test_zc, predict_var = TRUE, predict_response = TRUE)
    aux_zc <- gp_pred_zc$get_aux_pars()
    # Independent reference: adaptive numerical integration of the conditional moments over the predictive density of eta
    ref_zc <- sapply(seq_along(pred_lat_zc$mu), function(i) {
      m <- pred_lat_zc$mu[i]
      s <- sqrt(pred_lat_zc$var[i])
      int <- function(second) integrate(function(e) moments_zc(e, aux_zc[["shape"]], aux_zc[["xi"]], second) * dnorm(e, m, s),
                                        m - 10 * s, m + 10 * s, rel.tol = 1e-10)$value
      m1 <- int(FALSE)
      c(m1, int(TRUE) - m1^2)
    })
    expect_lt(max(abs(pred_resp_zc$mu - ref_zc[1, ])), TOLERANCE_MEDIUM)
    expect_lt(max(abs(pred_resp_zc$var - ref_zc[2, ])), TOLERANCE_LOOSE)
  })

  test_that("zero_censored_shifted_gamma likelihood for linear and GPBoost models ", {

    n_zg <- 100
    group_zg <- rep(1:10, each = 10)
    X_zg <- cbind(rep(1, n_zg), sim_rand_unif(n = n_zg, init_c = 0.4137))
    beta_zg <- c(0.4, 1.1)
    shape_zg <- 1.5
    xi_zg <- 0.5
    b_gr_zg <- qnorm(sim_rand_unif(n = 10, init_c = 0.6218))
    eta_true_zg <- as.vector(X_zg %*% beta_zg) + sqrt(0.5) * b_gr_zg[group_zg]
    y_zg <- pmax(qgamma(sim_rand_unif(n = n_zg, init_c = 0.2731), shape = shape_zg, scale = exp(eta_true_zg) / shape_zg) - xi_zg, 0)
    expect_equal(sum(y_zg == 0), 11L)
    X_test_zg <- cbind(rep(1, 3), c(0.1, 0.4, 0.8))
    group_test_zg <- c(1, 3, 11)

    # Likelihood evaluated at given (not estimated) parameters: a pure formula check, independent of any optimizer
    nll_given_zg <- GPModel(group_data = group_zg, likelihood = "zero_censored_shifted_gamma")$neg_log_likelihood(
      cov_pars = 0.4, y = y_zg, fixed_effects = as.vector(X_zg %*% c(0.2, 0.9)), aux_pars = c(1.2, 0.4))
    expect_lt(abs(nll_given_zg - 216.88404284), TOLERANCE_MEDIUM)

    ###################
    ## Linear regression model with a grouped random effect
    ###################
    capture.output(gp_model_zg <- fitGPModel(group_data = group_zg, likelihood = "zero_censored_shifted_gamma",
                                             y = y_zg, X = X_zg, params = OPTIM_PARAMS_BFGS), file = "NUL")
    coef_zg <- as.vector(gp_model_zg$get_coef(std_err = FALSE))
    expect_equal(length(coef_zg), 2L)
    expected_coef_zg <- c(0.56843079, 0.70272302)
    expect_lt(sum(abs(coef_zg - expected_coef_zg)), TOLERANCE_MEDIUM)
    coef_zg_std_err <- gp_model_zg$get_coef(std_err = TRUE)
    expect_equal(dim(coef_zg_std_err), c(2L, 2L))
    expect_true(all(coef_zg_std_err["Std. err.", ] > 0))
    expect_lt(sum(abs(as.vector(coef_zg_std_err["Std. err.", ]) - c(0.25754821, 0.33937624))), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_zg$get_cov_pars(std_err = FALSE)) - 0.23721131), TOLERANCE_MEDIUM)
    expect_equal(gp_model_zg$get_num_aux_pars(), 2L)
    expect_equal(names(gp_model_zg$get_aux_pars()), c("shape", "xi"))
    expect_lt(sum(abs(as.vector(gp_model_zg$get_aux_pars()) - c(1.00996354, 0.27004889))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_zg$get_current_neg_log_likelihood() - 214.80987505), TOLERANCE_MEDIUM)
    # Prediction: latent and response scale
    pred_zg <- predict(gp_model_zg, y = y_zg, group_data_pred = group_test_zg, X_pred = X_test_zg,
                       predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_zg$mu - c(1.86422932, 3.12883994, 3.23000417))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_zg$var - c(5.00107174, 12.79369889, 18.44122697))), TOLERANCE_LOOSE)
    re_pred_train_zg <- predict_training_data_random_effects(gp_model_zg)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_zg[, 1])) - c(0.07716494, 0.37692704, 0.33896718, -0.12219735, -0.29619186,
                                   -0.77824411, 0.50995056, 0.11566715, 0.17874348, -0.63858758))), TOLERANCE_MEDIUM)

    ###################
    ## GPBoost algorithm
    ###################
    gp_model_zg_boost <- GPModel(group_data = group_zg, likelihood = "zero_censored_shifted_gamma")
    gp_model_zg_boost$set_optim_params(params = OPTIM_PARAMS_BFGS)
    dtrain_zg <- gpb.Dataset(data = X_zg[, 2, drop = FALSE], label = y_zg)
    bst_zg <- gpb.train(data = dtrain_zg, gp_model = gp_model_zg_boost, nrounds = 20, learning_rate = 0.05,
                        max_depth = 2, min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    pred_zg_boost <- predict(bst_zg, data = X_zg[1:3, 2, drop = FALSE], group_data_pred = group_test_zg,
                             predict_var = TRUE, pred_latent = FALSE)
    expect_lt(abs(as.vector(gp_model_zg_boost$get_cov_pars(std_err = FALSE)) - 0.22095375), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_zg_boost$response_mean - c(2.19683451, 4.33235647, 1.97728379))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_zg_boost$response_var - c(5.75228874, 19.58583983, 6.60125004))), TOLERANCE_LOOSE)
  })

  test_that("zero_censored_shifted_gamma_varying_shape likelihood for linear and GPBoost models ", {

    n_zv <- 100
    group_zv <- rep(1:10, each = 10)
    X_zv <- cbind(rep(1, n_zv), sim_rand_unif(n = n_zv, init_c = 0.5231))
    beta_mean_zv <- c(0.4, 1.1)
    beta_shape_zv <- c(0.5, -0.7)
    xi_zv <- 0.5
    b_gr_zv <- qnorm(sim_rand_unif(n = 10, init_c = 0.7314))
    eta_true_zv <- as.vector(X_zv %*% beta_mean_zv) + sqrt(0.5) * b_gr_zv[group_zv]
    shape_true_zv <- exp(as.vector(X_zv %*% beta_shape_zv))
    y_zv <- pmax(qgamma(sim_rand_unif(n = n_zv, init_c = 0.1837), shape = shape_true_zv, scale = exp(eta_true_zv) / shape_true_zv) - xi_zv, 0)
    expect_equal(sum(y_zv == 0), 15L)
    X_test_zv <- cbind(rep(1, 3), c(0.1, 0.4, 0.8))
    group_test_zv <- c(1, 3, 11)

    # Likelihood evaluated at given (not estimated) parameters. The two blocks are the response mean and log(shape)
    fe_given_zv <- c(as.vector(X_zv %*% c(0.2, 0.9)), as.vector(X_zv %*% c(0.4, -0.5)))
    nll_given_zv <- GPModel(group_data = group_zv, likelihood = "zero_censored_shifted_gamma_varying_shape")$neg_log_likelihood(
      cov_pars = 0.4, y = y_zv, fixed_effects = fe_given_zv, aux_pars = 0.4)
    expect_lt(abs(nll_given_zv - 206.93868664), TOLERANCE_MEDIUM)

    # A fixed-effects-only log(shape) requires a fixed effects term (covariates and / or GPBoost boosting)
    expect_error(capture.output(fitGPModel(group_data = group_zv, likelihood = "zero_censored_shifted_gamma_varying_shape",
                                           y = y_zv, params = list(maxit = 2, init_coef_aux_pars_from_iid_model = FALSE)), file = "NUL"))

    ###################
    ## Linear regression model (the mean has a grouped random effect, log(shape) is fixed-effects only)
    ###################
    capture.output(gp_model_zv <- fitGPModel(group_data = group_zv, likelihood = "zero_censored_shifted_gamma_varying_shape",
                                             y = y_zv, X = X_zv, params = OPTIM_PARAMS_BFGS), file = "NUL")
    coef_zv <- as.vector(gp_model_zv$get_coef(std_err = FALSE))
    expect_equal(length(coef_zv), 4L)
    coef_zv_std_err <- gp_model_zv$get_coef(std_err = TRUE)
    expect_equal(colnames(coef_zv_std_err), c("Covariate_1", "Covariate_2", "Covariate_1_shape", "Covariate_2_shape"))
    expect_true(all(coef_zv_std_err["Std. err.", ] > 0))
    expect_lt(sum(abs(coef_zv - c(0.20194321, 1.14080025, 0.26618832, -0.30528010))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(coef_zv_std_err["Std. err.", ]) - c(0.35042355, 0.35883319, 0.28338584, 0.50824456))), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_zv$get_cov_pars(std_err = FALSE)) - 0.88127088), TOLERANCE_MEDIUM)
    expect_equal(gp_model_zv$get_num_aux_pars(), 1L)
    expect_equal(names(gp_model_zv$get_aux_pars()), "xi")
    expect_lt(abs(as.vector(gp_model_zv$get_aux_pars()) - 0.26324236), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_zv$get_current_neg_log_likelihood() - 204.73954422), TOLERANCE_MEDIUM)
    pred_zv <- predict(gp_model_zv, y = y_zv, group_data_pred = group_test_zv, X_pred = X_test_zv,
                       predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_zv$mu - c(0.99281426, 0.36869895, 4.48832117))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_zv$var - c(1.39505484, 0.31770929, 84.54632964))), TOLERANCE_LOOSE)
    re_pred_train_zv <- predict_training_data_random_effects(gp_model_zv)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_zv[, 1])) - c(-0.14330267, 0.86328997, -1.24540468, -1.58338835, 1.09760260,
                                   0.14407352, -0.70360020, 0.29502892, 1.16287144, -0.25488799))), TOLERANCE_MEDIUM)

    ###################
    ## With an intercept-only design (the same X is reused for every block) the varying-shape fit
    ## must reproduce the constant-shape fit
    ###################
    X_int_zv <- matrix(1, nrow = n_zv, ncol = 1)
    capture.output(gp_const_zv <- fitGPModel(group_data = group_zv, likelihood = "zero_censored_shifted_gamma",
                                             y = y_zv, X = X_int_zv, params = OPTIM_PARAMS_BFGS), file = "NUL")
    capture.output(gp_var_zv <- fitGPModel(group_data = group_zv, likelihood = "zero_censored_shifted_gamma_varying_shape",
                                           y = y_zv, X = X_int_zv, params = OPTIM_PARAMS_BFGS), file = "NUL")
    expect_lt(abs(gp_const_zv$get_current_neg_log_likelihood() - gp_var_zv$get_current_neg_log_likelihood()), TOLERANCE_MEDIUM)

    ###################
    ## GPBoost algorithm with two tree ensembles (mean and log-shape)
    ###################
    gp_model_zv_boost <- GPModel(group_data = group_zv, likelihood = "zero_censored_shifted_gamma_varying_shape")
    gp_model_zv_boost$set_optim_params(params = OPTIM_PARAMS_BFGS)
    dtrain_zv <- gpb.Dataset(data = X_zv[, 2, drop = FALSE], label = y_zv)
    bst_zv <- gpb.train(data = dtrain_zv, gp_model = gp_model_zv_boost, nrounds = 20, learning_rate = 0.05,
                        max_depth = 2, min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    pred_zv_boost <- predict(bst_zv, data = X_zv[1:3, 2, drop = FALSE], group_data_pred = group_test_zv,
                             predict_var = TRUE, pred_latent = FALSE)
    expect_lt(abs(as.vector(gp_model_zv_boost$get_cov_pars(std_err = FALSE)) - 0.93526425), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_zv_boost$response_mean - c(2.17895836, 0.33196983, 3.64783242))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_zv_boost$response_var - c(6.24164921, 0.27871762, 59.22645884))), TOLERANCE_LOOSE)
  })

  test_that("zero_censored_power_transformed_normal_heteroscedastic likelihood for linear and GPBoost models ", {

    likelihood <- "zero_censored_power_transformed_normal_heteroscedastic"
    n_zcp <- 100
    group_zcp <- rep(1:10, each = 10)
    X_zcp <- cbind(rep(1, n_zcp), sim_rand_unif(n = n_zcp, init_c = 0.4231))
    beta_mean_zcp <- c(0.4, 1.1)
    beta_scale_zcp <- c(-0.3, 0.7)
    lambda_zcp <- 0.75
    gr_var_zcp <- 0.5
    b_gr_zcp <- qnorm(sim_rand_unif(n = 10, init_c = 0.6412))
    mean_true_zcp <- as.vector(X_zcp %*% beta_mean_zcp) + sqrt(gr_var_zcp) * b_gr_zcp[group_zcp]
    log_sigma_true_zcp <- as.vector(X_zcp %*% beta_scale_zcp)
    x_lat_zcp <- mean_true_zcp + qnorm(sim_rand_unif(n = n_zcp, init_c = 0.2871)) * exp(log_sigma_true_zcp)
    y_zcp <- pmax(0, x_lat_zcp)^lambda_zcp
    expect_equal(mean(y_zcp == 0), 0.2)

    # Likelihood evaluated at given (not estimated) parameters: a pure formula check, independent of any optimizer
    fe_given_zcp <- c(as.vector(X_zcp %*% c(0.2, 0.9)), as.vector(X_zcp %*% c(-0.2, 0.6)))
    nll_given_zcp <- GPModel(group_data = group_zcp, likelihood = likelihood)$neg_log_likelihood(
      cov_pars = 0.4, y = y_zcp, fixed_effects = fe_given_zcp, aux_pars = 0.8)
    expect_lt(abs(nll_given_zcp - 121.92107768), TOLERANCE_MEDIUM)

    # A fixed-effects-only log standard deviation requires a fixed effects term (covariates and / or GPBoost boosting)
    expect_error(capture.output(fitGPModel(group_data = group_zcp, likelihood = likelihood, y = y_zcp,
                                           params = list(maxit = 2, init_coef_aux_pars_from_iid_model = FALSE)), file = "NUL"))

    ###################
    ## Linear regression model (mean has a grouped random effect, log(sigma) is fixed-effects only)
    ###################
    capture.output(gp_model_zcp <- fitGPModel(group_data = group_zcp, likelihood = likelihood, y = y_zcp, X = X_zcp,
                                              params = OPTIM_PARAMS_BFGS), file = "NUL")
    coef_zcp <- as.vector(gp_model_zcp$get_coef(std_err = FALSE))
    expect_equal(length(coef_zcp), 4L)
    coef_zcp_std_err <- gp_model_zcp$get_coef(std_err = TRUE)
    expect_equal(dim(coef_zcp_std_err), c(2L, 4L))
    # Note: std. errs. must be strictly positive; a plain is.finite() check would not catch a regression where the
    # log(sigma) block's std. errs. are silently left at their R-side zero-initialized default (0 is finite)
    expect_true(all(coef_zcp_std_err["Std. err.", ] > 0))
    expected_coef_zcp <- c(0.36913847, 1.60521687, -0.31152835, 0.91712942)
    expect_lt(sum(abs(coef_zcp - expected_coef_zcp)), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_zcp$get_cov_pars(std_err = FALSE)) - 0.28100193), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_zcp$get_aux_pars()) - 0.69356420), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_zcp$get_current_neg_log_likelihood() - 117.92994175), TOLERANCE_MEDIUM)
    # Prediction: response mean and variance
    X_test_zcp <- cbind(rep(1, 3), c(0.1, 0.4, 0.8))
    group_test_zcp <- c(1, 3, 11)
    pred_zcp <- predict(gp_model_zcp, y = y_zcp, group_data_pred = group_test_zcp, X_pred = X_test_zcp,
                        predict_var = TRUE, predict_response = TRUE)
    expected_mu_zcp <- c(0.58594067, 1.16162775, 1.36155518)
    expected_var_zcp <- c(0.27375042, 0.45183331, 0.76270513)
    expect_lt(sum(abs(pred_zcp$mu - expected_mu_zcp)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_zcp$var - expected_var_zcp)), TOLERANCE_MEDIUM)
    X_zero_zcp <- matrix(0, nrow = n_zcp, ncol = ncol(X_zcp))
    re_pred_train_zcp <- predict_training_data_random_effects(gp_model_zcp)
    expected_re_pred_train_zcp <- c(-0.11992737, 0.81411976, 0.31298099, -0.05903340, -0.26484985,
                                    0.41429315, -0.00512130, -0.57966632, 0.22550739, -0.71550754)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_zcp[, 1])) - expected_re_pred_train_zcp)), TOLERANCE_MEDIUM)
    re_pred_train_zcp_var <- predict_training_data_random_effects(gp_model_zcp, predict_var = TRUE)
    expected_re_pred_train_zcp_var <- c(0.07868473, 0.09165282, 0.08926593, 0.08715446, 0.09628326,
                                        0.08000501, 0.07111215, 0.09147585, 0.08963245, 0.10161664)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_zcp_var[, 2])) - expected_re_pred_train_zcp_var)), TOLERANCE_MEDIUM)
    pred_train_re_zcp <- predict(gp_model_zcp, y = y_zcp, group_data_pred = group_zcp, X_pred = X_zero_zcp,
                                 predict_response = FALSE, predict_var = FALSE)
    expect_lt(sum(abs(as.vector(re_pred_train_zcp[, 1]) - pred_train_re_zcp$mu)), TOLERANCE_STRICT)
    # Predicting requires covariate data for the model's linear predictors (mean and log(sigma))
    expect_error(predict(gp_model_zcp, y = y_zcp, group_data_pred = group_test_zcp,
                         predict_var = TRUE, predict_response = TRUE))

    # The log(sigma) block gradient combines direct-score, log-determinant and implicit-mode terms. Verify that the full
    # Laplace objective is stationary in every coefficient direction (mean block and log(sigma) block) at the optimum.
    # This needs its own tightly converged fit: with the default delta_rel_conv = 1e-6, the optimizer
    # stops while the gradient is still ~3e-2 in all directions, including the long-established mean block, so such a fit
    # would measure the optimizer's stopping tolerance rather than the correctness of the gradient
    capture.output(gp_model_zcp_tight <- fitGPModel(group_data = group_zcp, likelihood = likelihood, y = y_zcp, X = X_zcp, params = c(OPTIM_PARAMS_BFGS, list(delta_rel_conv = 1e-12))), file = "NUL")
    coef_zcp_fd <- as.vector(gp_model_zcp_tight$get_coef(std_err = FALSE))
    expect_lt(sum(abs(coef_zcp_fd - c(0.36984959, 1.60513986, -0.31105647, 0.91666661))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_zcp_tight$get_current_neg_log_likelihood() - 117.92993064), TOLERANCE_MEDIUM)
    cov_pars_zcp_fd <- as.vector(gp_model_zcp_tight$get_cov_pars(std_err = FALSE))
    aux_pars_zcp_fd <- as.vector(gp_model_zcp_tight$get_aux_pars())
    gp_model_zcp_fd <- GPModel(group_data = group_zcp, likelihood = likelihood)
    nll_zcp_fd <- function(coef_vec) gp_model_zcp_fd$neg_log_likelihood(cov_pars = cov_pars_zcp_fd, y = y_zcp, fixed_effects = as.vector(cbind(X_zcp %*% coef_vec[1:2], X_zcp %*% coef_vec[3:4])), aux_pars = aux_pars_zcp_fd)
    step_zcp_fd <- 1e-4
    gradient_zcp_fd <- sapply(1:4, function(k) { coef_plus <- coef_minus <- coef_zcp_fd; coef_plus[k] <- coef_plus[k] + step_zcp_fd; coef_minus[k] <- coef_minus[k] - step_zcp_fd; (nll_zcp_fd(coef_plus) - nll_zcp_fd(coef_minus)) / (2 * step_zcp_fd) })
    expect_lt(max(abs(gradient_zcp_fd)), 1e-3)

    ###################
    ## No random effects at all (iid model, pure linear regression for the mean and log(sigma))
    ###################
    capture.output(gp_model_zcp_iid <- fitGPModel(likelihood = likelihood, y = y_zcp, X = X_zcp,
                                                  params = OPTIM_PARAMS_BFGS), file = "NUL")
    expected_coef_zcp_iid <- c(0.33506594, 1.67738386, -0.14336405, 0.78367073)
    expect_lt(sum(abs(as.vector(gp_model_zcp_iid$get_coef(std_err = FALSE)) - expected_coef_zcp_iid)), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_zcp_iid$get_aux_pars()) - 0.69198615), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_zcp_iid$get_current_neg_log_likelihood() - 121.71631164), TOLERANCE_MEDIUM)

    ###################
    ## GPBoost algorithm (tree-boosting): mean via a grouped random effect + trees, log(sigma) via a second tree ensemble
    ###################
    gp_model_zcp_boost <- GPModel(group_data = group_zcp, likelihood = likelihood)
    gp_model_zcp_boost$set_optim_params(params = OPTIM_PARAMS_BFGS)
    dtrain_zcp <- gpb.Dataset(data = X_zcp[, 2, drop = FALSE], label = y_zcp)
    bst_zcp <- gpb.train(data = dtrain_zcp, gp_model = gp_model_zcp_boost, nrounds = 20, learning_rate = 0.05,
                         max_depth = 2, min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    pred_zcp_boost <- predict(bst_zcp, data = X_zcp[1:3, 2, drop = FALSE], group_data_pred = group_test_zcp,
                              predict_var = TRUE, pred_latent = FALSE)
    expect_lt(abs(as.vector(gp_model_zcp_boost$get_cov_pars(std_err = FALSE)) - 0.33950629), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_zcp_boost$get_aux_pars()) - 0.67688028), TOLERANCE_MEDIUM)
    expected_response_mean_boost_zcp <- c(0.78901313, 1.12780115, 0.99632895)
    expected_response_var_boost_zcp <- c(0.36894119, 0.36648812, 0.57118598)
    expect_lt(sum(abs(pred_zcp_boost$response_mean - expected_response_mean_boost_zcp)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_zcp_boost$response_var - expected_response_var_boost_zcp)), TOLERANCE_MEDIUM)

    ###################
    ## Gaussian processes
    ###################
    n_zcp2 <- 100
    X_zcp2 <- cbind(rep(1, n_zcp2), sim_rand_unif(n = n_zcp2, init_c = 0.1937))
    coords_zcp2 <- matrix(sim_rand_unif(n = n_zcp2 * 2, init_c = 0.5713), ncol = 2)
    Sigma_zcp2 <- 0.6 * exp(-as.matrix(dist(coords_zcp2)) / 0.15) + diag(1e-10, n_zcp2)
    b_gp_zcp2 <- as.vector(t(chol(Sigma_zcp2)) %*% qnorm(sim_rand_unif(n = n_zcp2, init_c = 0.8123)))
    mean_true_zcp2 <- as.vector(X_zcp2 %*% c(0.3, 1.0)) + b_gp_zcp2
    log_sigma_true_zcp2 <- as.vector(X_zcp2 %*% c(-0.25, 0.6))
    x_lat_zcp2 <- mean_true_zcp2 + qnorm(sim_rand_unif(n = n_zcp2, init_c = 0.3499)) * exp(log_sigma_true_zcp2)
    y_zcp2 <- pmax(0, x_lat_zcp2)^lambda_zcp
    expect_equal(mean(y_zcp2 == 0), 0.18)
    optim_params_zcp2 <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 300,
                              init_coef_aux_pars_from_iid_model = FALSE)

    # Likelihood evaluated at given (not estimated) parameters
    nll_given_gp_zcp <- GPModel(gp_coords = coords_zcp2, cov_function = "exponential",
                                likelihood = likelihood)$neg_log_likelihood(
      cov_pars = c(1, mean(dist(coords_zcp2)) / 3), y = y_zcp2, fixed_effects = rep(0, 2 * n_zcp2), aux_pars = 0.8)
    expect_lt(abs(nll_given_gp_zcp - 138.13953310), TOLERANCE_MEDIUM)

    ## Dense GP ("Stable")
    capture.output(gp_model_gp_zcp <- fitGPModel(gp_coords = coords_zcp2, cov_function = "exponential",
                                                 likelihood = likelihood, y = y_zcp2, X = X_zcp2,
                                                 params = optim_params_zcp2), file = "NUL")
    expected_coef_gp_zcp <- c(0.39728925, 1.59833428, -0.49364796, 0.97995228)
    expect_lt(sum(abs(as.vector(gp_model_gp_zcp$get_coef(std_err = FALSE)) - expected_coef_gp_zcp)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(as.vector(gp_model_gp_zcp$get_cov_pars(std_err = FALSE)) - c(0.27394758, 0.18410689))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(as.vector(gp_model_gp_zcp$get_aux_pars()) - 0.74483908), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_gp_zcp$get_current_neg_log_likelihood() - 119.57095193), relax_tolerance_nll(TOLERANCE_MEDIUM))
    coord_test_zcp <- coords_zcp2[1:3, , drop = FALSE] + 1e-3
    pred_gp_zcp <- predict(gp_model_gp_zcp, y = y_zcp2, gp_coords_pred = coord_test_zcp,
                           X_pred = X_zcp2[1:3, , drop = FALSE], predict_var = TRUE, predict_response = TRUE)
    expected_mu_gp_zcp <- c(0.85507602, 1.93294194, 1.52508240)
    expected_var_gp_zcp <- c(0.32915567, 0.81366804, 0.42213707)
    expect_lt(sum(abs(pred_gp_zcp$mu - expected_mu_gp_zcp)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(pred_gp_zcp$var - expected_var_gp_zcp)), TOLERANCE_NON_CONVEX)

    ## Iterative methods for grouped random effects
    group_zcp_crossed <- cbind(group_zcp, rep(1:5, times = n_zcp / 5))
    capture.output(gp_model_grouped_chol_zcp <- fitGPModel(group_data = group_zcp_crossed, likelihood = likelihood, matrix_inversion_method = "cholesky", y = y_zcp, X = X_zcp, params = OPTIM_PARAMS_BFGS), file = "NUL")
    expect_lt(sum(abs(as.vector(gp_model_grouped_chol_zcp$get_coef(std_err = FALSE)) - c(0.35884118, 1.61697557, -0.37683985, 0.98417865))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_grouped_chol_zcp$get_cov_pars(std_err = FALSE)) - c(0.30231863, 0.05341490))), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_grouped_chol_zcp$get_aux_pars()) - 0.69638277), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_grouped_chol_zcp$get_current_neg_log_likelihood() - 117.48723971), TOLERANCE_MEDIUM)
    capture.output(gp_model_grouped_iter_zcp <- fitGPModel(group_data = group_zcp_crossed, likelihood = likelihood, matrix_inversion_method = "iterative", y = y_zcp, X = X_zcp, params = c(OPTIM_PARAMS_BFGS, list(seed_rand_vec_trace = 1))), file = "NUL")
    expect_lt(sum(abs(as.vector(gp_model_grouped_iter_zcp$get_coef(std_err = FALSE)) - as.vector(gp_model_grouped_chol_zcp$get_coef(std_err = FALSE)))), TOLERANCE_ITERATIVE)
    expect_lt(sum(abs(as.vector(gp_model_grouped_iter_zcp$get_cov_pars(std_err = FALSE)) - as.vector(gp_model_grouped_chol_zcp$get_cov_pars(std_err = FALSE)))), TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model_grouped_iter_zcp$get_current_neg_log_likelihood() - gp_model_grouped_chol_zcp$get_current_neg_log_likelihood()), relax_tolerance_nll(TOLERANCE_ITERATIVE))

    ## GP with a Vecchia approximation. With num_neighbors = n - 1, Vecchia is exact and must match the dense GP fit
    capture.output(gp_model_vecchia_zcp <- fitGPModel(gp_coords = coords_zcp2, cov_function = "exponential",
                                                      likelihood = likelihood, gp_approx = "vecchia",
                                                      num_neighbors = n_zcp2 - 1, vecchia_ordering = "none",
                                                      matrix_inversion_method = "cholesky",
                                                      y = y_zcp2, X = X_zcp2, params = optim_params_zcp2), file = "NUL")
    expect_lt(sum(abs(as.vector(gp_model_vecchia_zcp$get_coef(std_err = FALSE)) - expected_coef_gp_zcp)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(as.vector(gp_model_vecchia_zcp$get_cov_pars(std_err = FALSE)) - c(0.27394758, 0.18410689))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_vecchia_zcp$get_current_neg_log_likelihood() - 119.57095193), relax_tolerance_nll(TOLERANCE_MEDIUM))

    ## GP with an FITC approximation
    capture.output(gp_model_fitc_zcp <- fitGPModel(gp_coords = coords_zcp2, cov_function = "exponential",
                                                   likelihood = likelihood, gp_approx = "fitc", num_ind_points = 30,
                                                   y = y_zcp2, X = X_zcp2, params = optim_params_zcp2), file = "NUL")
    expected_coef_fitc_zcp <- c(0.39434746, 1.58272678, -0.53801543, 1.00172670)
    expect_lt(sum(abs(as.vector(gp_model_fitc_zcp$get_coef(std_err = FALSE)) - expected_coef_fitc_zcp)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(as.vector(gp_model_fitc_zcp$get_cov_pars(std_err = FALSE)) - c(0.32711281, 0.16804725))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(as.vector(gp_model_fitc_zcp$get_aux_pars()) - 0.74748851), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_fitc_zcp$get_current_neg_log_likelihood() - 119.41168559), relax_tolerance_nll(TOLERANCE_MEDIUM))
  }) #end zero_censored_power_transformed_normal_heteroscedastic likelihood

}
