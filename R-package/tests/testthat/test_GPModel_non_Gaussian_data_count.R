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

  test_that("Poisson regression ", {

    # Single level grouped random effects
    mu <- exp(Z1 %*% b_gr_1)
    y <- qpois(sim_rand_unif(n=n, init_c=0.04532), lambda = mu)
    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "poisson",
                                           y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE, lr_cov=0.1, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    cov_pars <- c(0.4033406)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.07765297, -0.87488533, -0.87488533, 0.00000000)
    expected_cov <- c(0.07526284, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                      0.15041230, 0.15041230, 0.00000000, 0.00000000, 0.15041230,
                      0.15041230, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.40334058)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(1.1221925, 0.4494731, 0.4494731, 1.2234446)
    expected_var <- c(1.2206301, 0.4822647, 0.4822647, 1.9670879)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-140.4554806),TOLERANCE_MEDIUM)

    # Multiple random effects
    mu <- exp(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- qpois(sim_rand_unif(n=n, init_c=0.74532), lambda = mu)
    init_cov_pars <- rep(1,3)
    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "poisson", matrix_inversion_method = "cholesky",
                                           y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE,
                                                                lr_cov=0.1, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    cov_pars <- c(0.4069344, 1.6988978, 1.3415016)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 7)
    # Prediction
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.92620057, -0.08200469, 0.00000000)
    expected_cov <- c(0.07730896, 0.04403442, 0.00000000, 0.04403442, 0.11600469,
                      0.00000000, 0.00000000, 0.00000000, 1.80800000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)

    # Gaussian process model
    mu <- exp(L %*% b_1)
    y <- qpois(sim_rand_unif(n=n, init_c=0.435), lambda = mu)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- c(1,mean(dist(coords))/3)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "poisson",
                                           y = y, params = params)
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(1.1853922, 0.1500197))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.4329068, 0.4042531, 0.6833738)
    expected_cov <- c(6.550626e-01, 5.553938e-01, -8.406290e-06, 5.553938e-01, 6.631295e-01, -7.658261e-06, -8.406290e-06, -7.658261e-06, 4.170417e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(2.139213, 2.087188, 2.439748)
    expected_var <- c(6.373433, 6.185895, 5.519896)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_MEDIUM)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-195.03708036),TOLERANCE_STRICT)

    ## Grouped random effects model with a linear predictor
    mu_lin <- exp(Z1 %*% b_gr_1 + X%*%beta)
    y_lin <- qpois(sim_rand_unif(n=n, init_c=0.84532), lambda = mu_lin)
    gp_model <- fitGPModel(group_data = group, likelihood = "poisson",
                           y = y_lin, X=X, params = list(optimizer_cov = "gradient_descent",
                                                         optimizer_coef = "gradient_descent", lr_cov = 0.1, lr_coef = 0.1,
                                                         use_nesterov_acc = TRUE, acc_rate_cov = 0.5, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.2977336946)
    coef <- c(-0.1491220786, 2.1209270742)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 23)
  })

  test_that("Gamma regression ", {

    params <- OPTIM_PARAMS_BFGS
    params$init_aux_pars = 1.
    params$estimate_aux_pars = FALSE
    params_shape <- params
    params_shape$estimate_aux_pars <- TRUE
    shape <- 1

    # Single level grouped random effects
    mu <- exp(Z1 %*% b_gr_1)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.04532), scale = mu/shape, shape = shape)
    # Cannot have 0 in response variable
    y_zero <- y
    y_zero[1] <- 0
    expect_error(gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                        y = y_zero, params = params))
    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params)
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.5175032387)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 4)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test,
                    predict_cov_mat = TRUE, predict_response = FALSE, cov_pars = 0.6)
    expected_mu <- c(0.2141580841, -0.9414716643, -0.9414716643, 0.0000000)
    expected_cov <- c(0.08316978821, 0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000, 0.09903395128, 0.09903395128, 0.00000000000, 0.00000000000, 0.09903395128, 0.09903395128, 0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000, 0.60000000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test,
                    predict_var=TRUE, predict_response = TRUE, cov_pars = 0.6)
    expected_mu <- c(1.2914207620, 0.4098538326, 0.4098538326, 1.3498588076)
    expected_var <- c(1.9570462290, 0.2029549058, 0.2029549058, 4.8181150451)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_MEDIUM)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-105.676137),TOLERANCE_MEDIUM)
    # Also estimate shape parameter
    params_shape$optimizer_cov <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.5141632)
    aux_pars <- c(0.9719373)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-105.1597249), TOLERANCE_MEDIUM)
    # Also estimate shape parameter with lbfgs
    params_shape$optimizer_cov <- "lbfgs"
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params_shape), file='NUL')
    cov_pars <- c(0.5141245271 )
    aux_pars <- c(0.9719437296 )
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    # Can set learning rate for auxiliary parameters via lr_cov
    params_shape$optimizer_cov <- "gradient_descent"
    params_temp <- params_shape
    params_temp$maxit = 1
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params_temp), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.9058829)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-0.9297985)),TOLERANCE_STRICT)
    params_temp$lr_cov = 0.001
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params_temp), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.998025)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-0.9985453)),TOLERANCE_STRICT)
    # fix some covariance parameters
    params_loc <- params_shape
    params_loc$optimizer_cov = "lbfgs"
    params_loc$estimate_cov_par_index <- c(0)
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y, params = params_loc), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-1)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-0.9762847373)),TOLERANCE_STRICT)

    # Multiple random effects
    mu <- exp(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    params$init_cov_pars <- rep(1,3)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.04532), scale = mu/shape, shape = shape)
    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    cov_pars <- c(0.5080507200, 1.2045682905, 0.5297377706)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    # Prediction
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1121777, 0.1972216, 0.0000000)
    expected_var <- c(0.2405621, 0.2259258, 1.8080000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    # Also estimate shape parameter
    params_shape$optimizer_cov <- "nelder_mead"
    params_shape$init_cov_pars <- rep(1,3)
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_shape, matrix_inversion_method = "cholesky"), file='NUL')
    cov_pars <- c(0.5050897, 1.2026241, 0.5232070)
    aux_pars <- c(0.9819755)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),0.01)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-123.517723), TOLERANCE_MEDIUM)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_shape, matrix_inversion_method = "cholesky"), file='NUL')
    cov_pars <- c(0.5065183, 1.2028488, 0.5360939)
    aux_pars <- c(0.9827199)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 31)
    # Also estimate shape parameter with adam
    params_shape$optimizer_cov <- "adam"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_shape, matrix_inversion_method = "cholesky"), file='NUL')
    cov_pars <- c(0.5052794, 1.2018843, 0.5230190)
    aux_pars <- c(0.9820493)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 279)
    # Also estimate shape parameter with lbfgs
    params_shape$optimizer_cov <- "lbfgs"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_shape, matrix_inversion_method = "cholesky"), file='NUL')
    cov_pars <- c(0.5052899481, 1.2018984119, 0.5230376096)
    aux_pars <- c(0.9820532321 )
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-123.5177217), TOLERANCE_STRICT)
    # Also estimate shape parameter with gradient descent using internal initialization
    params_shape_no_init <- params_shape
    params_shape_no_init$init_aux_pars <- NULL
    params_shape_no_init$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_shape_no_init, matrix_inversion_method = "cholesky"), file='NUL')
    cov_pars <- c(0.5064068, 1.2028118, 0.5355322)
    aux_pars <- c(0.9826897)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 34)
    # fix some covariance parameters
    params_loc <- params_shape
    params_loc$optimizer_cov = "lbfgs"
    params_loc$estimate_cov_par_index <- c(0,0,1)
    params_loc$init_cov_pars <- c(1,1,1)
    cov_pars_fix <- c(1,1,0.5122295)
    aux_pars_fix <- 0.9857217
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_loc, matrix_inversion_method = "cholesky"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))[1:2]-params_loc$init_cov_pars[1:2])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_fix)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_fix)),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = "gamma",
                                           y = y, params = params_loc, matrix_inversion_method = "iterative"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))[1:2]-params_loc$init_cov_pars[1:2])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_fix)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_fix)),TOLERANCE_LOOSE)

    # Gaussian process model
    mu <- exp(L %*% b_1)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.435), scale = mu/shape, shape = shape)
    params_gp <- params
    params_gp$init_cov_pars <- c(1,mean(dist(coords))/3)
    params_shape_gp <- params_shape
    params_shape_gp$init_cov_pars <- c(1,mean(dist(coords))/3)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "gamma", y = y, params = params_gp)
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(1.0649277352, 0.2738906496))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    predict_cov_mat = TRUE, predict_response = FALSE, cov_pars = c(1,0.3))
    expected_mu <- c(0.3402190964, 0.3032421536, 0.8049749290)
    expected_cov <- c(0.4115761683055, 0.3656963345817, -0.0002730842313, 0.3656963345817, 0.4093969989207, -0.0002761274978, -0.0002730842313, -0.0002761274978, 0.3034576099586)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-154.4561783),TOLERANCE_STRICT)
    # Also estimate shape parameter
    params_shape_gp$optimizer_cov <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "gamma", y = y, params = params_shape_gp)
                    , file='NUL')
    cov_pars <- c(1.0445949478, 0.2971884204)
    aux_pars <- c(0.9400943304)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 115)
    # Also estimate shape parameter with gradient descent
    params_shape_gp$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "gamma", y = y, params = params_shape_gp)
                    , file='NUL')
    cov_pars <- c(1.0323441289, 0.2898716638)
    aux_pars <- c(0.9413081183)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 26)
    # fix some covariance parameters
    params_loc <- params_shape_gp
    params_loc$optimizer_cov = "lbfgs"
    params_loc$estimate_cov_par_index <- c(0,0)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "gamma", y = y, params = params_loc), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))[1:2]-params_loc$init_cov_pars[1:2])),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(1,0.1786481))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-0.9902641)),TOLERANCE_STRICT)

    params_loc$init_aux_pars <- 2
    params_loc$fitc_piv_chol_preconditioner_rank <- 60
    params_loc$num_rand_vec_trace = 100
    for (estimate_cov_par_index in list(c(0,0),c(1,0),c(0,1))) {
      params_loc$estimate_cov_par_index <- estimate_cov_par_index
      for(gp_approx in c("none", "vecchia", "full_scale_vecchia", "fitc")) {
        for (matrix_inversion_method in c("cholesky", "iterative")) {
          if (matrix_inversion_method == "iterative" && !(gp_approx %in% c("vecchia", "full_scale_vecchia"))) {
            next
          }
          capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                 likelihood = "gamma", y = y, params = params_loc, gp_approx=gp_approx,
                                                 num_ind_points =50, num_neighbors = 20, matrix_inversion_method = matrix_inversion_method), file='NUL')
          ind_not <- which(estimate_cov_par_index == 0)
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))[ind_not]-params_loc$init_cov_pars[ind_not])),TOLERANCE_STRICT)
          if (length(ind_not) == 2) {
            expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-1)),0.05)
          }
        }
      }
    }

    ## Grouped random effects model with a linear predictor
    params_shape$init_cov_pars <- params$init_cov_pars <- NULL
    mu_lin <- exp(Z1 %*% b_gr_1 + X%*%beta)
    y_lin <- qgamma(sim_rand_unif(n=n, init_c=0.532), scale = mu_lin/shape, shape = shape)
    gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                           y = y_lin, X=X, params = params)
    cov_pars <- c(0.4758553032)
    coef <- c(-0.07265257707, 1.89902842379)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    # Also estimate shape parameter
    params_shape$optimizer_cov <- "nelder_mead"
    params_shape$optimizer_coef <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                                           y = y_lin, X=X, params = params_shape), file='NUL')
    cov_pars <- c(0.5097316)
    coef <- c(-0.08623548, 1.90033132)
    aux_pars <- c(1.350364)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 256)
    # Also estimate shape parameter with gradient descent
    params_shape$optimizer_cov <- "gradient_descent"
    params_shape$optimizer_coef <- "gradient_descent"
    gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                           y = y_lin, X=X, params = params_shape)
    cov_pars <- c(0.5143204465 )
    coef <- c(-0.08618963691, 1.90049146102)
    aux_pars <- c(1.35146943)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_MEDIUM)

    ## Combined grouped random effects and Gaussian process model
    mu <- exp(L %*% b_1 + Z1 %*% b_gr_1)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.987), scale = mu/shape, shape = shape)
    params_cb <- params
    params_cb$init_cov_pars <- c(1,1,mean(dist(coords))/3)
    params_shape_cb <- params_shape
    params_shape_cb$init_cov_pars <- c(1,1,mean(dist(coords))/3)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           group_data = group, likelihood = "gamma", y = y, params = params_cb)
                    , file='NUL')
    cov_pars <- c(0.56752917723, 0.62600814972, 0.08320722994)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    group_test <- c(1,3,3)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred=group_test,
                    predict_cov_mat = TRUE, predict_response = FALSE, cov_pars = c(0.5,0.6,0.1))
    expected_mu <- c(0.25198471751, -0.69948330411, 0.09027905426)
    expected_cov <- c(0.593567787342, 0.420156335862, 0.007140959113, 0.420156335862, 0.630682357298, 0.119642413505, 0.007140959113, 0.119642413505, 0.474857495120)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.6,0.9,0.2),y=y)
    expect_lt(abs(nll-123.3965559),TOLERANCE_STRICT)
    # Also estimate shape parameter with gradient descent
    params_shape_cb$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           group_data = group, likelihood = "gamma", y = y, params = params_shape_cb)
                    , file='NUL')
    cov_pars <- c(0.62143448, 0.98703748, 0.07443428)
    aux_pars <- c(1.707991)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 27)

    # Gaussian process model with Vecchia approximation
    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative"){
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
      } else{
        tolerance_loc_1 <- 0.01
        tolerance_loc_2 <- 0.01
      }
      mu <- exp(0.75 * L %*% b_1)
      y <- qgamma(sim_rand_unif(n=n, init_c=0.7654), scale = mu/shape, shape = shape)
      params$init_cov_pars <- c(1,mean(dist(coords))/3)
      # Estimation
      if(inv_method=="iterative"){
        params$cg_delta_conv = 1e-6
        params$num_rand_vec_trace=500
      }
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             likelihood = "gamma", y = y, params = params,
                                             gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering = "random",
                                             matrix_inversion_method = inv_method), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(0.94860599912, 0.07302133047))),tolerance_loc_2)
      if(inv_method != "iterative"){
        expect_lt(gp_model$get_num_optim_iter(), 7)
        expect_gt(gp_model$get_num_optim_iter(), 4)
      }
      # Prediction
      coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
      gp_model$set_prediction_data(nsim_var_pred = 10000)
      pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                      predict_cov_mat = TRUE, predict_response = FALSE, cov_pars = c(1,0.1))
      expected_mu <- c(-0.1635515155, -0.1513173578, -0.2696781117)
      expected_cov <- c( 7.535277673e-01, 1.531939015e-01, -4.980911538e-06, 1.531939015e-01, 7.492727114e-01, -4.779232329e-06, -4.980911538e-06, -4.779232329e-06, 6.259940393e-01)
      expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
      adjust_tol <- 2
      if (inv_method == "iterative") adjust_tol <- 1.5
      expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),adjust_tol*tolerance_loc_1)
      # Evaluate approximate negative marginal log-likelihood
      nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
      nll_exp <- 159.9221359
      if(inv_method=="iterative"){
        expect_lt(abs(nll-nll_exp),0.4)
      } else{
        expect_lt(abs(nll-nll_exp),0.05)
      }
      # Also estimate shape parameter
      params_shape$optimizer_cov <- "nelder_mead"
      params_shape$init_cov_pars <- c(1,mean(dist(coords))/3)
      if(inv_method=="iterative"){
        params_shape$cg_delta_conv = 1e-6
        params_shape$num_rand_vec_trace=500
      }
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             likelihood = "gamma", y = y, params = params_shape,
                                             gp_approx = "vecchia", matrix_inversion_method = inv_method,
                                             num_neighbors = 30, vecchia_ordering = "random")
                      , file='NUL')
      cov_pars <- c(1.14184253, 0.03605877)
      aux_pars <- c(1.328749)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_2)
      # Also estimate shape parameter with gradient descent
      params_shape$optimizer_cov <- "gradient_descent"
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             likelihood = "gamma", y = y, params = params_shape,
                                             gp_approx = "vecchia", matrix_inversion_method = inv_method,
                                             num_neighbors = 30, vecchia_ordering = "random")
                      , file='NUL')
      cov_pars <- c(1.13722505, 0.03706853)
      aux_pars <- c(1.321834)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_2)
      expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),tolerance_loc_2)
      if(inv_method!="iterative"){
        expect_gt(gp_model$get_num_optim_iter(), 55)
        expect_lt(gp_model$get_num_optim_iter(), 60)
      }
    }
  }) # end Gamma regression

  test_that("negative binomial regression ", {

    params <- OPTIM_PARAMS_BFGS
    params$estimate_aux_pars <- TRUE
    params$init_aux_pars <- 1.
    params_shape <- params
    params_shape$estimate_aux_pars <- TRUE
    shape <- 1.8
    likelihood <- "negative_binomial"

    # Single level grouped random effects
    mu <- exp(Z1 %*% b_gr_1)
    y <- qnbinom(sim_rand_unif(n=n, init_c=0.156), mu = mu, size = shape)
    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, params = params)
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.3369416592)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-1.735168729)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-145.0521408)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1871783331, -0.4055575401, -0.4055575401, 0.0000000000)
    expected_cov <- c(0.09699323301, 0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
                      0.13423334750, 0.13423334750, 0.00000000000, 0.00000000000, 0.13423334750,
                      0.13423334750, 0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000, 0.33694165920)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(1.265762807, 0.7128809334, 0.7128809334, 1.183493703)
    expected_var <- c(2.44633493, 1.120845684, 1.120845684, 2.875311496)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_MEDIUM)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-145.8340641),TOLERANCE_MEDIUM)
    # Estimation with "nelder_mead"
    params_shape$optimizer_cov <- "nelder_mead"
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, params = params_shape), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.33714316)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-1.73506598)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 46)

    # Multiple random effects
    mu <- exp(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- qnbinom(sim_rand_unif(n=n, init_c=0.1468), mu = mu, size = shape)
    params_shape$init_cov_pars <- params$init_cov_pars <- rep(1,3)
    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = likelihood,
                                           y = y, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    cov_pars <- c(0.5427548465, 2.667802488, 0.6444668618)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-2.386787856)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-170.1430598)),TOLERANCE_STRICT)
    # Prediction
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(0.3670135621, -1.632614919, 0.000000000)
    expected_var <- c(0.2679508409, 0.3941603558, 1.8080000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # Also estimate shape parameter with gradient descent using internal initialization
    params_shape_no_init <- params_shape
    params_shape_no_init$init_aux_pars <- NULL
    params_shape_no_init$optimizer_cov <- "gradient_descent"
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, likelihood = likelihood,
                                           y = y, params = params_shape_no_init, matrix_inversion_method = "cholesky"), file='NUL')
    cov_pars <- c(0.5420803744, 2.667356769, 0.646631823)
    aux_pars <- c(2.386915728)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 10)

    # Gaussian process model
    mu <- exp(L %*% b_1)
    y <- qnbinom(sim_rand_unif(n=n, init_c=0.546), mu = mu, size = shape)
    params_gp <- params
    params_gp$init_cov_pars <- c(1,mean(dist(coords))/3)
    params_shape_gp <- params_shape
    params_shape_gp$init_cov_pars <- c(1,mean(dist(coords))/3)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = likelihood, y = y, params = params_gp)
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(1.324833289, 0.1613527915))),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-1.093074798)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-179.9164837)),TOLERANCE_STRICT)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.3960494203, -0.3888316723, 0.5173410546)
    expected_var <- c(0.9060617929, 0.8997384656, 0.5903066685)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-180.5949853),TOLERANCE_STRICT)

    ## Grouped random effects model with a linear predictor
    params_shape$init_cov_pars <- params$init_cov_pars <- NULL
    mu_lin <- exp(Z1 %*% b_gr_1 + X%*%beta)
    y_lin <- qnbinom(sim_rand_unif(n=n, init_c=0.13278), mu = mu_lin, size = shape)
    gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                           y = y_lin, X=X, params = params)
    cov_pars <- c(0.2448821863 )
    coef <- c(-0.02576650122, 2.242682035)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 12)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-1.911699282 )),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-177.3531204)),TOLERANCE_STRICT)

    # Gaussian process model with Vecchia approximation
    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative"){
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
      } else{
        tolerance_loc_1 <- 0.01
        tolerance_loc_2 <- 0.01
      }
      mu <- exp(0.75 * L %*% b_1)
      y <- qnbinom(sim_rand_unif(n=n, init_c=0.4819), mu = mu, size = shape)
      # Estimation
      if(inv_method=="iterative"){
        params$cg_delta_conv = 1e-6
        params$num_rand_vec_trace=500
      }
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             likelihood = likelihood, y = y, params = params_gp,
                                             gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering = "random",
                                             matrix_inversion_method = inv_method), file='NUL')
      cov_pars_exp <- c(0.4609968677, 0.11958972)
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tolerance_loc_2)
      expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-1.165048377 )),relax_tolerance(tolerance_loc_2, 1.165048377))
      expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-163.2316193)),relax_tolerance(tolerance_loc_2, 163.2316193))
      # Prediction
      coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
      gp_model$set_prediction_data(nsim_var_pred = 10000)
      pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
      expected_mu <- c(-0.206434751, -0.2358747563, 0.4079959528)
      expected_cov <- c(0.4013543338, 0.1621820831, 6.924098979e-05, 0.1621820831, 0.4044927521,
                        6.996043585e-05, 6.924098979e-05, 6.996043585e-05, 0.2974335868)
      expect_lt(sum(abs(pred$mu-expected_mu)),2*tolerance_loc_1)
      adjust_tol <- 2
      if (inv_method == "iterative") adjust_tol <- 4
      expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),adjust_tol*tolerance_loc_1)
      # Evaluate approximate negative marginal log-likelihood
      nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
      nll_exp <- 164.169506
      if(inv_method=="iterative"){
        expect_lt(abs(nll-nll_exp),2)
      } else{
        expect_lt(abs(nll-nll_exp),0.05)
      }
      # "vecchia_latent"
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             likelihood = likelihood, y = y, params = params_gp,
                                             gp_approx = "vecchia_latent", num_neighbors = 30, vecchia_ordering = "random",
                                             matrix_inversion_method = inv_method), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tolerance_loc_2)

    }
  }) # end negative binomial regression

  test_that("Saving a GPModel and loading from file works for non-Gaussian data", {

    # Binary regression
    probs <- pnorm(Z1 %*% b_gr_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.542) < probs)
    # Train model
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = OPTIM_PARAMS_BFGS)
    # Make predictions
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    # Predict response
    pred_resp <- predict(gp_model, y=y, group_data_pred = group_test,
                         X_pred = X_test, predict_var = TRUE, predict_response = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model,filename = filename)
    # Delete model
    rm(gp_model)
    # Load from file and make predictions again
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, group_data_pred = group_test,
                           X_pred = X_test, predict_cov_mat = TRUE, predict_response = FALSE)
    pred_resp_loaded <- predict(gp_model_loaded, y=y, group_data_pred = group_test,
                                X_pred = X_test, predict_var = TRUE, predict_response = TRUE)

    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    expect_equal(pred_resp$mu, pred_resp_loaded$mu)
    expect_equal(pred_resp$var, pred_resp_loaded$var)

    # Gamma regression
    mu <- exp(Z1 %*% b_gr_1 + X%*%beta)
    y <- qgamma(sim_rand_unif(n=n, init_c=0.146), scale = mu, shape = 10)
    # Train model
    gp_model <- fitGPModel(group_data = group, likelihood = "gamma", y = y, X = X, params = OPTIM_PARAMS_BFGS)
    # Make predictions
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    # Predict response
    pred_resp <- predict(gp_model, y=y, group_data_pred = group_test,
                         X_pred = X_test, predict_var = TRUE, predict_response = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    # Delete model
    rm(gp_model)
    # Load from file and make predictions again
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, group_data_pred = group_test,
                           X_pred = X_test, predict_cov_mat = TRUE, predict_response = FALSE)
    pred_resp_loaded <- predict(gp_model_loaded, y=y, group_data_pred = group_test,
                                X_pred = X_test, predict_var = TRUE, predict_response = TRUE)

    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    expect_equal(pred_resp$mu, pred_resp_loaded$mu)
    expect_equal(pred_resp$var, pred_resp_loaded$var)

    # t likelihood
    # Train model
    gp_model <- fitGPModel(group_data = group, likelihood = "t", y = y, X = X, params = OPTIM_PARAMS_BFGS)
    # Make predictions
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    # Predict response
    pred_resp <- predict(gp_model, y=y, group_data_pred = group_test,
                         X_pred = X_test, predict_var = TRUE, predict_response = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    # Delete model
    rm(gp_model)
    # Load from file and make predictions again
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, group_data_pred = group_test,
                           X_pred = X_test, predict_cov_mat = TRUE, predict_response = FALSE)
    pred_resp_loaded <- predict(gp_model_loaded, y=y, group_data_pred = group_test,
                                X_pred = X_test, predict_var = TRUE, predict_response = TRUE)

    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    expect_equal(pred_resp$mu, pred_resp_loaded$mu)
    expect_equal(pred_resp$var, pred_resp_loaded$var)
  })

}
