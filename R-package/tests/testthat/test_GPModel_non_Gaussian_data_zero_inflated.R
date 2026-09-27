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

  test_that("hurdle_gamma regression ", {

    params <- OPTIM_PARAMS_BFGS
    likelihood <- "hurdle_gamma"

    # Single level grouped random effects
    shape <- 2
    p0 <- 0.4
    eta <- Z1 %*% b_gr_1 + 0.5*X%*%beta
    mu <- exp(eta)
    y <- rep(NA,n)
    zeros <- sim_rand_unif(n=n, init_c=0.237985) <= p0
    y[zeros] <- 0
    y[!zeros] <- qgamma(sim_rand_unif(n=sum(!zeros), init_c=0.9632),
                        rate = shape / mu[!zeros], shape = shape)

    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky")
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(shape, p0))
    expect_lt(abs(nll-183.969936787735),TOLERANCE_STRICT)

    # Label needs to have the correct support
    yt <- y
    yt[100] <- -1e-10
    expect_error(gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                        y = yt, X=X, params = params, matrix_inversion_method = "cholesky"))

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.320275336481987)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_aux_pars()-c(2.44668106228388, 0.41))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(0.110570439418453, 1.14091349210305))),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-149.740800397881)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.496045136495701, 0.487026288640262, 0.611858349604019, 2.42053564764981)
    expected_var <- c(0.378967631666401, 0.398766848901543, 0.629384466420825, 13.4113083877069)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var))/sum(abs(expected_var)),TOLERANCE_STRICT)

    # Setting initial values and saving to file
    params_init <- params
    params_init$init_aux_pars <- c(shape, p0)
    params_init$init_cov_pars <- 1
    params_init$init_coef <- beta
    params_init$maxit <- 0
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params_init, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-1)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_aux_pars()-c(shape, p0))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-beta)),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.318398500262595, 0.267251264012491, 0.398692036129682, 8.07824282100101)
    expected_var <- c(0.175688205479334, 0.142030791688895, 0.316095340009824, 378.216129908876)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    gp_model_loaded <- loadGPModel(filename = filename)
    expect_lt(sum(abs(gp_model_loaded$get_cov_pars(std_err = FALSE)-1)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model_loaded$get_aux_pars()-c(shape, p0))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model_loaded$get_coef(std_err = FALSE))-beta)),TOLERANCE_STRICT)
    pred <- predict(gp_model_loaded, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.336767936372357)),TOLERANCE_MEDIUM)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-0.692053205065586, 0.460314799450466, 0.564020495003286, 1.47672658281785))),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.301097074771284, 0.379042216360261, 0.420461653760034, 3.05713379562945))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.125668266783157, 0.218015764301311, 0.26826592998371, 20.1983185214085))), TOLERANCE_MEDIUM)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              reuse_learning_rates_gp_model = FALSE) )
    expect_lt(sum(abs(cvbst$best_score-1.63537873073932)),TOLERANCE_MEDIUM)
    expect_gte(cvbst$best_iter, 12)
    expect_lte(cvbst$best_iter, 14)

  }) # end hurdle_gamma regression

  test_that("zoctn regression ", {

    params <- OPTIM_PARAMS_BFGS
    likelihood <- "zoctn"

    # Single level grouped random effects
    sd <- 0.5
    a <- -0.5
    b <- 1.2
    mu <- Z1 %*% b_gr_1 + 0.5*X%*%beta
    y <- qnorm(sim_rand_unif(n=n, init_c=0.74), mean = mu, sd = sd)
    logistic <- function(t) 1 / (1 + exp(-t))
    logit    <- function(p) log(p / (1 - p))
    y[y<0] <- 0
    y[y>1] <- 1
    y[y>0 & y<1] <- logistic(a + b*logit(y[y>0 & y<1]))

    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky")
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(sd, a, b))
    expect_lt(abs(nll-116.2406869),TOLERANCE_STRICT)

    # Label needs to have the correct support
    yt <- y
    yt[1] <- -1e-10
    expect_error(gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                        y = yt, X=X, params = params, matrix_inversion_method = "cholesky"))
    yt[1] <- 1+1e-10
    expect_error(gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                        y = yt, X=X, params = params, matrix_inversion_method = "cholesky"))

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    cov_pars <- 0.2916780257
    aux_pars <- c(0.5046217166, -0.7148127765, 1.2386879955)
    coef <- c(0.02781854661, 1.01645519976 )
    nll <- 59.97448286
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_aux_pars()-aux_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 15)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.09604337830, 0.08452576696, 0.14822281001, 0.70876044016)
    expected_var <- c(0.04435684115, 0.03864208307, 0.06746643149, 0.14055331039)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## coef and aux_par initialization with iid model
    params_init <- params
    params_init$init_coef_aux_pars_from_iid_model <- TRUE
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params_init, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(gp_model$get_aux_pars()-aux_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll)),TOLERANCE_MEDIUM)
    gp_model_iid <- fitGPModel(y = y, X = X, likelihood = likelihood, params=params_init)
    params_init$maxit <- 0
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params_init, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-gp_model_iid$get_coef(std_err = FALSE))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-gp_model_iid$get_aux_pars())),TOLERANCE_STRICT)
    # init_aux_pars given and only coefs initialized
    params_init_aux <- params_init
    params_init_aux$init_aux_pars <- aux_pars
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params_init_aux, matrix_inversion_method = "cholesky")
                    , file='NUL')
    params_ref <- params_init_aux
    params_ref$maxit <- 1000
    params_ref$estimate_aux_pars <- FALSE
    params_ref$init_coef_aux_pars_from_iid_model <- NULL
    gp_model_iid_ref <- fitGPModel(y = y, X = X, likelihood = likelihood, params = params_ref)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-gp_model_iid_ref$get_coef(std_err = FALSE))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)
    params_init_coef <- list(maxit = 0, init_coef = c(0.2, 0.3), init_aux_pars = aux_pars,
                             init_coef_aux_pars_from_iid_model = TRUE)
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params_init_coef, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-params_init_coef$init_coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_STRICT)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.3189194079)),TOLERANCE_MEDIUM)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.17448221639, 0.04343393005, 0.05223871382, 0.83825928407))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.06533991097, 0.01486182361, 0.01823024348, 0.08730518547))), TOLERANCE_MEDIUM)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              deterministic = TRUE) )
    expect_lte(cvbst$best_score,0.788817272181965*(1+0.05))
    expect_gte(cvbst$best_score,0.788817272181965*(1-0.05))
    expect_lte(cvbst$best_iter, 11)
    expect_gte(cvbst$best_iter, 9)

  }) # end zoctn regression

  test_that("zero_one_censored_transformed_beta regression ", {

    params <- OPTIM_PARAMS_BFGS
    likelihood <- "zero_one_censored_transformed_beta"

    # Single level grouped random effects
    sd <- 0.5
    phi <- 20
    u <- 0.15
    mu <- Z1 %*% b_gr_1 + 0.5*X%*%beta
    p <- 1 / (1 + exp(-mu))
    y <- qbeta(sim_rand_unif(n=n, init_c=0.23474), shape1 = p * phi, shape2 = (1 - p) * phi)
    y <- -u + (1+2*u) * y
    y[y<0] <- 0
    y[y>1] <- 1

    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky")
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(phi, u))
    expect_lt(abs(nll-54.04809846),3e-5)

    # The censoring probabilities are the beta CDF at t0 and t1. With an essentially zero random effect
    #   variance, the negative log-likelihood has to agree with the censored beta log-likelihood
    #   evaluated at a zero random effect, which is calculated here with 'pbeta' and 'dbeta'
    mu_beta <- 0.4
    phi_beta <- 8
    u_beta <- 0.4
    a_beta <- mu_beta * phi_beta
    b_beta <- (1 - mu_beta) * phi_beta
    t0_beta <- u_beta / (1 + 2 * u_beta)
    t1_beta <- (1 + u_beta) / (1 + 2 * u_beta)
    y_beta <- c(0, 0, 1, 1, 0.2, 0.5, 0.8, 0.95)
    t_beta <- (y_beta + u_beta) / (1 + 2 * u_beta)
    ll_beta <- sum(ifelse(y_beta <= 0, log(pbeta(t0_beta, a_beta, b_beta)),
                          ifelse(y_beta >= 1, log(1 - pbeta(t1_beta, a_beta, b_beta)),
                                 dbeta(t_beta, a_beta, b_beta, log = TRUE) - log(1 + 2 * u_beta))))
    gp_model_beta <- GPModel(group_data = 1:length(y_beta), likelihood = likelihood,
                             matrix_inversion_method = "cholesky")
    nll_beta <- gp_model_beta$neg_log_likelihood(cov_pars = c(1e-12), y = y_beta,
                                                 aux_pars = c(phi_beta, u_beta),
                                                 fixed_effects = rep(log(mu_beta / (1 - mu_beta)), length(y_beta)))
    expect_lt(abs(nll_beta + ll_beta), 1e-6)

    # Label needs to have the correct support
    yt <- y
    yt[1] <- -1e-10
    expect_error(gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                        y = yt, X=X, params = params, matrix_inversion_method = "cholesky"))
    yt[1] <- 1+1e-10
    expect_error(gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                        y = yt, X=X, params = params, matrix_inversion_method = "cholesky"))

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    # Note: the log-likelihood of this model is very flat in the precision parameter, and the optimizer
    #   stops at a different point along that direction on another compiler. The negative log-likelihood
    #   is almost unchanged there, the estimates are not: with gcc the covariance parameter deviates by
    #   0.013 and the regression coefficients by 0.039
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.2682095671)),relax_tolerance(0.01, 0.2682095671))
    expect_lt(sum(abs(gp_model$get_aux_pars()-c(22.879799528, 0.168605624))),relax_tolerance(0.6, c(22.879799528, 0.168605624)))
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.11240688283, 0.88192071991))),relax_tolerance(0.008, c(-0.11240688283, 0.88192071991)))
    nll <- -44.08117687
    # Along that flat direction the optimizer also stops slightly differently with another number of
    # threads: 0.0038 has been measured on the reference platform itself
    expect_lt(sum(abs((gp_model$get_current_neg_log_likelihood()-nll))),relax_tolerance_nll(0.01))
    expect_gt(gp_model$get_num_optim_iter(), 0)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.3900424970, 0.3251828138, 0.3809477867, 0.7292149088)
    expected_var <- c(0.01993931983, 0.01913466436, 0.02000007302, 0.03469011762)
    # see the note on the precision parameter above: 0.006 has been measured on the Linux CI
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01, expected_mu))
    # 0.00081 has been measured with clang + libc++ in the clang-asan container of R-hub
    expect_lt(sum(abs(pred$var-expected_var)),relax_tolerance(0.0008, expected_var))

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    # Note: with the default convergence tolerance, the covariance parameter estimated here depends on the order in
    #   which floating point numbers are summed, i.e. on the number of OpenMP threads. A tighter tolerance makes the
    #   result essentially thread-independent, but not bit-identical: deviations of up to about 0.01 have been
    #   observed for the covariance parameter and the predicted means below, which the tolerances have to accommodate
    gp_model$set_optim_params(params=modifyList(OPTIM_PARAMS_BFGS, list(delta_rel_conv = 1e-10)))
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    # The predicted values are only compared with the expected values on the reference platform: they react
    #   much more sensitively to the summation order than the estimate itself
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE) - 0.0972135292)), relax_tolerance(0.05, 0.0972135292))
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    if (USE_STRICT_TOLERANCES) {
      expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.3781333136, 0.3284435388, 0.1879960730, 0.7152683344))),0.05)
      expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.015931476492, 0.015692558118, 0.013436139072, 0.033427809073))), 0.05)
    } else {
      # Note: the response is censored at 0 and 1, so a predicted variance can be 0
      expect_true(all(is.finite(pred$response_mean)) &&
                    all(pred$response_mean >= 0) && all(pred$response_mean <= 1),
                  info = paste(pred$response_mean, collapse = ", "))
      expect_true(all(is.finite(pred$response_var)) && all(pred$response_var >= 0),
                  info = paste(pred$response_var, collapse = ", "))
    }

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              deterministic = TRUE) )
    expect_lte(cvbst$best_score,-0.309881596335*0.5)
    expect_gte(cvbst$best_score,-0.309881596335*2)
    # Note: which iteration is selected here depends on validation scores that differ in the last digits between
    #   runs with different numbers of OpenMP threads, so only a range is checked
    expect_lte(cvbst$best_iter, 10)
    expect_gte(cvbst$best_iter, 4)

  }) # end zero_one_censored_transformed_beta regression

  test_that("zero_one_censored_shifted_gamma regression ", {

    params <- OPTIM_PARAMS_BFGS
    likelihood <- "zero_one_censored_shifted_gamma"

    # Single level grouped random effects
    shape <- 5
    xi <- 0.1
    scale <- exp(Z1 %*% b_gr_1 + 0.25*X%*%beta) / shape
    y <- qgamma(sim_rand_unif(n=n, init_c=0.1346), scale = scale, shape = shape)
    y <- y - xi
    y[y<0] <- 0
    y[y>1] <- 1

    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky")
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(shape,xi))
    expect_lt(abs(nll-76.53696381),TOLERANCE_STRICT)

    # Label needs to have the correct support
    yt <- y
    yt[1] <- -1e-10
    expect_error(gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                        y = yt, X=X, params = params, matrix_inversion_method = "cholesky"))
    yt[1] <- 1+1e-10
    expect_error(gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                        y = yt, X=X, params = params, matrix_inversion_method = "cholesky"))

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.3549807283)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_aux_pars()-c(4.1078363130, 0.1028633593))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.1336940622, 0.6941017071))),TOLERANCE_STRICT)
    expect_lt(sum(abs((gp_model$get_current_neg_log_likelihood()-36.60875527))),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 21)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.4995770821, 0.6219404142, 0.6904084431, 0.8666146253)
    expected_var <- c(0.07514258820, 0.08229697620, 0.07972402790, 0.05700373880)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.1980027810 )),TOLERANCE_LOOSE)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.6729528034, 0.6549747141, 0.6407357560, 0.7476455234))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.06386184510, 0.06425764320, 0.06445518680, 0.08349093790))), TOLERANCE_LOOSE)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              deterministic = TRUE) )
    expect_lte(cvbst$best_score,0.7915884743*(1+TOLERANCE_LOOSE))
    expect_gte(cvbst$best_score,0.7915884743*(1-TOLERANCE_LOOSE))
    nit <- 5
    expect_lte(cvbst$best_iter, nit+4)
    expect_gte(cvbst$best_iter, nit-1)

  }) # end zero_one_censored_shifted_gamma regression

}
