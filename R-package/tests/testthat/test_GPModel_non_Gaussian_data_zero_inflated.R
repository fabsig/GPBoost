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

  test_that("iid model ", {

    params <- OPTIM_PARAMS_BFGS
    y <- X %*% beta + qnorm(sim_rand_unif(n=n, init_c=0.91468), sd=sqrt(0.01))
    likelihood <- "gaussian"

    # Estimation
    capture.output( gp_model <- fitGPModel(likelihood = likelihood, X=X, y = y, params = params) , file='NUL')
    cov_pars_exp <- c(7.654507e-03, 1.000000e-20)
    coef_exp <- c(0.094720436, 0.008837829, 1.987728662, 0.012498577)
    nll_opt_exp <- -101.7291793
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    # Prediction
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, X_pred = X_test, predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-0.8991438945,  0.4922661688,  2.0824490983)
    expected_var <- c(1e-20, 1e-20, 1e-20)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    # mod <- lm(y~X2, data=data.frame(y=y,X))
    # summary(mod)
    # predict(mod, newdata=data.frame(X_test))

    likelihood <- "t_fix_df"
    capture.output( gp_model <- fitGPModel(likelihood = likelihood, X=X, y = y, params = params) , file='NUL')
    aux_pars_exp <- c(0.0652430469, 2)
    coef_exp <- c(0.094283734360, 0.009319580548, 1.992402552983, 0.011695985542)
    nll_opt_exp <- -92.6701562
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 21)
    # Prediction
    pred <- predict(gp_model, X_pred = X_test, predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-0.9019175421, 0.4927642450, 2.0866862873)
    expected_var <- c(1e-20, 1e-20, 1e-20)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    likelihood <- "binary_logit"
    y_bin <- as.numeric(sim_rand_unif(n=n, init_c=0.468) < 1/(1+exp(-X %*% beta)))
    capture.output( gp_model <- fitGPModel(likelihood = likelihood, X=X, y = y_bin, params = params) , file='NUL')
    coef_exp <- c(0.08910433727, 0.22947935529, 1.57411916970, 0.35649689071)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-56.6742427)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 4)
    # Prediction
    pred <- predict(gp_model, X_pred = X_test, predict_var=TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-c(-0.6979552476, 0.4039281712, 1.6632235070))),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-c(1e-20, 1e-20, 1e-20))),TOLERANCE_STRICT)
    pred_resp <- predict(gp_model, X_pred = X_test, predict_var=TRUE, predict_response = TRUE)
    pred_exp <- c(0.3322656738, 0.5996311078, 0.8406703427)
    expect_lt(sum(abs(pred_resp$mu-pred_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_resp$var-pred_exp*(1-pred_exp))),TOLERANCE_STRICT)

    # mod <- glm(y~X2, data=data.frame(y=y_bin,X), family = binomial(link = "logit"))
    # summary(mod)
    # predict(mod, newdata=data.frame(X_test))

    likelihood <- "gamma"
    capture.output( gp_model <- fitGPModel(likelihood = likelihood, X=X, y = exp(y), params = params) , file='NUL')
    coef_exp <- c(0.098623234, 0.008821832, 1.986899634, 0.012429806)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-131.0965634)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()--72.4258)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 30)
    # Prediction
    pred <- predict(gp_model, X_pred = X_test, predict_var=TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-c(-0.8948265830, 0.4960031607, 2.0855228678))),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-c(1e-20, 1e-20, 1e-20))),TOLERANCE_STRICT)
    pred_resp <- predict(gp_model, X_pred = X_test, predict_var=TRUE, predict_response = TRUE)
    pred_exp <- c(0.4086784643, 1.6421447481, 8.0487988395)
    expect_lt(sum(abs(pred_resp$mu-pred_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_resp$var-c(0.001274008127, 0.020569870819, 0.494163699509))),TOLERANCE_STRICT)

    # mod <- glm(y~X2, data=data.frame(y=exp(y),X), family = Gamma(link = "log"))
    # summary(mod)
    # predict(mod, newdata=data.frame(X_test), type ="response")

  }) # end iid model

  test_that("asymmetric_laplace likelihood ", {

    params <- OPTIM_PARAMS_BFGS
    likelihood <- "asymmetric_laplace"

    quantile_asym_laplace <- function(q, alpha, lambda) {
      if (length(q) == 1) {
        if (q <= alpha) {
          return(log(q/alpha) * lambda / (1-alpha))
        } else {
          return(-log((1-q)/(1-alpha)) * lambda / alpha)
        }
      } else {
        res <- rep(NA,length(q))
        ind <- q <= alpha
        res[ind] <- log(q[ind]/alpha) * lambda / (1-alpha)
        res[!ind] <- -log((1-q[!ind])/(1-alpha)) * lambda / alpha
        return(res)
      }
    }

    # Single level grouped random effects
    quantile <- 0.5
    quantile_up <- 0.975
    lambda = 0.25
    error <- quantile_asym_laplace(q=sim_rand_unif(n=n, init_c=0.651), alpha=quantile, lambda = lambda)
    y <- Z1 %*% b_gr_1 + X%*%beta + error

    matrix_inversion_method <- "cholesky"
    # matrix_inversion_method_loop <- c("cholesky", "iterative")
    # for (matrix_inversion_method in matrix_inversion_method_loop) {
    if(matrix_inversion_method == "iterative") {
      tolerance_loc_1 <- TOLERANCE_STRICT
      tolerance_loc_2 <- TOLERANCE_LOOSE
      tolerance_loc_3 <- 0.1
    } else {
      tolerance_loc_1 <- TOLERANCE_STRICT
      tolerance_loc_2 <- TOLERANCE_LOOSE
      tolerance_loc_3 <- 0.1
    }

    expect_error(GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = matrix_inversion_method,
                         likelihood_additional_param = 1.1), "must be a quantile q with 0 < q < 1", fixed = TRUE)
    expect_error(GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = matrix_inversion_method,
                         likelihood_additional_param = -0.1), "must be a quantile q with 0 < q < 1", fixed = TRUE)
    expect_error(GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = matrix_inversion_method,
                         likelihood_additional_param = NA_real_), "must be a finite quantile q with 0 < q < 1", fixed = TRUE)
    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    nll_exp <- 273.0138019
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(lambda))
    expect_lt(abs(nll-nll_exp),tolerance_loc_1)
    expect_error(GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = matrix_inversion_method),
                 "No value was provided for 'likelihood_additional_param'", fixed = TRUE)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile_up)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(lambda))
    expect_lt(abs(nll-302.8484089),tolerance_loc_1)
    gp_model <- GPModel(group_data = group, likelihood = "asymmetric_laplace_tkc",
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    nll_exp2 <- 271.2555943
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(lambda))
    expect_lt(abs(nll-nll_exp2),tolerance_loc_1)
    gp_model <- GPModel(group_data = group, likelihood = "asymmetric_laplace_tkc_fisher_mode_finding",
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(lambda))
    expect_lt(abs(nll-nll_exp2),tolerance_loc_1)
    gp_model <- GPModel(group_data = group, likelihood = "asymmetric_laplace_tkc_not_fisher_mode_finding",
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y, aux_pars = c(lambda))
    expect_lt(abs(nll-270.8276752),tolerance_loc_1)

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                           y = y, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.8153285415)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model$get_aux_pars()-0.2688162279 )),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.3044197085, 2.0765502256))),tolerance_loc_1)
    # no standard errors are calculated for quantile regression (the approximate marginal likelihood is a
    #   pseudo-likelihood which is not smooth)
    expect_false(gp_model$can_calculate_standard_errors_coef())
    expect_false(gp_model$can_calculate_standard_errors_cov_pars())
    expect_false(gp_model$can_calculate_standard_errors_aux_pars())
    expect_equal(gp_model$get_coef(std_err = TRUE), gp_model$get_coef(std_err = FALSE))
    expect_equal(gp_model$get_cov_pars(std_err = TRUE), gp_model$get_cov_pars(std_err = FALSE))
    expect_equal(gp_model$get_aux_pars(std_err = TRUE), gp_model$get_aux_pars(std_err = FALSE))
    expect_lt(sum(abs((gp_model$get_current_neg_log_likelihood()-117.1840987))),tolerance_loc_1)
    expect_equal(gp_model$get_num_optim_iter(), 12)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-1.3426948213, -0.2126176767, 0.2026923684, 1.7721305171)
    expected_var <- c(0.02791522088, 0.02791522088, 0.02791522088, 0.81532854155)
    expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
    expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_1)

    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace_var_cor_pred_freq_asym", likelihood_additional_param = quantile,
                                           y = y, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
    expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_1)

    # Estimation with other options
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace_tkc", likelihood_additional_param = quantile,
                                           y = y, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.8230834628)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model$get_aux_pars()-0.2712208559  )),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.1344325423,  2.0358682043))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model$get_current_neg_log_likelihood()-116.1218566))),tolerance_loc_1)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-1.1523666445, -0.1685696229, 0.2386040180, 1.9014356621)
    expected_var <- c(0.03667225147, 0.03667225147, 0.03667225147, 0.82308346280)
    expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
    expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_1)
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace_tkc_not_fisher_mode_finding", likelihood_additional_param = quantile,
                                           y = y, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.7758838459)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model$get_aux_pars()-0.2545291278 )),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.0654253707, 2.0782768412))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model$get_current_neg_log_likelihood()-114.9476588))),tolerance_loc_1)
    expect_equal(gp_model$get_num_optim_iter(), 15)
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace_tkc_var_cor_pred_freq_asym", likelihood_additional_param = quantile,
                                           y = y, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_var_cor <- c(0.02840872148, 0.02840872148, 0.02840872148, 0.82308346280)
    expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
    expect_lt(sum(abs(pred$var-expected_var_cor)),tolerance_loc_1)

    # Initializing coefficients and auxiliary parameters from an iid model
    #   (this is the default, all fits above use 'init_coef_aux_pars_from_iid_model = FALSE')
    params_init_iid <- params
    params_init_iid$init_coef_aux_pars_from_iid_model <- TRUE
    capture.output( gp_model_iid_init <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                    y = y, X=X, params = params_init_iid, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_iid_init$get_cov_pars(std_err = FALSE)-0.4132224030)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_iid_init$get_aux_pars()-0.2690114464)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_iid_init$get_coef(std_err = FALSE))-c(-0.0468007689, 2.0773708217))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_iid_init$get_current_neg_log_likelihood()-116.2091326))),tolerance_loc_1)
    # the initialization from an iid model finds a better optimum than the one from the marginal sample quantile alone
    expect_lt(gp_model_iid_init$get_current_neg_log_likelihood(), 117.1840987)

    # Restarts of lbfgs ('max_num_restarts_lbfgs'). The approximate marginal likelihood of the 'asymmetric_laplace'
    #   likelihood is not smooth. The line search of lbfgs can thus fail, in which case lbfgs terminates without
    #   having converged (this happens for the fits above) and the variance of the random effects is estimated too large
    # "Cold" restarts (the default): the regression coefficients, the auxiliary parameters, and the modes are reset
    #   to their initial values and only the covariance parameters are kept
    params_restart <- params
    params_restart$max_num_restarts_lbfgs <- 4L
    capture.output( gp_model_restart <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                   y = y, X=X, params = params_restart, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_restart$get_cov_pars(std_err = FALSE)-0.4043949065)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_restart$get_aux_pars()-0.2691821831)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_restart$get_coef(std_err = FALSE))-c(-0.1675955800, 2.0823192468))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_restart$get_current_neg_log_likelihood()-116.0987752))),tolerance_loc_1)
    # the restarts find a better optimum than the fit without restarts (nll = 117.1840987, cov_par = 0.8153285415)
    expect_lt(gp_model_restart$get_current_neg_log_likelihood(), 117.1840987)
    # "Warm" restarts: the optimization simply continues from the current parameters with a re-initialized approximate
    #   Hessian. For the data below, the restarts do not find a better optimum (this is not the case in general)
    params_restart$cold_restart_lbfgs <- FALSE
    capture.output( gp_model_restart <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                   y = y, X=X, params = params_restart, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_restart$get_cov_pars(std_err = FALSE)-0.8144100464)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_restart$get_aux_pars()-0.2688516601)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_restart$get_coef(std_err = FALSE))-c(-0.3029737374, 2.0753408270))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_restart$get_current_neg_log_likelihood()-117.1792005))),tolerance_loc_1)
    # no restarts are done by default -> same results as above (for both 'cold_restart_lbfgs' options)
    params_restart$cold_restart_lbfgs <- TRUE
    params_restart$max_num_restarts_lbfgs <- 0L
    capture.output( gp_model_restart <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                   y = y, X=X, params = params_restart, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_restart$get_cov_pars(std_err = FALSE)-0.8153285415)),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_restart$get_current_neg_log_likelihood()-117.1840987))),tolerance_loc_1)
    expect_error(fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                            y = y, X=X, params = list(max_num_restarts_lbfgs = -1L)), "max_num_restarts_lbfgs is not >= 0", fixed = TRUE)

    # Line search of Nocedal and Wright (the default line search of lbfgs is a backtracking one). In contrast to the
    #   backtracking line search, this line search can return the best point found so far instead of the point that
    #   has been evaluated last (namely if the strong Wolfe condition is not satisfied when the maximal number of line
    #   search iterations is reached). The modes of the Laplace approximations then need to be restored accordingly,
    #   otherwise they correspond to a point that has been rejected by the line search (see 'SaveModesLo()' in
    #   optim_utils.h). This matters in particular for non-smooth likelihoods such as this one, for which mode finding
    #   is start-dependent
    params_nw <- params
    params_nw$optimizer_cov <- "lbfgs_linesearch_nocedal_wright"
    params_nw$optimizer_coef <- "lbfgs_linesearch_nocedal_wright"
    capture.output( gp_model_nw <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                              y = y, X=X, params = params_nw, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_nw$get_cov_pars(std_err = FALSE)-0.8064792556)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_nw$get_aux_pars()-0.2679067269)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_nw$get_coef(std_err = FALSE))-c(-0.2678415466, 2.0747656120))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_nw$get_current_neg_log_likelihood()-117.1057618))),tolerance_loc_1)
    expect_equal(gp_model_nw$get_num_optim_iter(), 13)
    # this line search finds a better optimum than the backtracking one for the data below (nll = 117.1840987)
    expect_lt(gp_model_nw$get_current_neg_log_likelihood(), 117.1840987)

    # Non-zero true intercept: the initial intercept is the marginal sample quantile of y and estimation is
    #   thus equivariant under a location shift of y (the results below are those of the fits above with the
    #   intercept shifted by 'shift'). Note: when initializing the intercept with zero (which was done before),
    #   the intercept stays far away from its true value and the estimated variance of the random effects is
    #   much too large (approx. 42 instead of approx. 0.82 for the data below)
    shift <- 9.9 # true intercept becomes 0.1 + 9.9 = 10
    y_shift <- y + shift
    capture.output( gp_model_shift <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                 y = y_shift, X=X, params = params, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_shift$get_cov_pars(std_err = FALSE)-0.8153285415)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_shift$get_aux_pars()-0.2688162279)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_shift$get_coef(std_err = FALSE))-c(-0.3044197085 + shift, 2.0765502256))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_shift$get_current_neg_log_likelihood()-117.1840987))),tolerance_loc_1)
    # same when initializing from an iid model
    capture.output( gp_model_shift <- fitGPModel(group_data = group, likelihood = likelihood, likelihood_additional_param = quantile,
                                                 y = y_shift, X=X, params = params_init_iid, matrix_inversion_method = matrix_inversion_method)
                    , file='NUL')
    expect_lt(sum(abs(gp_model_shift$get_cov_pars(std_err = FALSE)-0.4132224030)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_shift$get_aux_pars()-0.2690114464)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_shift$get_coef(std_err = FALSE))-c(-0.0468007689 + shift, 2.0773708217))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_shift$get_current_neg_log_likelihood()-116.2091326))),tolerance_loc_1)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    tolerance_gpboost <- 0.16
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.4967306854)),tolerance_gpboost)
    expect_lt(sum(abs(gp_model$get_aux_pars()-0.2378196222)),tolerance_gpboost)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = TRUE)
    # larger tolerance for the same reason as for the random effect means below: most runs reproduce the
    #   values below exactly, but deviations of about 0.18 have been observed
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-0.9585873450, 0.4262452431, 0.9630491993, 2.0000429180))), 0.3)
    # larger tolerance: the GP model is refitted in every boosting iteration and the parallel
    #   reductions in this refit are not bit-wise reproducible, so the random effect means vary
    #   slightly between runs ('deterministic = TRUE' only makes the tree building deterministic).
    #   Deviations of about 0.3 have been observed with the default tolerance of 0.16
    expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(0.2324522564, -0.2957041199, -0.2957041199, 0.0000000000))), 0.5)
    expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c( 0.02163779030, 0.02163779030, 0.02163779030, 0.49673068540))), tolerance_gpboost)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    set.seed(1)
    capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                    nrounds = 100, early_stopping_rounds = 5, metric="test_neg_log_likelihood",
                                    use_gp_model_for_validation = TRUE, folds = folds, verbose = 0), file='NUL')
    # the CV results below vary between runs since the GP model is refitted in every boosting iteration
    #   (see the comment above): scores of 1.460 - 1.585 and best iterations of 23 - 35 have been observed
    expect_lte(cvbst$best_score,1.52*(1+tolerance_loc_3))
    expect_gte(cvbst$best_score,1.52*(1-tolerance_loc_3))
    nit <- 29
    expect_lte(cvbst$best_iter, nit+10)
    expect_gte(cvbst$best_iter, nit-10)

    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = matrix_inversion_method, likelihood_additional_param = quantile)
    set.seed(1)
    capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                    nrounds = 100, early_stopping_rounds = 5, metric="quantile",
                                    use_gp_model_for_validation = TRUE, folds = folds, verbose = 0), file='NUL')
    # same as above: scores of 0.390 - 0.436 and best iterations of 23 - 42 have been observed.
    # The upper bound on the iteration is wider than the lower one because the boosting
    # trajectory turned out to be sensitive to the ordering of the sparse Cholesky: with CHOLMOD
    # instead of Eigen's SimplicialLLT the early stopping lands beyond nit+15. The score bounds
    # below are the substantive check, the iteration bounds are only a sanity range
    expect_lte(cvbst$best_score,0.413*(1+tolerance_loc_3))
    expect_gte(cvbst$best_score,0.413*(1-tolerance_loc_3))
    nit <- 32
    expect_lte(cvbst$best_iter, nit+30)
    expect_gte(cvbst$best_iter, nit-15)

    # }

  }) # end asymmetric_laplace regression

  test_that("asymmetric_laplace likelihood with the SSN-ALM mode refinement ", {

    # The (Fisher) quasi-Newton mode finding of the asymmetric Laplace likelihood uses the endpoint convention
    #   for the score at the kinks of the check loss and can stall at points that are not the exact non-smooth
    #   MAP. The '_ssn_alm' suffix enables a semismooth Newton method applied to the subproblems of an augmented
    #   Lagrangian method, which is run after the quasi-Newton loop if an exact KKT check fails ('_ssn_alm_always'
    #   skips the check). The refinement can never return a worse mode, so the negative log-likelihood must
    #   decrease (weakly) for every random effects structure and every matrix approximation
    quantile_asym_laplace <- function(q, alpha, lambda) {
      res <- rep(NA, length(q))
      ind <- q <= alpha
      res[ind] <- log(q[ind]/alpha) * lambda / (1-alpha)
      res[!ind] <- -log((1-q[!ind])/(1-alpha)) * lambda / alpha
      return(res)
    }
    quantile <- 0.7
    lambda <- 0.25
    error <- quantile_asym_laplace(q = sim_rand_unif(n = n, init_c = 0.651), alpha = quantile, lambda = lambda)
    y <- as.vector(Z1 %*% b_gr_1 + X %*% beta + error)
    fixed_effects <- as.vector(X %*% beta)
    tol <- relax_tolerance_nll(TOLERANCE_STRICT)

    # 'nll_ssn' returns the negative log-likelihood without and with the refinement. The two variants of the
    #   suffix must agree here: the exact KKT check never certifies a mode that is not the exact MAP
    nll_ssn <- function(cov_pars, base_likelihood = "asymmetric_laplace", optim_params = NULL, ...) {
      vapply(c("", "_ssn_alm", "_ssn_alm_always"), function(sfx) {
        # 'capture.output': the FITC variants below warn that inducing points coincide
        #   with data points, which is expected here and would only clutter the test output
        capture.output( gp_model <- GPModel(likelihood = paste0(base_likelihood, sfx),
                                            likelihood_additional_param = quantile, ...), file = 'NUL')
        if (!is.null(optim_params)) gp_model$set_optim_params(params = optim_params)
        capture.output( nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars, y = y, aux_pars = c(lambda),
                                                           fixed_effects = fixed_effects), file = 'NUL')
        nll
      }, numeric(1))
    }
    # Checks that the refinement lowers the negative log-likelihood by the expected amount and that the
    #   gated and the unconditional variant give the same result
    expect_ssn <- function(nll, expected_base, expected_ssn) {
      expect_lt(abs(nll[[1]] - expected_base), tol)
      expect_lt(abs(nll[[2]] - expected_ssn), tol)
      expect_equal(nll[[2]], nll[[3]])
      expect_lte(nll[[2]], nll[[1]])
    }

    ## One grouped random effect (Z is an incidence matrix, the SSN system is diagonal)
    expect_ssn(nll_ssn(c(0.9), group_data = group), 138.9898225, 138.9648429)
    ## Same with the triangular kernel curvature Laplace approximation: the refinement changes the mode, the
    ##   inferential curvature of the determinant must still be the TKC one and not the SSN active set
    expect_ssn(nll_ssn(c(0.9), base_likelihood = "asymmetric_laplace_tkc", group_data = group),
               141.5811521, 140.6629934)
    ## Two crossed grouped random effects (general sparse Z)
    expect_ssn(nll_ssn(c(0.9, 0.6), group_data = cbind(group, group2)), 148.5871268, 148.3692190)
    expect_ssn(nll_ssn(c(0.9, 0.6), group_data = cbind(group, group2),
                       matrix_inversion_method = "iterative"), 148.4716563, 148.2538536)
    ## Gaussian process, all matrix approximations
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential"),
               174.8555513, 174.3157800)
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                       gp_approx = "vecchia", num_neighbors = 20), 174.6832113, 174.2094239)
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                       gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = "iterative"),
               174.7652059, 174.2813998)
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                       gp_approx = "fitc", num_ind_points = 30), 172.4871945, 172.1303044)
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                       gp_approx = "full_scale_vecchia", num_ind_points = 30, num_neighbors = 20),
               174.7149656, 174.2474945)
    expect_ssn(nll_ssn(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                       gp_approx = "full_scale_vecchia", num_ind_points = 30, num_neighbors = 20,
                       matrix_inversion_method = "iterative",
                       optim_params = list(fitc_piv_chol_preconditioner_rank = 30, seed_rand_vec_trace = 1,
                                           num_rand_vec_trace = 200)),
               174.7950602, 174.3383760)
    ## Grouped random effects combined with a GP
    expect_ssn(nll_ssn(c(0.9, 0.9, 0.2), group_data = group, gp_coords = coords,
                       cov_function = "exponential"), 147.5915422, 146.6669310)

    ## Sample weights, including zero weights (an observation with a zero weight must not enter the active
    ##   set of the SSN system even though its prox value is zero)
    weights <- rep(1, n)
    weights[1:10] <- 0
    weights[11:20] <- 3
    nll_w <- vapply(c("", "_ssn_alm_always"), function(sfx) {
      # 'capture.output': notes that the weights do not sum to the number of data points,
      #   which is intended here and would only clutter the test output
      capture.output( gp_model <- GPModel(group_data = group, likelihood = paste0("asymmetric_laplace", sfx),
                                          likelihood_additional_param = quantile, weights = weights), file = 'NUL')
      capture.output( nll <- gp_model$neg_log_likelihood(cov_pars = c(0.9), y = y, aux_pars = c(lambda),
                                                         fixed_effects = fixed_effects), file = 'NUL')
      nll
    }, numeric(1))
    expect_lt(abs(nll_w[[1]] - 157.4794379), tol)
    expect_lt(abs(nll_w[[2]] - 157.4537383), tol)

    ## Estimation. Solving the mode finding problem exactly makes the approximate marginal likelihood a
    ##   well-defined function of the parameters, which is what the outer optimizer needs
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace",
                                           likelihood_additional_param = quantile, y = y, X = X,
                                           params = OPTIM_PARAMS_BFGS), file = 'NUL')
    capture.output( gp_model_ssn <- fitGPModel(group_data = group, likelihood = "asymmetric_laplace_ssn_alm",
                                               likelihood_additional_param = quantile, y = y, X = X,
                                               params = OPTIM_PARAMS_BFGS), file = 'NUL')
    expect_lte(gp_model_ssn$get_current_neg_log_likelihood(),
               gp_model$get_current_neg_log_likelihood() + relax_tolerance_nll(TOLERANCE_MEDIUM))
    expect_lt(abs(gp_model_ssn$get_current_neg_log_likelihood() - 136.6565088), relax_tolerance_nll(TOLERANCE_STRICT))
    expect_equal(gp_model_ssn$get_likelihood_name(), "asymmetric_laplace")

    ## Many observations exactly on a kink. With Z = I the exact mode has residuals that are exactly zero, and the
    ##   mode is then only stationary for an interior subgradient of the check loss, which the endpoint convention of
    ##   the quasi-Newton phase cannot produce. The refinement publishes the certifying subgradient instead, so that
    ##   'Q b = Z^T first_deriv_ll_' continues to hold and the gradients stay consistent with the mode
    y_kink <- round(y * 4) / 4# lots of duplicated responses
    nll_kink <- vapply(c("", "_ssn_alm", "_ssn_alm_always"), function(sfx) {
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                          likelihood = paste0("asymmetric_laplace", sfx), likelihood_additional_param = quantile)
      gp_model$neg_log_likelihood(cov_pars = c(0.9, 0.2), y = y_kink, aux_pars = c(lambda),
                                  fixed_effects = fixed_effects)
    }, numeric(1))
    expect_lt(abs(nll_kink[[1]] - 174.6327729), tol)
    expect_lt(abs(nll_kink[[2]] - 173.4317231), tol)
    expect_equal(nll_kink[[2]], nll_kink[[3]])
    expect_lte(nll_kink[[2]], nll_kink[[1]])

    ## A conjugate gradient budget that is far too small for the semismooth Newton systems. An inaccurate Newton
    ##   direction must not be reported as a successful solve: the refinement then gives up, keeps the mode of the
    ##   quasi-Newton iteration, and does not advertise a certified score. 'cg_max_num_it' also throttles the mode
    ##   finding itself, so the comparison has to be made against the same setting without the refinement
    nll_cg <- vapply(c("", "_ssn_alm", "_ssn_alm_always"), function(sfx) {
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "vecchia",
                          num_neighbors = 20, matrix_inversion_method = "iterative",
                          likelihood = paste0("asymmetric_laplace", sfx), likelihood_additional_param = quantile)
      gp_model$set_optim_params(params = list(cg_max_num_it = 1, seed_rand_vec_trace = 1))
      gp_model$neg_log_likelihood(cov_pars = c(0.9, 0.2), y = y, aux_pars = c(lambda),
                                  fixed_effects = fixed_effects)
    }, numeric(1))
    expect_true(all(is.finite(nll_cg)))
    expect_equal(nll_cg[[2]], nll_cg[[1]])
    expect_equal(nll_cg[[3]], nll_cg[[1]])

    ## ADMM warm start ('_admm_ssn_alm') and ADMM alone ('_admm'). The warm start first runs an alternating
    ##   direction method of multipliers at a fixed penalty, which costs one factorization in total instead of
    ##   one per semismooth Newton step, and then hands the multiplier and a penalty matched to the accuracy it
    ##   reached over to the semismooth Newton iterations. It solves the same problem and must therefore reach
    ##   the same optimum up to the KKT tolerance. ADMM alone is a baseline for measuring what the warm start
    ##   contributes: it is cheap, but its iterates are dual feasible and primal infeasible, so it can fail to
    ##   improve the exact MAP objective at all, in which case the quasi-Newton mode is returned unchanged
    nll_admm <- function(cov_pars, y_use = y, optim_params = NULL, ...) {
      vapply(c("_admm_ssn_alm", "_admm"), function(sfx) {
        # 'capture.output': see the comment in 'nll_ssn' above
        capture.output( gp_model <- GPModel(likelihood = paste0("asymmetric_laplace", sfx),
                                            likelihood_additional_param = quantile, ...), file = 'NUL')
        if (!is.null(optim_params)) gp_model$set_optim_params(params = optim_params)
        capture.output( nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars, y = y_use, aux_pars = c(lambda),
                                                           fixed_effects = fixed_effects), file = 'NUL')
        nll
      }, numeric(1))
    }
    # Neither variant may return a mode that is worse than the one of the quasi-Newton iteration
    expect_admm <- function(nll, expected_base, expected_admm_ssn, expected_admm) {
      expect_lt(abs(nll[[1]] - expected_admm_ssn), tol)
      expect_lt(abs(nll[[2]] - expected_admm), tol)
      expect_lte(nll[[1]], expected_base + tol)
      expect_lte(nll[[2]], expected_base + tol)
    }
    expect_admm(nll_admm(c(0.9), group_data = group), 138.9898225, 138.9648479, 138.9713257)
    expect_admm(nll_admm(c(0.9, 0.6), group_data = cbind(group, group2)),
                148.5871268, 148.3692208, 148.5871268)
    expect_admm(nll_admm(c(0.9, 0.6), group_data = cbind(group, group2),
                         matrix_inversion_method = "iterative"), 148.4716563, 148.2537400, 148.4716563)
    expect_admm(nll_admm(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential"),
                174.8555513, 174.3158050, 174.3158050)
    expect_admm(nll_admm(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                         gp_approx = "vecchia", num_neighbors = 20),
                174.6832113, 174.2094198, 174.2094198)
    expect_admm(nll_admm(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                         gp_approx = "vecchia", num_neighbors = 20, matrix_inversion_method = "iterative"),
                174.7652059, 174.2784319, 174.2784312)
    expect_admm(nll_admm(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                         gp_approx = "fitc", num_ind_points = 30), 172.4871945, 172.1302166, 172.1302166)
    expect_admm(nll_admm(c(0.9, 0.2), gp_coords = coords, cov_function = "exponential",
                         gp_approx = "full_scale_vecchia", num_ind_points = 30, num_neighbors = 20),
                174.7149656, 174.2474984, 174.2474984)
    expect_admm(nll_admm(c(0.9, 0.9, 0.2), group_data = group, gp_coords = coords,
                         cov_function = "exponential"), 147.5915422, 146.6669727, 146.6669727)
    ## Many observations exactly on a kink, see the comment above
    expect_admm(nll_admm(c(0.9, 0.2), y_use = y_kink, gp_coords = coords, cov_function = "exponential"),
                174.6327729, 173.4317200, 173.4317200)

    ## The suffixes are only supported for the asymmetric Laplace likelihood
    expect_error(GPModel(group_data = group, likelihood = "poisson_ssn_alm"),
                 "The '_ssn_alm' mode refinement is currently only supported", fixed = TRUE)
    expect_error(GPModel(group_data = group, likelihood = "poisson_admm_ssn_alm"),
                 "The '_ssn_alm' mode refinement is currently only supported", fixed = TRUE)
    expect_error(GPModel(group_data = group, likelihood = "poisson_admm"),
                 "The '_ssn_alm' mode refinement is currently only supported", fixed = TRUE)

  }) # end asymmetric_laplace with SSN-ALM mode refinement

  test_that("Standard errors for non-Gaussian likelihoods ", {

    ##################################################################################
    ## Single-level grouped random effects model with large data:
    ## t_fix_df (df = 100, ~ Gaussian) vs. Gaussian likelihood
    ##################################################################################

    # Large data
    n_L <- 1e6 # number of samples
    m_L <- n_L/10 # number of categories / levels for grouping variable
    group_L <- rep(1,n_L) # grouping variable
    for(i in 1:m_L) group_L[((i-1)*n_L/m_L+1):(i*n_L/m_L)] <- i
    keps <- 1E-10
    b1_L <- qnorm(sim_rand_unif(n=m_L, init_c=0.846)*(1-keps) + keps/2)
    X_L <- cbind(rep(1,n_L),sim_rand_unif(n=n_L, init_c=0.341)) # design matrix / covariate data for fixed effect
    beta <- c(2,2) # regression coefficients
    xi_L <- sqrt(0.5) * qnorm(sim_rand_unif(n=m_L, init_c=0.321)*(1-keps) + keps/2)
    y_L <- b1_L[group_L] + X_L%*%beta + xi_L

    # Gaussian likelihood
    gp_model <- fitGPModel(group_data = group_L, y = y_L, X = X_L, params = OPTIM_PARAMS_BFGS)
    cov_pars <- c(0.494977742806986, 0.000737869253510783, 1.00023218861287, 0.00469511495626555)
    coef <- c(2.00139224119177, 0.00348515144516913, 1.9982547154621, 0.00257213144546817)
    nll <- 1220035.31884647
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = TRUE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 8)

    # t likelihood with a fixed, large degrees-of-freedom parameter (df = 100), i.e., almost Gaussian noise
    gp_model_t <- fitGPModel(group_data = group_L, y = y_L, X = X_L, likelihood = "t_fix_df",
                             likelihood_additional_param = 100, params = OPTIM_PARAMS_BFGS)
    cov_pars_t <- c(0.99507942001268, 0.00466152106361322)
    aux_pars_t <- c(0.697555658265811, 0.000526826413633507, 100, NaN)
    coef_t <- c(2.00089388635637, 0.00360116458425975, 1.99824865983513, 0.00257268179943032)
    nll_t <- 1219982.93643412
    cov_pars_t_result <- as.vector(gp_model_t$get_cov_pars(std_err = TRUE))
    aux_pars_t_result <- as.vector(gp_model_t$get_aux_pars(std_err = TRUE))
    expect_lt(sum(abs(cov_pars_t_result-cov_pars_t)),TOLERANCE_STRICT)
    expect_lt(sum(abs(aux_pars_t_result[1:3]-aux_pars_t[1:3])),TOLERANCE_STRICT)
    expect_true(is.nan(aux_pars_t_result[4])) # no standard error for the fixed (not estimated) degrees-of-freedom parameter
    expect_lt(sum(abs(as.vector(gp_model_t$get_coef(std_err = TRUE))-coef_t)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model_t$get_current_neg_log_likelihood()-nll_t),TOLERANCE_MEDIUM)
    expect_equal(gp_model_t$get_num_optim_iter(), 7)

    # Compare the Gaussian and t_fix_df (df=100, ~ Gaussian) models: since a t distribution with a
    # large degrees-of-freedom parameter is very close to a Gaussian distribution, both the parameter
    # estimates and their standard errors should be approximately equal
    # Random effect (grouped) variance: same parametrization in both models -> compare directly
    expect_lt(abs(cov_pars_t_result[1] - cov_pars[3]), TOLERANCE_LOOSE) # estimate
    expect_lt(abs(cov_pars_t_result[2] - cov_pars[4]), TOLERANCE_LOOSE) # standard error
    # Idiosyncratic error: the Gaussian likelihood models this as a variance (cov_pars[1]), the t
    # likelihood as a scale parameter (aux_pars[1]) of a t distribution with fixed degrees of freedom.
    # Convert the t scale parameter (and its standard error, via the delta method) to the implied
    # variance of the noise term (Var = scale^2 * df / (df - 2)) for a fair, like-for-like comparison
    implied_var <- aux_pars_t_result[1]^2 * 100 / 98
    se_implied_var <- 2 * aux_pars_t_result[1] * (100 / 98) * aux_pars_t_result[2]
    expect_lt(abs(implied_var - cov_pars[1]), TOLERANCE_LOOSE) # estimate
    expect_lt(abs(se_implied_var - cov_pars[2]), TOLERANCE_LOOSE) # standard error
    # Linear regression coefficients: same parametrization in both models -> compare directly
    expect_lt(sum(abs(coef_t[c(1,3)] - coef[c(1,3)])), TOLERANCE_LOOSE) # estimates
    expect_lt(sum(abs(coef_t[c(2,4)] - coef[c(2,4)])), TOLERANCE_LOOSE) # standard errors

    ##################################################################################
    ## Single-level grouped random effects model with large data:
    ## zoctn likelihood (only hard-coded values since a comparison to another likelihood is not meaningful)
    ##################################################################################
    sd <- 0.5
    a <- -0.5
    b <- 1.2
    mu_zoctn_L <- b1_L[group_L] + 0.5*X_L%*%beta
    y_zoctn_L <- qnorm(sim_rand_unif(n=n_L, init_c=0.74), mean = mu_zoctn_L, sd = sd)
    logistic <- function(t) 1 / (1 + exp(-t))
    logit    <- function(p) log(p / (1 - p))
    y_zoctn_L[y_zoctn_L<0] <- 0
    y_zoctn_L[y_zoctn_L>1] <- 1
    y_zoctn_L[y_zoctn_L>0 & y_zoctn_L<1] <- logistic(a + b*logit(y_zoctn_L[y_zoctn_L>0 & y_zoctn_L<1]))

    gp_model_zoctn <- fitGPModel(group_data = group_L, y = y_zoctn_L, X = X_L, likelihood = "zoctn",
                                 params = OPTIM_PARAMS_BFGS)
    cov_pars_zoctn <- c(0.946230982256346, 0.00572224587906622)
    aux_pars_zoctn <- c(0.500451821354835, 0.000810341663144462, -0.501553021136885,
                        0.00411718642658039, 1.20666369143565, 0.00206520719955995)
    coef_zoctn <- c(0.996946818579902, 0.00364230082868289, 1.02535498713713, 0.00295663078420993)
    nll_zoctn <- 500973.000797625
    expect_lt(sum(abs(as.vector(gp_model_zoctn$get_cov_pars(std_err = TRUE))-cov_pars_zoctn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model_zoctn$get_aux_pars(std_err = TRUE))-aux_pars_zoctn)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model_zoctn$get_coef(std_err = TRUE))-coef_zoctn)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model_zoctn$get_current_neg_log_likelihood()-nll_zoctn),TOLERANCE_MEDIUM)
    expect_equal(gp_model_zoctn$get_num_optim_iter(), 17)

    ##################################################################################
    ## Gaussian process model with a Vecchia approximation:
    ## t_fix_df (df = 100, ~ Gaussian) vs. Gaussian likelihood
    ## Note: for the Gaussian likelihood, the Fisher information for the covariance parameters under a
    ## Vecchia approximation is, by default, calculated using a stochastic trace estimator
    ## (use_stochastic_trace_for_Fisher_information_Vecchia_ = TRUE in the C++ code); this introduces some
    ## additional (deterministic, seeded) noise into the standard errors of the covariance parameters, so a
    ## looser tolerance than for the (exact) grouped random effects model above is used for the comparisons
    ## below. In addition, standard errors for the linear regression coefficients are calculated very
    ## differently for the two likelihoods (an analytic formula for the Gaussian likelihood vs. a numerical
    ## approximation for the t likelihood), and these were found to disagree more under a Vecchia
    ## approximation than in the exact case above, so an even looser tolerance is used for that comparison
    ##################################################################################
    n_v <- 500 # number of samples
    d_v <- 2 # dimension of GP locations
    coords_v <- matrix(sim_rand_unif(n=n_v*d_v, init_c=0.15), ncol=d_v)
    sigma2_v <- 1 # marginal variance of GP
    rho_v <- 0.1 # range parameter
    D_v <- as.matrix(dist(coords_v))
    Sigma_v <- sigma2_v * exp(-D_v/rho_v) + diag(1E-20,n_v)
    L_v <- t(chol(Sigma_v))
    b_v <- qnorm(sim_rand_unif(n=n_v, init_c=0.741))
    X_v <- cbind(rep(1,n_v), sim_rand_unif(n=n_v, init_c=0.642)) # design matrix / covariate data for fixed effect
    beta_v <- c(1,1) # regression coefficients
    xi_v <- sqrt(0.1) * qnorm(sim_rand_unif(n=n_v, init_c=0.951)) # idiosyncratic (nugget) error
    y_v <- as.vector(L_v %*% b_v) + as.vector(X_v %*% beta_v) + xi_v
    num_neighbors_v <- 30

    tol_vecchia <- 0.035 # covariance / auxiliary parameters and coefficient estimates
    tol_vecchia_coef_se <- 0.12 # standard errors of regression coefficients (see note above)
    # The standard errors of covariance parameters of a non-Gaussian likelihood are obtained from a Hessian that is
    # approximated with finite differences of a gradient which itself relies on an iterative mode finding algorithm
    # (see 'CalcHessianCovParAuxPars'). They are thus much less accurate than the estimates themselves and are only
    # compared with a loose tolerance (the standard error of the GP variance below differs by about 40%)
    tol_vecchia_cov_pars_se <- relax_tolerance(0.1)
    # The standard errors of the regression coefficients are NaN if the numerically approximated Hessian is not
    # positive definite (see 'CalcStdDevCoefNonGaussian', which warns and returns NaN in that case). The Hessian is
    # obtained from finite differences of an approximated gradient, so whether it is positive definite depends on
    # the exact point the optimizer ends up at, which differs between compilers. Only check them if they exist
    coef_se_available <- function(x) all(is.finite(x[c(2, 4)]))

    # Gaussian likelihood
    gp_model_gauss_v <- fitGPModel(gp_coords = coords_v, cov_function = "exponential",
                                   gp_approx = "vecchia", num_neighbors = num_neighbors_v,
                                   y = y_v, X = X_v, params = OPTIM_PARAMS_BFGS)
    cov_pars_v <- c(0.0578304572593316, 0.0330021964490000, 0.715705182971271,
                    0.081226357385000, 0.0483181738202239, 0.00771178320850000)
    coef_v <- c(0.962932883927947, 0.110839163937625, 1.10211572960383, 0.0925165508666763)
    nll_v <- 537.536566769249
    expect_lt(sum(abs(as.vector(gp_model_gauss_v$get_cov_pars(std_err = TRUE))-cov_pars_v)),relax_tolerance(TOLERANCE_STRICT, cov_pars_v))
    expect_lt(sum(abs(as.vector(gp_model_gauss_v$get_coef(std_err = TRUE))-coef_v)),relax_tolerance(TOLERANCE_STRICT, coef_v))
    expect_lt(abs(gp_model_gauss_v$get_current_neg_log_likelihood()-nll_v),relax_tolerance(TOLERANCE_MEDIUM, nll_v))
    if (USE_STRICT_TOLERANCES) expect_equal(gp_model_gauss_v$get_num_optim_iter(), 14)

    # t likelihood with a fixed, large degrees-of-freedom parameter (df = 100), i.e., almost Gaussian noise
    # 'capture.output': warns that the standard deviations of the coefficients cannot be
    #   calculated when the approximated Hessian is not positive definite, which the check
    #   through 'coef_se_available' above already allows for
    capture.output( gp_model_t_v <- fitGPModel(gp_coords = coords_v, cov_function = "exponential",
                                               gp_approx = "vecchia", num_neighbors = num_neighbors_v,
                                               likelihood = "t_fix_df", likelihood_additional_param = 100,
                                               y = y_v, X = X_v, params = OPTIM_PARAMS_BFGS), file = 'NUL')
    cov_pars_t_v <- c(0.731926050658421, 0.113198966078325, 0.0469233950753127, 0.0132851890399170)
    aux_pars_t_v <- c(0.215485446742063, 0.134997889593886, 100, NaN)
    coef_t_v <- c(0.958738169312722, 0.0196912670065139, 1.09862570873013, 0.0328225530709028)
    nll_t_v <- 535.896092489469
    cov_pars_t_v_result <- as.vector(gp_model_t_v$get_cov_pars(std_err = TRUE))
    aux_pars_t_v_result <- as.vector(gp_model_t_v$get_aux_pars(std_err = TRUE))
    coef_t_v_result <- as.vector(gp_model_t_v$get_coef(std_err = TRUE))
    expect_lt(sum(abs(cov_pars_t_v_result[c(1,3)]-cov_pars_t_v[c(1,3)])),relax_tolerance(TOLERANCE_STRICT, cov_pars_t_v[c(1,3)]))# estimates
    # The standard errors of covariance and auxiliary parameters are obtained from a Hessian that is approximated
    #   with finite differences of a gradient which itself relies on an iterative mode finding algorithm (see
    #   'CalcHessianCovParAuxPars'). Within one build they are reproducible to about 1e-6, also with a different
    #   number of threads, but the optimizer of this model reaches a different stationary point on another
    #   compiler (see the note on the negative log-likelihood below), and the standard errors there differ by far
    #   more than the estimates do: 0.23 in the sum below has been measured with clang and libc++ in the
    #   sanitizer container of R-hub, for standard errors of the order of 0.01 - 0.11. They are therefore
    #   compared with 'tol_vecchia_cov_pars_se', the tolerance that the comparisons further below use for the
    #   standard errors of the same covariance parameters
    expect_lt(sum(abs(cov_pars_t_v_result[c(2,4)]-cov_pars_t_v[c(2,4)])),tol_vecchia_cov_pars_se)# standard errors
    expect_lt(sum(abs(aux_pars_t_v_result[1]-aux_pars_t_v[1])),relax_tolerance(TOLERANCE_STRICT, aux_pars_t_v[1]))# estimate
    expect_lt(sum(abs(aux_pars_t_v_result[2:3]-aux_pars_t_v[2:3])),relax_tolerance(TOLERANCE_MEDIUM, aux_pars_t_v[2:3]))# standard error and fixed df
    expect_true(is.nan(aux_pars_t_v_result[4])) # no standard error for the fixed (not estimated) degrees-of-freedom parameter
    expect_lt(sum(abs(coef_t_v_result[c(1,3)]-coef_t_v[c(1,3)])),relax_tolerance(TOLERANCE_STRICT, coef_t_v[c(1,3)]))
    if (coef_se_available(coef_t_v_result)) expect_lt(sum(abs(coef_t_v_result[c(2,4)]-coef_t_v[c(2,4)])),relax_tolerance(TOLERANCE_STRICT, coef_t_v[c(2,4)]))
    # This model converges to a clearly different stationary point on other compilers: the negative
    # log-likelihood is about 1.4 higher than the 535.9 below (0.26%), which is also the reason why the
    # approximated Hessian for the coefficient standard errors is not positive definite there (see above)
    expect_lt(abs(gp_model_t_v$get_current_neg_log_likelihood()-nll_t_v),
              if (USE_STRICT_TOLERANCES) TOLERANCE_MEDIUM else 2)
    if (USE_STRICT_TOLERANCES) expect_equal(gp_model_t_v$get_num_optim_iter(), 15)

    # Compare the Gaussian and t_fix_df (df=100, ~ Gaussian) models (see note above on tolerances)
    # GP variance and range: same parametrization in both models -> compare directly
    expect_lt(abs(cov_pars_t_v_result[1] - cov_pars_v[3]), tol_vecchia) # GP variance estimate
    expect_lt(abs(cov_pars_t_v_result[2] - cov_pars_v[4]), tol_vecchia_cov_pars_se) # GP variance standard error
    expect_lt(abs(cov_pars_t_v_result[3] - cov_pars_v[5]), tol_vecchia) # GP range estimate
    expect_lt(abs(cov_pars_t_v_result[4] - cov_pars_v[6]), tol_vecchia_cov_pars_se) # GP range standard error
    # Idiosyncratic (nugget) error: convert the t scale parameter (and its standard error, via the delta
    # method) to the implied noise variance, as in the grouped random effects model above
    implied_var_v <- aux_pars_t_v_result[1]^2 * 100 / 98
    se_implied_var_v <- 2 * aux_pars_t_v_result[1] * (100 / 98) * aux_pars_t_v_result[2]
    expect_lt(abs(implied_var_v - cov_pars_v[1]), tol_vecchia) # estimate
    expect_lt(abs(se_implied_var_v - cov_pars_v[2]), tol_vecchia_cov_pars_se) # standard error
    # Linear regression coefficients: estimates agree well; standard errors are compared with a much
    # looser tolerance since they are calculated very differently for the two likelihoods (see note above)
    expect_lt(sum(abs(coef_t_v_result[c(1,3)] - coef_v[c(1,3)])), tol_vecchia) # estimates
    if (coef_se_available(coef_t_v_result)) expect_lt(abs(coef_t_v_result[2] - coef_v[2]), tol_vecchia_coef_se) # standard error (intercept)
    if (coef_se_available(coef_t_v_result)) expect_lt(abs(coef_t_v_result[4] - coef_v[4]), tol_vecchia_coef_se) # standard error (slope)

  })

  test_that("saving and loading models with several fixed effects predictors ", {
    # Likelihoods with more than one fixed effects predictor (e.g., the mean and the log-variance for
    # 'gaussian_heteroscedastic'). A model is loaded by passing the saved coefficients as 'init_coef'
    # to a pseudo call to 'fit' (with maxit = 0), which is why both are tested here together

    n_sl <- 100
    group_sl <- rep(1:10, each = 10)
    X_sl <- cbind(rep(1, n_sl), sim_rand_unif(n = n_sl, init_c = 0.256))
    b_gr_sl <- qnorm(sim_rand_unif(n = 10, init_c = 0.741))
    u_sl <- sim_rand_unif(n = n_sl, init_c = 0.369)
    mean_sl <- as.vector(X_sl %*% c(0.3, 0.7)) + b_gr_sl[group_sl]
    # Second fixed effects predictor: log-variance / log-shape
    log_scale_sl <- as.vector(X_sl %*% c(-0.5, 1.2))
    y_het_sl <- mean_sl + qnorm(u_sl) * exp(0.5 * log_scale_sl)
    y_gamma_sl <- qgamma(u_sl, shape = exp(log_scale_sl), rate = exp(log_scale_sl) / exp(mean_sl))
    y_hurdle_sl <- ifelse(sim_rand_unif(n = n_sl, init_c = 0.271) < 0.3, 0, y_gamma_sl)
    X_test_sl <- cbind(rep(1, 3), c(0.1, 0.4, 0.8))
    group_test_sl <- c(1, 3, 11)

    cases_sl <- list(
      list(likelihood = "gamma", y = y_gamma_sl, num_sets_fe = 1L), # single predictor, as a control
      list(likelihood = "gaussian_heteroscedastic", y = y_het_sl, num_sets_fe = 2L),
      list(likelihood = "gamma_varying_shape", y = y_gamma_sl, num_sets_fe = 2L),
      list(likelihood = "hurdle_regression_gamma_varying_shape", y = y_hurdle_sl, num_sets_fe = 3L)
    )
    for (case_sl in cases_sl) {
      info_sl <- case_sl$likelihood
      num_coef_sl <- ncol(X_sl) * case_sl$num_sets_fe
      capture.output(gp_model_sl <- fitGPModel(group_data = group_sl, likelihood = info_sl,
                                               y = case_sl$y, X = X_sl,
                                               params = modifyList(OPTIM_PARAMS_BFGS, list(maxit = 20))),
                     file = "NUL")
      coef_sl <- gp_model_sl$get_coef(std_err = FALSE)
      expect_equal(length(coef_sl), num_coef_sl, info = info_sl)
      expect_true(all(is.finite(coef_sl)), info = info_sl)
      cov_pars_sl <- as.vector(gp_model_sl$get_cov_pars(std_err = FALSE))
      aux_pars_sl <- gp_model_sl$get_aux_pars()
      nll_sl <- gp_model_sl$get_current_neg_log_likelihood()
      pred_sl <- predict(gp_model_sl, group_data_pred = group_test_sl, X_pred = X_test_sl,
                         predict_var = TRUE, predict_response = TRUE)

      # Saving and loading must reproduce the model exactly, including the coefficients of all
      # fixed effects predictor blocks and the predictions
      filename_sl <- tempfile(fileext = ".json")
      saveGPModel(gp_model_sl, filename = filename_sl)
      gp_model_loaded_sl <- loadGPModel(filename = filename_sl)
      coef_loaded_sl <- gp_model_loaded_sl$get_coef(std_err = FALSE)
      expect_equal(as.vector(coef_loaded_sl), as.vector(coef_sl), tolerance = TOLERANCE_STRICT, info = info_sl)
      expect_equal(names(coef_loaded_sl), names(coef_sl), info = info_sl)
      expect_equal(as.vector(gp_model_loaded_sl$get_cov_pars(std_err = FALSE)), cov_pars_sl,
                   tolerance = TOLERANCE_STRICT, info = info_sl)
      expect_equal(as.vector(gp_model_loaded_sl$get_aux_pars()), as.vector(aux_pars_sl),
                   tolerance = TOLERANCE_STRICT, info = info_sl)
      expect_equal(gp_model_loaded_sl$get_current_neg_log_likelihood(), nll_sl,
                   tolerance = TOLERANCE_STRICT, info = info_sl)
      pred_loaded_sl <- predict(gp_model_loaded_sl, group_data_pred = group_test_sl, X_pred = X_test_sl,
                                predict_var = TRUE, predict_response = TRUE)
      expect_equal(pred_loaded_sl$mu, pred_sl$mu, tolerance = TOLERANCE_STRICT, info = info_sl)
      expect_equal(pred_loaded_sl$var, pred_sl$var, tolerance = TOLERANCE_STRICT, info = info_sl)

      # The same when the model is loaded from a list instead of a file (as done when a
      # 'gpb.Booster' with a 'GPModel' is loaded)
      gp_model_list_sl <- gpboost:::gpb.GPModel$new(model_list = gp_model_sl$model_to_list())
      expect_equal(as.vector(gp_model_list_sl$get_coef(std_err = FALSE)), as.vector(coef_sl),
                   tolerance = TOLERANCE_STRICT, info = info_sl)

      # 'init_coef' must be used for all fixed effects predictor blocks: with maxit = 0, the
      # coefficients of the fitted model are exactly the provided initial values
      init_coef_sl <- rep(c(0.1, -0.2), length.out = num_coef_sl)
      capture.output(gp_model_init_sl <- fitGPModel(group_data = group_sl, likelihood = info_sl,
                                                    y = case_sl$y, X = X_sl,
                                                    params = list(maxit = 0, init_coef = init_coef_sl,
                                                                  init_coef_aux_pars_from_iid_model = FALSE)),
                     file = "NUL")
      expect_equal(as.vector(gp_model_init_sl$get_coef(std_err = FALSE)), init_coef_sl,
                   tolerance = TOLERANCE_STRICT, info = info_sl)
    }

    # 'init_coef' can also be provided before the covariate data is known. Its length is then the
    # total number of coefficients, from which the number of covariates is derived
    gp_model_pre_sl <- GPModel(group_data = group_sl, likelihood = "gaussian_heteroscedastic")
    gp_model_pre_sl$set_optim_params(params = list(init_coef = c(0.1, -0.2, 0.3, -0.4)))
    capture.output(gp_model_pre_sl$fit(y = y_het_sl, X = X_sl, params = list(maxit = 0)), file = "NUL")
    expect_equal(as.vector(gp_model_pre_sl$get_coef(std_err = FALSE)), c(0.1, -0.2, 0.3, -0.4),
                 tolerance = TOLERANCE_STRICT)
    # A length that is not a multiple of the number of fixed effects predictors is an error
    expect_error(GPModel(group_data = group_sl, likelihood = "gaussian_heteroscedastic")$set_optim_params(
      params = list(init_coef = c(0.1, -0.2, 0.3))), "init_coef")

  })

}
