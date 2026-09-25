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

  test_that("beta regression ", {

    params <- OPTIM_PARAMS_BFGS
    likelihood <- "beta"

    # Single level grouped random effects
    mu <- 1 / (1 + exp(-(Z1 %*% b_gr_1 + 0.5*X%*%beta)))
    phi = 2
    y <- qbeta(sim_rand_unif(n=n, init_c=0.135456), shape1 = mu * phi, shape2 = (1 - mu) * phi)

    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll--31.05453707),TOLERANCE_STRICT)

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.4001315457)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-1.868524016 )),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.1282965526, 1.1881972770 ))),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()--54.4500614 )),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var= TRUE, predict_response = FALSE)
    expected_mu <- c(-1.1826158504, -0.1320929747, 0.1055464807, 1.0599007244)
    expected_var <- c(0.10336229497, 0.08644181625, 0.08644181625, 0.40013154573)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.2393651554, 0.4677054534, 0.5258171071, 0.7262142368)
    expected_var <- c(0.06565030867, 0.09013797079, 0.09027893547, 0.07849860055)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_MEDIUM)

    ## GPBoost algorithm
    y_gpb <- qbeta(sim_rand_unif(n=n, init_c=0.1456), shape1 = mu * phi, shape2 = (1 - mu) * phi)
    dtrain <- gpb.Dataset(data = X, label = y_gpb)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.3202558)),TOLERANCE_MEDIUM)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-0.83438828037, -0.11965478176, -0.02962818377, 1.26103671054))),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.4202132157, 0.2823993747, 0.3006671296, 0.7650395471))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.07663208127, 0.06368025701, 0.06616115505, 0.06105965388))), TOLERANCE_MEDIUM)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y_gpb)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              reuse_learning_rates_gp_model = FALSE) )
    expect_lt(sum(abs(cvbst$best_score--0.345594760036318)),TOLERANCE_LOOSE)
    expect_lte(cvbst$best_iter, 16)
    expect_gte(cvbst$best_iter, 15)

  }) # end beta regression

  test_that("negative_binomial_1 regression ", {

    params <- OPTIM_PARAMS_BFGS
    # 'negative_binomial_1' now DEFAULTS to a 'combined' approximation (quasi-Fisher mode finding + observed-Hessian
    # determinant), which shifts the mode / estimates by ~1e-5 relative to the pure observed-Hessian Laplace these strict
    # (1e-6) golden values were generated with. The strict golden checks below therefore pin the '_laplace'
    # (observed-information) version; the default 'combined' approximation is separately exercised at the end of this.
    likelihood <- "negative_binomial_1_laplace"

    # Single level grouped random effects
    mu <- exp(Z1 %*% b_gr_1 + 0.5*X%*%beta)
    phi = 0.5
    y <- qnbinom(sim_rand_unif(n=n, init_c=0.135456), size = mu / phi, prob = 1/(1+phi))

    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-178.2504468),TOLERANCE_STRICT)

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.479443183)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-0.3875111886)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.1869209845, 1.2215795573))),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-147.4626638)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 12)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var= TRUE, predict_response = FALSE)
    expected_mu <- c(-1.50813623680, -0.06547232544, 0.17884358603, 1.03465857279)
    expected_var <- c(0.13214360292, 0.09038251055, 0.09038251055, 0.47944318296)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.2364391412, 0.9799232074, 1.2511146091, 3.5764838904)
    expected_var <- c(0.3359595595, 1.4504871955, 1.8840006227, 12.8312580231)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.5959292609 )),TOLERANCE_STRICT)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(0.22626493197, -0.02387452881, -0.02387452881, 1.37497338251))),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.5074674531, 0.7102847977, 0.7102847977, 5.3277979647))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.7090701094, 0.9862357452, 0.9862357452, 30.1741534567))), TOLERANCE_MEDIUM)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              reuse_learning_rates_gp_model = FALSE) )
    expect_lt(sum(abs(cvbst$best_score-1.49474040330875)),TOLERANCE_MEDIUM)
    expect_equal(cvbst$best_iter, 34)

    # Exercise the DEFAULT 'negative_binomial_1' approximation ('combined': quasi-Fisher mode finding + observed-Hessian
    # determinant). It must fit and stay close to the '_laplace' (observed-information) golden values checked above; the
    # loose tolerance absorbs the ~1e-5 difference between quasi-Fisher and observed-Hessian mode finding.
    nll_comb <- GPModel(group_data = group, likelihood = "negative_binomial_1",
                        matrix_inversion_method = "cholesky")$neg_log_likelihood(cov_pars=c(0.9), y=y)
    expect_lt(abs(nll_comb-178.2504468), TOLERANCE_MEDIUM)
    capture.output( gp_comb <- fitGPModel(group_data = group, likelihood = "negative_binomial_1",
                                          y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(abs(as.vector(gp_comb$get_cov_pars(std_err = FALSE))-0.479443183), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_comb$get_aux_pars())-0.3875111886), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_comb$get_coef(std_err = FALSE))-c(-0.1869209845, 1.2215795573))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_comb$get_current_neg_log_likelihood()-147.4626638), TOLERANCE_MEDIUM)

  }) # end negative_binomial_1 regression

  test_that("binomial regression ", {

    params <- OPTIM_PARAMS_BFGS
    likelihood <- "binomial_logit"

    # Single level grouped random effects
    mu <- Z1 %*% b_gr_1 + 0.5*X%*%beta
    p <- 1 / (1 + exp(-mu))
    ntrial <- qpois(sim_rand_unif(n=n, init_c=0.9146), lambda=5)
    y <- qbinom(sim_rand_unif(n=n, init_c=0.146), size = ntrial, prob = p) / ntrial

    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky", weights = ntrial)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-164.4059537),TOLERANCE_STRICT)

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood, weights = ntrial,
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.2744642669 )),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.005279993048, 0.798354476357))),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-145.3393856)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var= TRUE, predict_response = FALSE)
    expected_mu <- c(-0.05764418646, -0.10010510651, 0.05956578876, 0.79307448331)
    expected_var <- c(0.06017870123, 0.08217586719, 0.08217586719, 0.27446426691)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.4858032665, 0.4754871830, 0.5145933378, 0.6784515040)
    expected_var <- c(0.2497984528, 0.2493991218, 0.2497870345, 0.2181550607)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky", weights = ntrial)
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.2204588084)),TOLERANCE_MEDIUM)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-0.7067572973, 0.5773264214, 0.3702024902, 0.7135313663))),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.4041701424, 0.5694021742, 0.5189985431, 0.6635301968))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.2408166384, 0.2451833382, 0.2496390554, 0.2232578747))), TOLERANCE_MEDIUM)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky", weights = ntrial)
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              reuse_learning_rates_gp_model = FALSE, metric="l2") )
    expect_lt(sum(abs(cvbst$best_score-0.084122414240285)),TOLERANCE_MEDIUM)
    expect_gte(cvbst$best_iter, 13)
    expect_lte(cvbst$best_iter, 14)

    ## Probit link
    likelihood <- "binomial_probit"

    # Single level grouped random effects
    p <- pnorm(mu)
    ntrial <- qpois(sim_rand_unif(n=n, init_c=0.9146), lambda=5)
    y <- qbinom(sim_rand_unif(n=n, init_c=0.146), size = ntrial, prob = p) / ntrial

    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky", weights = ntrial)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-184.0923436),TOLERANCE_STRICT)

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood, weights = ntrial,
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.3378497604)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.0184972324, 0.8934546473 ))),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-133.3944314)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.4262816428, 0.3898570045, 0.4581958369, 0.7753118545)
    expected_var <- c(0.2445656038, 0.2378685206, 0.2482524119, 0.1742033828)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky", weights = ntrial)
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.3128922659)),TOLERANCE_STRICT)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-0.6164866217, 0.3334834448, 0.1848957926, 0.8861782263))),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.3631685341, 0.4376649241, 0.3812511453, 0.7803583995))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.2312771499, 0.2461143383, 0.2358987095, 0.1713991678))), TOLERANCE_MEDIUM)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky", weights = ntrial)
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              reuse_learning_rates_gp_model = FALSE, metric="l2") )
    expect_lt(sum(abs(cvbst$best_score-0.0791837827384148)),TOLERANCE_LOOSE)
    expect_gte(cvbst$best_iter, 13)
    expect_lte(cvbst$best_iter, 14)

  }) # end binomial regression

  test_that("lognormal regression ", {

    params <- OPTIM_PARAMS_BFGS
    likelihood <- "lognormal"

    # Single level grouped random effects
    eta <- Z1 %*% b_gr_1 + 0.5*X%*%beta
    logvar = 0.5
    qlognorm_eta <- function(p, eta, logvar) {
      if (any(p < 0 | p > 1)) stop("'p' must be in [0, 1].")
      m <- eta - 0.5 * logvar
      s <- sqrt(logvar)
      exp(m + s * qnorm(p))
    }
    y <- qlognorm_eta(sim_rand_unif(n=n, init_c=0.913468), eta=eta, logvar=logvar)

    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-132.6707012),TOLERANCE_STRICT)

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood,
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.4529120267)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-0.4737246483 )),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.0817856977,0.8909274795))),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-93.36814818)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 13)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(1.110683450, 1.134531268, 1.355818163, 2.816789595)
    expected_var <- c(0.8343419502, 0.8705554101, 1.2432726330, 12.1077403376)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.5574238512)),TOLERANCE_MEDIUM)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-0.06829123608, 0.85388995665, 1.11622428955, 1.25895496521))),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(1.239467432, 2.244944617, 2.918027810, 4.653025502))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.7815570058, 2.5639035513,  4.3318087504, 33.4175885494))), TOLERANCE_MEDIUM)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, learning_rate=0.01, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              reuse_learning_rates_gp_model = FALSE) )
    # The cross-validation score aggregates a boosting run over several folds and depends on the number of OpenMP
    # threads: measured against the value below, the deviation is 2.5e-4 with a single thread and 1.5e-3 with 16, i.e.
    # the thread-induced change alone exceeds TOLERANCE_MEDIUM (1e-3). This is the only assertion in the suite whose
    # pass/fail outcome flips with the thread count, so it needs a tolerance that accommodates that variation
    expect_lt(sum(abs(cvbst$best_score-1.22029815715316)),TOLERANCE_LOOSE)
    expect_equal(cvbst$best_iter, 8)

  }) # end lognormal regression

  test_that("betabinomial regression ", {

    params <- OPTIM_PARAMS_BFGS
    likelihood <- "betabinomial"

    # Single level grouped random effects
    eta <- Z1 %*% b_gr_1 + 0.5*X%*%beta
    mu <- 1/(1+exp(-eta))
    phi <- 2
    a <- mu * phi
    b <- (1-mu) * phi
    p <- qbeta(sim_rand_unif(n=n, init_c=0.5940), shape1=a, shape2=b)
    ntrial <- qpois(sim_rand_unif(n=n, init_c=0.15468), lambda=5) + 1
    y <- qbinom(sim_rand_unif(n=n, init_c=0.146), size = ntrial, prob = p) / ntrial

    # Evaluate negative log-likelihood
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky", weights = ntrial)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-220.9211521),TOLERANCE_STRICT)

    # Estimation
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = likelihood, weights = ntrial,
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky")
                    , file='NUL')
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.1184719163)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(0.005406537788, 0.698069670326 ))),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-180.6305215)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 12)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.4109065594, 0.4323400890, 0.4662659760, 0.6645252609)
    expected_var <- c(0.2420650863, 0.2454235695, 0.2488623825, 0.2229510881)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky", weights = ntrial)
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.1436621527)),TOLERANCE_MEDIUM)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-0.60461313705, -0.04195378862, -0.04195378862, 0.72437041248))),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.3458223512, 0.3714938330, 0.3714938330, 0.6680860736))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.2262380800, 0.2334917974, 0.2334917974, 0.2217771309))), TOLERANCE_MEDIUM)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood,
                        matrix_inversion_method = "cholesky", weights = ntrial)
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              reuse_learning_rates_gp_model = FALSE, metric="l2") )
    expect_lt(sum(abs(cvbst$best_score-0.126457411513177)),TOLERANCE_MEDIUM)
    expect_gte(cvbst$best_iter, 23)
    expect_lte(cvbst$best_iter, 26)

  }) # end betabinomial regression

}
