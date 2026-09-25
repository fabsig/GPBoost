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

  test_that("GLMM with an offset", {

    #####################
    ## Binary classification
    #####################
    n <- 250000 # number of samples
    m <- n / 500 # number of categories / levels for grouping variable
    group <- rep(1,n) # grouping variable
    for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
    b_gr <- sqrt(0.5) * qnorm(sim_rand_unif(n=m, init_c=0.5455))
    offset <- (2*(sim_rand_unif(n=m, init_c=0.54) - 0.5))[group]
    group_test <- c(1,3,9999)
    X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
    X_test <- cbind(rep(1,3),c(-0.5,0.4,1))
    beta <- c(0.1,2) # regression coefficients
    probs <- pnorm(b_gr[group])
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.23431) < probs)
    probs_o <- pnorm(b_gr[group] + offset)
    y_o <- as.numeric(sim_rand_unif(n=n, init_c=0.23431) < probs_o)

    nrounds <- 5
    cov_pars <- c(0.4872681027)
    expected_mu <- c(0.03985967082, -0.42595827038, 0.00000000)
    expected_cov <- c(0.003123267296 , 0.000000000, 0.000000000, 0.000000000,
                      0.003334889393 , 0.000000000, 0.000000000, 0.000000000, 0.4872681027)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, params = DEFAULT_OPTIM_PARAMS)
    pred <- predict(gp_model, group_data_pred = group_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y_o, params = DEFAULT_OPTIM_PARAMS, offset = offset)
    pred <- predict(gp_model, group_data_pred = group_test, offset = offset,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.03)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred_offset_not_provided <- predict(gp_model, group_data_pred = group_test,
                                        predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-pred_offset_not_provided$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-as.vector(pred_offset_not_provided$cov))),TOLERANCE_STRICT)
    # Saving model to file and not providing offset for prediction
    cov_pars_before_save <- as.vector(gp_model$get_cov_pars(std_err = FALSE))
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, group_data_pred = group_test,
                           predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-pred_loaded$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-as.vector(pred_loaded$cov))),TOLERANCE_STRICT)
    expect_lt(sum(abs(cov_pars_before_save - as.vector(gp_model_loaded$get_cov_pars(std_err = FALSE)))),TOLERANCE_STRICT)

    # With linear predictor and offset
    probs <- pnorm(b_gr[group] + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.23431) < probs)
    probs_o <- pnorm(b_gr[group] + offset + X%*%beta)
    y_o <- as.numeric(sim_rand_unif(n=n, init_c=0.23431) < probs_o)

    nrounds <- 6
    cov_pars <- c(0.4484032861)
    coefs <- c(0.028274040843, 0.030146676645, 2.006213492633, 0.006633363776)
    expected_mu <- c(-0.8414334263, 0.5596772562, 2.0344875335)
    expected_cov <- c(0.005217061783 , 0.000000000000, 0.000000000000, 0.000000000000, 0.00573413605 , 0.000000000000, 0.000000000000, 0.000000000000, 0.4484032861)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = DEFAULT_OPTIM_PARAMS)
    pred <- predict(gp_model, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coefs)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y_o, X=X, params = DEFAULT_OPTIM_PARAMS, offset = offset)
    pred <- predict(gp_model, group_data_pred = group_test, X_pred = X_test, offset = offset,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),0.05)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coefs)),0.4)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.15)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.05)
    pred_offset_not_provided <- predict(gp_model, group_data_pred = group_test, X_pred = X_test,
                                        predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-pred_offset_not_provided$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-as.vector(pred_offset_not_provided$cov))),TOLERANCE_STRICT)
    # Saving model to file and not providing offset for prediction
    cov_pars_before_save <- as.vector(gp_model$get_cov_pars(std_err = FALSE))
    coef_before_save <- as.vector(gp_model$get_coef(std_err = TRUE))
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, group_data_pred = group_test, X_pred = X_test,
                           predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-pred_loaded$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-as.vector(pred_loaded$cov))),TOLERANCE_STRICT)
    expect_lt(sum(abs(cov_pars_before_save - as.vector(gp_model_loaded$get_cov_pars(std_err = FALSE)))),TOLERANCE_STRICT)
    expect_lt(sum(abs(coef_before_save - as.vector(gp_model_loaded$get_coef(std_err = TRUE)))),0.005)

    #####################
    ## Poisson regression
    #####################
    n <- 100000 # number of samples
    m <- 1000
    group <- rep(1,n) # grouping variable
    for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
    b_gr <- sqrt(0.5) * qnorm(sim_rand_unif(n=m, init_c=0.5455))
    offset <- (2*(sim_rand_unif(n=m, init_c=0.54) - 0.5))[group]
    group_test <- c(1,3,9999)
    X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
    X_test <- cbind(rep(1,3),c(-0.5,0.4,1))
    beta <- c(0.1,2) # regression coefficients
    mu <- exp(b_gr[group])
    y <- qpois(sim_rand_unif(n=n, init_c=0.468), lambda = mu)
    mu_o <- exp(b_gr[group] + offset)
    y_o <- qpois(sim_rand_unif(n=n, init_c=0.468), lambda = mu_o)

    cov_pars <- 0.4949265643
    nll_opt <- 132766.8144
    nrounds <- 5
    expected_mu <- c(-0.0197946765, -0.4943165282, 0.00000000)
    expected_cov <- c(0.009993953963 , 0.000000000000, 0.000000000000, 0.000000000000, 0.01586816243 , 0.000000000000, 0.000000000000, 0.000000000000, 0.4949265643)
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "poisson",
                                           y = y, params = DEFAULT_OPTIM_PARAMS), file='NUL')
    pred <- predict(gp_model, group_data_pred = group_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "poisson",
                                           y = y_o, params = DEFAULT_OPTIM_PARAMS, offset = offset), file='NUL')
    pred <- predict(gp_model, group_data_pred = group_test, offset = offset,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    nll_opt_o <- 133702.5947
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_o),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.05, expected_mu))
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)
    # # Compare to lme4
    # mod <- glmer(y ~ -1 + (1|group), data=data.frame(y=y_o,group),family=poisson(), offset = offset)
    # summary(mod)

    # With linear predictor and offset
    mu <- exp(b_gr[group] + X%*%beta)
    y <- qpois(sim_rand_unif(n=n, init_c=0.468), lambda = mu)
    mu_o <- exp(b_gr[group] + offset + X%*%beta)
    y_o <- qpois(sim_rand_unif(n=n, init_c=0.468), lambda = mu_o)
    cov_pars <- 0.5014601251
    coefs <- c(0.122983736, 2.006020280)
    nll_opt <- 143780.2423
    nrounds <- 83
    expected_mu <- c(-0.8712505423, 0.3962539667, 2.1290040162)
    expected_cov <- c(0.00534826066 , 0.000000000000, 0.000000000000, 0.000000000000, 0.02125426934 , 0.000000000000, 0.000000000000, 0.000000000000, 0.50146012507)
    gp_model <- fitGPModel(group_data = group, likelihood = "poisson",
                           y = y, X = X, params = DEFAULT_OPTIM_PARAMS)
    pred <- predict(gp_model, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    params = DEFAULT_OPTIM_PARAMS
    params$optimizer_cov <- "lbfgs"
    params$optimizer_coef <- "lbfgs"
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "poisson",
                                           y = y_o, X = X, params = params, offset = offset), file='NUL')
    pred <- predict(gp_model, group_data_pred = group_test, X_pred = X_test, offset = offset,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),0.1)
    nll_opt_o <- 144626.2556
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_o),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.1)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_LOOSE)

    # # Compare to lme4
    # mod <- glmer(y ~ X2 + (1|group), data=data.frame(y=y_o,X,group),family=poisson(), offset = offset)
    # summary(mod)
    # summary(gp_model)

  })

  test_that("Binary classification with multiple grouped random effects ", {

    vec_chol_or_iterative <- c("cholesky","iterative")
    for (inv_method in vec_chol_or_iterative) {
      if(inv_method == "iterative") {
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_LOOSE
        tolerance_loc_3 <- TOLERANCE_ITERATIVE
        tolerance_loc_4 <- 0.2
        loop_cg_PC = c("ssor", "zic")
      } else {
        tolerance_loc_1 <- TOLERANCE_STRICT
        tolerance_loc_2 <- TOLERANCE_STRICT
        tolerance_loc_3 <- TOLERANCE_MEDIUM
        tolerance_loc_4 <- TOLERANCE_STRICT
        loop_cg_PC = c("ssor")
      }
      for (cg_preconditioner_type in loop_cg_PC) {
        probs <- pnorm(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
        y <- as.numeric(sim_rand_unif(n=n, init_c=0.57341) < probs)
        init_cov_pars <- rep(1,3)

        capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                               y = y, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method,
                                               params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                             lr_cov = 0.2, use_nesterov_acc = FALSE, cg_preconditioner_type=cg_preconditioner_type,
                                                             num_rand_vec_trace=100, init_coef_aux_pars_from_iid_model = FALSE))
                        , file='NUL')
        expected_values <- c(0.3060671, 0.9328884, 0.3146682)
        nll_opt <- 59.33113628
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),tolerance_loc_1)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt),tolerance_loc_3)

        # Predict training data random effects
        cov_pars <- gp_model$get_cov_pars(std_err = FALSE)
        all_training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
        first_occurences_1 <- match(unique(group), group)
        first_occurences_2 <- match(unique(group2), group2)
        pred_random_effects <- all_training_data_random_effects[first_occurences_1,c(1,4)]
        pred_random_slopes <- all_training_data_random_effects[first_occurences_1,c(3,6)]
        pred_random_effects_crossed <- all_training_data_random_effects[first_occurences_2,c(2,5)]
        group_unique <- unique(group)
        group_data_pred = cbind(group_unique,rep(-1,length(group_unique)))
        x_pr = rep(0,length(group_unique))
        preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr,
                         predict_response = FALSE, predict_var = TRUE)
        expect_lt(sum(abs(pred_random_effects[,1] - preds$mu)),TOLERANCE_STRICT)
        expect_lt(sum(abs(pred_random_effects[,2] - (preds$var-cov_pars[2]))),tolerance_loc_1)
        # Check whether random slopes are correct
        x_pr = rep(1,length(group_unique))
        preds2 <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr,
                          predict_response = FALSE)
        expect_lt(sum(abs(pred_random_slopes[,1] - (preds2$mu-preds$mu))),TOLERANCE_STRICT)
        # Check whether crossed random effects are correct
        group_unique <- unique(group2)
        group_data_pred = cbind(rep(-1,length(group_unique)),group_unique)
        x_pr = rep(0,length(group_unique))
        preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr,
                         predict_response = FALSE, predict_var = TRUE)
        expect_lt(sum(abs(pred_random_effects_crossed[,1] - preds$mu)),TOLERANCE_MEDIUM)
        expect_lt(sum(abs(pred_random_effects_crossed[,2] - (preds$var-cov_pars[1]))),tolerance_loc_1)

        # Prediction
        group_data_pred = cbind(c(1,1,77),c(2,1,98))
        group_rand_coef_data_pred = c(0,0.1,0.3)
        gp_model <- GPModel(likelihood = "bernoulli_probit", group_data = cbind(group,group2),
                            group_rand_coef_data = x, ind_effect_group_rand_coef = 1, matrix_inversion_method = inv_method)
        expected_mu <- c(0.5195889, -0.6411954, 0.0000000)
        expected_cov <- c(0.3422367, 0.1554011, 0.0000000, 0.1554011,
                          0.3457334, 0.0000000, 0.0000000, 0.0000000, 1.8080000)
        pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                                 cov_pars = c(0.9,0.8,1.2), predict_cov_mat = TRUE, predict_response = FALSE)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
        expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),tolerance_loc_4)
        # Predict variances
        pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                                 cov_pars = c(0.9,0.8,1.2), predict_var = TRUE, predict_response = FALSE)
        expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
        expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
        if(inv_method=="cholesky"){
          # Multiple random effects: training with Nelder-Mead
          capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                                 y = y, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method,
                                                 params = list(optimizer_cov = "nelder_mead", delta_rel_conv=1e-6, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
                          , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(0.3055487, 0.9300562, 0.3048811))),TOLERANCE_STRICT)
        }
        # Multiple random effects: training with BFGS
        capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                               y = y, likelihood = "bernoulli_probit", matrix_inversion_method = inv_method,
                                               params = list(optimizer_cov = "lbfgs", init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(0.3030687897, 0.9292636103, 0.3037924600))),tolerance_loc_1)
        # Evaluate negative log-likelihood
        nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.8,1.2),y=y)
        expect_lt(abs(nll-60.6422359),tolerance_loc_3)

        ## Two crossed random effects
        probs_2 <- pnorm(Z1 %*% b_gr_1 + Z2 %*% b_gr_2)
        y_2 <- as.numeric(sim_rand_unif(n=n, init_c=0.156) < probs_2)
        params = DEFAULT_OPTIM_PARAMS
        params$init_cov_pars <- rep(1,2)
        params$cg_preconditioner_type=cg_preconditioner_type
        params$num_rand_vec_trace=100
        capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), matrix_inversion_method = inv_method,
                                               y = y_2, likelihood = "bernoulli_probit", params = params)
                        , file='NUL')
        expected_values <- c(0.1950790008, 0.5496159992)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),5*tolerance_loc_2)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-64.37229209),tolerance_loc_3)
        # summary(gp_model)
        # # Compare to lme4
        # library(lme4)
        # mod <- glmer(y ~ -1 + (1|group) + (1|group2), data=data.frame(y=y_2,group,group2),family=binomial(link="probit"))
        # summary(mod)
      }
    } # end loop over matrix_inversion_method

    # Multiple cluster_ids
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y, cluster_ids = cluster_ids, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky",
                                           params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                         lr_cov = 0.2, use_nesterov_acc = FALSE, maxit=100, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expected_values <- c(0.1634433, 0.8952201, 0.3219087)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 42)
    # Prediction
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                        cluster_ids = cluster_ids, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), cluster_ids_pred = cluster_ids_pred, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.2159939, 0.0000000, 0.0000000)
    expected_cov <- c(0.4547941, 0.0000000, 0.0000000, 0.0000000,
                      1.7120000, 0.0000000, 0.0000000, 0.0000000, 1.8080000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)

    # Only one RE and random coefficient
    probs <- pnorm(Z1 %*% b_gr_1 + Z3 %*% b_gr_3)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.957341) < probs)
    init_cov_pars <- c(1,1)
    capture.output( gp_model <- fitGPModel(group_data = group, group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky",
                                           params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE, maxit=100, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    expected_values <- c(1.00742383, 0.02612587)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 100)

    # Random coefficients with intercept random effect dropped
    probs <- pnorm(Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.8341) < probs)
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                                           ind_effect_group_rand_coef = 1, drop_intercept_group_rand_effect = c(TRUE,FALSE),
                                           y = y, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky",
                                           params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expected_values <- c(1.0044712, 0.6549656)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 18)
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
    first_occurences_1 <- match(unique(group), group)
    first_occurences_2 <- match(unique(group2), group2)
    pred_random_slopes <- all_training_data_random_effects[first_occurences_1,2]
    pred_random_effects_crossed <- all_training_data_random_effects[first_occurences_2,1]
    group_unique <- unique(group)
    group_data_pred = cbind(group_unique,rep(-1,length(group_unique)))
    # Check whether random slopes are correct
    x_pr = rep(1,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr, predict_response = FALSE)
    expect_lt(sum(abs(pred_random_slopes - preds$mu)),TOLERANCE_MEDIUM)
    # Check whether crossed random effects are correct
    group_unique <- unique(group2)
    group_data_pred = cbind(rep(-1,length(group_unique)),group_unique)
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr, predict_response = FALSE)
    expect_lt(sum(abs(pred_random_effects_crossed - preds$mu)),TOLERANCE_MEDIUM)
    # Prediction
    gp_model <- GPModel(likelihood = "bernoulli_probit", group_data = cbind(group,group2),
                        group_rand_coef_data = x, ind_effect_group_rand_coef = 1, matrix_inversion_method = "cholesky",
                        drop_intercept_group_rand_effect = c(TRUE,FALSE))
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.8,1.2), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.8493404, -0.2338359, 0.0000000)
    expected_cov <- c(0.206019606, -0.001276366, 0.0000000, -0.001276366,
                      0.155209578, 0.0000000, 0.0000000, 0.0000000, 0.908000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.8,1.2), predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)

    # Including linear fixed effects
    probs <- pnorm(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3 + X%*%beta)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- rep(1,3)
    y_lin <- as.numeric(sim_rand_unif(n=n, init_c=0.41) < probs)
    capture.output( gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y_lin, X=X, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky", params = params)
                    , file='NUL')
    cov_pars <- c(0.8047844, 1.5684941, 1.8099834)
    coef <- c(-0.4002821736, 2.5025630022)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),3*TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    # Prediction
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    X_test <- cbind(rep(1,3),c(-0.5,0.4,1))
    pred <- gp_model$predict(group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             X_pred = X_test, predict_cov_mat = TRUE, predict_response = FALSE, cov_pars = c(0.8,1.5,1.8))
    expected_mu <- c(-0.5401923644, 0.8816074298, 2.1022806795)
    expected_cov <- c(0.5808497995, 0.1935342214, 0.0000000000, 0.1935342214, 0.5933682927, 0.0000000000, 0.0000000000, 0.0000000000, 2.4620000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)

  })

}
