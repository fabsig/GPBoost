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

  test_that("gaussian_latent likelihood agrees with gaussian likelihood ", {
    n_lat <- 40
    group_lat <- rep(1:8, each = 5)
    group2_lat <- rep(1:5, length.out = n_lat)
    coords_lat <- cbind(seq(0.02, 0.98, length.out = n_lat),
                        sin(seq(0.1, 2.9, length.out = n_lat)))
    y_lat <- 0.3 + 0.7 * seq_len(n_lat) / n_lat + cos(seq_len(n_lat) / 4)
    err_var <- 0.35
    gr_var_1 <- 0.7
    gr_var_2 <- 0.45
    gp_var <- 0.8
    gp_range <- 0.25
    tol_lat <- 5e-5
    expect_gaussian_latent_equal <- function(model_gauss, model_latent,
                                             cov_pars_gauss, cov_pars_latent,
                                             aux_pars_gauss = NULL,
                                             aux_pars_latent = err_var,
                                             tolerance = tol_lat) {
      expect_equal(model_latent$get_num_aux_pars(), 1L)
      nll_gauss <- model_gauss$neg_log_likelihood(cov_pars = cov_pars_gauss, y = y_lat, aux_pars = aux_pars_gauss)
      nll_latent <- model_latent$neg_log_likelihood(cov_pars = cov_pars_latent, y = y_lat, aux_pars = aux_pars_latent)
      expect_equal(nll_latent, nll_gauss, tolerance = tolerance)
    }
    expect_gaussian_latent_predictions_equal <- function(model_gauss, model_latent,
                                                         cov_pars_gauss, cov_pars_latent,
                                                         group_data_pred = NULL,
                                                         gp_coords_pred = NULL,
                                                         aux_pars_gauss = NULL,
                                                         aux_pars_latent = err_var,
                                                         tolerance = tol_lat) {
      if (!is.null(aux_pars_gauss)) {
        model_gauss$set_optim_params(params = list(init_aux_pars = aux_pars_gauss,
                                                   init_coef_aux_pars_from_iid_model = FALSE))
      }
      model_latent$set_optim_params(params = list(init_aux_pars = aux_pars_latent,
                                                  init_coef_aux_pars_from_iid_model = FALSE))
      pred_gauss <- predict(model_gauss, y = y_lat, group_data_pred = group_data_pred,
                            gp_coords_pred = gp_coords_pred, cov_pars = cov_pars_gauss,
                            predict_var = TRUE, predict_response = TRUE)
      pred_latent <- predict(model_latent, y = y_lat, group_data_pred = group_data_pred,
                             gp_coords_pred = gp_coords_pred, cov_pars = cov_pars_latent,
                             predict_var = TRUE, predict_response = TRUE)
      expect_equal(pred_latent$mu, pred_gauss$mu, tolerance = tolerance)
      expect_equal(pred_latent$var, pred_gauss$var, tolerance = tolerance)
    }

    expect_gaussian_latent_equal(
      GPModel(group_data = group_lat, likelihood = "gaussian"),
      GPModel(group_data = group_lat, likelihood = "gaussian_latent"),
      c(err_var, gr_var_1),
      c(gr_var_1)
    )
    expect_gaussian_latent_equal(
      GPModel(group_data = cbind(group_lat, group2_lat), likelihood = "gaussian"),
      GPModel(group_data = cbind(group_lat, group2_lat), likelihood = "gaussian_latent"),
      c(err_var, gr_var_1, gr_var_2),
      c(gr_var_1, gr_var_2)
    )
    expect_gaussian_latent_equal(
      GPModel(group_data = group_lat, gp_coords = coords_lat, cov_function = "exponential",
              likelihood = "gaussian", matrix_inversion_method = "cholesky"),
      GPModel(group_data = group_lat, gp_coords = coords_lat, cov_function = "exponential",
              likelihood = "gaussian_latent", matrix_inversion_method = "cholesky"),
      c(err_var, gr_var_1, gp_var, gp_range),
      c(gr_var_1, gp_var, gp_range)
    )
    expect_gaussian_latent_equal(
      GPModel(gp_coords = coords_lat, cov_function = "exponential", likelihood = "gaussian",
              gp_approx = "vecchia_latent", num_neighbors = 5, vecchia_ordering = "none",
              matrix_inversion_method = "cholesky"),
      GPModel(gp_coords = coords_lat, cov_function = "exponential", likelihood = "gaussian_latent",
              gp_approx = "vecchia", num_neighbors = 5, vecchia_ordering = "none",
              matrix_inversion_method = "cholesky"),
      c(gp_var, gp_range),
      c(gp_var, gp_range),
      aux_pars_gauss = err_var
    )

    group_pred_lat <- c(1, 4, 99)
    group_pred_cross_lat <- cbind(c(1, 4, 99), c(1, 3, 99))
    coords_pred_lat <- coords_lat[c(2, 11, 30), ] + 0.01
    expect_gaussian_latent_predictions_equal(
      GPModel(group_data = group_lat, likelihood = "gaussian"),
      GPModel(group_data = group_lat, likelihood = "gaussian_latent"),
      c(err_var, gr_var_1),
      c(gr_var_1),
      group_data_pred = group_pred_lat
    )
    expect_gaussian_latent_predictions_equal(
      GPModel(group_data = cbind(group_lat, group2_lat), likelihood = "gaussian"),
      GPModel(group_data = cbind(group_lat, group2_lat), likelihood = "gaussian_latent"),
      c(err_var, gr_var_1, gr_var_2),
      c(gr_var_1, gr_var_2),
      group_data_pred = group_pred_cross_lat
    )
    expect_gaussian_latent_predictions_equal(
      GPModel(group_data = group_lat, gp_coords = coords_lat, cov_function = "exponential",
              likelihood = "gaussian", matrix_inversion_method = "cholesky"),
      GPModel(group_data = group_lat, gp_coords = coords_lat, cov_function = "exponential",
              likelihood = "gaussian_latent", matrix_inversion_method = "cholesky"),
      c(err_var, gr_var_1, gp_var, gp_range),
      c(gr_var_1, gp_var, gp_range),
      group_data_pred = group_pred_lat,
      gp_coords_pred = coords_pred_lat
    )
    expect_gaussian_latent_predictions_equal(
      GPModel(gp_coords = coords_lat, cov_function = "exponential", likelihood = "gaussian",
              gp_approx = "vecchia_latent", num_neighbors = 5, vecchia_ordering = "none",
              matrix_inversion_method = "cholesky"),
      GPModel(gp_coords = coords_lat, cov_function = "exponential", likelihood = "gaussian_latent",
              gp_approx = "vecchia", num_neighbors = 5, vecchia_ordering = "none",
              matrix_inversion_method = "cholesky"),
      c(gp_var, gp_range),
      c(gp_var, gp_range),
      gp_coords_pred = coords_pred_lat,
      aux_pars_gauss = err_var
    )

    ## Iterative methods for crossed random effects (stochastic -> lower tolerance)
    tol_lat_iterative <- TOLERANCE_ITERATIVE
    iterative_optim_params <- list(cg_delta_conv = 1e-6, num_rand_vec_trace = 500,
                                   seed_rand_vec_trace = 1, init_coef_aux_pars_from_iid_model = FALSE)
    gp_model_gauss_it <- GPModel(group_data = cbind(group_lat, group2_lat), likelihood = "gaussian",
                                 matrix_inversion_method = "iterative")
    gp_model_latent_it <- GPModel(group_data = cbind(group_lat, group2_lat), likelihood = "gaussian_latent",
                                  matrix_inversion_method = "iterative")
    gp_model_gauss_it$set_optim_params(params = iterative_optim_params)
    gp_model_latent_it$set_optim_params(params = iterative_optim_params)
    expect_gaussian_latent_equal(
      gp_model_gauss_it,
      gp_model_latent_it,
      c(err_var, gr_var_1, gr_var_2),
      c(gr_var_1, gr_var_2),
      tolerance = tol_lat_iterative
    )
    expect_gaussian_latent_predictions_equal(
      gp_model_gauss_it,
      gp_model_latent_it,
      c(err_var, gr_var_1, gr_var_2),
      c(gr_var_1, gr_var_2),
      group_data_pred = group_pred_cross_lat,
      tolerance = tol_lat_iterative
    )

    ## Vecchia + GP with iterative methods (stochastic -> lower tolerance)
    gp_model_gauss_vecchia_it <- GPModel(gp_coords = coords_lat, cov_function = "exponential", likelihood = "gaussian",
                                        gp_approx = "vecchia_latent", num_neighbors = 5, vecchia_ordering = "none",
                                        matrix_inversion_method = "iterative")
    gp_model_latent_vecchia_it <- GPModel(gp_coords = coords_lat, cov_function = "exponential", likelihood = "gaussian_latent",
                                         gp_approx = "vecchia", num_neighbors = 5, vecchia_ordering = "none",
                                         matrix_inversion_method = "iterative")
    gp_model_gauss_vecchia_it$set_optim_params(params = iterative_optim_params)
    gp_model_latent_vecchia_it$set_optim_params(params = iterative_optim_params)
    expect_gaussian_latent_equal(
      gp_model_gauss_vecchia_it,
      gp_model_latent_vecchia_it,
      c(gp_var, gp_range),
      c(gp_var, gp_range),
      aux_pars_gauss = err_var,
      tolerance = tol_lat_iterative
    )
    expect_gaussian_latent_predictions_equal(
      gp_model_gauss_vecchia_it,
      gp_model_latent_vecchia_it,
      c(gp_var, gp_range),
      c(gp_var, gp_range),
      gp_coords_pred = coords_pred_lat,
      aux_pars_gauss = err_var,
      tolerance = tol_lat_iterative
    )

    capture.output(gp_model_gauss_fit <- fitGPModel(
      group_data = group_lat, likelihood = "gaussian", y = y_lat,
      params = list(optimizer_cov = "lbfgs", maxit = 100,
                    init_cov_pars = c(err_var, gr_var_1),
                    init_coef_aux_pars_from_iid_model = FALSE)), file = "NUL")
    capture.output(gp_model_latent_fit <- fitGPModel(
      group_data = group_lat, likelihood = "gaussian_latent", y = y_lat,
      params = list(optimizer_cov = "lbfgs", maxit = 100,
                    init_cov_pars = c(gr_var_1), init_aux_pars = c(err_var),
                    init_coef_aux_pars_from_iid_model = FALSE)), file = "NUL")
    expect_equal(gp_model_latent_fit$get_current_neg_log_likelihood(),
                 gp_model_gauss_fit$get_current_neg_log_likelihood(),
                 tolerance = TOLERANCE_MEDIUM)
    expect_equal(c(as.vector(gp_model_latent_fit$get_aux_pars()),
                   as.vector(gp_model_latent_fit$get_cov_pars(std_err = FALSE))),
                 as.vector(gp_model_gauss_fit$get_cov_pars(std_err = FALSE)),
                 tolerance = TOLERANCE_MEDIUM)
    pred_gauss_fit <- predict(gp_model_gauss_fit, y = y_lat, group_data_pred = group_pred_lat,
                              predict_var = TRUE, predict_response = TRUE)
    pred_latent_fit <- predict(gp_model_latent_fit, y = y_lat, group_data_pred = group_pred_lat,
                               predict_var = TRUE, predict_response = TRUE)
    expect_equal(pred_latent_fit$mu, pred_gauss_fit$mu, tolerance = TOLERANCE_MEDIUM)
    expect_equal(pred_latent_fit$var, pred_gauss_fit$var, tolerance = TOLERANCE_MEDIUM)
  })

  test_that("Binary classification with Gaussian process model ", {
    probs <- pnorm(L %*% b_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    init_cov_pars <- c(1,mean(dist(coords))/3)
    # Label needs to have correct format
    expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential",
                            likelihood = "bernoulli_probit",
                            y = b_1, params = list(optimizer_cov = "gradient_descent", init_coef_aux_pars_from_iid_model = FALSE)))
    yw <- y
    yw[3] <- yw[3] + 1E-6
    expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential",
                            likelihood = "bernoulli_probit",
                            y = yw, params = list(optimizer_cov = "gradient_descent", init_coef_aux_pars_from_iid_model = FALSE)))
    # Only gradient descent can be used
    expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential",
                            likelihood = "bernoulli_probit",
                            y = y, params = list(optimizer_cov = "fisher_scoring", init_coef_aux_pars_from_iid_model = FALSE)))
    # Estimation using gradient descent
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent",
                                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                       convergence_criterion = "relative_change_in_parameters",
                                                       init_cov_pars = init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(0.9419234, 0.1866877)
    nll_opt <- 63.61263619
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 40)
    # Can switch between likelihoods
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        likelihood = "gaussian", matrix_inversion_method = "cholesky")
    gp_model$set_likelihood("bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent",
                                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                       convergence_criterion = "relative_change_in_parameters",
                                                       init_cov_pars = init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent",
                                                       lr_cov = 0.01, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5, init_cov_pars = init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars2 <- c(0.9646422, 0.1844797)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars2)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 26)
    # Estimation using Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "nelder_mead", delta_rel_conv=1e-6,
                                                       init_cov_pars = init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    cov_pars3 <- c(0.9998047, 0.1855072)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars3)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Estimation using lbfgs
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "lbfgs", init_cov_pars = init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars_lbfgs <- c(0.9418327551, 0.1866904020)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_lbfgs)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_LOOSE)
    # Estimation using Adam
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "adam", init_cov_pars = init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars_adam <- c(0.9419081, 0.1866883)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_adam)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 200)

    # Prediction
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           y = y, matrix_inversion_method = "cholesky", params = list(optimizer_cov = "gradient_descent",
                                                                                                      lr_cov=0.01, use_nesterov_acc=FALSE, init_cov_pars = init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.6595663, -0.6638940, 0.4997690)
    expected_cov <- c(0.6482224576, 0.5765285950, -0.0001030520, 0.5765285950,
                      0.6478191338, -0.0001163496, -0.0001030520, -0.0001163496, 0.4435551436)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.3037139, 0.3025143, 0.6612807)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_mu*(1-expected_mu))),TOLERANCE_STRICT)

    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    preds <- predict(gp_model, gp_coords_pred = coords,
                     predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),TOLERANCE_STRICT)

    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-63.6205917),TOLERANCE_STRICT)

    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
    opt <- optim(par=c(1,0.1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
    cov_pars <- c(0.9419234, 0.1866877)
    expect_lt(sum(abs(opt$par-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(abs(opt$value-(63.6126363)),TOLERANCE_MEDIUM)
    expect_equal(as.integer(opt$counts[1]), 47)

    ###################
    ## Random coefficient GPs
    ###################
    probs_RC <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3))
    y_RC <- as.numeric(sim_rand_unif(n=n, init_c=0.543) < probs_RC)
    init_cov_pars_RC <- rep(init_cov_pars, 3)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           y = y_RC, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov=0.5, maxit=1000, init_cov_pars=init_cov_pars_RC, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    expected_values <- c(0.3701097, 0.2846740, 2.1160325, 0.3305266, 0.1241462, 0.1846456)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 39)
    # Prediction
    gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential", likelihood = "bernoulli_probit")
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    pred <- gp_model$predict(y = y_RC, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(1,0.1,0.8,0.15,1.1,0.08),
                             predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.18346008, 0.03479258, -0.17247579)
    expected_cov <- c(1.039879e+00, 7.521981e-01, -3.256500e-04, 7.521981e-01,
                      8.907289e-01, -6.719282e-05, -3.256500e-04, -6.719282e-05, 9.147899e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(1,0.1,0.8,0.15,1.1,0.08),y=y_RC)
    expect_lt(abs(nll-65.1768199),TOLERANCE_MEDIUM)

    ###################
    ##  Multiple cluster IDs
    ###################
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, cluster_ids = cluster_ids,likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky",
                                           params = list(optimizer_cov = "gradient_descent", lr_cov=0.2,
                                                         use_nesterov_acc=FALSE, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    cov_pars <- c(1.0132099, 0.2121574)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 4)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        cluster_ids = cluster_ids,likelihood = "bernoulli_probit")
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.1612322, 0.0000000, 0.9866054)
    expected_cov <- c(1.2200315255, 0.0000000000, 0.0003369428, 0.0000000000, 1.5000000000, 0.0000000000, 0.0003369428, 0.0000000000, 1.0744784423)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)

    # Matern with shape estimated
    if (!SKIP_BESSEL_COV_TESTS) {
      params = OPTIM_PARAMS_BFGS
      params$init_cov_pars <- c(1,mean(dist(coords))/3,1.5)
      params$maxit=10
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern_estimate_shape", matrix_inversion_method = "cholesky",
                                             cov_fct_shape = 1.5, y = y, params = params, likelihood = "bernoulli_probit") , file='NUL')
      cov_pars_other <- c(0.6289098, 0.1786315, 70.6673764)
      num_it_other <- 10
      nll_opt_other <- 63.07716
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_other)),TOLERANCE_MEDIUM)
      expect_equal(gp_model$get_num_optim_iter(), num_it_other)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_other), TOLERANCE_MEDIUM)
    }

    ###########################
    ## Use of weights
    ###########################
    nws <- 50
    gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5,0.1),y=y) + 0.
    params = OPTIM_PARAMS_BFGS
    params$init_cov_pars <- c(1,mean(dist(coords))/3)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 1.5, y = y, params = params, likelihood = "bernoulli_probit") , file='NUL')
    deltas <- c(0,1e-5,1e-1)
    for (i in 1:length(deltas)) {
      delta <- deltas[i]
      weights = c(rep(1+delta,nws),rep(1-delta,n-nws))
      gp_model_weights <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                  weights = weights, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky")
      nll_weighted <- gp_model_weights$neg_log_likelihood(cov_pars=c(0.5,0.1),y=y)
      capture.output( gp_model_weights <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                                     cov_fct_shape = 1.5, y = y, weights = weights,
                                                     params = params, likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky") , file='NUL')
      if (delta == 0) {
        expect_lt(abs(nll-nll_weighted), 1e-12)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-gp_model_weights$get_cov_pars(std_err = FALSE))),1e-12)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-gp_model_weights$get_current_neg_log_likelihood()), 1e-8)
      } else if (delta <= 1e-5) {
        expect_lt(abs(nll-nll_weighted), 3e-5)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-gp_model_weights$get_cov_pars(std_err = FALSE))),1e-5)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-gp_model_weights$get_current_neg_log_likelihood()), 3e-5)
      } else if (delta <= 1e-1) {
        expect_lt(abs(nll-nll_weighted), 0.3)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-gp_model_weights$get_cov_pars(std_err = FALSE))),0.1)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-gp_model_weights$get_current_neg_log_likelihood()), 0.4)
      }
    }

    ###########################
    ## likelihood_learning_rate parameter
    ###########################
    deltas <- c(1e-9,1e-4)
    for (i in 1:length(deltas)) {
      if (delta <= 1e-9) {
        expect_lt(abs(nll-nll_weighted), 1e-7)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-gp_model_weights$get_cov_pars(std_err = FALSE))),1e-8)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-gp_model_weights$get_current_neg_log_likelihood()), 1e-7)
      } else if (delta <= 1e-5) {
        expect_lt(abs(nll-nll_weighted), 1e-2)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-gp_model_weights$get_cov_pars(std_err = FALSE))),1e-3)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-gp_model_weights$get_current_neg_log_likelihood()), 1e-2)
      }
    }

    ###########################
    # Prediction with var_cor_pred option
    ###########################
    cov_par_pred <- c(0.5,0.1)
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky") , file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, cov_par=cov_par_pred, predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.3741743, -0.3737473, 0.3367670)
    expected_var <- c(0.4366123, 0.4365858, 0.3563977)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", matrix_inversion_method = "cholesky",
                                        likelihood = "bernoulli_probit_var_cor_pred_lr", likelihood_learning_rate = 1) , file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, cov_par=cov_par_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", matrix_inversion_method = "cholesky",
                                        likelihood = "bernoulli_probit_var_cor_pred_lr", likelihood_learning_rate = 1+1e-6) , file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, cov_par=cov_par_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", matrix_inversion_method = "cholesky",
                                        likelihood = "bernoulli_probit_var_cor_pred_lr", likelihood_learning_rate = 2) , file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, cov_par=cov_par_pred, predict_var = TRUE, predict_response = FALSE)
    expected_var <- c(0.4046473, 0.4049691, 0.3088420)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
  })

  test_that("Binary classification with Gaussian process model with multiple observations at the same location", {

    eps_multiple <- as.vector(L_multiple %*% b_multiple)
    probs <- pnorm(eps_multiple)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.9341) < probs)
    params = DEFAULT_OPTIM_PARAMS
    init_cov_pars = c(1,mean(dist(unique(coords_multiple)))/3)
    params$init_cov_pars = init_cov_pars

    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           y = y,likelihood = "bernoulli_probit", matrix_inversion_method = "cholesky",
                                           params = params), file='NUL')
    cov_pars <- c(0.6857065, 0.2363754)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.2633282, -0.2637633, -0.2637633)
    expected_cov <- c(0.9561355, 0.8535206, 0.8535206, 0.8535206, 1.0180227,
                      1.0180227, 0.8535206, 1.0180227, 1.0180227)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred_resp <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                                  cov_pars = c(1.5,0.15), predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_resp$mu-c(0.4253296, 0.4263502, 0.4263502))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_resp$var-c(0.2444243, 0.2445757, 0.2445757))),TOLERANCE_STRICT)

    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    preds <- predict(gp_model, gp_coords_pred = coords_multiple,
                     predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),TOLERANCE_STRICT)

    # Multiple cluster IDs and multiple observations
    coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    cluster_ids_pred = c(0L,3L,3L)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_cov <- c(0.9561355, 0.0000000, 0.0000000, 0.0000000, 1.5000000,
                      1.5000000, 0.0000000, 1.5000000, 1.5000000)
    expect_lt(sum(abs(pred$mu-c(-0.2633282, rep(0,2)))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred_resp <- gp_model$predict(y = y, gp_coords_pred = coord_test, cluster_ids_pred = cluster_ids_pred,
                                  cov_pars = c(1.5,0.15), predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred_resp$mu-c(0.4253296, 0.5000000, 0.5000000))),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_resp$var-c(0.2444243, 0.2500000, 0.2500000))),TOLERANCE_STRICT)

    # With linear regression term
    probs <- pnorm(eps_multiple + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.67981) < probs)
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none", matrix_inversion_method = "cholesky",
                                           y = y, X=X, params = params), file='NUL')
    cov_pars <- c(0.7462918, 0.0500844)
    coefs <- c(0.8545078, 1.7286015)
    num_it <- 168
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 39)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(0.07792923414, 1.27274858973, 2.65562981184)
    expected_var <- c(0.7267864819, 0.7329004392, 0.7329004392)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2), y=y)
    expect_lt(abs(nll-59.9183192),TOLERANCE_STRICT)
    # With fixed effects
    fixed_effects <- as.numeric(X%*%beta)
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2), y=y, fixed_effects=fixed_effects)
    expect_lt(abs(nll-42.8518187),TOLERANCE_STRICT)
  })

  test_that("Binary classification with one grouped random effects ", {

    probs <- pnorm(Z1 %*% b_gr_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.823431) < probs)
    init_cov_pars <- c(1)

    # Estimation using gradient descent
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters",
                                       init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.40255)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 62)
    # Can switch between likelihoods
    gp_model <- GPModel(group_data = group, likelihood = "gaussian")
    gp_model$set_likelihood("bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters",
                                       init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                       acc_rate_cov = 0.5, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars2 <- c(0.4012595)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars2)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 10)

    # Estimation using gradient descent and too large learning rate
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 10, use_nesterov_acc = FALSE, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 5)

    # Prediction
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, params = list(optimizer_cov = "gradient_descent",
                                                use_nesterov_acc = FALSE, lr_cov = 0.1, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.000000, -0.796538, -0.796538, 0.000000)
    expected_cov <- c(0.1133436, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
                      0.1407783, 0.1407783, 0.0000000, 0.0000000, 0.1407783,
                      0.1407783, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.4070775)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,6,11,16)])),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_response = TRUE)
    expected_mu <- c(0.5000000, 0.2279027, 0.2279027, 0.5000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    # Prediction for only new groups
    group_test <- c(-1,-1,-2,-2)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-rep(0,4))),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-rep(0,0.4070775))),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-rep(0.5,4))),TOLERANCE_STRICT)
    # Prediction for only new cluster_ids
    cluster_ids_pred <- c(-1L,-1L,-2L,-2L)
    group_test <- c(1,99999,3,3)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, cluster_ids_pred = cluster_ids_pred,
                    predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-rep(0,4))),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-rep(0.4070775261,4))),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, cluster_ids_pred = cluster_ids_pred,
                    predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-rep(0.5,4))),TOLERANCE_STRICT)

    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences,]
    group_unique <- unique(group)
    preds <- predict(gp_model, group_data_pred = group_unique,
                     predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),1E-6)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),1E-6)

    # Estimation using Nelder-Mead
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "nelder_mead", delta_rel_conv=1e-6, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.4027452)),TOLERANCE_STRICT)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-c(0.0000000, -0.7935873, -0.7935873, 0.0000000))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-c(0.1130051, 0.1401125, 0.1401125, 0.4027452))),TOLERANCE_STRICT)

    # Estimation using lbfgs
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "lbfgs", init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.4025750768)),TOLERANCE_STRICT)

    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-65.8590638),TOLERANCE_STRICT)

    # Do optimization using optim
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    opt <- optim(par=c(2), fn=gp_model$neg_log_likelihood, y=y, method="Brent", lower=0, upper=1E9)
    cov_pars <- c(0.40255)
    expect_lt(sum(abs(opt$par-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(abs(opt$value-(65.2599674)),TOLERANCE_MEDIUM)

    ###########################
    # Use of weights
    ###########################
    nws <- 50
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5),y=y) + 0.
    params = OPTIM_PARAMS_BFGS
    params$init_cov_pars <- c(1.)
    capture.output( gp_model <- fitGPModel(group_data = group, y = y, params = params, likelihood = "bernoulli_probit") , file='NUL')

    deltas <- c(0,1e-5,1e-1)
    for (i in 1:length(deltas)) {
      delta <- deltas[i]
      weights = c(rep(1+delta,nws),rep(1-delta,n-nws))
      gp_model_weights <- GPModel(group_data = group, weights = weights, likelihood = "bernoulli_probit")
      nll_weighted <- gp_model_weights$neg_log_likelihood(cov_pars=c(0.5),y=y)
      capture.output( gp_model_weights <- fitGPModel(group_data = group, y = y, weights = weights,
                                                     params = params, likelihood = "bernoulli_probit") , file='NUL')
      if (delta == 0) {
        expect_lt(abs(nll-nll_weighted), 1e-99)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-gp_model_weights$get_cov_pars(std_err = FALSE))),1e-12)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-gp_model_weights$get_current_neg_log_likelihood()), 1e-12)
      } else if (delta <= 1e-5) {
        expect_lt(abs(nll-nll_weighted), 1e-4)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-gp_model_weights$get_cov_pars(std_err = FALSE))),1e-5)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-gp_model_weights$get_current_neg_log_likelihood()), 9e-5)
      } else if (delta <= 1e-1) {
        expect_lt(abs(nll-nll_weighted), 1)
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-gp_model_weights$get_cov_pars(std_err = FALSE))),0.05)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-gp_model_weights$get_current_neg_log_likelihood()), 1)
      }
    }

    ###########################
    # Prediction with var_cor_pred option
    ###########################
    cov_par_pred <- c(0.5)
    group_test <- c(1,3,9999)
    capture.output( gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit") , file='NUL')
    pred <- predict(gp_model, y=y, group_data_pred = group_test, cov_par=cov_par_pred, predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c(0.0000000, -0.8518212, 0.0000000)
    expected_var <- c(0.1195286, 0.1536399, 0.5000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    capture.output( gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit_var_cor_pred_lr", likelihood_learning_rate = 1) , file='NUL')
    pred <- predict(gp_model, y=y, group_data_pred = group_test, cov_par=cov_par_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    capture.output( gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit_var_cor_pred_lr", likelihood_learning_rate = 1+1e-6) , file='NUL')
    pred <- predict(gp_model, y=y, group_data_pred = group_test, cov_par=cov_par_pred, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
    capture.output( gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit_var_cor_pred_lr", likelihood_learning_rate = 2) , file='NUL')
    pred <- predict(gp_model, y=y, group_data_pred = group_test, cov_par=cov_par_pred, predict_var = TRUE, predict_response = FALSE)
    expected_var <- c(0.06787762, 0.09076508, 0.50000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_STRICT)
  })

  test_that("Binary classification for combined Gaussian process and grouped random effects ", {

    probs <- pnorm(L %*% b_1 + Z1 %*% b_gr_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.67341) < probs)
    init_cov_pars <- c(1,1,mean(dist(coords))/3)

    # Estimation using gradient descent
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        group_data = group, likelihood = "bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                       lr_cov = 0.2, use_nesterov_acc = FALSE,
                                                       convergence_criterion = "relative_change_in_parameters", init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    cov_pars <- c(0.3181509, 1.2788456, 0.1218680)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 55)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-63.74320741),TOLERANCE_STRICT)

    # Prediction
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           group_data = group, y = y, params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                                                    use_nesterov_acc = FALSE, lr_cov = 0.2, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    coord_test <- cbind(c(0.1,0.21,0.7),c(0.9,0.91,0.55))
    group_test <- c(1,3,9999)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1217634, -0.9592585, -0.2694489)
    expected_cov <- c(1.0745455607, 0.2190063794, 0.0040797451, 0.2190063794,
                      1.0089298170, 0.0000629706, 0.0040797451, 0.0000629706, 1.0449941968)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test,
                    predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test, predict_response = TRUE)
    expected_mu <- c(0.5336859, 0.2492699, 0.4252731)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)

    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model)
    pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]), predict_response = FALSE)
    expect_lt(sum(abs(training_data_random_effects[,2] - pred_GP$mu)),1E-6)
    # Grouped REs
    preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords, predict_response = FALSE)
    pred_RE <- preds$mu - pred_GP$mu
    expect_lt(sum(abs(training_data_random_effects[,1] - pred_RE)),1E-6)

    # Estimation using Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        group_data = group, likelihood = "bernoulli_probit")
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "nelder_mead", delta_rel_conv=1E-8, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(0.3181320, 1.2795124, 0.1218866))),TOLERANCE_STRICT)

    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(1.1,0.9,0.2),y=y)
    expect_lt(abs(nll-65.7219266),TOLERANCE_STRICT)

    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        group_data = group, likelihood = "bernoulli_probit")
    capture.output( opt <- optim(par=c(0.5,1,0.1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead"), file='NUL')
    cov_pars <- c(0.3181509, 1.2788456, 0.1218680)
    expect_lt(sum(abs(opt$par-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(abs(opt$value-(63.7432077)),TOLERANCE_MEDIUM)
    expect_equal(as.integer(opt$counts[1]), 88)
  })

  test_that("Combined GP and grouped random effects model with random coefficients ", {

    probs <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3) +
                     Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.9867234) < probs)
    init_cov_pars <- c(rep(1,3),rep(c(1,mean(dist(coords))/3),3))

    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                         lr_cov = 0.2, use_nesterov_acc = FALSE, maxit=10, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    expected_values <- c(0.09859312, 0.35813763, 0.50164573, 0.67372019,
                         0.08825524, 0.77807532, 0.10896128, 1.03921290, 0.09538707)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 10)

    # Prediction
    gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential", likelihood = "bernoulli_probit",
                        group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1)
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    group_data_pred = cbind(c(1,1,7),c(2,1,3))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(1.612451, 1.147407, -1.227187)
    expected_cov <- c(1.63468526, 1.02982815, -0.01916993, 1.02982815,
                      1.43601348, -0.03404720, -0.01916993, -0.03404720, 1.55017397)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)

    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-71.4286594),TOLERANCE_MEDIUM)
  })

  test_that("Combined GP and grouped random effects model with cluster_id's not constant ", {

    probs <- pnorm(L %*% b_1 + Z1 %*% b_gr_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    init_cov_pars <- c(1,1,mean(dist(coords[cluster_ids==1,]))/3)

    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                                           y = y, cluster_ids = cluster_ids,likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", lr_cov=0.2, use_nesterov_acc = FALSE,
                                                         init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    cov_pars <- c(0.276476226, 0.007278016, 0.132195703)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 261)

    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    group_data_pred = c(1,1,9999)
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                        cluster_ids = cluster_ids,likelihood = "bernoulli_probit")
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,1,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1074035, 0.0000000, 0.2945508)
    expected_cov <- c(0.98609786, 0.00000000, -0.02013244, 0.00000000,
                      2.50000000, 0.00000000, -0.02013244, 0.00000000, 2.28927616)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
  })

  test_that("FITC preconditioner for several clusters ", {

    # The inducing points of the preconditioner are determined separately for every cluster. Determining
    # them only for the first cluster left the other clusters without preconditioner components
    probs_pc <- pnorm(L %*% b_1)
    y_pc <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs_pc)
    cov_pars_pc <- c(1, 0.1)
    params_pc <- list(cg_preconditioner_type = "fitc", fitc_piv_chol_preconditioner_rank = 15,
                      num_rand_vec_trace = 500, reuse_rand_vec_trace = TRUE, seed_rand_vec_trace = 1,
                      cg_delta_conv = 1E-6)
    nll_pc <- function(inv_method, gp_approx_loc, cluster_ids_loc) {
      capture.output( gp_loc <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        likelihood = "bernoulli_probit", gp_approx = gp_approx_loc,
                                        num_neighbors = 20, num_ind_points = 20, vecchia_ordering = "none",
                                        cluster_ids = cluster_ids_loc,
                                        matrix_inversion_method = inv_method), file = 'NUL')
      gp_loc$set_optim_params(params = params_pc)
      capture.output( nll_loc <- gp_loc$neg_log_likelihood(cov_pars = cov_pars_pc, y = y_pc), file = 'NUL')
      nll_loc
    }
    for (gp_approx_pc in c("vecchia", "full_scale_vecchia")) {
      nll_chol_pc <- nll_pc("cholesky", gp_approx_pc, cluster_ids)
      nll_iter_pc <- nll_pc("iterative", gp_approx_pc, cluster_ids)
      expect_true(is.finite(nll_iter_pc))
      # the log determinant is estimated stochastically, so the two do not agree exactly
      expect_lt(abs(nll_iter_pc - nll_chol_pc), 2 * TOLERANCE_ITERATIVE)
    }

  })

  test_that("Repeated predictions with the same model give the same result ", {

    # The predictive covariance matrix of several grouped random effects is estimated stochastically when
    # the iterative methods are used. The random number generator has to be seeded at every prediction:
    # otherwise a prediction that is repeated with unchanged arguments returns a different covariance matrix
    y_rp <- as.numeric(sim_rand_unif(n = n, init_c = 0.2341) < pnorm(Z1 %*% b_gr_1 + Z2 %*% b_gr_2))
    cov_pars_rp <- c(0.8, 0.6)
    group_test_rp <- cbind(c(1, 2, 9999), c(1, 3, 9999))
    for (inv_method_rp in c("cholesky", "iterative")) {
      capture.output( gp_model_rp <- GPModel(group_data = cbind(group, group2), likelihood = "bernoulli_probit",
                                             matrix_inversion_method = inv_method_rp), file = 'NUL')
      gp_model_rp$set_optim_params(params = list(num_rand_vec_trace = 100, seed_rand_vec_trace = 1))
      capture.output( pred_rp_1 <- predict(gp_model_rp, y = y_rp, group_data_pred = group_test_rp,
                                           cov_pars = cov_pars_rp, predict_cov_mat = TRUE,
                                           predict_response = FALSE), file = 'NUL')
      capture.output( pred_rp_2 <- predict(gp_model_rp, y = y_rp, group_data_pred = group_test_rp,
                                           cov_pars = cov_pars_rp, predict_cov_mat = TRUE,
                                           predict_response = FALSE), file = 'NUL')
      expect_lt(sum(abs(pred_rp_1$mu - pred_rp_2$mu)), TOLERANCE_STRICT, label = paste0("predictive mean (", inv_method_rp, ")"))
      expect_lt(sum(abs(as.vector(pred_rp_1$cov) - as.vector(pred_rp_2$cov))), TOLERANCE_STRICT,
                label = paste0("predictive covariance (", inv_method_rp, ")"))
    }

  })

  test_that("FITC preconditioner with inducing points from the cover tree ", {

    # The cover tree determines the number of inducing points of the preconditioner itself, which then
    # differs from 'fitc_piv_chol_preconditioner_rank'. The random vectors for the stochastic estimate
    # of the log determinant have to have the rank of the preconditioner that was actually constructed
    probs_ctp <- pnorm(L %*% b_1)
    y_ctp <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs_ctp)
    cov_pars_ctp <- c(1, 0.1)
    params_ctp <- list(cg_preconditioner_type = "fitc", fitc_piv_chol_preconditioner_rank = 15,
                       num_rand_vec_trace = 500, reuse_rand_vec_trace = TRUE, seed_rand_vec_trace = 1,
                       cg_delta_conv = 1E-6)
    nll_ctp <- function(inv_method, gp_approx_loc) {
      capture.output( gp_loc <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        likelihood = "bernoulli_probit", gp_approx = gp_approx_loc,
                                        num_neighbors = 20, num_ind_points = 20, vecchia_ordering = "none",
                                        ind_points_selection = "cover_tree", cover_tree_radius = 0.2,
                                        matrix_inversion_method = inv_method), file = 'NUL')
      gp_loc$set_optim_params(params = params_ctp)
      capture.output( nll_loc <- gp_loc$neg_log_likelihood(cov_pars = cov_pars_ctp, y = y_ctp), file = 'NUL')
      nll_loc
    }
    for (gp_approx_ctp in c("vecchia", "full_scale_vecchia")) {
      nll_chol_ctp <- nll_ctp("cholesky", gp_approx_ctp)
      nll_iter_ctp <- nll_ctp("iterative", gp_approx_ctp)
      expect_true(is.finite(nll_iter_ctp))
      # the log determinant is estimated stochastically, so the two do not agree exactly
      expect_lt(abs(nll_iter_ctp - nll_chol_ctp), 2 * TOLERANCE_ITERATIVE)
    }

  })

  test_that("Binary classification with linear predictor and grouped random effects model ", {

    probs <- pnorm(Z1 %*% b_gr_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.542) < probs)
    init_cov_pars = c(1)

    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                     optimizer_coef = "gradient_descent", lr_cov = 0.05, lr_coef = 1,
                                                     use_nesterov_acc = TRUE, acc_rate_cov = 0.2, acc_rate_coef = 0.1,
                                                     init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.4072025)
    coef <- c(-0.1113238, 1.5178339)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 43)

    # Estimation using Nelder-Mead
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead", delta_rel_conv=1e-12,
                                                     init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.399973)
    coef <- c(-0.1109516, 1.5149596)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    # init_cov_pars not given
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead", delta_rel_conv=1e-12, init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.399973)
    coef <- c(-0.1109516, 1.5149596)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)

    # Estimation using lbfgs
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", init_coef_aux_pars_from_iid_model = FALSE))
    cov_pars <- c(0.3996146704)
    coef <- c(-0.1109363315, 1.5150072519)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 9)

    # Prediction
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                     optimizer_coef = "gradient_descent",
                                                     use_nesterov_acc=FALSE, lr_coef=1, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.81132150, -0.08574588, 0.21768684, 1.40591430)
    expected_cov <- c(0.1380238, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.1688248, 0.1688248,
                      0.0000000, 0.0000000, 0.1688248, 0.1688248, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.4051185)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test, predict_response = TRUE)
    expected_mu <- c(0.2234684, 0.4683923, 0.5797886, 0.8821984)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)

    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences,]
    group_unique <- unique(group)
    X_zero <- cbind(rep(0,length(group_unique)),rep(0,length(group_unique)))
    preds <- predict(gp_model, group_data_pred = group_unique, X_pred = X_zero,
                     predict_response = FALSE, predict_var = TRUE)
    expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)),TOLERANCE_STRICT)

    # Standard deviations
    capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                                           y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                                     optimizer_coef = "gradient_descent", init_cov_pars=init_cov_pars,
                                                                     use_nesterov_acc = TRUE, lr_cov = 0.1, lr_coef = 1, init_coef_aux_pars_from_iid_model = FALSE)),
                    file='NUL')
    cov_pars <- c(0.4016599868 )
    coef <- c(-0.1116235586,  0.2568338470 , 1.5161515464,  0.2637361920)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)

    # Providing initial covariance parameters and coefficients
    cov_pars <- c(1)
    coef <- c(2,5)
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(maxit=0, init_cov_pars=cov_pars, init_coef=coef,
                                                     optimizer_cov = "gradient_descent", init_coef_aux_pars_from_iid_model = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)

    # Large data
    n_L <- 1e6 # number of samples
    m_L <- n_L/10 # number of categories / levels for grouping variable
    group_L <- rep(1,n_L) # grouping variable
    for(i in 1:m_L) group_L[((i-1)*n_L/m_L+1):(i*n_L/m_L)] <- i
    keps <- 1E-10
    b1_L <- qnorm(sim_rand_unif(n=m_L, init_c=0.671)*(1-keps) + keps/2)
    X_L <- cbind(rep(1,n_L),sim_rand_unif(n=n_L, init_c=0.8671)-0.5) # design matrix / covariate data for fixed effect
    probs_L <- pnorm(b1_L[group_L] + X_L%*%beta)
    y_L <- as.numeric(sim_rand_unif(n=n_L, init_c=0.12378)*(1-keps) + keps/2 < probs_L)
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- fitGPModel(group_data = group_L, likelihood = "bernoulli_probit",
                           y = y_L, X=X_L, params = list(optimizer_cov = "gradient_descent",
                                                         optimizer_coef = "gradient_descent", lr_cov = 0.05, lr_coef = 0.1,
                                                         use_nesterov_acc = TRUE, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-0.9757876802)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(0.09848153264, 1.99446139138))),TOLERANCE_MEDIUM)

  })

  test_that("Binary classification with linear predictor and Gaussian process model ", {

    probs <- pnorm(L %*% b_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.199) < probs)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- c(1,mean(dist(coords))/3)

    # Estimation
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = params)
    cov_pars <- c(1.2660987164, 0.2854664658)
    coefs <- c(0.2041076447, 1.4663366438)
    nll <- 48.41567975
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)

    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = c(1,0.2))
    expected_mu <- c(-0.6873889499, 0.3334397127, 2.5116340251)
    expected_var <- c(0.7205439641, 0.7196871780, 0.4591627357)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # Estimation using Nelder-Mead
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead",
                                                     maxit=1000, delta_rel_conv=1e-12, init_cov_pars = params$init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(1.2717516, 0.2875537))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(0.1999365, 1.4666199))),TOLERANCE_MEDIUM)

    # Standard deviations
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           y = y, X=X, params = params),
                    file='NUL')
    coef <- c(0.2041076447, 0.5402831971, 1.4663366438, 0.3028191307)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)

  })

  test_that("Binary classification with Gaussian process model and logit link function", {

    probs <- 1/(1+exp(- L %*% b_1))
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- c(1,mean(dist(coords))/3)
    params$lr_cov=0.01

    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_logit",
                                           y = y, params = params)
                    , file='NUL')
    cov_pars <- c(1.4300136, 0.1891952)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 85)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.7792960, -0.7876208, 0.5476390)
    expected_cov <- c(1.024266883e+00, 9.215203622e-01, 5.561463409e-05, 9.215203622e-01, 1.022897212e+00, 2.028646043e-05, 5.561463409e-05, 2.028646043e-05, 7.395745025e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.3442815, 0.3426873, 0.6159933)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_mu*(1-expected_mu))),TOLERANCE_STRICT)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-66.299571),TOLERANCE_STRICT)
  })

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

  test_that("test negative log-likelihood metric of the GPBoost algorithm for the gaussian_heteroscedastic likelihood ", {

    # The "test_neg_log_likelihood" metric of the GPBoost algorithm on validation data, together with an independent
    # calculation: for every validation point, the integral of a reference density (written with base R functions) over
    # the latent predictive distribution of the first predictor, given the tree-ensemble values of the other predictors
    validation_test_nll <- function(likelihood, y, x, group, log_dens, additional_likelihood_data = NULL, nrounds = 5) {
      tr <- seq(1, length(y), by = 2)
      va <- seq(2, length(y), by = 2)
      dtrain <- gpb.Dataset(data = x[tr, , drop = FALSE], label = y[tr])
      dvalid <- gpb.Dataset.create.valid(dtrain, data = x[va, , drop = FALSE], label = y[va])
      gp_model <- GPModel(group_data = group[tr], likelihood = likelihood,
                          additional_likelihood_data = if (is.null(additional_likelihood_data)) NULL else additional_likelihood_data[tr])
      gp_model$set_optim_params(params = list(optimizer_cov = "lbfgs", maxit = 300, init_coef_aux_pars_from_iid_model = FALSE))
      gp_model$set_prediction_data(group_data_pred = group[va])
      bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = nrounds, learning_rate = 0.1, max_depth = 2, min_data_in_leaf = 5,
                       valids = list(valid = dvalid), verbose = 0, deterministic = TRUE)
      metric <- unlist(bst$record_evals$valid$test_neg_log_likelihood$eval)[nrounds]
      pred <- predict(bst, data = x[va, , drop = FALSE], group_data_pred = group[va], predict_var = TRUE, pred_latent = TRUE, num_iteration = nrounds)
      raw <- predict(bst, data = x[va, , drop = FALSE], ignore_gp_model = TRUE, pred_latent = TRUE, num_iteration = nrounds)
      num_blocks <- length(raw) / length(va)
      extra <- if (num_blocks > 1) matrix(sapply(2:num_blocks, function(k) raw[(seq_along(va) - 1) * num_blocks + k]), nrow = length(va)) else matrix(0, length(va), 1)
      mean_eta <- pred$fixed_effect + pred$random_effect_mean
      sd_eta <- sqrt(pred$random_effect_cov)
      aux <- gp_model$get_aux_pars()
      reference <- mean(sapply(seq_along(va), function(i) {
        integrand <- function(e) exp(log_dens(y[va][i], e, extra[i, ], aux)) * dnorm(e, mean_eta[i], sd_eta[i])
        -log(integrate(integrand, mean_eta[i] - 12 * sd_eta[i], mean_eta[i] + 12 * sd_eta[i], rel.tol = 1e-10)$value)
      }))
      c(metric = unname(metric), reference = reference)
    }
    n_v <- 200
    group_v <- rep(1:20, each = 10)
    x_v <- matrix(sim_rand_unif(n_v, 0.61), ncol = 1)
    eta_v <- 0.2 + 0.6 * x_v[, 1] + 0.4 * qnorm(sim_rand_unif(20, 0.27))[group_v]
    u1_v <- sim_rand_unif(n_v, 0.44)
    u2_v <- sim_rand_unif(n_v, 0.83)
    zeta_v <- -0.5 + 1.2 * x_v[, 1]
    zeta2_v <- 0.3 + 0.8 * x_v[, 1]
    # Reference log-densities, vectorized in the first predictor e (z: values of the other predictors, a: auxiliary parameters)
    cases <- list(
      gaussian_heteroscedastic = list(y = qnorm(u2_v, eta_v, exp(zeta_v / 2)),
        log_dens = function(y, e, z, a) dnorm(y, e, exp(z[1] / 2), log = TRUE),
        tol_reference = 1e-6))
    expected <- c(gaussian_heteroscedastic = 1.71305334)
    for (lik in names(cases)) {
      res <- validation_test_nll(lik, cases[[lik]]$y, x_v, group_v, cases[[lik]]$log_dens)
      expect_lt(abs(res[["metric"]] - res[["reference"]]), cases[[lik]]$tol_reference)
      expect_lt(abs(res[["metric"]] - expected[[lik]]), TOLERANCE_MEDIUM)
    }
  })

}
