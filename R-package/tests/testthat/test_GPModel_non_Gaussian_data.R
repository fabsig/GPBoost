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
  # Same for tolerances that are defined locally in a test: only ever RELAX them, never tighten them
  relax_tolerance <- function(tol) if (USE_STRICT_TOLERANCES) tol else max(2 * tol, 0.5)
  # Separate helper for ABSOLUTE differences of negative log-likelihoods (scale 100-1000 here)
  relax_tolerance_nll <- function(tol) if (USE_STRICT_TOLERANCES) tol else max(3 * tol, 3)
  # Separate helper for the very strict tolerances (1e-6). 'relax_tolerance' must not be used for
  # these, since its lower bound of 0.5 would make such a test meaningless. Deviations of a few 1e-6
  # occur under valgrind in particular, which does not reproduce floating point arithmetic bit-wise
  # (it rounds the 80 bit intermediate results of x87 to 64 bit and its libm differs)
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
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.05))
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

  test_that("Binary classification Gaussian process model with Vecchia approximation", {
    params_vecchia <- c(DEFAULT_OPTIM_PARAMS, cg_delta_conv = sqrt(1e-6),
                        num_rand_vec_trace = 500, cg_preconditioner_type = "pivoted_cholesky",
                        fitc_piv_chol_preconditioner_rank = dim(coords)[1] - 1 )
    init_cov_pars = c(1,mean(dist(coords))/3)
    params_vecchia$init_cov_pars = init_cov_pars
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars = init_cov_pars
    params_mult <- DEFAULT_OPTIM_PARAMS
    init_cov_pars_mult = c(1,mean(dist(unique(coords_multiple)))/3)
    params_mult$init_cov_pars = init_cov_pars_mult
    params_vecchia_mult <- params_vecchia
    params_vecchia_mult$init_cov_pars = init_cov_pars_mult
    params_vecchia_mult$fitc_piv_chol_preconditioner_rank <- dim(unique(coords_multiple))[1]

    # Simulate data and define expected values
    probs <- pnorm(L %*% b_1) # note: linear predictor is not included in simulation
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.19341) < probs)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    init_cov_pars <- c(1,mean(dist(coords))/3)
    cov_pars_pred_eval = c(1,0.2)
    cov_pars <- c(0.92350821208, 0.05944214192)
    coefs <- c(0.3983333, -0.2653886)
    num_it <- 17
    expected_mu <- c(0.3389905, 0.1512445, -0.1039307)
    expected_cov <- c(0.6193228722, 0.5503216948, -0.0001420698, 0.5503216948,
                      0.6159348965, -0.0001556274, -0.0001420698, -0.0001556274, 0.4291674143)
    expected_mu_resp <- c(0.6050312, 0.5473537, 0.4653610)
    expected_var_resp <- c(0.2389684, 0.2477576, 0.2488001)
    expected_nll <- 67.18342059
    # Estimation, prediction, and likelihood evaluation without Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, X = X, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE,
                    predict_var = TRUE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu_resp)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var_resp)),TOLERANCE_MEDIUM)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
    expect_lt(abs(nll-expected_nll),TOLERANCE_STRICT)
    # No linear regression term without Vecchia approximation
    cov_pars_no_X <- c(0.6875476, 0.1062862 )
    mu_no_X <- c(0.01874013, 0.01200800, 0.20498871)
    var_no_X <- c(0.6105248, 0.6093745, 0.4235374)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_no_X)),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval)
    expect_lt(sum(abs(pred$mu-mu_no_X)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-var_no_X)),TOLERANCE_MEDIUM)
    # With duplicates and linear regression term without Vecchia approximation
    eps_multiple <- as.vector(L_multiple %*% b_multiple)
    probs_multiple <- pnorm(eps_multiple)
    y_multiple <- as.numeric(sim_rand_unif(n=n, init_c=0.2818) < probs_multiple)
    coord_test_multiple <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    cov_pars_multiple <- c(0.8263711, 0.1240696 )
    coefs_multiple <- c( 0.6168877, 0.1381717)
    num_it_multiple <- 17
    expected_mu_multiple <- c(-0.01076580, 0.07873293, 0.18927032)
    expected_var_multiple <- c(0.5653402, 0.6019163, 0.6019163)
    nll_multiple <- 58.671494
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y_multiple, X = X, params = params_mult), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_multiple)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_multiple)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_multiple)
    pred <- predict(gp_model, y=y_multiple, gp_coords_pred = coord_test_multiple, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred_eval)
    expect_lt(sum(abs(pred$mu-expected_mu_multiple)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var_multiple)),TOLERANCE_MEDIUM)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y_multiple)
    expect_lt(abs(nll-nll_multiple),TOLERANCE_STRICT)

    for (inv_method in c("cholesky", "iterative")) {
      if(inv_method == "iterative") {
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
        tolerance_loc_3 <- 2*TOLERANCE_ITERATIVE
        loop_cg_PC = c("pivoted_cholesky", "vadu", "fitc")
      } else {
        tolerance_loc_1 <- TOLERANCE_STRICT
        tolerance_loc_2 <- TOLERANCE_MEDIUM
        tolerance_loc_3 <-TOLERANCE_STRICT
        loop_cg_PC = c("vadu")
      }
      nsim_var_pred <- 10000
      for (cg_preconditioner_type in loop_cg_PC) {
        params_vecchia$cg_preconditioner_type <- cg_preconditioner_type
        # Vecchia approximation with no ordering
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                            num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        capture.output( fit(gp_model, y = y, X = X, params = params_vecchia)
                        , file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),tolerance_loc_1)
        if(inv_method != "iterative") {
          expect_equal(gp_model$get_num_optim_iter(), num_it)
        }
        # Prediction
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                     num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_cov_mat = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_var = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_response = TRUE, predict_var = TRUE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu_resp)),tolerance_loc_1)
          expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
        }
        # Likelihood evaluation
        nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
        expect_lt(abs(nll-expected_nll),tolerance_loc_1)

        if(inv_method == "iterative" && cg_preconditioner_type == "pivoted_cholesky"){
          ## Cannot change cg_preconditioner_type after a model has been fitted
          expect_error( capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", init_cov_pars=init_cov_pars,
                                                                           lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                                           convergence_criterion = "relative_change_in_parameters",
                                                                           cg_delta_conv = 1e-6, num_rand_vec_trace = 500,
                                                                           cg_preconditioner_type = "vadu", init_coef_aux_pars_from_iid_model = FALSE)), file='NUL'))
        }

        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {# some tests are only run for one preconditioner
          ############################
          # Vecchia approximation with random ordering
          ############################
          capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                 vecchia_ordering="random", likelihood = "bernoulli_probit",
                                                 gp_approx = "vecchia",  num_neighbors = n-1,
                                                 y = y, X = X, params = params_vecchia,
                                                 matrix_inversion_method = inv_method), file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_1)
          if(inv_method != "iterative") {
            expect_equal(gp_model$get_num_optim_iter(), num_it)
          }
          # Prediction
          gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                       num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_cov_mat = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
          expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),tolerance_loc_3)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_var = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
          expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_response = TRUE, predict_var = TRUE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu_resp)),tolerance_loc_1)
          expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
          # Likelihood evaluation
          nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
          expect_lt(abs(nll-expected_nll),2*tolerance_loc_1)

          #######################
          ## Less neighbors than observations
          #######################
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                              num_neighbors = 30, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method), file='NUL')
          capture.output( fit(gp_model, y = y, X = X, params = params_vecchia)
                          , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_2)
          if(inv_method != "iterative") {
            expect_equal(gp_model$get_num_optim_iter(), num_it)
          }
          # Prediction
          gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                       num_neighbors_pred = 30, nsim_var_pred = nsim_var_pred)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_cov_mat = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          mu_less_neig <- c(0.3368557, 0.1492578, -0.1034736)
          cov_less_neig <- c(0.6193174862, 0.5503175873, -0.0001440701, 0.5503175873,
                             0.6159313469, -0.0001546077, -0.0001440701, -0.0001546077, 0.4292547351)
          mu_resp_less_neig <- c(0.6043853, 0.5467346, 0.4655140)
          var_resp_less_neig <- c(0.2391037, 0.2478159, 0.2488107)
          expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$cov)-cov_less_neig)),0.2)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_var = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_1)
          expect_lt(sum(abs(as.vector(pred$var)-cov_less_neig[c(1,5,9)])),tolerance_loc_1)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_response = TRUE, predict_var = TRUE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-mu_resp_less_neig)),tolerance_loc_1)
          expect_lt(sum(abs(pred$var-var_resp_less_neig)),tolerance_loc_1)
          # Use vecchia_pred_type = "order_obs_first_cond_all"
          gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
          pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                          predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
          expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_1)
          expect_lt(sum(abs(as.vector(pred$cov)-cov_less_neig)),tolerance_loc_1)
          pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE,
                          predict_var = TRUE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
          expect_lt(sum(abs(pred$mu-mu_resp_less_neig)),tolerance_loc_1)
          expect_lt(sum(abs(pred$var-var_resp_less_neig)), tolerance_loc_1)
          # Use vecchia_pred_type = "latent_order_obs_first_cond_obs_only"
          gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_obs_only",
                                       nsim_var_pred = 2000)
          pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                          predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
          expected_cov_loc <- c(0.6193174862, 0.2835405301, -0.0001440701, 0.2835405301, 0.6159312648,
                                -0.0001525779, -0.0001440701, -0.0001525779, 0.4292547351)
          expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_loc)),tolerance_loc_1)
          # Use vecchia_pred_type = "order_obs_first_cond_obs_only"
          gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only")
          pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                          predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
          expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_loc)),1.5*tolerance_loc_1)
        }

        ############################
        # Predict training data random effects
        ############################
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                               num_neighbors = 30, vecchia_ordering = "none",
                                               matrix_inversion_method = inv_method,
                                               y = y, params = params_vecchia), file='NUL')
        training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
        gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only")
        preds <- predict(gp_model, gp_coords_pred = coords, predict_response = FALSE,
                         predict_var = TRUE)
        expect_lt(sum(abs(training_data_random_effects[,1] - preds$mu)),tolerance_loc_1)
        if(inv_method == "iterative"){
          expect_lt(mean(abs(training_data_random_effects[,2] - preds$var)), tolerance_loc_1) #Different RNG-Status
        } else {
          expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)), tolerance_loc_1)
        }

        ############################
        # No linear regression term
        ############################
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                               num_neighbors = n-1, vecchia_ordering = "random",
                                               matrix_inversion_method = inv_method,
                                               y = y, params = params_vecchia), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_no_X)),tolerance_loc_1)
        pred <- capture.output( predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                                        predict_response = FALSE, cov_pars = cov_pars_pred_eval), file='NUL')
        expect_lt(sum(abs(pred$mu-mu_no_X)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$var)-var_no_X)),tolerance_loc_2)

        ############################
        # With duplicates and linear regression term
        ############################
        capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                               likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                               num_neighbors = n-1, vecchia_ordering = "none",
                                               matrix_inversion_method = inv_method,
                                               y = y_multiple, X = X, params = params_vecchia_mult), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_multiple)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_multiple)),tolerance_loc_1)
        if(inv_method != "iterative") {
          expect_equal(gp_model$get_num_optim_iter(), num_it_multiple)
        }
        # Prediction
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                     num_neighbors_pred = n/4+1, nsim_var_pred = nsim_var_pred)
        pred <- predict(gp_model, y=y_multiple, gp_coords_pred = coord_test_multiple, X_pred = X_test,
                        predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred_eval)
        expect_lt(sum(abs(pred$mu-expected_mu_multiple)),tolerance_loc_2)
        expect_lt(sum(abs(pred$var-expected_var_multiple)),tolerance_loc_2)
        # Likelihood evaluation
        nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y_multiple)
        expect_lt(abs(nll-nll_multiple),tolerance_loc_1)
        # Predict training data random effects
        training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
        gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only",
                                     num_neighbors_pred = n/4, nsim_var_pred = nsim_var_pred)
        preds <- predict(gp_model, gp_coords_pred = coords_multiple, predict_response = FALSE,
                         predict_var = TRUE, X_pred = X)
        pred_mu_exp <- preds$mu - X %*% gp_model$get_coef(std_err = FALSE)
        expect_lt(sum(abs(training_data_random_effects[,1] - pred_mu_exp)),tolerance_loc_1)
        if(inv_method == "iterative"){
          expect_lt(mean(abs(training_data_random_effects[,2] - preds$var)), tolerance_loc_1) #Different RNG-Status
        } else {
          expect_lt(sum(abs(training_data_random_effects[,2] - preds$var)), tolerance_loc_1)
        }

      }# end loop cg_preconditioner_type in loop_cg_PC
    }# end loop inv_method in c("cholesky", "iterative")

    ## "vecchia" preconditioner
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                        num_neighbors = 30, vecchia_ordering = "none",
                                        matrix_inversion_method = "iterative"), file='NUL')
    gp_model$set_optim_params(params = params_vecchia)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
    expect_lt(abs(nll-expected_nll),tolerance_loc_1)

    #######################
    ## Other covariance functions
    #######################
    cov_pars_matern <- c(0.98944996176, 0.04986090038)
    coefs_matern <- c(0.4250887028, -0.2722344688)
    num_it_matern <- 18
    nll_opt_matern <- 64.59961544
    nll_matern <- 68.10706059
    mu_matern <- c(0.3603830, 0.1577247, -0.1189037)
    var_matern <- c(0.4497997, 0.4460163, 0.2566184)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, X = X, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_matern)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_matern)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_matern)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_matern), TOLERANCE_STRICT)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
    expect_lt(abs(nll-nll_matern),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-mu_matern)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-var_matern)),TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5 + 1E-4,
                                             likelihood = "bernoulli_probit", gp_approx = "none",
                                             y = y, X = X, params = params), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_matern)),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_matern)),TOLERANCE_MEDIUM)
      expect_equal(gp_model$get_num_optim_iter(), num_it_matern)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_matern), TOLERANCE_MEDIUM)
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
      expect_lt(abs(nll-nll_matern),TOLERANCE_MEDIUM)
      pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                      predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
      expect_lt(sum(abs(pred$mu-mu_matern)),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(pred$var)-var_matern)),TOLERANCE_MEDIUM)
    }
    # With Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                                           likelihood = "bernoulli_probit", gp_approx = "vecchia", num_neighbors = n-1,
                                           y = y, X = X, params = params, matrix_inversion_method = "cholesky"), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_matern)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_matern)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_matern)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_matern), TOLERANCE_STRICT)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
    expect_lt(abs(nll-nll_matern),TOLERANCE_STRICT)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test) , file='NUL')
    expect_lt(sum(abs(pred$mu-mu_matern)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-var_matern)),TOLERANCE_STRICT)
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5 + 1E-4,
                                             likelihood = "bernoulli_probit", gp_approx = "vecchia", num_neighbors = n-1,
                                             y = y, X = X, params = params, matrix_inversion_method = "cholesky"), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_matern)),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_matern)),TOLERANCE_MEDIUM)
      expect_equal(gp_model$get_num_optim_iter(), num_it_matern)
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_matern), TOLERANCE_MEDIUM)
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
      expect_lt(abs(nll-nll_matern),TOLERANCE_MEDIUM)
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                                      predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test) , file='NUL')
      expect_lt(sum(abs(pred$mu-mu_matern)),TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(pred$var)-var_matern)),TOLERANCE_MEDIUM)
    }

    ###################
    ## Random coefficient GPs
    ###################
    probs <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3))
    y_rand_coef <- as.numeric(sim_rand_unif(n=n, init_c=0.543) < probs)
    init_cov_pars_RC <- rep(init_cov_pars, 3)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           y = y_rand_coef, likelihood = "bernoulli_probit", gp_approx = "vecchia",
                                           num_neighbors = n-1, vecchia_ordering = "none", matrix_inversion_method = "cholesky",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov=0.5, maxit=1000, init_cov_pars=init_cov_pars_RC, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expected_values <- c(0.3701097, 0.2846740, 2.1160323, 0.3305266, 0.1241462, 0.1846456)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 39)
    # Same estimation without Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           y = y_rand_coef, likelihood = "bernoulli_probit", gp_approx = "none",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov=0.5, maxit=1000, init_cov_pars=init_cov_pars_RC, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-expected_values)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 39)
    # Prediction
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC,
                                        cov_function = "exponential", likelihood = "bernoulli_probit",
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none", matrix_inversion_method = "cholesky"), file='NUL')
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- gp_model$predict(y = y_rand_coef, gp_coords_pred = coord_test, gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.18346009, 0.03479259, -0.17247579)
    expected_cov <- c(1.039879e+00, 7.521981e-01, -3.256500e-04, 7.521981e-01,
                      8.907289e-01, -6.719282e-05, -3.256500e-04, -6.719282e-05, 9.147899e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Same prediction without Veccchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC,
                                        cov_function = "exponential", likelihood = "bernoulli_probit",
                                        gp_approx = "none"), file='NUL')
    pred <- gp_model$predict(y = y_rand_coef, gp_coords_pred = coord_test, gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)

    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(1,0.1,0.8,0.15,1.1,0.08),y=y_rand_coef)
    expect_lt(abs(nll-65.1768199),TOLERANCE_MEDIUM)

    ###################
    ##  Multiple cluster IDs
    ###################
    probs <- pnorm(L %*% b_1)
    y_clus <- as.numeric(sim_rand_unif(n=n, init_c=0.2978341) < probs)
    init_cov_pars <- c(1,mean(dist(coords[cluster_ids==1,]))/3)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y_clus, cluster_ids = cluster_ids, likelihood = "bernoulli_probit",
                                           gp_approx = "vecchia", num_neighbors = n-1,
                                           vecchia_ordering = "none", matrix_inversion_method = "cholesky",
                                           params = list(optimizer_cov = "gradient_descent", lr_cov=0.2,
                                                         use_nesterov_acc = FALSE, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(0.5085134, 0.2011667)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 20)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    cluster_ids_pred = c(1,3,1)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        cluster_ids = cluster_ids,likelihood = "bernoulli_probit"), file='NUL')
    pred <- gp_model$predict(y = y_clus, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1509569, 0.0000000, 0.9574946)
    expected_cov <- c(1.2225959453, 0.0000000000, 0.0003074858, 0.0000000000,
                      1.5000000000, 0.0000000000, 0.0003074858, 0.0000000000, 1.0761874845)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)

  })

  test_that("Binary classification Gaussian process model with VIF (or Full scale Vecchia) approximation", {
    params_vif <- c(DEFAULT_OPTIM_PARAMS, cg_delta_conv = sqrt(1e-6),
                    num_rand_vec_trace = 500, cg_preconditioner_type = "fitc")
    init_cov_pars = c(1,mean(dist(coords))/3)
    params_vif$init_cov_pars = init_cov_pars
    params_vif$fitc_piv_chol_preconditioner_rank = dim(coords)[1] - 1
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars = init_cov_pars
    params_mult <- DEFAULT_OPTIM_PARAMS
    init_cov_pars_mult = c(1,mean(dist(unique(coords_multiple)))/3)
    params_mult$init_cov_pars = init_cov_pars_mult
    params_vif_mult <- params_vif
    params_vif_mult$init_cov_pars = init_cov_pars_mult

    # Simulate data and define expected values
    probs <- pnorm(L %*% b_1) # note: linear predictor is not included in simulation
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.19341) < probs)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    init_cov_pars <- c(1,mean(dist(coords))/3)
    cov_pars_pred_eval = c(1,0.2)
    cov_pars <- c(0.92350821208, 0.05944214192)
    coefs <- c(0.3983333, -0.2653886)
    num_it <- 17
    expected_mu <- c(0.3389905, 0.1512445, -0.1039307)
    expected_cov <- c(0.6193228722, 0.5503216948, -0.0001420698, 0.5503216948,
                      0.6159348965, -0.0001556274, -0.0001420698, -0.0001556274, 0.4291674143)
    expected_mu_resp <- c(0.6050312, 0.5473537, 0.4653610)
    expected_var_resp <- c(0.2389684, 0.2477576, 0.2488001)
    expected_nll <- 67.18342059
    # Estimation, prediction, and likelihood evaluation without VIF approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, X = X, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE,
                    predict_var = TRUE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu_resp)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var_resp)),TOLERANCE_MEDIUM)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
    expect_lt(abs(nll-expected_nll),TOLERANCE_STRICT)
    # No linear regression term without VIF approximation
    cov_pars_no_X <- c(0.6875476, 0.1062862 )
    mu_no_X <- c(0.01874013, 0.01200800, 0.20498871)
    var_no_X <- c(0.6105248, 0.6093745, 0.4235374)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_no_X)),TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval)
    expect_lt(sum(abs(pred$mu-mu_no_X)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-var_no_X)),TOLERANCE_MEDIUM)
    # With duplicates and linear regression term without Vecchia approximation
    eps_multiple <- as.vector(L_multiple %*% b_multiple)
    probs_multiple <- pnorm(eps_multiple)
    y_multiple <- as.numeric(sim_rand_unif(n=n, init_c=0.2818) < probs_multiple)
    coord_test_multiple <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    cov_pars_multiple <- c(0.8263711, 0.1240696 )
    coefs_multiple <- c( 0.6168877, 0.1381717)
    num_it_multiple <- 17
    expected_mu_multiple <- c(-0.01076580, 0.07873293, 0.18927032)
    expected_var_multiple <- c(0.5653402, 0.6019163, 0.6019163)
    nll_multiple <- 58.671494
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           y = y_multiple, X = X, params = params_mult), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_multiple)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_multiple)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_multiple)
    pred <- predict(gp_model, y=y_multiple, gp_coords_pred = coord_test_multiple, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = cov_pars_pred_eval)
    expect_lt(sum(abs(pred$mu-expected_mu_multiple)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var_multiple)),TOLERANCE_MEDIUM)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y_multiple)
    expect_lt(abs(nll-nll_multiple),TOLERANCE_STRICT)
    tolerance_loc_1 <- TOLERANCE_ITERATIVE
    tolerance_loc_2 <- TOLERANCE_ITERATIVE
    tolerance_loc_3 <- 2*TOLERANCE_ITERATIVE
    loop_cg_PC = c("vifdu", "fitc")
    nsim_var_pred <- 10000
    for (cg_preconditioner_type in loop_cg_PC) {
      params_vif$cg_preconditioner_type <- cg_preconditioner_type
      # vif approximation with no ordering
      capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                          likelihood = "bernoulli_probit", gp_approx = "full_scale_vecchia",
                                          num_neighbors = n-1, num_ind_points = 20,vecchia_ordering = "none",
                                          matrix_inversion_method = "iterative"), file='NUL')
      capture.output( fit(gp_model, y = y, X = X, params = params_vif)
                      , file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_1)
      # Prediction
      gp_model$set_prediction_data(num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                      predict_var = TRUE, predict_response = FALSE,
                                      cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
      expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
      # Likelihood evaluation
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
      expect_lt(abs(nll-expected_nll),tolerance_loc_3)


      if (cg_preconditioner_type == "fitc") {# some tests are only run for one preconditioner
        ############################
        # VIF approximation with correlation-based neighbor search
        ############################
        capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                               vecchia_ordering="random", likelihood = "bernoulli_probit",
                                               gp_approx = "full_scale_vecchia_correlation_based",
                                               num_neighbors = n-1, num_ind_points = 20,
                                               y = y, X = X, params = params_vif,
                                               matrix_inversion_method = "iterative"), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_1)
        # Prediction
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_var = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_1)
        # This stochastic predictive variance varies strongly with the number of OpenMP threads: it is 0.04 with 16
        # threads but 0.19 with a single one, so it used to pass by only 4% of a 2*tolerance_loc_1 (= 0.2) budget while
        # the thread-induced change alone is 0.15. Use 4*tolerance_loc_1 so that intermediate thread counts also pass
        expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),4*tolerance_loc_1)
        # Likelihood evaluation
        nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y)
        expect_lt(abs(nll-expected_nll),tolerance_loc_1)

        #######################
        ## Less neighbors than observations
        #######################
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = "bernoulli_probit", gp_approx = "full_scale_vecchia",
                                            num_neighbors = 10, num_ind_points = 20, vecchia_ordering = "none",
                                            matrix_inversion_method = "iterative"), file='NUL')
        capture.output( fit(gp_model, y = y, X = X, params = params_vif)
                        , file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_3)
        # Prediction
        mu_less_neig <- c(0.3362000,  0.1499488, -0.1014509)
        var_resp_less_neig <- c(0.6036511, 0.6025073, 0.4220153)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_var = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-mu_less_neig)),tolerance_loc_1)
        expect_lt(sum(abs(as.vector(pred$var)-var_resp_less_neig)),relax_tolerance(2*tolerance_loc_1))
      }


    }# end loop cg_preconditioner_type in loop_cg_PC
  })

  test_that("Binary classification Gaussian process model with Wendland covariance function", {

    probs <- pnorm(L %*% b_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    init_cov_pars <- c(mean(dist(coords))/3)
    # Estimation using gradient descent
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland",
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 0.1,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, init_cov_pars=init_cov_pars, init_coef_aux_pars_from_iid_model = FALSE)), file='NUL')
    cov_pars <- c(0.5553221)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 33)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.05440076, -0.05767809, 0.05060592)
    expected_cov <- c(0.5539199, 0.4080647, 0.0000000, 0.4080647, 0.5533222, 0.0000000, 0.0000000, 0.0000000, 0.5146614)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE)
    expected_mu <- c(0.4825954, 0.4815441, 0.5163995)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
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

  test_that("Tapering for binary classification", {
    probs <- pnorm(L %*% b_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.199) < probs)
    params = DEFAULT_OPTIM_PARAMS
    params$init_cov_pars <- c(1,mean(dist(coords))/3)
    params_mult = DEFAULT_OPTIM_PARAMS
    params_mult$init_cov_pars <- c(1,mean(dist(unique(coords_multiple)))/3)

    # No tapering
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
                    predict_var = TRUE, predict_response = FALSE, cov_pars = c(1, 0.2))
    expected_mu <- c(-0.6873889499, 0.3334397127, 2.5116340251)
    expected_var <- c(0.7205439641, 0.7196871780 ,0.4591627357)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # With tapering and very large tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                                           gp_approx = "tapering", cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X=X, params = params), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - nll), TOLERANCE_STRICT)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = c(1, 0.2))
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # With tapering and small tapering range
    params_25 <- params
    params_25$init_cov_pars <- c(1, mean(dist(coords))/5.9 / 2 * sqrt(5))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", likelihood = "bernoulli_probit",
                                           cov_fct_shape = 2.5,
                                           gp_approx = "tapering", cov_fct_taper_shape = 1, cov_fct_taper_range = 0.5,
                                           y = y, X=X, params = params_25), file='NUL')
    cov_pars <- c(0.8066310, 0.4394089)
    coefs <- c(0.3572927, 1.3868262)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 9)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = c(1,0.2))
    expected_mu <- c( -0.4479216, 0.5456168, 2.4365937)
    expected_var <- c(0.7188332, 0.7297977, 0.3826909)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # Multiple observations at the same location
    eps_multiple <- as.vector(L_multiple %*% b_multiple)
    probs <- pnorm(eps_multiple + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.41) < probs)
    #No tapering
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "none",
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X=X, params = params_mult), file='NUL')
    cov_pars <- c(1.10087285407, 0.08210071565)
    coefs <- c(0.5546205072, 1.7750831945)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_MEDIUM)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE, cov_pars = c(1,0.1))
    expected_mu <- c( -0.4444039547, 0.7819372016, 2.2020037559)
    expected_var <- c(0.8182217994, 0.8492281399, 0.8492281399)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

    # With tapering and very large tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "tapering",
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           y = y, X=X, params = params_mult), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),1e-1)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),1e-1)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.1)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),1)

    # With tapering and small tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", gp_approx = "tapering",
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 0.5,
                                           y = y, X=X, params = params_mult), file='NUL')
    cov_pars <- c(1.1116570369, 0.1605364915)
    coefs <- c(0.5578846856, 1.7690518846)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),1e-2)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var = TRUE, predict_response = FALSE)
    expected_mu <- c( -0.4310628596 , 0.7944185893 , 2.2096600970)
    expected_var <- c( 0.9198750706, 0.9556430171, 0.9556430171)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE_MEDIUM)

  })

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
      expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-1.165048377 )),relax_tolerance(tolerance_loc_2))
      expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-163.2316193)),relax_tolerance(tolerance_loc_2))
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

  test_that("Space-time Gaussian process model with linear regression term ", {
    probs <- pnorm(eps_ST)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.165) < probs)
    likelihood <- "bernoulli_logit"
    cov_pars_nll <- c(1.6,0.07,0.2)
    cov_pars_nll2 <- c(1.6,10,0.01)
    coord_test <- rbind(c(200,0.2,0.9), cbind(time, coords)[c(1,10),])
    coord_test[-1,c(2:3)] <- coord_test[-1,c(2:3)] + 0.01
    X_test <- cbind(rep(1,3),c(0,0,0))
    cov_pars_pred <- c(1,0.1,0.1)
    params = OPTIM_PARAMS_BFGS
    params$init_cov_pars <- c(1,mean(dist(time))/3,mean(dist(coords))/3)

    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                        cov_function = "matern_space_time", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 70.2364458
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                           cov_function = "matern_space_time", cov_fct_shape = 0.5,
                           y = y, X = X, params = params)
    cov_pars <- c(0.13319234812, 0.06333494877, 0.12906707148)
    coef <- c(0.1363328524, 0.2142364703, 0.2661459983, 0.2975975894)
    nrounds <- 15
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu <- c(0.1363328524, 0.4163590207, 0.6388916187)
    expected_cov <- c(1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.85149434352, 0.01824729944, 0.00000000000, 0.01824729944, 0.81056965538)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = TRUE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expected_mu_resp <- c(0.5281428989, 0.5872303341, 0.6330448814)
    expected_var_resp <- c(0.2492079772, 0.2423908688, 0.2322990595)
    expect_lt(sum(abs(pred$mu-expected_mu_resp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_resp)),TOLERANCE_STRICT)

    ##############
    ## With Vecchia approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                                        cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none", matrix_inversion_method = "cholesky"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                                           cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky"),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = TRUE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu_resp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_resp)),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)

    ## Less neighbors
    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative"){
        tolerance_loc <- TOLERANCE_ITERATIVE
      } else{
        tolerance_loc <- TOLERANCE_STRICT
      }
      nsim_var_pred <- 10000
      # Evaluate negative log-likelihood
      num_neighbors <- 50
      capture.output( gp_model <- GPModel(gp_coords = cbind(time, coords), likelihood = likelihood,
                                          cov_function = "matern_space_time", cov_fct_shape = 0.5, matrix_inversion_method = inv_method,
                                          gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none"),
                      file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
      expect_lt(abs(nll-70.2364313),0.2)
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll2,y=y)
      expect_lt(abs(nll-70.6574683),0.2)
      # Fit model
      capture.output( gp_model <- fitGPModel(gp_coords = cbind(time, coords), likelihood = likelihood, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                             gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                             matrix_inversion_method = inv_method, y = y, X=X, params = params),
                      file='NUL')
      cov_pars_nn <- c(0.13310337502, 0.06332284601, 0.12921443605)
      coef_nn <- c(0.1370527248, 0.2142481946, 0.2677589771, 0.2976186564)
      nrounds_nn <- 15
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_nn)),tolerance_loc)
      if (inv_method=="cholesky") {
        expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
      }
      # Prediction
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=num_neighbors, nsim_var_pred=nsim_var_pred)
      pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                      X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
      expected_mu_nn <- c(0.1370527248, 0.4168275706, 0.6393660124)
      expected_cov_nn <- c(1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.8515104403 , 0.0182491973, 0.00000000, 0.0182491973, 0.8105919548)
      expect_lt(sum(abs(pred$mu-expected_mu_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_nn)),tolerance_loc)
      pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                      X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      expect_lt(sum(abs(pred$mu-expected_mu_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(pred$var)-expected_cov_nn[c(1,5,9)])),tolerance_loc)
    }

    ##############
    ## Multiple observations at the same location
    ##############
    coords_ST = cbind(time, coords)
    coords_ST[1:5,] <- coords_ST[(n-4):n,]
    params = OPTIM_PARAMS_BFGS
    params$init_cov_pars <- c(1,mean(dist(unique(coords_ST)[,1]))/3,mean(dist(unique(coords_ST)[,-1]))/3)
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                        likelihood = likelihood)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 70.85206038
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                           y = y, X=X, params = params, likelihood = likelihood)
    cov_pars <- c(0.0003103303859, 0.0160438298347, 0.0139448004490)
    coef <- c(0.1356549353, 0.2031592096, 0.2579334524, 0.2882626226)
    nrounds <- 19
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    ## With Vecchia approximation
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-6, vecchia_ordering = "none",
                                        likelihood = likelihood, matrix_inversion_method = "cholesky"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ST, cov_function = "matern_space_time", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-6, vecchia_ordering = "none",
                                           y = y, X=X, params = params, likelihood = likelihood, matrix_inversion_method = "cholesky"),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
  })

  test_that("ARD Gaussian process model with linear regression term ", {
    # Simulate data
    d <- 3 # dimension of GP locations
    coords_ARD <- matrix(sim_rand_unif(n=n*d, init_c=0.48231), ncol=d)
    sigma2_1 <- 0.75^2 # marginal variance of GP
    rhos <- c(0.1,0.2,0.1)
    coords_ARD_scaled <- coords_ARD
    for (i in 1:dim(coords_ARD)[2]) coords_ARD_scaled[,i] <- coords_ARD[,i] / rhos[i]
    D_ARD <- as.matrix(dist(coords_ARD_scaled))
    Sigma_ARD <- sigma2_1 * exp(-D_ARD) + diag(1E-20,n)
    # hist(Sigma_ARD)
    C_ARD <- t(chol(Sigma_ARD))
    b_ARD <- qnorm(sim_rand_unif(n=n, init_c=0.4658))
    eps_ARD <- as.vector(C_ARD %*% b_ARD)
    probs <- pnorm(eps_ARD)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.18354) < probs)
    likelihood <- "bernoulli_logit"
    params = OPTIM_PARAMS_BFGS
    init_cov_pars <- c(1)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars <- c(init_cov_pars, mean(dist(coords_ARD[,i])/3))
    params$init_cov_pars <- init_cov_pars

    cov_pars_nll <- c(0.7, 0.5 * rhos)
    coord_test <- rbind(c(10000,0.2,0.9), coords_ARD[c(1,10),])
    coord_test[-1,c(2:3)] <- coord_test[-1,c(2:3)] + 0.01
    X_test <- cbind(rep(1,3),c(0,0,0))
    cov_pars_pred <- c(sigma2_1, rhos)

    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood,
                        cov_function = "matern_ard", cov_fct_shape = 0.5)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 69.7023612
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood,
                           cov_function = "matern_ard", cov_fct_shape = 0.5, y = y, X = X, params = params)
    cov_pars <- c(0.13905428093, 0.06867025605, 0.04247690364, 0.15469536599)
    coef <- c(-0.2543743520, 0.1505760147)
    nrounds <- 15
    nll_opt <- 68.41713226
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expected_mu <- c(-0.25437435197, 0.06788130795, 0.01430265524)
    expected_cov <- c(0.5625000000000, 0.0000000000000, 0.0000000000000, 0.0000000000000, 0.4938848144137, 0.0002158338884, 0.0000000000000, 0.0002158338884, 0.4862042504205)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Prediction without prior model fitting
    exp_mu_no_coef <- c(0.00000000, 0.25771940, 0.17913289)
    exp_cov_no_coef <- c(0.56250000000, 0.00000000000, 0.00000000000, 0.00000000000, 0.49481305128, 0.00021588667, 0.00000000000, 0.00021588667, 0.48645327980)
    gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5)
    pred <- predict(gp_model, gp_coords_pred = coord_test, y = y, predict_response = FALSE,
                    predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-exp_mu_no_coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-exp_cov_no_coef)),TOLERANCE_STRICT)
    # Matern with shape estimated
    params_ARD_est_shape <- OPTIM_PARAMS_BFGS
    params_ARD_est_shape$init_cov_pars <- c(init_cov_pars,1.5)
    if (!SKIP_BESSEL_COV_TESTS) {
      capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard_estimate_shape",
                                             y = y, X = X, params = params_ARD_est_shape),
                      file='NUL')
      cov_pars_est_shape <- c(0.57108958797,  0.08471275821,  0.03304572501,  0.16194229745, 115.08702014148)
      coef_est_shape <- c(-0.2905450775, 0.2387123371, 0.1944576895, 0.3275844333)
      nrounds_est_shape <- 28
      nll_opt_est_shape <- 68.13569857
      capture.output( expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))[1:4]-cov_pars_est_shape[1:4])),TOLERANCE_STRICT), file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))[5]-cov_pars_est_shape[5])),TOLERANCE_MEDIUM)
      capture.output( expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = TRUE))-coef_est_shape )),TOLERANCE_MEDIUM), file='NUL')
      expect_equal(gp_model$get_num_optim_iter(), nrounds_est_shape )
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_est_shape), TOLERANCE_STRICT)
    }

    ##############
    ## With Vecchia approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood,
                                        cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none", matrix_inversion_method = "cholesky"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood,
                                           cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                           y = y, X=X, params = params, matrix_inversion_method = "cholesky"),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),relax_tolerance_strict(TOLERANCE_STRICT))
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    # Prediction
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance_strict(TOLERANCE_STRICT))
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance_strict(TOLERANCE_STRICT))
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred=n)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance_strict(TOLERANCE_STRICT))
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance_strict(TOLERANCE_STRICT))
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)
    # Prediction without prior model fitting
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood,
                                        cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                        matrix_inversion_method = "cholesky"),
                    file='NUL')
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=n+2)
    pred <- predict(gp_model, gp_coords_pred = coord_test, y = y, predict_response = FALSE,
                    predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-exp_mu_no_coef)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-exp_cov_no_coef)),TOLERANCE_STRICT)

    ## Less neighbors
    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative"){
        tolerance_loc <- TOLERANCE_ITERATIVE
      } else{
        tolerance_loc <- TOLERANCE_STRICT
      }
      nsim_var_pred <- 10000
      # Evaluate negative log-likelihood
      num_neighbors <- 50
      capture.output( gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood,
                                          cov_function = "matern_ard", cov_fct_shape = 0.5, matrix_inversion_method = inv_method,
                                          gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none"),
                      file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
      expect_lt(abs(nll-69.70236284),tolerance_loc)
      # Fit model
      capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                             gp_approx = "vecchia", num_neighbors = num_neighbors, vecchia_ordering = "none",
                                             y = y, X=X, params = params, matrix_inversion_method = inv_method),
                      file='NUL')
      cov_pars_nn <- c(0.19603539585, 0.06791498325, 0.03368011905, 0.15885250994)
      coef_nn <- c(-0.2701394756, 0.1619874679)
      nrounds_nn <- 25
      nll_opt_nn <- 68.41033632
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_nn)),tolerance_loc)
      if (inv_method == "cholesky") {
        expect_equal(gp_model$get_num_optim_iter(), nrounds_nn)
      }
      expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), tolerance_loc)
      # Prediction
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all", num_neighbors_pred=num_neighbors, nsim_var_pred=nsim_var_pred)
      pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                      X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
      expected_mu_nn <- c(-0.270139475620, 0.056010991285, 0.004015366351)
      expected_cov_nn <- c(0.5625000000000, 0.0000000000000, 0.0000000000000, 0.0000000000000, 0.4938560837495, 0.0002305907991, 0.0000000000000, 0.0002305907991, 0.4862229980931)
      expect_lt(sum(abs(pred$mu-expected_mu_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(pred$cov)-expected_cov_nn)),tolerance_loc)
      pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                      X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
      expect_lt(sum(abs(pred$mu-expected_mu_nn)),tolerance_loc)
      expect_lt(sum(abs(as.vector(pred$var)-expected_cov_nn[c(1,5,9)])),tolerance_loc)
    }

    ##############
    ## With FITC approximation
    ##############
    # Evaluate negative log-likelihood
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "fitc", num_ind_points = n, ind_points_selection = "random"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = n, ind_points_selection = "random",
                                           y = y, X = X, params = params),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT_LOWER)
    # Prediction
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_cov_mat = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    pred <- predict(gp_model, gp_coords_pred = coord_test, predict_response = FALSE,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_STRICT)

    ## Less inducing points
    # Evaluate negative log-likelihood
    num_ind_points <- 50
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "fitc", num_ind_points = num_ind_points, ind_points_selection = "kmeans++"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-69.8362518046058),TOLERANCE_STRICT)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD, likelihood = likelihood, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = num_ind_points, ind_points_selection = "kmeans++",
                                           y = y, X = X, params = params),
                    file='NUL')
    cov_pars_nn <- c(0.00006475, 0.02037307, 0.01531317, 0.14835818)
    coef_nn <- c(-0.25715745, 0.14726729)
    nll_opt_nn <- 68.46260798
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_nn)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_nn)),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_nn), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_ITERATIVE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), 0.2)

    ##############
    ## Multiple observations at the same location
    ##############
    coords_ARD_mult = coords_ARD
    coords_ARD_mult[1:5,] <- coords_ARD_mult[(n-4):n,]
    params = OPTIM_PARAMS_BFGS
    init_cov_pars_mult <- c(1)
    for (i in 1:dim(coords_ARD)[2]) init_cov_pars_mult <- c(init_cov_pars_mult, mean(dist(unique(coords_ARD_mult)[,i])/3))
    params$init_cov_pars <- init_cov_pars_mult
    # Evaluate negative log-likelihood
    gp_model <- GPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                        likelihood = likelihood)
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    nll_exp <- 69.34595415
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                           y = y, X=X, params = params, likelihood = likelihood)
    cov_pars <- c(0.44308197588, 0.09997589302, 0.03067100521, 0.12031834901)
    coef <- c(-0.2800805796, 0.1785353899)
    nrounds <- 22
    nll_opt <- 68.21587237
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_opt), TOLERANCE_STRICT)
    ## With Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "vecchia", num_neighbors = n-6, vecchia_ordering = "none",
                                        likelihood = likelihood, matrix_inversion_method = "cholesky"),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "vecchia", num_neighbors = n-6, vecchia_ordering = "none",
                                           y = y, X=X, params = params, likelihood = likelihood, matrix_inversion_method = "cholesky"),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
    ## With FITC approximation
    capture.output( gp_model <- GPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                        gp_approx = "fitc", num_ind_points = n - 5, ind_points_selection = "random",
                                        likelihood = likelihood),
                    file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_nll,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_MEDIUM)
    capture.output( gp_model <- fitGPModel(gp_coords = coords_ARD_mult, cov_function = "matern_ard", cov_fct_shape = 0.5,
                                           gp_approx = "fitc", num_ind_points = n - 5, ind_points_selection = "random",
                                           y = y, X=X, params = params, likelihood = likelihood),
                    file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef)),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), nrounds)
  })

  test_that("t likelihood", {

    params = OPTIM_PARAMS_BFGS
    init_cov_pars = c(1,mean(dist(coords))/3)
    params$init_cov_pars = init_cov_pars
    likelihood_additional_param = 1
    params_vecchia <- c(params, cg_delta_conv = sqrt(1e-6),
                        num_rand_vec_trace = 50, cg_preconditioner_type = "pivoted_cholesky",
                        fitc_piv_chol_preconditioner_rank = n-1)
    params_vecchia$init_cov_pars = init_cov_pars

    # Simulate data and define expected values
    y <- L %*% b_1 + qnorm(sim_rand_unif(n=n, init_c=0.1)) / 5
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    cov_pars_pred_eval = c(1,0.2)
    coefs_pred = c(0.5,0.1)
    aux_pars_pred_eval = c(1,3)
    expected_nll <- 144.1194745
    cov_pars_fix_df <- c(0.94274448822, 0.09483522922)
    coefs_fix_df <- c(0.2780976933, -0.1045530677)
    aux_pars_fix_df <- c(0.07765501731, 1.00000000000)
    num_it_fix_df <- 10
    nll_est_fix_df <- 112.1635624
    cov_pars <- c(1.00786933961, 0.09231304834)
    coefs <- c(0.30227595731, -0.09752032205)
    aux_pars <- c(0.00165718826, 1.63405265512)
    num_it <- 24
    nll_est <- 107.8275669
    expected_mu <- c(-0.046398775956, -0.003934498908, 0.789074244932)
    expected_cov <- c(0.5895448972540, 0.5224498197427, -0.0001390612641, 0.5224498197427, 0.5897209705595, -0.0001486534570, -0.0001390612641, -0.0001486534570, 0.4058804336013)
    expected_var_resp <- expected_cov[c(1,5,9)] + aux_pars_pred_eval[1]^2
    # expected_var_resp <- c(3.589544897, 3.589720971, 3.405880434)
    # Estimation, prediction, and likelihood evaluation without Vecchia approximation
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "t", gp_approx = "none")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y, aux_pars = aux_pars_pred_eval)
    expect_lt(abs(nll-expected_nll),TOLERANCE_STRICT)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "t_fix_df", gp_approx = "none",
                                           y = y, X = X, params = params, likelihood_additional_param=likelihood_additional_param), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_fix_df)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_fix_df)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_fix_df)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_fix_df),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_fix_df)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "t", gp_approx = "none",
                                           y = y, X = X, params = params, likelihood_additional_param=likelihood_additional_param), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_LOOSE)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),2*TOLERANCE_LOOSE)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est),TOLERANCE_MEDIUM)
    # Prediction
    gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE,
                                    predict_var = TRUE, cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-expected_var_resp)),TOLERANCE_MEDIUM)
    ############################
    # With duplicates and linear regression term without Vecchia approximation
    ############################
    y_multiple <- L_multiple %*% b_multiple + qnorm(sim_rand_unif(n=n, init_c=0.2818)) / 5
    coord_test_multiple <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
    expected_nll_multiple <- 126.5295458
    cov_pars_multiple <- c(0.7281991570720, 0.0007068132731)
    coefs_multiple <- c(0.572603702130, 0.007961020259)
    aux_pars_multiple <- c(0.1803089654, 6.9281212623)
    num_it_multiple <- 40
    nll_est_multiple <- 34.7799636
    expected_mu_multiple <- c(-0.01186332228, 0.04582440444, 0.12582440444)
    expected_var_multiple <- c(0.5599871156, 0.5964929698, 0.5964929698)
    gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential", likelihood = "t", gp_approx = "none")
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y_multiple, aux_pars = aux_pars_pred_eval)
    expect_lt(abs(nll-expected_nll_multiple),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                           likelihood = "t", gp_approx = "none",
                                           y = y_multiple, X = X, params = params, likelihood_additional_param=likelihood_additional_param), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_multiple)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_multiple)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_multiple)),relax_tolerance(TOLERANCE_STRICT_LOWER))
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_multiple),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it_multiple)
    gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
    pred <- predict(gp_model, y=y_multiple, gp_coords_pred = coord_test_multiple, predict_var = TRUE,
                    predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    expect_lt(sum(abs(pred$mu-expected_mu_multiple)),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var_multiple)),TOLERANCE_MEDIUM)

    for(inv_method in c("cholesky", "iterative")){
      if(inv_method == "iterative") {
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
        tolerance_loc_3 <- 0.5
        loop_cg_PC = c("pivoted_cholesky", "vadu", "fitc")
      } else {
        tolerance_loc_1 <- TOLERANCE_MEDIUM
        tolerance_loc_2 <- TOLERANCE_LOOSE
        tolerance_loc_3 <- TOLERANCE_LOOSE
        loop_cg_PC = c("vadu")
      }
      nsim_var_pred <- 10000
      for (cg_preconditioner_type in loop_cg_PC) {
        params_vecchia$cg_preconditioner_type <- cg_preconditioner_type

        # Likelihood evaluation
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = "t", gp_approx = "vecchia",
                                            num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        gp_model$set_optim_params(params = params_vecchia)
        capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y, aux_pars = aux_pars_pred_eval), file='NUL')
        expect_lt(abs(nll-expected_nll),2*tolerance_loc_3)
        # Estimation
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = "t", gp_approx = "vecchia",
                                            num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method, likelihood_additional_param=likelihood_additional_param), file='NUL')
        capture.output( fit(gp_model, y = y, X = X, params = params_vecchia) , file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_ITERATIVE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_ITERATIVE)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_ITERATIVE)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est),3*tolerance_loc_3)
        # Prediction
        gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                     num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_cov_mat = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_var = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_response = TRUE, predict_var = TRUE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
          expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
        }

        ############################
        # With duplicates and linear regression term
        ############################
        capture.output( gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential", likelihood = "t",
                                            gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        params_vecchia_mult <- params_vecchia
        params_vecchia_mult$fitc_piv_chol_preconditioner_rank <- dim(unique(coords_multiple))[1]
        gp_model$set_optim_params(params = params_vecchia_mult)
        capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y_multiple, aux_pars = aux_pars_pred_eval), file='NUL')
        expect_lt(abs(nll-expected_nll_multiple),tolerance_loc_3)
        capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                               likelihood = "t", gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering = "none",
                                               matrix_inversion_method = inv_method,
                                               y = y_multiple, X = X, params = params_vecchia_mult, likelihood_additional_param=likelihood_additional_param), file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_multiple)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_multiple)),TOLERANCE_ITERATIVE)
        expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_multiple)),0.3)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_multiple),tolerance_loc_2)
        gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                     num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
        capture.output( pred <- predict(gp_model, y=y_multiple, gp_coords_pred = coord_test_multiple, predict_var = TRUE,
                                        predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu_multiple)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var_multiple)),tolerance_loc_1)

        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {# some tests are only run for one preconditioner

          #######################
          ## Less neighbors than observations
          #######################
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = "t", gp_approx = "vecchia",
                                              num_neighbors = 20, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method), file='NUL')
          gp_model$set_optim_params(params = params_vecchia)
          capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y, aux_pars = aux_pars_pred_eval), file='NUL')
          expected_nll_less_nn <- 144.099563
          expect_lt(abs(nll-expected_nll_less_nn),2*tolerance_loc_3)
          # Estimation
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = "t", gp_approx = "vecchia",
                                              num_neighbors = 20, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method, likelihood_additional_param=likelihood_additional_param), file='NUL')
          capture.output( fit(gp_model, y = y, X = X, params = params_vecchia) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_ITERATIVE)
          expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_ITERATIVE)
          nll_est_less_nn <- 107.8264387
          expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_less_nn),3*tolerance_loc_3)
          # Prediction
          gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
          gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", nsim_var_pred = nsim_var_pred)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_cov_mat = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),0.2)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_var = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),tolerance_loc_1)
          if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {
            capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                            predict_response = TRUE, predict_var = TRUE,
                                            cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
            expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
            expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
          }
        }

      }# end loop cg_preconditioner_type in loop_cg_PC
    }# end loop inv_method in c("cholesky", "iterative")
  }) #end t-likelihood

  test_that("gaussian_heteroscedastic_fixed_and_random likelihood", {
    params = OPTIM_PARAMS_BFGS
    init_cov_pars = c(1,mean(dist(coords))/3,0.1,mean(dist(coords))/3)
    params$init_cov_pars = init_cov_pars
    params_vecchia <- c(params, cg_delta_conv = sqrt(1e-6),
                        num_rand_vec_trace = 50, cg_preconditioner_type = "pivoted_cholesky")
    params_vecchia$init_cov_pars = init_cov_pars
    likelihood <- "gaussian_heteroscedastic_fixed_and_random"

    # Simulate data and define expected values
    Sigma2 <- 0.1 * exp(-D/0.2) + diag(1E-20,n)
    L2 <- t(chol(Sigma))
    b_2 <- qnorm(sim_rand_unif(n=n, init_c=0.834))
    y <- L %*% b_1 + qnorm(sim_rand_unif(n=n, init_c=0.1234)) * exp(0.5 * L2 %*% b_2)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    cov_pars_pred_eval = c(1,0.2,0.1,0.2)
    coefs_pred = c(c(0.5,0.1),c(0.5,0.1))
    expected_nll <- 199.6831947
    cov_pars <- c(0.29257505689, 0.16019690150, 0.20398810623, 0.02123292904)
    coefs <- c(0.2573774906, -0.1120390282, 0.6360477105, 0.2961457581)
    num_it <- 15
    nll_est <- 191.2306375
    expected_mu <- c(0.06126291, 0.07337373, 0.30807230)
    expected_var <- c(0.5994207, 0.6014515, 0.3936357)
    expected_var_resp <- c(2.147623, 2.268682, 2.010216)

    #  # Estimation, prediction, and likelihood evaluation without Vecchia approximation
    # gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = likelihood , gp_approx = "none")
    # nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y, aux_pars = aux_pars_pred_eval)
    # expect_lt(abs(nll-expected_nll),TOLERANCE_STRICT)
    # # Estimation
    # capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
    #                                        likelihood = "t_fix_df", gp_approx = "none",
    #                                        y = y, X = X, params = params, likelihood_additional_param=likelihood_additional_param), file='NUL')
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_fix_df)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs_fix_df)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars_fix_df)),TOLERANCE_STRICT)
    # expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_fix_df),TOLERANCE_STRICT)
    # expect_equal(gp_model$get_num_optim_iter(), num_it_fix_df)
    # capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
    #                                        likelihood = likelihood , gp_approx = "none",
    #                                        y = y, X = X, params = params, likelihood_additional_param=likelihood_additional_param), file='NUL')
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_MEDIUM)
    # expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_LOOSE)
    # expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_pars)),TOLERANCE_LOOSE)
    # expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est),TOLERANCE_MEDIUM)
    # # Prediction
    # gp_model$set_optim_params(params = list(init_aux_pars = aux_pars_pred_eval, init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
    # pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
    #                 predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    # expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    # expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_MEDIUM)
    # pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE,
    #                 predict_response = FALSE, cov_pars = cov_pars_pred_eval, X_pred = X_test)
    # expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    # expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE_MEDIUM)
    # capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE,
    #                                 predict_var = TRUE, cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
    # expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
    # expect_lt(sum(abs(pred$var-expected_var_resp)),TOLERANCE_MEDIUM)
    for(inv_method in c("cholesky")){#, "iterative"
      if(inv_method == "iterative") {
        tolerance_loc_1 <- TOLERANCE_ITERATIVE
        tolerance_loc_2 <- TOLERANCE_ITERATIVE
        tolerance_loc_3 <- 0.5
        loop_cg_PC = c("pivoted_cholesky", "vadu", "fitc")
      } else {
        tolerance_loc_1 <- TOLERANCE_MEDIUM
        tolerance_loc_2 <- TOLERANCE_LOOSE
        tolerance_loc_3 <- TOLERANCE_LOOSE
        loop_cg_PC = c("vadu")
      }
      nsim_var_pred <- 10000
      for (cg_preconditioner_type in loop_cg_PC) {
        params_vecchia$cg_preconditioner_type <- cg_preconditioner_type

        # Likelihood evaluation
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = likelihood , gp_approx = "vecchia",
                                            num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        gp_model$set_optim_params(params = params_vecchia)
        capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y), file='NUL')
        expect_lt(abs(nll-expected_nll),tolerance_loc_3)
        # Estimation
        capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                            likelihood = likelihood , gp_approx = "vecchia",
                                            num_neighbors = n-1, vecchia_ordering = "none",
                                            matrix_inversion_method = inv_method), file='NUL')
        capture.output( fit(gp_model, y = y, X = X, params = params_vecchia) , file='NUL')
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),TOLERANCE_ITERATIVE)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_ITERATIVE)
        expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est),tolerance_loc_3)
        if (inv_method != "iterative") {
          expect_equal(gp_model$get_num_optim_iter(), num_it)
        }
        # Prediction
        gp_model$set_optim_params(params = list(init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
        gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all",
                                     num_neighbors_pred = n+2, nsim_var_pred = nsim_var_pred)
        capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                        predict_var = TRUE, predict_response = FALSE,
                                        cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
        expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(pred$var)-expected_var)),tolerance_loc_1)
        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_response = TRUE, predict_var = TRUE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
          expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_1)
        }

        if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {# some tests are only run for one preconditioner

          #######################
          ## Less neighbors than observations
          #######################
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = likelihood , gp_approx = "vecchia",
                                              num_neighbors = 20, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method), file='NUL')
          gp_model$set_optim_params(params = params_vecchia)
          capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_pred_eval, y=y), file='NUL')
          expected_nll_less_nn <- 199.6932499
          expect_lt(abs(nll-expected_nll_less_nn),tolerance_loc_3)
          # Estimation
          capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = likelihood , gp_approx = "vecchia",
                                              num_neighbors = 30, vecchia_ordering = "none",
                                              matrix_inversion_method = inv_method), file='NUL')
          capture.output( fit(gp_model, y = y, X = X, params = params_vecchia) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coefs)),TOLERANCE_ITERATIVE)
          nll_est_less_nn <- 191.2393688
          expect_lt(abs(gp_model$get_current_neg_log_likelihood()-nll_est_less_nn),tolerance_loc_3)
          # Prediction
          gp_model$set_optim_params(params = list(init_coef = coefs_pred, init_coef_aux_pars_from_iid_model = FALSE))
          gp_model$set_prediction_data(vecchia_pred_type = "latent_order_obs_first_cond_all", nsim_var_pred = nsim_var_pred)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                          predict_var = TRUE, predict_response = FALSE,
                                          cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(pred$var)-expected_var)),tolerance_loc_2)
          if (inv_method != "iterative" || cg_preconditioner_type == "vadu") {
            capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                                            predict_response = TRUE, predict_var = TRUE,
                                            cov_pars = cov_pars_pred_eval, X_pred = X_test), file='NUL')
            expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_2)
            expect_lt(sum(abs(pred$var-expected_var_resp)),tolerance_loc_2)
          }
        }

      }# end loop cg_preconditioner_type in loop_cg_PC
    }# end loop inv_method in c("cholesky", "iterative")
  }) #end gaussian_heteroscedastic_fixed_and_random likelihood

  test_that("gaussian_heteroscedastic likelihood (fixed effects only) for linear and GPBoost models ", {

    n_het <- 100
    group_het <- rep(1:10, each = 10)
    X_het <- cbind(rep(1, n_het), sim_rand_unif(n = n_het, init_c = 0.256))
    beta_mean <- c(0.3, 0.7)
    beta_var <- c(-0.5, 1.2)
    gr_var_het <- 1
    b_gr_het <- qnorm(sim_rand_unif(n = 10, init_c = 0.741))
    mean_true <- as.vector(X_het %*% beta_mean) + sqrt(gr_var_het) * b_gr_het[group_het]
    log_var_true <- as.vector(X_het %*% beta_var)
    y_het <- mean_true + qnorm(sim_rand_unif(n = n_het, init_c = 0.369)) * exp(0.5 * log_var_true)

    # Likelihood evaluated at given (not estimated) parameters: a pure formula check, independent of any optimizer
    cov_pars_given_het <- 0.3
    fixed_effects_given_het <- as.vector(cbind(X_het %*% c(0.2, 0.5), X_het %*% c(-0.3, 0.8)))
    nll_given_het <- GPModel(group_data = group_het, likelihood = "gaussian_heteroscedastic")$neg_log_likelihood(
      cov_pars = cov_pars_given_het, y = y_het, fixed_effects = fixed_effects_given_het)
    expect_lt(abs(nll_given_het - 157.80743264), TOLERANCE_MEDIUM)

    # A fixed-effects-only variance requires a fixed effects term (covariates and / or GPBoost boosting):
    # without any covariates and without the GPBoost algorithm, fitting should raise an informative error
    expect_error(capture.output(fitGPModel(group_data = group_het, likelihood = "gaussian_heteroscedastic",
                                           y = y_het, params = list(maxit = 2, init_coef_aux_pars_from_iid_model = FALSE)),
                                file = "NUL"))

    ###################
    ## Linear regression model (mean has a grouped random effect, variance is fixed-effects only)
    ###################
    capture.output(gp_model_het <- fitGPModel(group_data = group_het, likelihood = "gaussian_heteroscedastic",
                                              y = y_het, X = X_het,
                                              params = OPTIM_PARAMS_BFGS), file = "NUL")
    coef_het <- as.vector(gp_model_het$get_coef(std_err = FALSE))
    expect_equal(length(coef_het), 4L)
    coef_het_std_err <- gp_model_het$get_coef(std_err = TRUE)
    expect_equal(dim(coef_het_std_err), c(2L, 4L))
    # Note: std. errs. must be strictly positive; a plain is.finite() check would not catch a regression where the
    # variance block's std. errs. are silently left at their R-side zero-initialized default (0 is finite)
    expect_true(all(coef_het_std_err["Std. err.", ] > 0))
    expected_coef_het <- c(-0.16843105, 1.05258998, -0.64123490, 1.54924057)
    expect_lt(sum(abs(coef_het - expected_coef_het)), TOLERANCE_MEDIUM)
    expect_lt(abs(as.vector(gp_model_het$get_cov_pars(std_err = FALSE)) - 0.24994751), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_het$get_current_neg_log_likelihood() - 155.27522914), TOLERANCE_MEDIUM)
    # Prediction: response mean and variance
    X_test_het <- cbind(rep(1, 3), c(0.1, 0.4, 0.8))
    group_test_het <- c(1, 3, 11)
    pred_het <- predict(gp_model_het, y = y_het, group_data_pred = group_test_het, X_pred = X_test_het,
                        predict_var = TRUE, predict_response = TRUE)
    expected_mu_het <- c(0.35476713, 0.16102877, 0.67364093)
    expected_var_het <- c(0.69153035, 1.04948914, 2.06871225)
    expect_lt(sum(abs(pred_het$mu - expected_mu_het)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_het$var - expected_var_het)), TOLERANCE_MEDIUM)
    X_zero_het <- matrix(0, nrow = n_het, ncol = ncol(X_het))
    re_pred_train_het <- predict_training_data_random_effects(gp_model_het)
    expected_re_pred_train_het <- c(0.41793918, 0.11415140, -0.09157617, -0.06884991, 0.53482262,
                                    -0.64437448, 0.20923218, -0.85328655, 0.28497061, 0.09707038)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_het[, 1])) - expected_re_pred_train_het)), TOLERANCE_MEDIUM)
    re_pred_train_het_var <- predict_training_data_random_effects(gp_model_het, predict_var = TRUE)
    expected_re_pred_train_het_var <- c(0.07663970, 0.06660995, 0.07079751, 0.07706457, 0.07193486,
                                        0.06797221, 0.08127999, 0.07331034, 0.06945477, 0.07959953)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_het_var[, 2])) - expected_re_pred_train_het_var)), TOLERANCE_MEDIUM)
    pred_train_re_het <- predict(gp_model_het, y = y_het, group_data_pred = group_het, X_pred = X_zero_het,
                                 predict_response = FALSE, predict_var = FALSE)
    expect_lt(sum(abs(as.vector(re_pred_train_het[, 1]) - pred_train_re_het$mu)), TOLERANCE_STRICT)
    # Predicting requires covariate data for the model's linear predictor (mean and variance)
    expect_error(predict(gp_model_het, y = y_het, group_data_pred = group_test_het,
                         predict_var = TRUE, predict_response = TRUE))

    ###################
    ## No random effects at all (iid model, pure linear regression for mean and variance)
    ###################
    capture.output(gp_model_het_iid <- fitGPModel(likelihood = "gaussian_heteroscedastic",
                                                  y = y_het, X = X_het,
                                                  params = OPTIM_PARAMS_BFGS), file = "NUL")
    coef_het_iid <- as.vector(gp_model_het_iid$get_coef(std_err = FALSE))
    expected_coef_het_iid <- c(-0.18164405, 1.06906319, -0.14266627, 0.97312331)
    expect_lt(sum(abs(coef_het_iid - expected_coef_het_iid)), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_het_iid$get_current_neg_log_likelihood() - 159.44268884), TOLERANCE_MEDIUM)

    ###################
    ## GPBoost algorithm (tree-boosting): mean via a grouped random effect + trees, variance via a second tree ensemble
    ###################
    gp_model_het_boost <- GPModel(group_data = group_het, likelihood = "gaussian_heteroscedastic")
    gp_model_het_boost$set_optim_params(params = OPTIM_PARAMS_BFGS)
    dtrain_het <- gpb.Dataset(data = X_het[, 2, drop = FALSE], label = y_het)
    bst_het <- gpb.train(data = dtrain_het, gp_model = gp_model_het_boost, nrounds = 20,
                         learning_rate = 0.01, max_depth = 2, min_data_in_leaf = 5,
                         verbose = 0, deterministic = TRUE)
    pred_het_boost <- predict(bst_het, data = X_het[1:3, 2, drop = FALSE], group_data_pred = group_test_het,
                              predict_var = TRUE, pred_latent = FALSE)
    expect_lt(abs(as.vector(gp_model_het_boost$get_cov_pars(std_err = FALSE)) - 0.15080798), TOLERANCE_MEDIUM)
    expected_response_mean_boost <- c(0.52600579, 0.24099045, 0.37506889)
    expected_response_var_boost <- c(1.43766912, 1.43641997, 1.58325054)
    expect_lt(sum(abs(pred_het_boost$response_mean - expected_response_mean_boost)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_het_boost$response_var - expected_response_var_boost)), TOLERANCE_MEDIUM)
    expect_error(predict_training_data_random_effects(gp_model_het_boost),
                 "predict_training_data_random_effects\\(bst\\)")
    re_pred_train_boost <- predict_training_data_random_effects(bst_het)
    expected_re_pred_train_boost <- c(0.27708188, 0.02084478, -0.06485732, -0.09881753, 0.40985823,
                                      -0.38168637, 0.10144342, -0.57590024, 0.17622270, 0.07567558)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_boost[, 1])) - expected_re_pred_train_boost)), TOLERANCE_MEDIUM)
    re_pred_train_boost_var <- predict_training_data_random_effects(bst_het, predict_var = TRUE)
    expect_lt(sum(abs(re_pred_train_boost_var[, 1] - re_pred_train_boost[, 1])), TOLERANCE_STRICT)
    expected_re_pred_train_boost_var <- c(0.07329236, 0.07204321, 0.07222443, 0.07247900, 0.07183686,
                                          0.07303205, 0.07294693, 0.07201704, 0.07256303)
    expect_lt(sum(abs(unique(as.vector(re_pred_train_boost_var[, 2])) - expected_re_pred_train_boost_var)), TOLERANCE_MEDIUM)

    ###################
    ## GPs
    ###################

    n_het2 <- 100
    X_het2 <- cbind(rep(1, n_het2), sim_rand_unif(n = n_het2, init_c = 0.187))
    beta_mean2 <- c(0.2, 0.5)
    beta_var2 <- c(-0.4, 0.9)
    log_var_true2 <- as.vector(X_het2 %*% beta_var2)
    coords_het2 <- matrix(sim_rand_unif(n = n_het2 * 2, init_c = 0.723), ncol = 2)
    D_het2 <- as.matrix(dist(coords_het2))
    gp_var2 <- 0.6
    gp_range2 <- 0.15
    Sigma_het2 <- gp_var2 * exp(-D_het2 / gp_range2) + diag(1E-10, n_het2)
    b_gp_het2 <- as.vector(t(chol(Sigma_het2)) %*% qnorm(sim_rand_unif(n = n_het2, init_c = 0.812)))
    mean_true2 <- as.vector(X_het2 %*% beta_mean2) + b_gp_het2
    y_het2 <- mean_true2 + qnorm(sim_rand_unif(n = n_het2, init_c = 0.234)) * exp(0.5 * log_var_true2)
    optim_params_het2_bfgs <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 300,
                                   init_coef_aux_pars_from_iid_model = FALSE)
    optim_params_het2_bfgs_iter <- c(optim_params_het2_bfgs, list(seed_rand_vec_trace = 1))

    ###################
    ## Dense GP ("Stable")
    ###################
    # Likelihood evaluated at given (not estimated) parameters: a pure formula check, independent of any optimizer
    nll_given_gp <- GPModel(gp_coords = coords_het2, cov_function = "exponential",
                            likelihood = "gaussian_heteroscedastic")$neg_log_likelihood(
      cov_pars = c(1, mean(dist(coords_het2)) / 3), y = y_het2, fixed_effects = rep(0, 2 * n_het2))
    expect_lt(abs(nll_given_gp - 172.94595730), TOLERANCE_MEDIUM)
    capture.output(gp_model_gp <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                             likelihood = "gaussian_heteroscedastic", y = y_het2, X = X_het2,
                                             params = optim_params_het2_bfgs), file = "NUL")
    coef_gp_bfgs <- as.vector(gp_model_gp$get_coef(std_err = FALSE))
    expected_coef_gp <- c(0.53370608, -0.06832911, -0.05858086, 0.76800104)
    expect_lt(sum(abs(coef_gp_bfgs - expected_coef_gp)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_gp$get_cov_pars(std_err = FALSE)) - c(0.10683991, 0.01031087))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_gp$get_current_neg_log_likelihood() - 163.47352630), TOLERANCE_MEDIUM)
    coord_test_gp <- coords_het2[1:3, , drop = FALSE] + 1e-3
    pred_gp <- predict(gp_model_gp, y = y_het2, gp_coords_pred = coord_test_gp, X_pred = X_het2[1:3, , drop = FALSE],
                       predict_var = TRUE, predict_response = TRUE)
    expected_mu_gp <- c(0.51363253, 0.49606308, 0.51154296)
    expected_var_gp <- c(1.18640601, 1.19690164, 1.42727248)
    expect_lt(sum(abs(pred_gp$mu - expected_mu_gp)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_gp$var - expected_var_gp)), TOLERANCE_MEDIUM)
    X_zero_het2 <- matrix(0, nrow = n_het2, ncol = ncol(X_het2))
    re_pred_train_gp <- predict_training_data_random_effects(gp_model_gp)
    expected_re_pred_train_gp <- c(-0.00843116, -0.02773224, 0.00938880, 0.03301499, 0.09514145)
    expect_lt(sum(abs(re_pred_train_gp[1:5, 1] - expected_re_pred_train_gp)), TOLERANCE_MEDIUM)
    pred_train_gp <- predict(gp_model_gp, y = y_het2, gp_coords_pred = coords_het2, X_pred = X_zero_het2,
                             predict_response = FALSE, predict_var = FALSE)
    expect_lt(sum(abs(as.vector(re_pred_train_gp[, 1]) - pred_train_gp$mu)), TOLERANCE_STRICT)

    ###################
    ## GP with Vecchia approximation
    ###################
    capture.output(gp_model_vecchia <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                                  likelihood = "gaussian_heteroscedastic", gp_approx = "vecchia",
                                                  num_neighbors = n_het2 - 1, vecchia_ordering = "none",
                                                  matrix_inversion_method = "cholesky",
                                                  y = y_het2, X = X_het2, params = optim_params_het2_bfgs), file = "NUL")
    coef_vecchia <- as.vector(gp_model_vecchia$get_coef(std_err = FALSE))
    # With num_neighbors = n - 1, Vecchia is exact and should match the dense GP fit (same optimizer) closely
    expect_lt(sum(abs(coef_vecchia - expected_coef_gp)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_vecchia$get_cov_pars(std_err = FALSE)) - c(0.10683991, 0.01031087))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_vecchia$get_current_neg_log_likelihood() - 163.47352630), TOLERANCE_MEDIUM)
    re_pred_train_vecchia <- predict_training_data_random_effects(gp_model_vecchia)
    capture.output( pred_train_vecchia <- predict(gp_model_vecchia, y = y_het2, gp_coords_pred = coords_het2, X_pred = X_zero_het2,
                                  predict_response = FALSE, predict_var = FALSE), file = "NUL")
    expect_lt(sum(abs(as.vector(re_pred_train_vecchia[, 1]) - pred_train_vecchia$mu)), TOLERANCE_STRICT)
    # matrix_inversion_method = "iterative" 
    capture.output(gp_model_vecchia_iter <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                                       likelihood = "gaussian_heteroscedastic", gp_approx = "vecchia",
                                                       num_neighbors = n_het2 - 1, vecchia_ordering = "none",
                                                       matrix_inversion_method = "iterative",
                                                       y = y_het2, X = X_het2, params = optim_params_het2_bfgs_iter), file = "NUL")
    coef_vecchia_iter <- as.vector(gp_model_vecchia_iter$get_coef(std_err = FALSE))
    expected_coef_vecchia_iter <- c(0.53573745, -0.07251654, -0.06497315, 0.77184161)
    expect_lt(sum(abs(coef_vecchia_iter - expected_coef_vecchia_iter)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(as.vector(gp_model_vecchia_iter$get_cov_pars(std_err = FALSE)) - c(0.11147938, 0.01147543))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_vecchia_iter$get_current_neg_log_likelihood() - 163.47418211), TOLERANCE_NON_CONVEX)

    ###################
    ## GP with FITC / VIF (full_scale_vecchia) approximation
    ###################
    capture.output(gp_model_fitc <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                               likelihood = "gaussian_heteroscedastic", gp_approx = "fitc",
                                               num_ind_points = 50, y = y_het2, X = X_het2, params = optim_params_het2_bfgs), file = "NUL")
    coef_fitc <- as.vector(gp_model_fitc$get_coef(std_err = FALSE))
    expected_coef_fitc <- c(0.53242708, -0.07016827, 0.04718839, 0.69725428)
    expect_lt(sum(abs(coef_fitc - expected_coef_fitc)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_fitc$get_cov_pars(std_err = FALSE)) - c(0.00873612, 0.00548896))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_fitc$get_current_neg_log_likelihood() - 163.49280003), TOLERANCE_MEDIUM)
    pred_fitc <- predict(gp_model_fitc, y = y_het2, gp_coords_pred = coord_test_gp, X_pred = X_het2[1:3, , drop = FALSE],
                         predict_var = TRUE, predict_response = TRUE)
    expected_mu_fitc <- c(0.51930607, 0.51859412, 0.50126126)
    expected_var_fitc <- c(1.20305368, 1.21152758, 1.43760007)
    expect_lt(sum(abs(pred_fitc$mu - expected_mu_fitc)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_fitc$var - expected_var_fitc)), TOLERANCE_MEDIUM)

    ###################
    ## Crossed / multiple grouped random effects (Woodbury identity)
    ###################
    group1_het2 <- rep(1:10, each = 10)
    group2_het2 <- rep(1:5, length.out = n_het2)
    # matrix_inversion_method = "cholesky" is required: "iterative" (the default for crossed / multiple grouped
    # random effects, i.e., num_re_group_total > 1) is not yet supported for this likelihood
    capture.output(gp_model_crossed <- fitGPModel(group_data = cbind(group1_het2, group2_het2),
                                                  likelihood = "gaussian_heteroscedastic", y = y_het2, X = X_het2,
                                                  matrix_inversion_method = "cholesky",
                                                  params = optim_params_het2_bfgs), file = "NUL")
    coef_crossed_bfgs <- as.vector(gp_model_crossed$get_coef(std_err = FALSE))
    expected_coef_crossed <- c(0.53302511, -0.07068343, 0.04484505, 0.70722753)
    expect_lt(sum(abs(coef_crossed_bfgs - expected_coef_crossed)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model_crossed$get_cov_pars(std_err = FALSE)) - c(0.00172197, 0.00177310))), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_crossed$get_current_neg_log_likelihood() - 163.49297024), TOLERANCE_MEDIUM)
    group_test_crossed <- cbind(c(1, 3, 11), c(1, 2, 6))
    pred_crossed <- predict(gp_model_crossed, y = y_het2, group_data_pred = group_test_crossed, X_pred = X_het2[1:3, , drop = FALSE],
                            predict_var = TRUE, predict_response = TRUE)
    expected_mu_crossed <- c(0.51311185, 0.51840682, 0.50163047)
    expected_var_crossed <- c(1.19716914, 1.20576336, 1.43534326)
    expect_lt(sum(abs(pred_crossed$mu - expected_mu_crossed)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred_crossed$var - expected_var_crossed)), TOLERANCE_MEDIUM)
    # With two crossed grouped random effects, predict_training_data_random_effects() returns one column per
    # random effect; their sum corresponds to the total random effect predicted via the X_pred = 0 trick
    re_pred_train_crossed <- predict_training_data_random_effects(gp_model_crossed)
    expected_re_pred_train_crossed_1 <- c(0.00145590, 0.00145590, 0.00145590, 0.00145590, 0.00145590)
    expected_re_pred_train_crossed_2 <- c(-0.00815135, 0.00366130, -0.00372236, 0.00801002, 0.00019450)
    expect_lt(sum(abs(re_pred_train_crossed[1:5, 1] - expected_re_pred_train_crossed_1)), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(re_pred_train_crossed[1:5, 2] - expected_re_pred_train_crossed_2)), TOLERANCE_MEDIUM)
    X_zero_crossed <- matrix(0, nrow = n_het2, ncol = ncol(X_het2))
    pred_train_crossed <- predict(gp_model_crossed, y = y_het2, group_data_pred = cbind(group1_het2, group2_het2), X_pred = X_zero_crossed,
                                  predict_response = FALSE, predict_var = FALSE)
    re_pred_train_crossed_sum <- as.vector(re_pred_train_crossed[, 1]) + as.vector(re_pred_train_crossed[, 2])
    expect_lt(sum(abs(re_pred_train_crossed_sum - pred_train_crossed$mu)), TOLERANCE_STRICT)
    # matrix_inversion_method = "iterative" 
    capture.output(gp_model_crossed_iter <- fitGPModel(group_data = cbind(group1_het2, group2_het2),
                                                       likelihood = "gaussian_heteroscedastic", y = y_het2, X = X_het2,
                                                       matrix_inversion_method = "iterative",
                                                       params = optim_params_het2_bfgs_iter), file = "NUL")
    coef_crossed_iter <- as.vector(gp_model_crossed_iter$get_coef(std_err = FALSE))
    expected_coef_crossed_iter <- c(0.53458908, -0.07561868, 0.05287152, 0.69402576)
    expect_lt(sum(abs(coef_crossed_iter - expected_coef_crossed_iter)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(as.vector(gp_model_crossed_iter$get_cov_pars(std_err = FALSE)) - c(0.00215241, 0.00234007))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_crossed_iter$get_current_neg_log_likelihood() - 163.49394794), TOLERANCE_NON_CONVEX)

    ###################
    ## Full scale Vecchia / VIF approximation (inducing points + Vecchia-approximated residual)
    ###################
    # Note: this is a hard, non-convex optimization problem (as with the dense GP above), so cholesky and iterative
    # (which uses a stochastic gradient) can converge to different stationary points with similar likelihood value;
    # we therefore compare the *exact* (cholesky-computed) negative log-likelihood at both solutions instead of coefficients
    optim_params_fsva <- c(optim_params_het2_bfgs, list(fitc_piv_chol_preconditioner_rank = 50))
    capture.output(gp_model_fsva <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                               likelihood = "gaussian_heteroscedastic", gp_approx = "full_scale_vecchia",
                                               num_ind_points = 30, num_neighbors = 19, cov_fct_taper_range = 0.5,
                                               matrix_inversion_method = "cholesky",
                                               y = y_het2, X = X_het2, params = optim_params_fsva), file = "NUL")
    coef_fsva <- as.vector(gp_model_fsva$get_coef(std_err = FALSE))
    cov_pars_fsva <- as.vector(gp_model_fsva$get_cov_pars())
    expected_coef_fsva <- c(0.51527546, 0.06715165, -0.24951696, 1.12150879)
    expect_lt(sum(abs(coef_fsva - expected_coef_fsva)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(cov_pars_fsva - c(0.73163534, 0.07601483))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_fsva$get_current_neg_log_likelihood() - 167.00513611), TOLERANCE_NON_CONVEX)
    pred_fsva <- predict(gp_model_fsva, y = y_het2, gp_coords_pred = coord_test_gp, X_pred = X_het2[1:3, , drop = FALSE],
                         predict_var = TRUE, predict_response = TRUE)
    expected_mu_fsva <- c(0.48902512, 0.42663286, 0.62958769)
    expected_var_fsva <- c(1.28537511, 1.37648473, 1.74035590)
    expect_lt(sum(abs(pred_fsva$mu - expected_mu_fsva)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(pred_fsva$var - expected_var_fsva)), TOLERANCE_NON_CONVEX)
    capture.output( gp_model_fsva_exact_ll <- GPModel(gp_coords = coords_het2, cov_function = "exponential",
                                      likelihood = "gaussian_heteroscedastic", gp_approx = "full_scale_vecchia",
                                      num_ind_points = 30, num_neighbors = 19, cov_fct_taper_range = 0.5,
                                      matrix_inversion_method = "cholesky"), file = "NUL")
    nll_fsva_at_chol_point <- gp_model_fsva_exact_ll$neg_log_likelihood(
      cov_pars = cov_pars_fsva, y = y_het2, fixed_effects = as.vector(cbind(X_het2 %*% coef_fsva[1:2], X_het2 %*% coef_fsva[3:4])))
    expect_equal(nll_fsva_at_chol_point, gp_model_fsva$get_current_neg_log_likelihood(), tolerance = TOLERANCE_MEDIUM)
    # matrix_inversion_method = "iterative" 
    optim_params_fsva_iter <- c(optim_params_fsva, list(seed_rand_vec_trace = 1))
    capture.output(gp_model_fsva_iter <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                                    likelihood = "gaussian_heteroscedastic", gp_approx = "full_scale_vecchia",
                                                    num_ind_points = 30, num_neighbors = 19, cov_fct_taper_range = 0.5,
                                                    matrix_inversion_method = "iterative",
                                                    y = y_het2, X = X_het2, params = optim_params_fsva_iter), file = "NUL")
    coef_fsva_iter <- as.vector(gp_model_fsva_iter$get_coef(std_err = FALSE))
    cov_pars_fsva_iter <- as.vector(gp_model_fsva_iter$get_cov_pars())
    expected_coef_fsva_iter <- c(0.36450404, 0.54980421, -1.02123877, 2.50972524)
    expect_lt(sum(abs(coef_fsva_iter - expected_coef_fsva_iter)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(cov_pars_fsva_iter - c(0.78854155, 0.06221294))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_fsva_iter$get_current_neg_log_likelihood() - 169.27528843), TOLERANCE_NON_CONVEX)
    nll_fsva_iter_at_exact_ll <- gp_model_fsva_exact_ll$neg_log_likelihood(
      cov_pars = cov_pars_fsva_iter, y = y_het2, fixed_effects = as.vector(cbind(X_het2 %*% coef_fsva_iter[1:2], X_het2 %*% coef_fsva_iter[3:4])))
    # The exact (cholesky) NLL at the iterative solution should be close to the exact NLL at the cholesky solution,
    # even though the coefficients themselves can differ substantially (flat likelihood ridge for this small dataset).
    # Tolerance is loose (relative to the ~150-200 scale of the NLL here) since this is a stochastic-gradient fit
    # converging on a hard, non-convex landscape; a real gradient bug would be expected to cause a much larger gap
    expect_lt(abs(nll_fsva_iter_at_exact_ll - nll_fsva_at_chol_point), 5)
    # matrix_inversion_method = "iterative" with the "vifdu" CG preconditioner (solves (Sigma^-1+W) directly, no push-through)
    optim_params_fsva_iter_vifdu <- c(optim_params_het2_bfgs, list(seed_rand_vec_trace = 1, cg_preconditioner_type = "vifdu"))
    capture.output(gp_model_fsva_iter_vifdu <- fitGPModel(gp_coords = coords_het2, cov_function = "exponential",
                                                          likelihood = "gaussian_heteroscedastic", gp_approx = "full_scale_vecchia",
                                                          num_ind_points = 30, num_neighbors = 19, cov_fct_taper_range = 0.5,
                                                          matrix_inversion_method = "iterative",
                                                          y = y_het2, X = X_het2, params = optim_params_fsva_iter_vifdu), file = "NUL")
    coef_fsva_iter_vifdu <- as.vector(gp_model_fsva_iter_vifdu$get_coef(std_err = FALSE))
    expected_coef_fsva_iter_vifdu <- c(0.36452982, 0.55026778, -1.02258335, 2.51333719)
    expect_lt(sum(abs(coef_fsva_iter_vifdu - expected_coef_fsva_iter_vifdu)), TOLERANCE_NON_CONVEX)
    expect_lt(sum(abs(as.vector(gp_model_fsva_iter_vifdu$get_cov_pars(std_err = FALSE)) - c(0.88243463, 0.06380923))), TOLERANCE_NON_CONVEX)
    expect_lt(abs(gp_model_fsva_iter_vifdu$get_current_neg_log_likelihood() - 170.19579450), TOLERANCE_NON_CONVEX)
  })

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

  test_that("linear covariance ", {

    params <- OPTIM_PARAMS_BFGS

    d_lin <- 50 # dimension of GP locations
    coords_lin <- matrix(sim_rand_unif(n=n*d_lin, init_c=0.1156), ncol=d_lin)
    beta_lin <- qnorm(sim_rand_unif(n=d_lin, init_c=0.1234),sd=1)
    lp_lin <- coords_lin %*% beta_lin + X %*% beta
    y <- lp_lin + qnorm(sim_rand_unif(n=n, init_c=0.2224), sd=0.1)
    coord_test <- matrix(sim_rand_unif(n=3*d_lin, init_c=0.19156), ncol=d_lin)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))

    likelihood <- "gaussian"
    for (cov_function in c("linear", "linear_no_woodbury")) {
      if (cov_function == "linear") {
        matrix_inversion_method_loop <- c("cholesky", "iterative")
      } else {
        matrix_inversion_method_loop <- c("cholesky")
      }
      for (matrix_inversion_method in matrix_inversion_method_loop) {
        if(matrix_inversion_method == "iterative") {
          tolerance_loc_1 <- 2
          tolerance_loc_2 <- 0.05
          tolerance_loc_3 <- 6
        } else {
          tolerance_loc_1 <- TOLERANCE_STRICT
          tolerance_loc_2 <- TOLERANCE_STRICT
          tolerance_loc_3 <- TOLERANCE_STRICT
          tolerance_loc_4 <- 0.002
        }

        # Evaluate negative log-likelihood
        gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                            matrix_inversion_method = matrix_inversion_method, cov_function = cov_function)
        # gp_model$set_optim_params(params = list(num_rand_vec_trace=500, init_coef_aux_pars_from_iid_model = FALSE))
        nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
        nll_exp <- 268.6641569
        expect_lt(abs(nll-nll_exp),tolerance_loc_1)
        # Estimation
        capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                               matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params) , file='NUL')
        cov_pars_exp <- c(0.01428942126, 0.92806146725)
        coef_exp <- c(0.08076221412, 1.97947766605)
        nll_opt_exp <- 81.26251299
        num_it <- 17
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),2*tolerance_loc_2)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tolerance_loc_1)
        if (matrix_inversion_method == "cholesky") expect_equal(gp_model$get_num_optim_iter(), num_it)
        # Prediction
        pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                        predict_var=TRUE, predict_response = FALSE)
        expected_mu <- c(4.671312214, 3.029084877, 7.400864491)
        expected_var <- c(0.01524446, 0.01621295, 0.01564379)
        expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(tolerance_loc_3))
        expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_4)

        # X_testd <- X
        # X_testd[,2] <- 0
        # X_testd[,1] <- 0
        # pred <- predict(gp_model, y=y, gp_coords_pred = coords_lin, X_pred = X_testd,
        #                 predict_var=TRUE, predict_response = FALSE)
        # b <- coords_lin %*% beta_lin
        # plot(b,pred$mu)
        # mean((b-pred$mu)^2) # 0.006641768

        ## Vecchia approximation
        if (matrix_inversion_method == "cholesky") {
          gp_approx <- "vecchia"
          num_neighbors <- n - 1
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors,
                              vecchia_ordering = "none")
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),tolerance_loc_1)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors,
                                                 vecchia_ordering = "none") , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tolerance_loc_2)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tolerance_loc_1)
          expect_equal(gp_model$get_num_optim_iter(), num_it)
          gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(tolerance_loc_3))
          expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_2)

          num_neighbors <- 50
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors, vecchia_ordering = "none")
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),relax_tolerance_nll(5))
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors, vecchia_ordering = "none") , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.2)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.3)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),45)
          expect_equal(gp_model$get_num_optim_iter(), 15)
          gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.05))
          expect_lt(sum(abs(pred$var-expected_var)),0.05)

          gp_approx <- "fitc"
          ind_points_selection <- "random"
          num_ind_points <- n-1
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),tolerance_loc_4)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),relax_tolerance(tolerance_loc_4))
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tolerance_loc_4)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tolerance_loc_4)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_4)
          expect_lt(sum(abs(pred$var-expected_var)),0.1)

          num_ind_points <- 50
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),1.2)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.02)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.05)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),2.5)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),0.15)
          expect_lt(sum(abs(pred$var-expected_var)),0.1)

          # VIF approximation
          gp_approx <- "vif"
          ind_points_selection <- "random"
          num_ind_points <- n-1
          num_neighbors <- 20
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                              ind_points_selection = ind_points_selection)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),tolerance_loc_4)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors,
                                                 num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),relax_tolerance(tolerance_loc_4))
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tolerance_loc_4)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tolerance_loc_4)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_4)
          expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_4)

          num_ind_points <- 50
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                              ind_points_selection = ind_points_selection)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),relax_tolerance_nll(0.1))
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors,
                                                 num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),relax_tolerance(tolerance_loc_4))
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),relax_tolerance(0.02))
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),relax_tolerance_nll(0.2))
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.05))
          expect_lt(sum(abs(pred$var-expected_var)),0.05)
        }

        ## GPBoost algorithm
        if (matrix_inversion_method == "cholesky") {
          dtrain <- gpb.Dataset(data = X, label = y)
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function)
          gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
          bst <- gpboost(data = dtrain, gp_model = gp_model,
                         nrounds = 30, learning_rate = 0.1, max_depth = 6,
                         min_data_in_leaf = 5, verbose = 0)
          expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-c(0.03919405941, 0.91870507429 ))),TOLERANCE_MEDIUM)
          # Prediction
          pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
                          predict_var = TRUE, pred_latent = TRUE)
          expect_lt(sum(abs(tail(pred$fixed_effect, n=3)-c(1.654867987, 2.755278195, 3.513302218))),TOLERANCE_MEDIUM)
          # Predict response
          pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
                          predict_var = TRUE, pred_latent = FALSE)
          expect_lt(sum(abs(tail(pred$response_mean, n=3)-c( 4.498812041, 2.449730254, 7.779354333))),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(tail(pred$response_var, n=3)-c(0.08045027559, 0.08266782334, 0.08156700720))), TOLERANCE_MEDIUM)
        }
      }
    }

    likelihood <- "t_fix_df"
    for (cov_function in c("linear", "linear_no_woodbury")) {
      if (cov_function == "linear") {
        matrix_inversion_method_loop <- c("cholesky", "iterative")
      } else {
        matrix_inversion_method_loop <- c("cholesky")
      }
      for (matrix_inversion_method in matrix_inversion_method_loop) {
        if(matrix_inversion_method == "iterative") {
          tolerance_loc_1 <- 2
          tolerance_loc_2 <- 0.05
          tolerance_loc_3 <- 4
        } else {
          tolerance_loc_1 <- TOLERANCE_STRICT
          tolerance_loc_2 <- TOLERANCE_STRICT
          tolerance_loc_3 <- TOLERANCE_STRICT
          tolerance_loc_4 <- 0.002
        }

        # Evaluate negative log-likelihood
        gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                            matrix_inversion_method = matrix_inversion_method, cov_function = cov_function)
        # gp_model$set_optim_params(params = list(num_rand_vec_trace=500, init_coef_aux_pars_from_iid_model = FALSE))
        nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5),y=y)
        nll_exp <- 227.5314805
        expect_lt(abs(nll-nll_exp),tolerance_loc_1)

        # Estimation
        if(matrix_inversion_method == "choklesky") {
          # Estimation is very slow for iterative methods
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,
                                                 matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                                 X=X, y = y, params = params) , file='NUL')
          cov_pars_exp <- c(0.9357944695)
          aux_par_exp <- c(0.09651268839, 2.00000000000)
          coef_exp <- c(0.1011884891, 1.9905600506)
          nll_opt_exp <- 82.49996414
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_par_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_MEDIUM)
          # Prediction
          pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                          predict_var=TRUE, predict_response = TRUE)
          expected_mu <- c(4.600315578, 3.029201064, 7.466329615)
          expected_var <- c(0.02586692444, 0.02691118187, 0.02630117411)
          expect_lt(sum(abs(pred$mu-expected_mu)),0.1)
          expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_MEDIUM)

          ## Vecchia approximation
          gp_approx <- "vecchia"
          num_neighbors <- n - 1
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5),y=y)
          expect_lt(abs(nll-nll_exp),tolerance_loc_1)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,
                                                 matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                                 X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_par_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_MEDIUM)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = TRUE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_MEDIUM)

          ## FITC approximation
          gp_approx <- "fitc"
          ind_points_selection <- "random"
          num_ind_points <- n-1
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, ind_points_selection = ind_points_selection, num_ind_points=num_ind_points)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5),y=y)
          expect_lt(abs(nll-nll_exp),1e-5)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,
                                                 matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                                 X=X, y = y, params = params,
                                                 gp_approx = gp_approx, ind_points_selection = ind_points_selection, num_ind_points=num_ind_points) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.5)
          expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_par_exp)),0.01)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.5)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),3)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = TRUE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),3)
          expect_lt(sum(abs(pred$var-expected_var)),0.1)

          # VIF approximation
          gp_approx <- "vif"
          ind_points_selection <- "kmeans++"
          num_ind_points <- 10
          num_neighbors <- 80
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                              ind_points_selection = ind_points_selection)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5),y=y)
          expect_lt(abs(nll-nll_exp),tolerance_loc_4)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors,
                                                 num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),3)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),4)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),120)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),2)
          expect_lt(sum(abs(pred$var-expected_var)),7)

          ## GPBoost algorithm
          dtrain <- gpb.Dataset(data = X, label = y)
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function)
          gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
          bst <- gpboost(data = dtrain, gp_model = gp_model,
                         nrounds = 30, learning_rate = 0.1, max_depth = 6,
                         min_data_in_leaf = 5, verbose = 0)
          expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-c(0.9269398031  ))),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-c(0.1895315932, 2.0000000000))),0.01)
          # Predict response
          pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
                          predict_var = TRUE, pred_latent = FALSE)
          expect_lt(sum(abs(tail(pred$response_mean, n=3)-c(4.510024051, 3.325082390, 7.658811482))),0.1)
          expect_lt(sum(abs(tail(pred$response_var, n=3)-c( 0.0982840077, 0.1011744166, 0.1000118224))), 0.01)
        }
      }
    }

  }) # end linear covariance

  test_that("hurst covariance ", {

    hurst_cov <- function(t, pars) {
      sigma2 <- pars[1]
      H <- pars[2]
      r  <- rowSums(t^2)
      rH <- r^H
      D2 <- as.matrix(dist(t))^2
      A <- outer(rH, rH, "+")
      K <- 0.5 * sigma2 * (A - D2^H)
      K
    }
    simulate_hurst <- function(t, pars, jitter = 1e-8) {
      K <- hurst_cov(t, pars)
      n  <- dim(t)[1]
      K <- K + jitter * diag(n)  # small jitter for numerical stability
      L <- chol(K)
      z <- qnorm(sim_rand_unif(n=n, init_c=0.1346), sd=0.1)
      y <- drop(L %*% z)
      y
    }

    params <- OPTIM_PARAMS_BFGS

    H_true <- 0.5
    sigma2_true <- 1
    pars <- c(sigma2_true, H_true)
    b <- simulate_hurst(coords, pars)
    y <- X %*% beta + b + qnorm(sim_rand_unif(n=n, init_c=0.1354), sd=sqrt(0.01))

    coord_test <- matrix(sim_rand_unif(n=3*2, init_c=0.19156), ncol=2)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))

    likelihood <- "gaussian"
    cov_function <- "hurst"
    matrix_inversion_method <- "cholesky"

    # Evaluate negative log-likelihood
    cov_pars_eval = c(0.01, sigma2_true, H_true)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    nll_exp <- 2508.161111
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params) , file='NUL')
    cov_pars_exp <- c(2.430011710e-02, 1.417072813e-07, 9.571564920e-01)
    coef_exp <- c(0.06807413795, 2.01626778203)
    nll_opt_exp <- -43.96963741
    num_it <- 26
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    if (matrix_inversion_method == "cholesky") expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-0.9400622610, 0.4713289372, 0.8745803091)
    expected_var <- c(1.416871849e-07, 1.416920045e-07, 1.417021983e-07)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## Vecchia approximation
    gp_approx <- "vecchia"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 50
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors,
                                        vecchia_ordering = "none") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),2)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           vecchia_ordering = "none") , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.3)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.01)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),0.1)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    gp_approx <- "vecchia_correlation"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 50
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-2512.097),10)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.3)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.01)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),0.1)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    gp_approx <- "fitc"
    ind_points_selection <- "random"
    num_ind_points <- n-1
    tol_fitc <- 0.001
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),tol_fitc)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_fitc)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    num_ind_points <- 50
    tol_fitc2 <- 0.15
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),300)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc2)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),1)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    # VIF approximation
    gp_approx <- "vif"
    ind_points_selection <- "random"
    num_ind_points <- n-1
    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                                        ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),1e-5)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_ind_points <- 50
    tol_vif <- 0.05
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                                        ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),relax_tolerance_nll(1))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_vif)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_vif)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_vif)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function) , file='NUL')
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-c(2.828394259e-02, 1.947589183e-10, 3.121883864e-01 ))),TOLERANCE_MEDIUM)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=3)-c(-0.6202239136, 0.3687055841, 0.7950425174))),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=3)-c(-0.6202239139, 0.3687055879, 0.7950425167))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(tail(pred$response_var, n=3)-c(0.02828394279, 0.02828394279, 0.02828394279))), TOLERANCE_MEDIUM)

    ####################
    ## non-Gaussian likelihood
    ####################
    likelihood <- "t_fix_df"

    # Evaluate negative log-likelihood
    cov_pars_eval = c(sigma2_true, H_true)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    nll_exp <- 196.6342458
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Estimation
    # Note: the optimizer ends up in a very flat region of the log-likelihood (the Hurst parameter goes to almost 0).
    #   Which point in this region it stops at depends on the order in which floating point numbers are summed, i.e.,
    #   on the number of OpenMP threads: measured against the values below (which were recorded with all threads of
    #   this machine), the deviations over 1, 2, 4 and 8 threads are up to 2e-4 for the covariance parameters, 7e-3
    #   for the coefficients, 2e-3 for the negative log-likelihood and 6e-2 for the predicted means, and the number
    #   of iterations varies between 59 and 69. The tolerances below have to accommodate this. Note: this must NOT be
    #   solved by setting 'num_parallel_threads' on the model, since that calls omp_set_num_threads() and thereby
    #   changes the thread count of every model built later in the same R process
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params) , file='NUL')
    cov_pars_exp <- c(0.01850628942, 0.008229165631)
    coef_exp <- c(0.087477519806 ,2.0203797574)
    nll_opt_exp <- -76.64280779
    tol_flat_region <- 0.05
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_flat_region)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_flat_region)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_flat_region)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-0.9401364534, 0.4826689913, 0.8811585466)
    expected_var <- c(0.0092433831081, 0.0091525152933, 0.0091867058615)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.15)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_LOOSE)

    ## Vecchia approximation
    gp_approx <- "vecchia"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # slow
    # capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
    #                                        matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
    #                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    # expect_equal(gp_model$get_num_optim_iter(), num_it)
    # gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    # capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
    #                                 predict_var=TRUE, predict_response = FALSE) , file='NUL')
    # expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors,
                                        vecchia_ordering = "none") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),0.5)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.1)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),10)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),0.2)
    expect_lt(sum(abs(pred$var-expected_var)),0.1)

    gp_approx <- "fitc"
    ind_points_selection <- "random"
    num_ind_points <- n-1
    tol_fitc <- 0.001
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),tol_fitc)
    #slow
    # capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
    #                                        matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
    #                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc)
    # expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc)
    # expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_fitc)
    # capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
    #                                 predict_var=TRUE, predict_response = FALSE) , file='NUL')
    # expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01))
    # expect_lt(sum(abs(pred$var-expected_var)),0.01)

    num_ind_points <- 20
    # Note: with only 20 inducing points, this is a very crude approximation and the fit is only compared to the
    #   exact one above to check that it is in the same ballpark. The exact fit ends up in a very flat region of
    #   the log-likelihood (see the note there), which the approximation does not reproduce, and the tolerances
    #   below thus have to be loose
    tol_fitc2 <- 2.5
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),2)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc2)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),50)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),1.)
    expect_lt(sum(abs(pred$var-expected_var)),0.1)

    # VIF approximation
    gp_approx <- "vif"
    ind_points_selection <- "kmeans++"
    num_ind_points <- 10
    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                                        ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),0.05)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),1.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.2)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),200)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),0.2)
    expect_lt(sum(abs(pred$var-expected_var)),2)

    # slow
    # ## GPBoost algorithm
    # dtrain <- gpb.Dataset(data = X, label = y)
    # capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
    #                                     matrix_inversion_method = matrix_inversion_method, cov_function = cov_function) , file='NUL')
    # gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    # bst <- gpboost(data = dtrain, gp_model = gp_model,
    #                nrounds = 2, learning_rate = 0.1, max_depth = 6,
    #                min_data_in_leaf = 5, verbose = 0)
    # expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-c(2.828394259e-02, 1.947589183e-10, 3.121883864e-01 ))),TOLERANCE_MEDIUM)
    # # Prediction
    # pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
    #                 predict_var = TRUE, pred_latent = TRUE)
    # expect_lt(sum(abs(tail(pred$fixed_effect, n=3)-c(-0.6202239136, 0.3687055841, 0.7950425174))),TOLERANCE_MEDIUM)
    # # Predict response
    # pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
    #                 predict_var = TRUE, pred_latent = FALSE)
    # expect_lt(sum(abs(tail(pred$response_mean, n=3)-c(-0.6202239139, 0.3687055879, 0.7950425167))),TOLERANCE_MEDIUM)
    # expect_lt(sum(abs(tail(pred$response_var, n=3)-c(0.02828394279, 0.02828394279, 0.02828394279))), TOLERANCE_MEDIUM)

    ####################
    ## Hurst ARD
    ####################
    likelihood <- "gaussian"
    cov_function <- "hurst_ard"
    matrix_inversion_method <- "cholesky"

    # Evaluate negative log-likelihood
    cov_pars_eval = c(0.01, sigma2_true, H_true, 1.5)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    nll_exp <- 2817.257021
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params) , file='NUL')
    cov_pars_exp <- c(2.430010340e-02, 1.159703688e-07, 9.612311400e-01, 8.006726088e-01)
    coef_exp <- c(0.06798478941, 2.01626275734)
    nll_opt_exp <- -43.96966225
    num_it <- 27
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    if (matrix_inversion_method == "cholesky") expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-0.9401500043, 0.4712381007, 0.8744885264)
    expected_var <- c(8.895122085e-08, 4.318616931e-08, 2.009318253e-08)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## Vecchia approximation
    gp_approx <- "vecchia"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors,
                                        vecchia_ordering = "none") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           vecchia_ordering = "none") , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),relax_tolerance_nll(5))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),3)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.01)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),1)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),0.025)
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    gp_approx <- "vecchia_correlation"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    # expect_lt(abs(nll-627.4222728),10) # some randomness
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),1)  some randomness
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.5)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),15)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),0.3)
    expect_lt(sum(abs(pred$var-expected_var)),0.3)

    gp_approx <- "fitc"
    ind_points_selection <- "random"
    num_ind_points <- n-1
    tol_fitc <- 0.005
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y) , file='NUL')
    expect_lt(abs(nll-nll_exp),tol_fitc)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_fitc)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    num_ind_points <- 50
    tol_fitc2 <- 6
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y) , file='NUL')
    expect_lt(abs(nll-nll_exp),600)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc2)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),1)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    # VIF approximation
    gp_approx <- "vif"
    ind_points_selection <- "random"
    num_ind_points <- n-1
    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                                        ind_points_selection = ind_points_selection) , file='NUL')
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y) , file='NUL')
    expect_lt(abs(nll-nll_exp),5e-5)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_ind_points <- 50
    tol_vif <- 0.06
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                                        ind_points_selection = ind_points_selection) , file='NUL')
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y) , file='NUL')
    expect_lt(abs(nll-nll_exp),relax_tolerance_nll(5))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_vif)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_vif)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_vif)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

  }) # end hurst covariance

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
    expect_lt(abs(nll-52.12617684),3e-5)

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
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.1945649727)),0.01)
    expect_lt(sum(abs(gp_model$get_aux_pars()-c(29.748038906, 0.289104109))),0.3)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.0813684525, 0.71460257263))),0.008)
    nll <- -46.42265403
    expect_lt(sum(abs((gp_model$get_current_neg_log_likelihood()-nll))),relax_tolerance_nll(0.002))
    expect_gt(gp_model$get_num_optim_iter(), 0)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.3927253661, 0.3321704886, 0.3861809101, 0.7298256767)
    expected_var <- c(0.02161799049, 0.02084689485, 0.02168851371, 0.04924037565)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.004)
    expect_lt(sum(abs(pred$var-expected_var)),0.0008)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    # Note: with the default convergence tolerance, the covariance parameter estimated here is bimodal: depending on
    #   the order in which floating point numbers are summed (i.e., on the number of OpenMP threads), the optimizer
    #   stops either at approximately 0.16 or at approximately 0.33. A tighter tolerance makes the result essentially
    #   thread-independent, but not bit-identical: deviations of up to about 0.01 have been observed for the
    #   covariance parameter and the predicted means below, which the tolerances have to accommodate
    gp_model$set_optim_params(params=modifyList(OPTIM_PARAMS_BFGS, list(delta_rel_conv = 1e-10)))
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    # Which of the two modes is reached depends on the platform, so the estimate is accepted at either of
    #   them (the reference platform stops at the first one, clang / libc++ at the second one). The
    #   predicted values are only compared with the expected values on the reference platform: they are
    #   not available for the second mode, and they react much more sensitively to the summation order
    #   than the estimate itself
    expect_lt(min(sum(abs(gp_model$get_cov_pars(std_err = FALSE) - 0.1589997424)),
                  sum(abs(gp_model$get_cov_pars(std_err = FALSE) - 0.3190))),
              relax_tolerance(0.05))
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    if (USE_STRICT_TOLERANCES) {
      expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.3889933901, 0.3305930951, 0.1999567510, 0.7131849737))),0.05)
      expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.019326910879, 0.018747560249, 0.015420479708, 0.044852818320))), 0.05)
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
    expect_lte(cvbst$best_score,-0.906197716909493*0.5)
    expect_gte(cvbst$best_score,-0.906197716909493*2)
    # Note: which iteration is selected here depends on validation scores that differ in the last digits between
    #   runs with different numbers of OpenMP threads (2 to 5 have been observed), so only a range is checked
    expect_lte(cvbst$best_iter, 5)
    expect_gte(cvbst$best_iter, 2)

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
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.4209158489)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_aux_pars()-c(3.50495674874, 0.06611314103 ))),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(-0.1713543234, 0.7616663663))),TOLERANCE_STRICT)
    expect_lt(sum(abs((gp_model$get_current_neg_log_likelihood()-36.79381797))),TOLERANCE_MEDIUM)
    expect_equal(gp_model$get_num_optim_iter(), 17)
    # Prediction
    group_test <- c(1,3,3,9999)
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = TRUE)
    expected_mu <- c(0.4938941250, 0.6200604917, 0.6895052787, 0.8658269508)
    expected_var <- c(0.07536757200, 0.08391117696, 0.08156129615, 0.05815172162)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-0.2014705208 )),TOLERANCE_LOOSE)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.6771125713, 0.6521416995, 0.6432654205, 0.7448076230))),TOLERANCE_LOOSE)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.06411222024, 0.06472276124, 0.06483977737, 0.08373115432))), TOLERANCE_LOOSE)

    # cv function
    dtrain <- gpb.Dataset(data = X, label = y)
    gp_model <- GPModel(group_data = group, likelihood = likelihood, matrix_inversion_method = "cholesky")
    output <- capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model,
                                              nrounds = 100, early_stopping_rounds = 5,
                                              use_gp_model_for_validation = TRUE, folds = folds, verbose = 0,
                                              deterministic = TRUE) )
    expect_lte(cvbst$best_score,0.821794098802474*(1+TOLERANCE_LOOSE))
    expect_gte(cvbst$best_score,0.821794098802474*(1-TOLERANCE_LOOSE))
    nit <- 3
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
    expect_lt(sum(abs(gp_model_nw$get_cov_pars(std_err = FALSE)-0.4107693823)),tolerance_loc_1)
    expect_lt(sum(abs(gp_model_nw$get_aux_pars()-0.2683705872)),tolerance_loc_1)
    expect_lt(sum(abs(as.vector(gp_model_nw$get_coef(std_err = FALSE))-c(-0.1347096134, 2.0887629560))),tolerance_loc_1)
    expect_lt(sum(abs((gp_model_nw$get_current_neg_log_likelihood()-116.1152356))),tolerance_loc_1)
    expect_equal(gp_model_nw$get_num_optim_iter(), 17)
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
    cov_pars_v <- c(0.0578304572593316, 0.0323206204551912, 0.715705182971271,
                    0.080146397671051, 0.0483181738202239, 0.00714845208930165)
    coef_v <- c(0.962932883927947, 0.110839163937625, 1.10211572960383, 0.0925165508666763)
    nll_v <- 537.536566769249
    expect_lt(sum(abs(as.vector(gp_model_gauss_v$get_cov_pars(std_err = TRUE))-cov_pars_v)),relax_tolerance(TOLERANCE_STRICT))
    expect_lt(sum(abs(as.vector(gp_model_gauss_v$get_coef(std_err = TRUE))-coef_v)),relax_tolerance(TOLERANCE_STRICT))
    expect_lt(abs(gp_model_gauss_v$get_current_neg_log_likelihood()-nll_v),relax_tolerance(TOLERANCE_MEDIUM))
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
    expect_lt(sum(abs(cov_pars_t_v_result[c(1,3)]-cov_pars_t_v[c(1,3)])),relax_tolerance(TOLERANCE_STRICT))# estimates
    # The standard errors of covariance and auxiliary parameters are obtained from a Hessian that is approximated
    #   with finite differences of a gradient which itself relies on an iterative mode finding algorithm (see
    #   'CalcHessianCovParAuxPars'). They are thus not reproducible to the same accuracy as the estimates: differences
    #   of a few 1e-6 have been observed between builds (the standard errors below are of the order of 0.02 - 0.17)
    expect_lt(sum(abs(cov_pars_t_v_result[c(2,4)]-cov_pars_t_v[c(2,4)])),relax_tolerance(TOLERANCE_MEDIUM))# standard errors
    expect_lt(sum(abs(aux_pars_t_v_result[1]-aux_pars_t_v[1])),relax_tolerance(TOLERANCE_STRICT))# estimate
    expect_lt(sum(abs(aux_pars_t_v_result[2:3]-aux_pars_t_v[2:3])),relax_tolerance(TOLERANCE_MEDIUM))# standard error and fixed df
    expect_true(is.nan(aux_pars_t_v_result[4])) # no standard error for the fixed (not estimated) degrees-of-freedom parameter
    expect_lt(sum(abs(coef_t_v_result[c(1,3)]-coef_t_v[c(1,3)])),relax_tolerance(TOLERANCE_STRICT))
    if (coef_se_available(coef_t_v_result)) expect_lt(sum(abs(coef_t_v_result[c(2,4)]-coef_t_v[c(2,4)])),relax_tolerance(TOLERANCE_STRICT))
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
