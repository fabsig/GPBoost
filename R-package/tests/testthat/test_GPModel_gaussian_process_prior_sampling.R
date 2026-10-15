context("GPModel_gaussian_process")

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  TOLERANCE_ITERATIVE <- 1E-1
  TOLERANCE_LOOSE <- 1E-2
  TOLERANCE_MEDIUM <- 1e-3
  TOLERANCE_STRICT <- 1E-5
  # Some of the optimization problems below are non-convex, and a different compiler or standard library
  # does not reproduce floating point arithmetic bit-wise. The tight tolerances therefore only hold on the
  # reference platform on which the expected values were calculated. 'relax_tolerance*()' of
  # helper-tolerances.R relaxes them elsewhere and reports once per test run which of the two is in force.
  # Covariance functions with a general (non-fixed) smoothness need 'std::cyl_bessel_k', which is a C++17
  # feature that is not provided by every standard library (in particular not by libc++, which is used by
  # clang on macOS and in the clang sanitizer containers of R-hub / CRAN)
  SKIP_BESSEL_COV_TESTS <- !gpboost:::has_std_cyl_bessel_k() &&
    Sys.getenv("GPBOOST_RUN_BESSEL_COV_TESTS") != "true"
  
  DEFAULT_OPTIM_PARAMS <- list(optimizer_cov = "gradient_descent",
                               lr_cov = 0.1, use_nesterov_acc = TRUE,
                               acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                               optimizer_coef = "gradient_descent", lr_coef = 0.1,
                               convergence_criterion = "relative_change_in_log_likelihood",
                               cg_delta_conv = 1E-6, cg_preconditioner_type = "predictive_process_plus_diagonal",
                               cg_max_num_it = 1000, cg_max_num_it_tridiag = 1000,
                               num_rand_vec_trace = 1000, reuse_rand_vec_trace = TRUE,
                               init_coef_aux_pars_from_iid_model = FALSE)
  DEFAULT_OPTIM_PARAMS_FISHER <- list(optimizer_cov = "fisher_scoring", delta_rel_conv = 1E-6,
                                      optimizer_coef = "gradient_descent", lr_coef = 0.1,
                                      convergence_criterion = "relative_change_in_log_likelihood",
                                      cg_delta_conv = 1E-6, cg_preconditioner_type = "predictive_process_plus_diagonal",
                                      cg_max_num_it = 1000, cg_max_num_it_tridiag = 1000,
                                      num_rand_vec_trace = 1000, reuse_rand_vec_trace = TRUE,
                                      seed_rand_vec_trace = 1,
                                      init_coef_aux_pars_from_iid_model = FALSE)
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
  C <- t(chol(Sigma))
  b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.8))
  eps <- as.vector(C %*% b_1)
  # Random coefficients
  Z_SVC <- matrix(sim_rand_unif(n=n*2, init_c=0.6), ncol=2) # covariate data for random coeffients
  colnames(Z_SVC) <- c("var1","var2")
  b_2 <- qnorm(sim_rand_unif(n=n, init_c=0.17))
  b_3 <- qnorm(sim_rand_unif(n=n, init_c=0.42))
  eps_svc <- as.vector(C %*% b_1 + Z_SVC[,1] * C %*% b_2 + Z_SVC[,2] * C %*% b_3)
  # Error term
  xi <- qnorm(sim_rand_unif(n=n, init_c=0.1)) / 5
  # Data for linear mixed effects model
  X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
  beta <- c(2,2) # regression coefficients
  # cluster_ids 
  cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))
  # GP with multiple observations at the same locations
  coords_multiple <- matrix(sim_rand_unif(n=n*d/4, init_c=0.1), ncol=d)
  coords_multiple <- rbind(coords_multiple,coords_multiple,coords_multiple,coords_multiple)
  D_multiple <- as.matrix(dist(coords_multiple))
  Sigma_multiple <- sigma2_1*exp(-D_multiple/rho)+diag(1E-10,n)
  C_multiple <- t(chol(Sigma_multiple))
  b_multiple <- qnorm(sim_rand_unif(n=n, init_c=0.8))
  eps_multiple <- as.vector(C_multiple %*% b_multiple)
  
  test_that("Training data random effects and prior sampling for a Gaussian Vecchia approximation", {

    # With all neighbors the Vecchia approximation is exact and can be compared with dense algebra.
    # This also covers the default ('lbfgs') optimizer, for which the preparations for the
    #   prediction must not try to calculate covariance matrices of individual components
    y <- eps + xi
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n - 1,
                                           vecchia_ordering = "none", y = y,
                                           params = list(init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    cov_pars <- as.numeric(gp_model$get_cov_pars())
    prior_cov <- cov_pars[2] * exp(-D / cov_pars[3])
    psi <- prior_cov + cov_pars[1] * diag(n)
    expected_mean <- as.vector(prior_cov %*% solve(psi, y))
    expected_var <- diag(prior_cov - prior_cov %*% solve(psi, prior_cov))
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    expect_lt(max(abs(training_data_random_effects[, 1] - expected_mean)), TOLERANCE_MEDIUM)
    expect_lt(max(abs(training_data_random_effects[, 2] - expected_var)), TOLERANCE_MEDIUM)
    capture.output( prior_samples <- predict(gp_model, gp_coords_pred = coords[1:3, ], sample_prior = TRUE,
                                             num_prior_samples = 2)$prior_samples, file = 'NUL')
    expect_equal(dim(prior_samples), c(n, 2))
    expect_true(all(is.finite(prior_samples)))

    # with weights: the error variance of observation i is cov_pars[1] / weights[i]
    weights <- 0.5 + sim_rand_unif(n = n, init_c = 0.914)
    capture.output( gp_model_w <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                             gp_approx = "vecchia", num_neighbors = n - 1,
                                             vecchia_ordering = "none", weights = weights, y = y,
                                             params = list(init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    cov_pars_w <- as.numeric(gp_model_w$get_cov_pars())
    prior_cov_w <- cov_pars_w[2] * exp(-D / cov_pars_w[3])
    psi_w <- prior_cov_w + cov_pars_w[1] * diag(1 / weights)
    expected_mean_w <- as.vector(prior_cov_w %*% solve(psi_w, y))
    expected_var_w <- diag(prior_cov_w - prior_cov_w %*% solve(psi_w, prior_cov_w))
    training_data_random_effects_w <- predict_training_data_random_effects(gp_model_w, predict_var = TRUE)
    expect_lt(max(abs(training_data_random_effects_w[, 1] - expected_mean_w)), TOLERANCE_MEDIUM)
    expect_lt(max(abs(training_data_random_effects_w[, 2] - expected_var_w)), TOLERANCE_MEDIUM)

  })


  test_that("Prior samples of the latent process do not contain the nugget effect ", {

    # 'predict_response = FALSE' has to draw from the covariance of the latent Gaussian process,
    # 'predict_response = TRUE' from the covariance of the observed process, which adds the nugget effect
    nugget_pr <- 0.8
    sigma2_pr <- 1.2
    cov_pars_pr <- c(nugget_pr, sigma2_pr, rho)
    num_samples_pr <- 30000
    for (gp_approx_pr in c("none", "vecchia", "full_scale_vecchia")) {
      capture.output( gp_model_pr <- GPModel(gp_coords = coords, cov_function = "exponential",
                                             gp_approx = gp_approx_pr, num_neighbors = n - 1,
                                             num_ind_points = 20, ind_points_selection = "random",
                                             vecchia_ordering = "none"), file = 'NUL')
      capture.output( latent_pr <- predict(gp_model_pr, gp_coords_pred = coords[1:3, ], cov_pars = cov_pars_pr,
                                           sample_prior = TRUE, num_prior_samples = num_samples_pr,
                                           predict_response = FALSE)$prior_samples, file = 'NUL')
      capture.output( response_pr <- predict(gp_model_pr, gp_coords_pred = coords[1:3, ], cov_pars = cov_pars_pr,
                                             sample_prior = TRUE, num_prior_samples = num_samples_pr,
                                             predict_response = TRUE)$prior_samples, file = 'NUL')
      expect_equal(dim(latent_pr), c(n, num_samples_pr))
      expect_lt(abs(mean(latent_pr^2) / sigma2_pr - 1), TOLERANCE_ITERATIVE)
      expect_lt(abs(mean(response_pr^2) / (sigma2_pr + nugget_pr) - 1), TOLERANCE_ITERATIVE)
    }

    # At duplicate locations the latent process takes the same value, the observed process does not
    coords_pr <- rbind(coords[1:10, ], coords[1:10, ])
    for (gp_approx_pr in c("none", "vecchia")) {
      capture.output( gp_model_pr <- GPModel(gp_coords = coords_pr, cov_function = "exponential",
                                             gp_approx = gp_approx_pr,
                                             num_neighbors = nrow(coords_pr) - 1,
                                             vecchia_ordering = "none"), file = 'NUL')
      capture.output( latent_pr <- predict(gp_model_pr, gp_coords_pred = coords[1:3, ], cov_pars = cov_pars_pr,
                                           sample_prior = TRUE, num_prior_samples = 1000,
                                           predict_response = FALSE)$prior_samples, file = 'NUL')
      expect_lt(max(abs(latent_pr[1:10, ] - latent_pr[11:20, ])), 1e-3)
    }
  })

  test_that("Prior samples for a non-Gaussian likelihood without an approximation ", {

    # The samples are drawn from the covariance of the latent process, which is calculated on the scale of the unique
    # locations if there are duplicate locations
    sigma2_ng <- 1.3
    num_samples_ng <- 20000
    coords_unique_ng <- coords[1:20, ]
    D_ng <- as.matrix(dist(coords_unique_ng))
    for (has_duplicates_ng in c(FALSE, TRUE)) {
      coords_ng <- if (has_duplicates_ng) rbind(coords_unique_ng, coords_unique_ng) else coords_unique_ng
      capture.output( gp_model_ng <- GPModel(gp_coords = coords_ng, cov_function = "exponential",
                                             likelihood = "bernoulli_probit"), file = 'NUL')
      capture.output( prior_ng <- predict(gp_model_ng, gp_coords_pred = coords[1:3, ], cov_pars = c(sigma2_ng, rho),
                                          sample_prior = TRUE, num_prior_samples = num_samples_ng,
                                          predict_response = FALSE)$prior_samples, file = 'NUL')
      expect_equal(dim(prior_ng), c(nrow(coords_ng), num_samples_ng))
      expect_lt(abs(mean(prior_ng^2) / sigma2_ng - 1), TOLERANCE_ITERATIVE)
      cov_emp_ng <- cov(t(prior_ng[1:20, ]))
      expect_lt(max(abs(cov_emp_ng - sigma2_ng * exp(-D_ng / rho))), 0.1)
      if (has_duplicates_ng) {
        expect_equal(prior_ng[1:20, ], prior_ng[21:40, ])
      }
    }

  })

  test_that("Prior samples when the prediction data contains a new cluster ", {

    # The prior samples are generated at the locations of the training data. A cluster that has no training data
    # has no prior samples and must not write its predictive means into them
    capture.output( gp_model_nc <- GPModel(gp_coords = coords, cov_function = "exponential", cluster_ids = cluster_ids),
                    file = 'NUL')
    capture.output( prior_nc <- predict(gp_model_nc, gp_coords_pred = coords[1:60, ], cluster_ids_pred = rep(1:3, each = 20),
                                        cov_pars = c(0.5, sigma2_1, rho), sample_prior = TRUE, num_prior_samples = 3,
                                        predict_response = FALSE)$prior_samples, file = 'NUL')
    expect_equal(dim(prior_nc), c(n, 3))
    expect_true(all(is.finite(prior_nc)))
    expect_false(any(prior_nc == 0))

  })

  test_that("Prior sampling is not supported for grouped random effects ", {

    group_ps <- rep(1:10, each = 10)
    group_2_ps <- rep(1:5, times = 20)
    for (group_data_ps in list(group_ps, cbind(group_ps, group_2_ps))) {
      capture.output( gp_model_gr <- GPModel(group_data = group_data_ps), file = 'NUL')
      group_data_pred_ps <- if (is.matrix(group_data_ps)) group_data_ps[1:3, ] else group_data_ps[1:3]
      expect_error(predict(gp_model_gr, group_data_pred = group_data_pred_ps, cov_pars = c(0.5, rep(1, NCOL(group_data_ps))),
                           sample_prior = TRUE, num_prior_samples = 2), "not implemented for grouped random effects")
    }

  })

}
