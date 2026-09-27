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
  
  test_that("Prediction does not leave state behind that changes later predictions", {

    y <- eps + xi
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", y = y,
                                           params = list(optimizer_cov = "fisher_scoring",
                                                         init_coef_aux_pars_from_iid_model = FALSE))
                    , file='NUL')
    training_data_random_effects <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    preds <- predict(gp_model, gp_coords_pred = coords, predict_var = TRUE, predict_response = FALSE)

    # Providing covariance parameters to predict() must not leave a factorization behind that is
    #   then reused by predict_training_data_random_effects() with the estimated parameters
    invisible(capture.output( predict(gp_model, gp_coords_pred = coords[1:3, ],
                                      cov_pars = 3 * as.numeric(gp_model$get_cov_pars()),
                                      predict_var = TRUE) , file='NUL'))
    training_data_random_effects_after <- predict_training_data_random_effects(gp_model, predict_var = TRUE)
    expect_lt(max(abs(training_data_random_effects_after - training_data_random_effects)), TOLERANCE_STRICT)

    # Sampling from the prior must not change the response data of the model and thus not the
    #   posterior predictions made afterwards
    prior_samples <- predict(gp_model, gp_coords_pred = coords[1:3, ], sample_prior = TRUE,
                             num_prior_samples = 2)$prior_samples
    expect_equal(dim(prior_samples), c(length(y), 2))
    expect_true(all(is.finite(prior_samples)))
    preds_after <- predict(gp_model, gp_coords_pred = coords, predict_var = TRUE, predict_response = FALSE)
    expect_lt(max(abs(preds_after$mu - preds$mu)), TOLERANCE_STRICT)
    expect_lt(max(abs(as.vector(preds_after$var) - as.vector(preds$var))), TOLERANCE_STRICT)

  })

  test_that("Calculating standard errors of covariance parameters does not change later predictions ", {

    # The standard errors are calculated with the factorization on the original scale, the rest of the
    # code expects it on the transformed scale with the error variance factored out. Leaving the
    # factorization behind on the original scale changed the prior samples drawn afterwards
    y_se <- eps + xi
    for (gp_approx_se in c("none", "vecchia")) {
      capture.output( gp_model_se <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                gp_approx = gp_approx_se, num_neighbors = n - 1,
                                                vecchia_ordering = "none", y = y_se,
                                                params = list(init_coef_aux_pars_from_iid_model = FALSE)),
                      file = 'NUL')
      cov_pars_se <- as.numeric(gp_model_se$get_cov_pars())
      capture.output( pred_se <- predict(gp_model_se, gp_coords_pred = coords[1:5, ], predict_var = TRUE,
                                         predict_response = FALSE), file = 'NUL')
      num_samples_se <- 2000
      capture.output( prior_before_se <- predict(gp_model_se, gp_coords_pred = coords[1:3, ], sample_prior = TRUE,
                                                 num_prior_samples = num_samples_se)$prior_samples, file = 'NUL')
      expect_equal(dim(prior_before_se), c(n, num_samples_se))
      invisible(gp_model_se$get_cov_pars(std_err = TRUE))
      capture.output( prior_after_se <- predict(gp_model_se, gp_coords_pred = coords[1:3, ], sample_prior = TRUE,
                                                num_prior_samples = num_samples_se)$prior_samples, file = 'NUL')
      capture.output( pred_after_se <- predict(gp_model_se, gp_coords_pred = coords[1:5, ], predict_var = TRUE,
                                               predict_response = FALSE), file = 'NUL')
      # The prior samples have the marginal variance of the Gaussian process
      expect_lt(abs(mean(prior_before_se^2) / cov_pars_se[2] - 1), TOLERANCE_ITERATIVE)
      expect_lt(abs(mean(prior_after_se^2) / cov_pars_se[2] - 1), TOLERANCE_ITERATIVE)
      # The posterior predictions must not change either
      expect_lt(max(abs(pred_after_se$mu - pred_se$mu)), TOLERANCE_STRICT)
      expect_lt(max(abs(as.vector(pred_after_se$var) - as.vector(pred_se$var))), TOLERANCE_STRICT)
    }
  })

  test_that("Repeated predictions with the same model give the same result ", {

    # The predictive variances of the full-scale approximations are estimated stochastically. The random
    # vectors have to depend only on the seed: drawing new ones at every call made a prediction that is
    # repeated with unchanged arguments return different predictive variances
    y_rp <- eps + xi
    cov_pars_rp <- c(0.05, sigma2_1, rho)
    coord_test_rp <- cbind(seq(0.05, 0.95, length.out = 20), seq(0.95, 0.05, length.out = 20))
    args_rp <- list(
      fitc = list(gp_approx = "fitc", num_ind_points = 20, ind_points_selection = "kmeans++",
                  matrix_inversion_method = "cholesky"),
      full_scale_tapering_cholesky = list(gp_approx = "full_scale_tapering", num_ind_points = 20,
                                          ind_points_selection = "kmeans++", cov_fct_taper_range = 0.3,
                                          cov_fct_taper_shape = 2, matrix_inversion_method = "cholesky"),
      full_scale_tapering_iterative = list(gp_approx = "full_scale_tapering", num_ind_points = 20,
                                           ind_points_selection = "kmeans++", cov_fct_taper_range = 0.3,
                                           cov_fct_taper_shape = 2, matrix_inversion_method = "iterative"))
    for (case_rp in names(args_rp)) {
      capture.output( gp_model_rp <- do.call(GPModel, c(list(gp_coords = coords, cov_function = "exponential"),
                                                        args_rp[[case_rp]])), file = 'NUL')
      capture.output( pred_rp_1 <- predict(gp_model_rp, y = y_rp, gp_coords_pred = coord_test_rp,
                                           cov_pars = cov_pars_rp, predict_var = TRUE), file = 'NUL')
      capture.output( pred_rp_2 <- predict(gp_model_rp, y = y_rp, gp_coords_pred = coord_test_rp,
                                           cov_pars = cov_pars_rp, predict_var = TRUE), file = 'NUL')
      expect_lt(sum(abs(pred_rp_1$mu - pred_rp_2$mu)), TOLERANCE_STRICT, label = paste0("predictive mean (", case_rp, ")"))
      expect_lt(sum(abs(as.vector(pred_rp_1$var) - as.vector(pred_rp_2$var))), TOLERANCE_STRICT,
                label = paste0("predictive variance (", case_rp, ")"))
    }

  })

  test_that("Sampling from the prior does not change the state of a full-scale Vecchia model ", {

    # Prior samples of the latent process need Vecchia factors without the nugget effect. They must not
    # replace the factors of the observed process, which are cached and reused by later calculations
    y_ps <- eps + xi
    capture.output( gp_model_ps <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                              gp_approx = "full_scale_vecchia", num_neighbors = 15,
                                              num_ind_points = 10, ind_points_selection = "random",
                                              vecchia_ordering = "none", y = y_ps,
                                              params = OPTIM_PARAMS_BFGS), file = 'NUL')
    cov_pars_ps <- as.numeric(gp_model_ps$get_cov_pars())
    pred_ps <- predict(gp_model_ps, gp_coords_pred = coords[1:5, ], predict_var = TRUE,
                       predict_response = FALSE)
    num_samples_ps <- 30000
    latent_ps <- predict(gp_model_ps, gp_coords_pred = coords[1:3, ], sample_prior = TRUE,
                         num_prior_samples = num_samples_ps,
                         predict_response = FALSE)$prior_samples
    expect_equal(dim(latent_ps), c(n, num_samples_ps))
    expect_lt(abs(mean(latent_ps^2) / cov_pars_ps[2] - 1), TOLERANCE_ITERATIVE)
    # the predictions must not change after the prior has been sampled
    pred_after_ps <- predict(gp_model_ps, gp_coords_pred = coords[1:5, ], predict_var = TRUE,
                             predict_response = FALSE)
    expect_lt(max(abs(pred_after_ps$mu - pred_ps$mu)), TOLERANCE_STRICT)
    expect_lt(max(abs(as.vector(pred_after_ps$var) - as.vector(pred_ps$var))), TOLERANCE_STRICT)
    # prior samples of the observed process drawn afterwards still contain the nugget effect
    response_ps <- predict(gp_model_ps, gp_coords_pred = coords[1:3, ], sample_prior = TRUE,
                           num_prior_samples = num_samples_ps,
                           predict_response = TRUE)$prior_samples
    expect_lt(abs(mean(response_ps^2) / (cov_pars_ps[1] + cov_pars_ps[2]) - 1), TOLERANCE_ITERATIVE)

  })

}
