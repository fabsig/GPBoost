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
  
  test_that("CUDA GPU", {
    
    y <- eps + X%*%beta + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    cov_pars_pred <- c(0.1,1,0.1)
    init_cov_pars <- c(var(y)/2,var(y)/2,mean(dist(coords))/3)
    params = OPTIM_PARAMS_BFGS
    params$init_cov_pars <- init_cov_pars
    
    # VIF without GPU
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", GPU_use = FALSE,
                                           gp_approx = "vif_correlation_based",num_ind_points = 50,num_neighbors = 20,
                                           y = y, X = X,  matrix_inversion_method = "cholesky",
                                           params = OPTIM_PARAMS_BFGS), file='NUL')
    cov_pars_without_GPU <- as.vector(gp_model$get_cov_pars(std_err = FALSE))
    coefs_without_GPU <- as.vector(gp_model$get_coef(std_err = FALSE))
    NLL_without_GPU <- gp_model$get_current_neg_log_likelihood()
    Num_optim_iter_without_GPU <- gp_model$get_num_optim_iter()
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    pred_mu_without_GPU <- pred$mu
    pred_var_without_GPU <- as.vector(pred$var)
    
    # VIF with GPU
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", GPU_use = TRUE,
                                           gp_approx = "vif_correlation_based",num_ind_points = 50,num_neighbors = 20,
                                           y = y, X = X,  matrix_inversion_method = "cholesky",
                                           params = OPTIM_PARAMS_BFGS), file='NUL')
    
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE)) - cov_pars_without_GPU)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE)) - coefs_without_GPU)),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood() - NLL_without_GPU),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), Num_optim_iter_without_GPU)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_var = TRUE, cov_pars = cov_pars_pred)
    expect_lt(sum(abs(pred$mu - pred_mu_without_GPU)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$var) - pred_var_without_GPU)),TOLERANCE_STRICT)
    
  })
   
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


  test_that("Standard errors of covariance parameters for FITC and full-scale tapering with several clusters ", {

    # Both approximations are exact here (FITC with as many inducing points as data points per cluster,
    # full-scale tapering with a taper range that covers all distances), so they have to give the same
    # standard errors as the model without an approximation. The scaling of the Fisher information with
    # the error variance is applied once to the sum over all clusters; applying it inside the loop over
    # the clusters rescaled the contributions of the previous clusters again
    n_cl_fi <- 4
    cluster_ids_fi <- rep(1:n_cl_fi, each = n / n_cl_fi)
    y_fi <- eps + xi
    params_fi <- c(OPTIM_PARAMS_BFGS,
                   list(init_cov_pars = c(var(y_fi) / 2, var(y_fi) / 2, mean(dist(coords)) / 3),
                        num_rand_vec_trace = 1000, reuse_rand_vec_trace = TRUE, seed_rand_vec_trace = 1))
    args_fi <- list(gp_coords = coords, cov_function = "exponential", cluster_ids = cluster_ids_fi,
                    y = y_fi, params = params_fi)
    capture.output( gp_model_exact_fi <- do.call(fitGPModel, args_fi), file = 'NUL')
    cov_pars_exact_fi <- gp_model_exact_fi$get_cov_pars(std_err = TRUE)
    args_approx_fi <- list(
      fitc = c(args_fi, list(gp_approx = "fitc", num_ind_points = n / n_cl_fi,
                             ind_points_selection = "random")),
      full_scale_tapering = c(args_fi, list(gp_approx = "full_scale_tapering",
                                            num_ind_points = n / n_cl_fi - 1,
                                            ind_points_selection = "random",
                                            cov_fct_taper_range = 1e6, cov_fct_taper_shape = 2,
                                            matrix_inversion_method = "cholesky")))
    for (gp_approx_fi in names(args_approx_fi)) {
      capture.output( gp_model_fi <- do.call(fitGPModel, args_approx_fi[[gp_approx_fi]]), file = 'NUL')
      cov_pars_fi <- gp_model_fi$get_cov_pars(std_err = TRUE)
      expect_lt(max(abs(cov_pars_fi[1, ] / cov_pars_exact_fi[1, ] - 1)), TOLERANCE_MEDIUM)
      # The Fisher information of these approximations is calculated with stochastic trace estimation,
      #   so the standard errors are only equal up to a Monte Carlo error
      expect_lt(max(abs(cov_pars_fi[2, ] / cov_pars_exact_fi[2, ] - 1)), TOLERANCE_ITERATIVE)
    }
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

  test_that("Full-scale Vecchia approximation with cover tree inducing points ", {

    # The cover tree determines the number of inducing points itself, so the result must not depend on
    # 'num_ind_points'. With all Vecchia neighbors the approximation is exact and can be compared with
    # the model without an approximation
    y_ct <- eps + xi
    cov_pars_ct <- c(0.05, sigma2_1, rho)
    gp_model_exact_ct <- GPModel(gp_coords = coords, cov_function = "exponential")
    nll_exact_ct <- gp_model_exact_ct$neg_log_likelihood(cov_pars = cov_pars_ct, y = y_ct)
    for (num_ind_points_ct in c(10, 20, 50)) {
      capture.output( gp_model_ct <- GPModel(gp_coords = coords, cov_function = "exponential",
                                             gp_approx = "full_scale_vecchia", num_neighbors = n - 1,
                                             vecchia_ordering = "none",
                                             num_ind_points = num_ind_points_ct,
                                             ind_points_selection = "cover_tree"), file = 'NUL')
      nll_ct <- gp_model_ct$neg_log_likelihood(cov_pars = cov_pars_ct, y = y_ct)
      expect_lt(abs(nll_ct - nll_exact_ct), TOLERANCE_MEDIUM)
    }
  })

  test_that("Inducing points from the cover tree for several clusters of different size ", {

    # The cover tree determines the number of inducing points separately for every cluster. Imposing the
    # number selected for one cluster on the next one made the construction of the smaller cluster fail,
    # depending on the order in which the clusters are processed
    n_small_ct2 <- 12
    # the small cluster is a tight blob, for which the cover tree selects clearly fewer inducing points
    #   than for the large cluster, which covers the whole domain
    coords_ct2 <- rbind(coords[1:(n - n_small_ct2), ], 0.02 * coords[1:n_small_ct2, ])
    y_ct2 <- eps + xi
    cov_pars_ct2 <- c(0.05, sigma2_1, rho)
    cluster_ids_ct2 <- list(large_cluster_first = c(rep(1, n - n_small_ct2), rep(2, n_small_ct2)),
                            small_cluster_first = c(rep(2, n - n_small_ct2), rep(1, n_small_ct2)))
    for (gp_approx_ct2 in c("fitc", "full_scale_tapering", "full_scale_vecchia")) {
      for (cluster_order_ct2 in names(cluster_ids_ct2)) {
        capture.output( gp_model_ct2 <- GPModel(gp_coords = coords_ct2, cov_function = "exponential",
                                                gp_approx = gp_approx_ct2,
                                                cluster_ids = cluster_ids_ct2[[cluster_order_ct2]],
                                                num_ind_points = 5, cover_tree_radius = 0.2,
                                                ind_points_selection = "cover_tree",
                                                num_neighbors = 10, vecchia_ordering = "none",
                                                cov_fct_taper_range = 0.5, cov_fct_taper_shape = 2,
                                                matrix_inversion_method = "cholesky"), file = 'NUL')
        capture.output( nll_ct2 <- gp_model_ct2$neg_log_likelihood(cov_pars = cov_pars_ct2, y = y_ct2),
                        file = 'NUL')
        expect_true(is.finite(nll_ct2))
      }
    }
  })

  test_that("Profiled-out parameters belong to the parameters that are returned ", {

    # When the line search of 'lbfgs_linesearch_nocedal_wright' runs out of iterations, it returns the best
    # point found so far and not the one that has been evaluated last. The profiled-out error variance and
    # regression coefficients have to be moved back to that point as well, otherwise they are the ones of the
    # rejected last candidate and do not belong to the covariance parameters that are returned (relative
    # error 1.2e-08 here, and 1.4e-02 when the line search is forced to stop after two iterations)
    y_po <- eps + as.vector(X %*% beta) + xi
    for (optimizer_po in c("lbfgs", "lbfgs_linesearch_nocedal_wright")) {
      capture.output( gp_model_po <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                                y = y_po, X = X,
                                                params = list(optimizer_cov = optimizer_po, maxit = 1000,
                                                              init_coef_aux_pars_from_iid_model = FALSE)),
                      file = 'NUL')
      cov_pars_po <- as.numeric(gp_model_po$get_cov_pars())
      coef_po <- as.numeric(gp_model_po$get_coef())
      # the closed-form expressions that are profiled out, with the error variance factored out of the
      #   covariance matrix (which is the scale on which the optimizer works)
      psi_po <- (cov_pars_po[2] / cov_pars_po[1]) * exp(-D / cov_pars_po[3]) + diag(n)
      resid_po <- as.vector(y_po - X %*% coef_po)
      sigma2_po <- as.numeric(t(resid_po) %*% solve(psi_po, resid_po)) / n
      coef_wls_po <- as.vector(solve(t(X) %*% solve(psi_po, X), t(X) %*% solve(psi_po, y_po)))
      expect_lt(abs(cov_pars_po[1] / sigma2_po - 1), 1E-10,
                label = paste0("profiled-out error variance (", optimizer_po, ")"))
      expect_lt(max(abs(coef_po / coef_wls_po - 1)), 1E-10,
                label = paste0("profiled-out regression coefficients (", optimizer_po, ")"))
    }

  })

  test_that("Cover tree inducing points with the default number of inducing points ", {

    # The cover tree determines the number of inducing points itself and ignores 'num_ind_points'. Checking
    # the requested number (here its default, 500 and 200) against the data rejected data sets for which the
    # cover tree selects an admissible number, so only the number that has been selected is checked
    y_ctd <- eps + xi
    cov_pars_ctd <- c(0.05, sigma2_1, rho)
    for (gp_approx_ctd in c("fitc", "full_scale_tapering", "full_scale_vecchia")) {
      capture.output( gp_model_ctd <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              gp_approx = gp_approx_ctd, ind_points_selection = "cover_tree",
                                              cover_tree_radius = 0.2, num_neighbors = 20,
                                              vecchia_ordering = "none", cov_fct_taper_range = 0.5,
                                              cov_fct_taper_shape = 2,
                                              matrix_inversion_method = "cholesky"), file = 'NUL')
      capture.output( nll_ctd <- gp_model_ctd$neg_log_likelihood(cov_pars = cov_pars_ctd, y = y_ctd), file = 'NUL')
      expect_true(is.finite(nll_ctd))
    }
    # a number of inducing points that does not work for the data is still rejected, also when the cover
    #   tree has selected it (here one inducing point per data point)
    expect_error( GPModel(gp_coords = coords, cov_function = "exponential",
                          gp_approx = "full_scale_tapering", ind_points_selection = "cover_tree",
                          cover_tree_radius = 1e-8, cov_fct_taper_range = 0.5, cov_fct_taper_shape = 2) )

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

  test_that("Redetermined inducing points of several clusters are those of their own cluster ", {

    # With an ARD covariance function the inducing points are redetermined during the estimation. The
    # kmeans++ algorithm is started from the inducing points of the last redetermination, which have to
    # be the ones of the same cluster: an empty cluster keeps its mean, so inducing points that lie
    # in the region of another cluster are never moved to the data and the approximation degenerates
    y_rd <- eps + xi
    cluster_ids_rd <- c(rep(1, n / 2), rep(2, n / 2))
    # the two clusters are in disjoint regions of the coordinate space
    coords_rd <- coords
    coords_rd[(n / 2 + 1):n, ] <- coords_rd[(n / 2 + 1):n, ] + 10
    capture.output( gp_model_rd <- fitGPModel(gp_coords = coords_rd, cov_function = "matern_ard",
                                              cov_fct_shape = 1.5, gp_approx = "fitc",
                                              num_ind_points = 20, ind_points_selection = "kmeans++",
                                              cluster_ids = cluster_ids_rd, y = y_rd,
                                              params = OPTIM_PARAMS_BFGS), file = 'NUL')
    marginal_var_rd <- as.numeric(gp_model_rd$get_cov_pars())[2]
    capture.output( pred_rd <- predict(gp_model_rd, gp_coords_pred = coords_rd,
                                       cluster_ids_pred = cluster_ids_rd, predict_var = TRUE,
                                       predict_response = FALSE), file = 'NUL')
    for (cluster_rd in c(1, 2)) {
      ind_rd <- which(cluster_ids_rd == cluster_rd)
      # the latent process is recovered in both clusters, it is not if the inducing points of a cluster
      #   lie in the region of the other one (the correlation is then close to 0 and the predictive
      #   variance close to the marginal variance)
      expect_gt(cor(pred_rd$mu[ind_rd], y_rd[ind_rd]), 0.8)
      expect_lt(mean(as.vector(pred_rd$var)[ind_rd]), 0.5 * marginal_var_rd)
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
