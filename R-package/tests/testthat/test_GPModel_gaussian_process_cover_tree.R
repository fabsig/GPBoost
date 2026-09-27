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

}
