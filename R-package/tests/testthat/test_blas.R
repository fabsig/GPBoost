context("blas")

# Restores whether OpenBLAS is used as it was before a test, also if an expectation fails
restore_blas <- function(blas_before) {
  if (blas_before == "builtin") {
    gpb.set.blas("builtin")
  } else {
    gpb.set.blas("auto")
  }
}

test_that("the BLAS library can be queried and switched off", {

  blas_before <- gpb.get.blas()
  on.exit(restore_blas(blas_before), add = TRUE)
  expect_true(is.character(blas_before) && length(blas_before) == 1L)
  gpb.set.blas("builtin")
  expect_equal(gpb.get.blas(), "builtin")
  gpb.set.blas("auto")
  expect_match(gpb.get.blas(), "^(builtin$|OpenBLAS)")
  expect_error(gpb.set.blas("openblas"))
  expect_error(gpb.set.blas(1))

})

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

  # Function that simulates uniform random variables
  sim_rand_unif <- function(n, init_c=0.1){
    mod_lcg <- 2^32 # modulus for linear congruential generator (random0 used)
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for(i in 2:n) sim[i] <- (22695477 * sim[i-1] + 1) %% mod_lcg
    return(sim / mod_lcg)
  }

  # Exact Gaussian process: large enough that OpenBLAS uses several threads for the Cholesky factorization
  # and the inverse, see include/GPBoost/blas_backend.h
  n <- 900
  n_pred <- 50
  coords_all <- matrix(sim_rand_unif(n = 2 * (n + n_pred), init_c = 0.423), ncol = 2)
  coords <- coords_all[1:n, ]
  coords_pred <- coords_all[n + 1:n_pred, ]
  cov_pars <- c(0.3, 1, 0.1) # error variance, marginal variance, range
  Sigma_all <- cov_pars[2] * exp(-as.matrix(dist(coords_all)) / cov_pars[3])
  Sigma <- Sigma_all[1:n, 1:n] + diag(cov_pars[1], n)
  y <- as.vector(t(chol(Sigma)) %*% qnorm(sim_rand_unif(n = n, init_c = 0.817)))

  test_that("OpenBLAS and GPBoost's own code give the same results for an exact Gaussian process", {

    blas_before <- gpb.get.blas()
    on.exit(restore_blas(blas_before), add = TRUE)
    # Negative log-likelihood and predictive distribution calculated in R
    L <- t(chol(Sigma))
    L_inv_y <- forwardsolve(L, y)
    nll_exact <- sum(log(diag(L))) + 0.5 * sum(L_inv_y^2) + n / 2 * log(2 * pi)
    Sigma_cross <- Sigma_all[n + 1:n_pred, 1:n]
    L_inv_Sigma_cross_t <- forwardsolve(L, t(Sigma_cross))
    pred_mean_exact <- as.vector(t(L_inv_Sigma_cross_t) %*% L_inv_y)
    pred_cov_exact <- Sigma_all[n + 1:n_pred, n + 1:n_pred] + diag(cov_pars[1], n_pred) -
      crossprod(L_inv_Sigma_cross_t)
    results <- list()
    for (blas in c("auto", "builtin")) {
      gpb.set.blas(blas)
      gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
      nll <- gp_model$neg_log_likelihood(cov_pars = cov_pars, y = y)
      expect_lt(abs(nll - nll_exact) / abs(nll_exact), 1E-10)
      pred <- predict(gp_model, y = y, gp_coords_pred = coords_pred, cov_pars = cov_pars,
                      predict_cov_mat = TRUE, predict_response = TRUE)
      expect_lt(max(abs(pred$mu - pred_mean_exact)), 1E-10)
      expect_lt(max(abs(pred$cov - pred_cov_exact)), 1E-10)
      # The gradient and the standard errors use the inverse of the covariance matrix
      capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", y = y) ,
                      file = 'NUL')
      results[[blas]] <- gp_model$get_cov_pars(std_err = TRUE)
    }
    # The estimates differ at most by the accuracy of the optimization
    expect_lt(max(abs(results[["auto"]] - results[["builtin"]])), 1E-5)

  })

  test_that("OpenBLAS and GPBoost's own code give the same results for sparse Cholesky factorizations", {

    blas_before <- gpb.get.blas()
    on.exit(restore_blas(blas_before), add = TRUE)
    # Vecchia approximation with a Bernoulli likelihood
    n_vecchia <- 2000
    coords_vecchia <- matrix(sim_rand_unif(n = 2 * n_vecchia, init_c = 0.291), ncol = 2)
    y_vecchia <- as.numeric(sim_rand_unif(n = n_vecchia, init_c = 0.734) <
                              pnorm(sin(6 * coords_vecchia[, 1]) + cos(5 * coords_vecchia[, 2])))
    # Crossed grouped random effects with a Poisson likelihood
    n_crossed <- 3000
    group_data <- cbind(floor(100 * sim_rand_unif(n = n_crossed, init_c = 0.156)),
                        floor(100 * sim_rand_unif(n = n_crossed, init_c = 0.672)))
    y_crossed <- qpois(sim_rand_unif(n = n_crossed, init_c = 0.538),
                       lambda = exp(0.5 * sin(group_data[, 1]) + 0.5 * cos(group_data[, 2])))
    nll <- list()
    for (blas in c("auto", "builtin")) {
      gpb.set.blas(blas)
      gp_model <- GPModel(gp_coords = coords_vecchia, cov_function = "exponential", likelihood = "bernoulli_probit",
                          gp_approx = "vecchia", num_neighbors = 30, matrix_inversion_method = "cholesky")
      nll[[paste0("vecchia_", blas)]] <- gp_model$neg_log_likelihood(cov_pars = c(1, 0.1), y = y_vecchia)
      gp_model <- GPModel(group_data = group_data, likelihood = "poisson", matrix_inversion_method = "cholesky")
      nll[[paste0("crossed_", blas)]] <- gp_model$neg_log_likelihood(cov_pars = c(0.5, 0.5), y = y_crossed)
    }
    # The modes of the Laplace approximations differ at most by the accuracy of the mode finding, which
    # changes the approximate likelihood only to second order
    expect_lt(abs(nll[["vecchia_auto"]] - nll[["vecchia_builtin"]]) / abs(nll[["vecchia_builtin"]]), 1E-8)
    expect_lt(abs(nll[["crossed_auto"]] - nll[["crossed_builtin"]]) / abs(nll[["crossed_builtin"]]), 1E-8)

  })

}
