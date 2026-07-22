context("GPModel_zero_inflated_hurdle")

# Tests for the zero-inflated count (zero_inflated_poisson / _negative_binomial / _negative_binomial_1)
# and hurdle positive (hurdle_gamma / hurdle_lognormal) likelihoods with a constant structural-zero
# probability. Covered model cases: single-level grouped RE, crossed grouped REs (Cholesky and
# iterative), Vecchia GP (Cholesky and iterative), and combined grouped RE + GP (no approximation).
# Test tasks: likelihood evaluation at given parameters, estimation (with the negative log-likelihood
# at the estimated parameters), and prediction including predictive variances.
# Note: the zero-inflated count likelihoods are genuinely non-log-concave at zero counts; with complex
# random-effect structures (crossed / combined) or the iterative method the aggregated Laplace Hessian
# can be indefinite, so those cases are exercised with the robust positive hurdle families.

if (Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS") {

  TOL_STRICT <- 1e-5
  TOL_MED <- 1e-3
  TOL_LOOSE <- 1e-2
  TOL_ITER <- 1e-1

  sim_rand_unif <- function(n, init_c = 0.1) {
    mod_lcg <- 2^32
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for (i in 2:n) sim[i] <- (22695477 * sim[i - 1] + 1) %% mod_lcg
    return(sim / mod_lcg)
  }

  n <- 500L
  m <- 50L
  group <- rep(1:m, each = n / m)
  group_crossed <- cbind(group, rep(1:(n / m), times = m))
  b1 <- qnorm(sim_rand_unif(m, 0.15))
  b2 <- qnorm(sim_rand_unif(n / m, 0.37))
  d <- 2L
  coords <- matrix(sim_rand_unif(n * d, 0.63), ncol = d)
  Dm <- as.matrix(dist(coords))
  Sig <- exp(-Dm / 0.2) + diag(1e-8, n)
  gp_re <- as.vector(t(chol(Sig)) %*% qnorm(sim_rand_unif(n, 0.9)))
  u1 <- sim_rand_unif(n, 0.271)
  u2 <- sim_rand_unif(n, 0.55)
  X <- matrix(1.0, n, 1)
  it_par <- list(cg_delta_conv = 1e-6, num_rand_vec_trace = 200, seed_rand_vec_trace = 1, reuse_rand_vec_trace = TRUE)

  is_count <- function(fam) grepl("poisson|negative_binomial", fam)
  # Use the explicit ordinary-Laplace suffix for the historical golden values. Unsuffixed ZI-count likelihoods use
  # combined Fisher-Laplace, which has the same observed-Hessian objective but a different mode-finding curvature.
  lik <- function(fam) if (is_count(fam) && !grepl("_regression_", fam)) paste0(fam, "_laplace") else fam
  sim_y <- function(fam, eta) {
    mu <- exp(eta)
    p0 <- 0.3
    if (fam == "zero_inflated_poisson") ifelse(u1 < p0, 0L, qpois(u2, mu))
    else if (fam == "zero_inflated_negative_binomial") ifelse(u1 < p0, 0L, qnbinom(u2, size = 2, mu = mu))
    else if (fam == "zero_inflated_negative_binomial_1") ifelse(u1 < p0, 0L, qnbinom(u2, size = mu / 0.6, mu = mu))
    else if (fam == "hurdle_gamma") ifelse(u1 < p0, 0, qgamma(u2, shape = 2, rate = 2 / mu))
    else ifelse(u1 < p0, 0, exp(qnorm(u2, log(mu) - 0.25, sqrt(0.5))))
  }
  aux0 <- function(fam) switch(fam, zero_inflated_poisson = 0.3, zero_inflated_negative_binomial = c(2, 0.3),
    zero_inflated_negative_binomial_1 = c(0.6, 0.3), hurdle_gamma = c(2, 0.3), hurdle_lognormal = c(0.5, 0.3))
  eta_true <- function(fam, re) if (is_count(fam)) (-0.2 + re) else (0.5 + re)
  fams <- c("zero_inflated_poisson", "zero_inflated_negative_binomial", "zero_inflated_negative_binomial_1",
            "hurdle_gamma", "hurdle_lognormal")

  # Golden values (deterministic; generated with the installed package)
  GRP <- list(
    zero_inflated_poisson = list(eval = 512.05, est = c(0.273824, -0.387411, 0.273181, 504.945),
      mu = c(0.371826, 0.698981, 0.565764), var = c(0.455206, 0.956973, 0.824791)),
    zero_inflated_negative_binomial = list(eval = 515.749, est = c(0.288551, -0.686769, 0.821462, 0.0012576, 509.700),
      mu = c(0.443883, 0.740021, 0.580566), var = c(0.764349, 1.57599, 1.24213)),
    zero_inflated_negative_binomial_1 = list(eval = 510.889, est = c(0.257704, -0.671678, 0.88514, 0.0173243, 503.876),
      mu = c(0.390796, 0.767784, 0.571036), var = c(0.766432, 1.53756, 1.17978)),
    hurdle_gamma = list(eval = 783.714, est = c(0.298766, 0.348024, 2.16364, 0.276, 774.962),
      mu = c(0.656338, 1.38272, 1.19058), var = c(0.500369, 2.11542, 2.44205)),
    hurdle_lognormal = list(eval = 761.289, est = c(0.290816, 0.344675, 0.460094, 0.276, 751.84),
      mu = c(0.636183, 1.3212, 1.18189), var = c(0.536284, 2.24374, 2.69135)))

  test_that("grouped RE: likelihood evaluation, estimation, and prediction for all families", {
    for (fam in fams) {
      y <- sim_y(fam, eta_true(fam, 0.7 * b1[group]))
      gpm <- GPModel(group_data = group, likelihood = lik(fam))
      nll <- gpm$neg_log_likelihood(cov_pars = 0.5, y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      expect_equal(nll, GRP[[fam]]$eval, tolerance = TOL_STRICT)

      gp <- fitGPModel(group_data = group, likelihood = lik(fam), y = y, X = X, params = list(maxit = 100, trace = FALSE))
      est <- c(gp$get_cov_pars(), as.numeric(gp$get_coef()), gp$get_aux_pars(), gp$get_current_neg_log_likelihood())
      expect_equal(as.numeric(est), GRP[[fam]]$est, tolerance = TOL_MED)

      pd <- predict(gp, group_data_pred = c(1L, 2L, max(group) + 1L), X_pred = matrix(1.0, 3, 1),
                    predict_var = TRUE, predict_response = TRUE)
      expect_equal(as.numeric(pd$mu), GRP[[fam]]$mu, tolerance = TOL_MED)
      expect_equal(as.numeric(pd$var), GRP[[fam]]$var, tolerance = TOL_MED)
    }
  })

  test_that("grouped RE: hurdle GPD/EGPD families (evaluation, estimation, prediction)", {
    qgpd <- function(u, sigma, xi) sigma / xi * ((1 - u)^(-xi) - 1)
    qegpd_power <- function(u, sigma, xi, kappa) qgpd(u^(1 / kappa), sigma, xi)
    sigma <- exp(0.2 + 0.7 * b1[group])
    # hurdle_gpd (auxiliary: shape, p0)
    yg <- ifelse(u1 < 0.3, 0, qgpd(u2, sigma, 0.2))
    gpm <- GPModel(group_data = group, likelihood = "hurdle_gpd")
    expect_equal(gpm$neg_log_likelihood(cov_pars = 0.5, y = yg, fixed_effects = rep(0.0, n), aux_pars = c(0.2, 0.3)), 769.366, tolerance = TOL_STRICT)
    gp <- fitGPModel(group_data = group, likelihood = "hurdle_gpd", y = yg, X = X, params = list(maxit = 100, trace = FALSE))
    est <- c(gp$get_cov_pars(), as.numeric(gp$get_coef()), gp$get_aux_pars(), gp$get_current_neg_log_likelihood())
    expect_equal(as.numeric(est), c(0.313569, 0.0602354, 0.181543, 0.276, 767.427), tolerance = TOL_MED)
    pd <- predict(gp, group_data_pred = c(1L, 2L, max(group) + 1L), X_pred = matrix(1.0, 3, 1), predict_var = TRUE, predict_response = TRUE)
    expect_equal(as.numeric(pd$mu), c(0.713328, 1.2867, 1.09925), tolerance = TOL_MED)
    expect_equal(as.numeric(pd$var), c(1.57714, 4.85236, 4.66652), tolerance = TOL_MED)
    # hurdle_egpd_power (auxiliary: shape, kappa, p0)
    ye <- ifelse(u1 < 0.3, 0, qegpd_power(u2, sigma, 0.2, 1.5))
    gpm2 <- GPModel(group_data = group, likelihood = "hurdle_egpd_power")
    expect_equal(gpm2$neg_log_likelihood(cov_pars = 0.5, y = ye, fixed_effects = rep(0.0, n), aux_pars = c(0.2, 1.5, 0.3)), 867.777, tolerance = TOL_STRICT)
    gp2 <- fitGPModel(group_data = group, likelihood = "hurdle_egpd_power", y = ye, X = X, params = list(maxit = 100, trace = FALSE))
    est2 <- c(gp2$get_cov_pars(), as.numeric(gp2$get_coef()), gp2$get_aux_pars(), gp2$get_current_neg_log_likelihood())
    expect_equal(as.numeric(est2), c(0.30056, -0.103469, 0.239158, 1.74669, 0.276, 864.526), tolerance = TOL_MED)
    # undocumented alias zero_inflated_gpd -> hurdle_gpd
    gp3 <- fitGPModel(group_data = group, likelihood = "zero_inflated_gpd", y = yg, X = X, params = list(maxit = 5, trace = FALSE))
    expect_equal(gp3$get_likelihood_name(), "hurdle_gpd")
  })

  test_that("regression (fixed-effects) zero model: evaluation, estimation (eta + zeta blocks), prediction", {
    xc <- 2 * sim_rand_unif(n, 0.42) - 1
    Xr <- cbind(1, xc)
    eta_r <- 0.5 + 0.7 * b1[group] + 0.6 * xc
    zeta_r <- -0.3 + 1.2 * xc
    y <- ifelse(u1 < plogis(zeta_r), 0.0, exp(qnorm(u2, log(exp(eta_r)) - 0.25, sqrt(0.5))))
    gpm <- GPModel(group_data = group, likelihood = "hurdle_regression_lognormal")
    expect_equal(gpm$neg_log_likelihood(cov_pars = 0.5, y = y, fixed_effects = rep(0.0, 2 * n), aux_pars = 0.5), 724.385, tolerance = TOL_STRICT)
    gp <- fitGPModel(group_data = group, likelihood = "hurdle_regression_lognormal", y = y, X = Xr, params = list(maxit = 100, trace = FALSE))
    # coef returns both blocks: [eta_intercept, eta_x, zeta_intercept (_zero), zeta_x (_zero)]
    cf <- gp$get_coef()
    expect_equal(length(cf), 4L)
    expect_true(all(endsWith(names(cf)[3:4], "_zero")))
    est <- c(gp$get_cov_pars(), as.numeric(cf), gp$get_aux_pars(), gp$get_current_neg_log_likelihood())
    expect_equal(as.numeric(est), c(0.305373, 0.352269, 0.734696, -0.33679, 1.22878, 0.451265, 648.497), tolerance = TOL_MED)
    pd <- predict(gp, group_data_pred = c(1L, max(group) + 1L), X_pred = rbind(c(1, 0.4), c(1, -0.4)), predict_var = TRUE, predict_response = TRUE)
    expect_equal(as.numeric(pd$mu), c(0.518035, 0.85956), tolerance = TOL_MED)
    expect_equal(as.numeric(pd$var), c(0.710859, 1.52345), tolerance = TOL_MED)
  })

  test_that("count regression (fixed-effects zero model): evaluation, estimation, prediction (grouped RE)", {
    xc <- 2 * sim_rand_unif(n, 0.42) - 1
    Xr <- cbind(1, xc)
    eta_r <- -0.2 + 0.7 * b1[group] + 0.6 * xc
    p0r <- plogis(-0.3 + 1.2 * xc)
    mu_r <- exp(eta_r)
    simc <- function(fam) {
      if (fam == "zero_inflated_regression_poisson") ifelse(u1 < p0r, 0L, qpois(u2, mu_r))
      else if (fam == "zero_inflated_regression_negative_binomial") ifelse(u1 < p0r, 0L, qnbinom(u2, size = 2, mu = mu_r))
      else ifelse(u1 < p0r, 0L, qnbinom(u2, size = mu_r / 0.6, mu = mu_r))
    }
    auxc <- function(fam) switch(fam, zero_inflated_regression_poisson = NULL,
      zero_inflated_regression_negative_binomial = 2, zero_inflated_regression_negative_binomial_1 = 0.6)
    # Golden values (deterministic; generated with the installed package). The Poisson case is well-identified at this n;
    # the NB regression variants are only weakly identified here (zero-inflation vs. NB overdispersion) but remain reproducible.
    CREG <- list(
      zero_inflated_regression_poisson = list(eval = 454.8418, est = c(0.1851334, -0.3526239, 0.6497278, -0.5526028, 1.452601, 443.291)),
      zero_inflated_regression_negative_binomial = list(eval = 430.0845, est = c(0.1524396, -0.7631289, 0.3785567, -3.035804, 3.555543, 0.6356567, 422.1173)),
      zero_inflated_regression_negative_binomial_1 = list(eval = 431.3405, est = c(0.1296883, -0.6813157, 0.528945, -2.079165, 2.584585, 0.8593076, 423.4761)))
    for (fam in names(CREG)) {
      y <- simc(fam)
      gpm <- GPModel(group_data = group, likelihood = lik(fam))
      ap <- auxc(fam)
      nll <- if (is.null(ap)) gpm$neg_log_likelihood(cov_pars = 0.5, y = y, fixed_effects = rep(0.0, 2 * n))
             else gpm$neg_log_likelihood(cov_pars = 0.5, y = y, fixed_effects = rep(0.0, 2 * n), aux_pars = ap)
      expect_equal(nll, CREG[[fam]]$eval, tolerance = TOL_STRICT)
      gp <- fitGPModel(group_data = group, likelihood = lik(fam), y = y, X = Xr, params = list(maxit = 100, trace = FALSE))
      est <- c(gp$get_cov_pars(), as.numeric(gp$get_coef()), gp$get_aux_pars(), gp$get_current_neg_log_likelihood())
      expect_equal(as.numeric(est), CREG[[fam]]$est, tolerance = TOL_MED)
      if (fam == "zero_inflated_regression_poisson") {
        pd <- predict(gp, group_data_pred = c(1L, max(group) + 1L), X_pred = rbind(c(1, 0.4), c(1, -0.4)), predict_var = TRUE, predict_response = TRUE)
        expect_equal(as.numeric(pd$mu), c(0.3792956, 0.4497843), tolerance = TOL_MED)
        expect_equal(as.numeric(pd$var), c(0.565987, 0.5692858), tolerance = TOL_MED)
      }
    }
  })

  test_that("count regression: coupled zeta gradient on crossed REs and Vecchia (Cholesky)", {
    xc <- 2 * sim_rand_unif(n, 0.42) - 1
    Xr <- cbind(1, xc)
    p0r <- plogis(-0.3 + 1.2 * xc)
    # crossed grouped REs (Cholesky) -> exercises the GroupedRE-cholesky coupled grad_F
    muc <- exp(-0.2 + 0.7 * b1[group] + 0.5 * b2[group_crossed[, 2]] + 0.6 * xc)
    yc <- ifelse(u1 < p0r, 0L, qpois(u2, muc))
    gpc <- fitGPModel(group_data = group_crossed, likelihood = "zero_inflated_regression_poisson", y = yc, X = Xr,
                      matrix_inversion_method = "cholesky", params = list(maxit = 100, trace = FALSE))
    expect_equal(as.numeric(c(gpc$get_cov_pars(), as.numeric(gpc$get_coef()), gpc$get_current_neg_log_likelihood())),
                 c(0.1362728, 0.1014989, -0.4518659, 0.4179769, -0.426195, 0.8633492, 424.9478), tolerance = TOL_MED)
    # Vecchia GP (Cholesky) -> exercises the Vecchia coupled grad_F
    muv <- exp(-0.2 + gp_re + 0.6 * xc)
    yv <- ifelse(u1 < p0r, 0L, qpois(u2, muv))
    gpv <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "vecchia", num_neighbors = 15L,
                      vecchia_ordering = "none", likelihood = "zero_inflated_regression_poisson", y = yv, X = Xr,
                      matrix_inversion_method = "cholesky", params = list(maxit = 100, trace = FALSE))
    expect_equal(as.numeric(c(gpv$get_cov_pars(), as.numeric(gpv$get_coef()), gpv$get_current_neg_log_likelihood())),
                 c(0.549663, 0.150791, 0.031629, 0.74816, -0.277397, 1.42993, 580.322), tolerance = TOL_MED)
  })

  test_that("count regression: full fisher_laplace enables the iterative method (crossed REs and Vecchia)", {
    # Full fisher_laplace has nonnegative determinant W, so iterative matrix inversion is allowed.
    xc <- 2 * sim_rand_unif(n, 0.42) - 1
    Xr <- cbind(1, xc)
    p0r <- plogis(-0.3 + 1.2 * xc)
    # crossed grouped REs: the (fisher) likelihood evaluation matches Cholesky, and fitting -- which exercises the
    # coupled zeta-block fixed-effect gradient (log-det + implicit terms) on the iterative grouped-RE path -- runs and
    # returns finite estimates. (Full crossed-RE iterative estimates are only weakly identified at this n and are
    # sensitive to the stochastic trace, so they are not compared elementwise to Cholesky here.)
    muc <- exp(-0.2 + 0.7 * b1[group] + 0.5 * b2[group_crossed[, 2]] + 0.6 * xc)
    yc <- ifelse(u1 < p0r, 0L, qpois(u2, muc))
    gc <- GPModel(group_data = group_crossed, likelihood = "zero_inflated_regression_poisson_fisher_laplace", matrix_inversion_method = "cholesky")
    nll_c <- gc$neg_log_likelihood(cov_pars = c(0.5, 0.3), y = yc, fixed_effects = rep(0.0, 2 * n))
    gi <- GPModel(group_data = group_crossed, likelihood = "zero_inflated_regression_poisson_fisher_laplace", matrix_inversion_method = "iterative")
    gi$set_optim_params(params = it_par)
    nll_i <- gi$neg_log_likelihood(cov_pars = c(0.5, 0.3), y = yc, fixed_effects = rep(0.0, 2 * n))
    expect_equal(nll_i, nll_c, tolerance = TOL_ITER)
    gfit <- fitGPModel(group_data = group_crossed, likelihood = "zero_inflated_regression_poisson_fisher_laplace", y = yc, X = Xr,
                       matrix_inversion_method = "iterative", params = c(list(maxit = 20, trace = FALSE), it_par))
    expect_true(all(is.finite(c(gfit$get_cov_pars(), as.numeric(gfit$get_coef())))))
    # Vecchia GP (distinct coordinates -> the !use_random_effects_indices_of_data_ zeta-gradient path): iterative
    # fitting must run and reproduce the Cholesky estimates.
    muv <- exp(-0.2 + gp_re + 0.6 * xc)
    yv <- ifelse(u1 < p0r, 0L, qpois(u2, muv))
    fit_vecchia <- function(mim, extra = list()) fitGPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "vecchia",
      num_neighbors = 15L, vecchia_ordering = "none", likelihood = "zero_inflated_regression_poisson_fisher_laplace",
      y = yv, X = Xr, matrix_inversion_method = mim, params = c(list(maxit = 100, trace = FALSE), extra))
    est_of <- function(gp) c(gp$get_cov_pars(), as.numeric(gp$get_coef()))
    est_vc <- est_of(fit_vecchia("cholesky"))
    est_vi <- est_of(fit_vecchia("iterative", it_par))
    expect_equal(est_vi, est_vc, tolerance = TOL_ITER)
  })

  test_that("observed-Hessian zero-inflated counts allow iterative setup and reject full-scale Vecchia", {
    y <- sim_y("zero_inflated_poisson", eta_true("zero_inflated_poisson", 0.7 * b1[group]))
    hessian_fams <- c(
      "zero_inflated_poisson_laplace",
      "zero_inflated_negative_binomial_laplace",
      "zero_inflated_poisson", # combined defaults: (quasi-)Fisher mode, observed-Hessian determinant
      "zero_inflated_negative_binomial",
      "zero_inflated_negative_binomial_1",
      "zero_inflated_negative_binomial_1_laplace",
      "zero_inflated_regression_poisson_laplace",
      "zero_inflated_regression_negative_binomial_laplace",
      "zero_inflated_regression_negative_binomial_1",
      "zero_inflated_regression_negative_binomial_1_laplace"
    )
    for (fam in hessian_fams) {
      mod_iter <- GPModel(group_data = group_crossed, likelihood = fam,
                          matrix_inversion_method = "iterative")
      expect_true(inherits(mod_iter, "GPModel"), info = fam)
    }
    expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential", gp_approx = "full_scale_vecchia",
                            num_neighbors = 15L, num_ind_points = 30L, likelihood = "zero_inflated_regression_poisson",
                            y = y, X = X, params = list(maxit = 1, trace = FALSE)),
                 regexp = "full_scale_vecchia")
  })

  test_that("zero-inflated counts: combined is the default and full fisher_laplace remains distinct", {
    FISH <- list(
      zero_inflated_poisson = list(eval = 512.132, est = c(0.270959, -0.381861, 0.289847, 504.594)),
      zero_inflated_negative_binomial = list(eval = 516.2, est = c(0.289737, -0.716779, 0.826161, 0.00125802, 509.484)))
    for (fam in names(FISH)) {
      y <- sim_y(fam, eta_true(fam, 0.7 * b1[group]))
      # The explicit full-Fisher objective generally differs from both the default combined method and ordinary Laplace.
      gpf <- GPModel(group_data = group, likelihood = paste0(fam, "_fisher_laplace"))
      nll_f <- gpf$neg_log_likelihood(cov_pars = 0.5, y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      expect_equal(nll_f, FISH[[fam]]$eval, tolerance = TOL_STRICT)
      default <- GPModel(group_data = group, likelihood = fam)
      combined <- GPModel(group_data = group, likelihood = paste0(fam, "_fisher_laplace_combined"))
      ordinary <- GPModel(group_data = group, likelihood = paste0(fam, "_laplace"))
      nll_default <- default$neg_log_likelihood(cov_pars = 0.5, y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      nll_combined <- combined$neg_log_likelihood(cov_pars = 0.5, y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      nll_ordinary <- ordinary$neg_log_likelihood(cov_pars = 0.5, y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      expect_equal(nll_default, nll_combined, tolerance = TOL_STRICT)
      expect_equal(nll_default, nll_ordinary, tolerance = TOL_MED)
      expect_false(isTRUE(all.equal(nll_f, nll_default, tolerance = 1e-6)))
      gp <- fitGPModel(group_data = group, likelihood = paste0(fam, "_fisher_laplace"), y = y, X = X, params = list(maxit = 100, trace = FALSE))
      est <- c(gp$get_cov_pars(), as.numeric(gp$get_coef()), gp$get_aux_pars(), gp$get_current_neg_log_likelihood())
      expect_equal(as.numeric(est), FISH[[fam]]$est, tolerance = TOL_MED)
    }
    # The Fisher information is nonnegative, so iterative matrix inversion is allowed (and matches Cholesky) for crossed grouped REs.
    y <- sim_y("zero_inflated_poisson", eta_true("zero_inflated_poisson", 0.7 * b1[group] + 0.5 * b2[group_crossed[, 2]]))
    gc <- GPModel(group_data = group_crossed, likelihood = "zero_inflated_poisson_fisher_laplace", matrix_inversion_method = "cholesky")
    nll_c <- gc$neg_log_likelihood(cov_pars = c(0.5, 0.3), y = y, fixed_effects = rep(0.0, n), aux_pars = 0.3)
    gi <- GPModel(group_data = group_crossed, likelihood = "zero_inflated_poisson_fisher_laplace", matrix_inversion_method = "iterative")
    gi$set_optim_params(params = it_par)
    nll_i <- gi$neg_log_likelihood(cov_pars = c(0.5, 0.3), y = y, fixed_effects = rep(0.0, n), aux_pars = 0.3)
    expect_equal(nll_c, 492.0144, tolerance = TOL_MED)
    expect_equal(nll_i, nll_c, tolerance = TOL_ITER)
  })



  test_that("crossed grouped REs: likelihood evaluation (Cholesky and iterative)", {
    # positive hurdle families are robust for crossed REs (Cholesky is deterministic)
    for (fam in c("hurdle_gamma", "hurdle_lognormal")) {
      y <- sim_y(fam, eta_true(fam, 0.7 * b1[group] + 0.5 * b2[group_crossed[, 2]]))
      gpc <- GPModel(group_data = group_crossed, likelihood = lik(fam), matrix_inversion_method = "cholesky")
      nll_c <- gpc$neg_log_likelihood(cov_pars = c(0.5, 0.3), y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      gpi <- GPModel(group_data = group_crossed, likelihood = lik(fam), matrix_inversion_method = "iterative")
      gpi$set_optim_params(params = it_par)
      nll_i <- gpi$neg_log_likelihood(cov_pars = c(0.5, 0.3), y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      golden <- if (fam == "hurdle_gamma") 744.788 else 722.554
      expect_equal(nll_c, golden, tolerance = TOL_MED)
      expect_equal(nll_i, golden, tolerance = TOL_ITER)
    }
  })

  test_that("crossed grouped REs: estimation and prediction (hurdle_lognormal)", {
    y <- sim_y("hurdle_lognormal", 0.5 + 0.7 * b1[group] + 0.5 * b2[group_crossed[, 2]])
    gp <- fitGPModel(group_data = group_crossed, likelihood = "hurdle_lognormal", y = y, X = X,
                     matrix_inversion_method = "cholesky", params = list(maxit = 100, trace = FALSE))
    est <- c(gp$get_cov_pars(), as.numeric(gp$get_coef()), gp$get_aux_pars(), gp$get_current_neg_log_likelihood())
    expect_equal(as.numeric(est), c(0.283842, 0.120793, 0.225625, 0.458416, 0.276, 716.915), tolerance = TOL_LOOSE)
  })

  test_that("Vecchia GP: likelihood evaluation (Cholesky, all families; iterative, positive families)", {
    vecchia_chol_golden <- c(zero_inflated_poisson = 699.975, zero_inflated_negative_binomial = 698.102,
      zero_inflated_negative_binomial_1 = 695.676, hurdle_gamma = 993.845, hurdle_lognormal = 975.656)
    for (fam in fams) {
      y <- sim_y(fam, eta_true(fam, gp_re))
      gpc <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = lik(fam), gp_approx = "vecchia",
                     num_neighbors = 15L, vecchia_ordering = "none", matrix_inversion_method = "cholesky")
      nll_c <- gpc$neg_log_likelihood(cov_pars = c(1.0, 0.2), y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      expect_equal(nll_c, as.numeric(vecchia_chol_golden[fam]), tolerance = TOL_STRICT)
    }
    for (fam in c("hurdle_gamma", "hurdle_lognormal")) {
      y <- sim_y(fam, eta_true(fam, gp_re))
      gpv <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = lik(fam), gp_approx = "vecchia",
                     num_neighbors = 15L, vecchia_ordering = "none", matrix_inversion_method = "iterative")
      gpv$set_optim_params(params = it_par)
      nll_i <- gpv$neg_log_likelihood(cov_pars = c(1.0, 0.2), y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      golden <- if (fam == "hurdle_gamma") 993.845 else 975.656
      expect_equal(nll_i, golden, tolerance = TOL_ITER)
    }
  })

  test_that("Vecchia GP: estimation and prediction with predictive variances (hurdle_lognormal)", {
    y <- sim_y("hurdle_lognormal", 0.5 + gp_re)
    gp <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "hurdle_lognormal", y = y, X = X,
                     gp_approx = "vecchia", num_neighbors = 15L, vecchia_ordering = "none",
                     matrix_inversion_method = "cholesky", params = list(maxit = 100, trace = FALSE))
    est <- c(gp$get_cov_pars(), as.numeric(gp$get_coef()), gp$get_aux_pars(), gp$get_current_neg_log_likelihood())
    expect_equal(as.numeric(est), c(0.793172, 0.142811, 0.573852, 0.489955, 0.276, 973.542), tolerance = TOL_LOOSE)
    pd <- predict(gp, gp_coords_pred = coords[1:3, , drop = FALSE], X_pred = matrix(1.0, 3, 1),
                  predict_var = TRUE, predict_response = TRUE)
    expect_equal(as.numeric(pd$mu), c(6.13081, 0.785188, 0.673447), tolerance = TOL_LOOSE)
    expect_equal(as.numeric(pd$var), c(66.751, 1.27586, 1.04426), tolerance = TOL_LOOSE)
  })

  test_that("combined grouped RE + GP (no approximation): likelihood evaluation", {
    for (fam in c("hurdle_gamma", "hurdle_lognormal")) {
      y <- sim_y(fam, eta_true(fam, 0.7 * b1[group] + gp_re))
      gpm <- GPModel(group_data = group, gp_coords = coords, cov_function = "exponential", likelihood = lik(fam),
                     matrix_inversion_method = "cholesky")
      nll <- gpm$neg_log_likelihood(cov_pars = c(0.5, 1.0, 0.2), y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      golden <- if (fam == "hurdle_gamma") 972.082 else 954.85
      expect_equal(nll, golden, tolerance = TOL_MED)
    }
  })

  test_that("Vecchia GP with multiple observations at the same location (Cholesky; iterative matches Cholesky)", {
    # Many observations share the same coordinates -> the use_random_effects_indices_of_data_ Vecchia path. The zero-inflated
    # count families use their default Fisher approximation here (plain names, no '_laplace'); the Fisher information is
    # nonnegative, so ZIP/ZINB additionally support the iterative method (which must reproduce the Cholesky estimates). ZINB1
    # ('combined' observed-Hessian determinant) is Cholesky-only; the hurdle families use the observed-Hessian Laplace.
    nu <- 100L
    coords_u <- matrix(sim_rand_unif(nu * 2, 0.53), ncol = 2)
    rep_idx <- rep(seq_len(nu), length.out = n)
    coords_rep <- coords_u[rep_idx, ]
    Du <- as.matrix(dist(coords_u))
    gp_re_u <- as.vector(t(chol(exp(-Du / 0.2) + diag(1e-8, nu))) %*% qnorm(sim_rand_unif(nu, 0.77)))
    gp_rep <- gp_re_u[rep_idx]
    REP <- list(
      zero_inflated_poisson = list(eval = 466.3513, est = c(0.7965833, 0.2478529, -0.6343344, 0.3410409, 463.8714),
        mu = c(1.474985, 0.3807137, 0.1189141), var = c(2.953984, 0.5049349, 0.1340482)),
      zero_inflated_negative_binomial = list(eval = 466.691, est = c(0.8613269, 0.2635387, -0.5679296, 7.076212, 0.4218029, 462.9031),
        mu = c(1.518201, 0.3812376, 0.1173187), var = c(4.291644, 0.5924335, 0.1413001)),
      zero_inflated_negative_binomial_1 = list(eval = 458.1655, est = c(0.8506113, 0.2441034, -0.5778237, 0.1627298, 0.4387794, 454.0112),
        mu = c(1.389454, 0.4017661, 0.1148457), var = c(3.509263, 0.664118, 0.1535877)),
      hurdle_gamma = list(eval = 672.4665, est = c(0.9458235, 0.2479103, -0.04470852, 2.02878, 0.276, 671.2249),
        mu = c(3.286106, 0.6348532, 0.08400707), var = c(13.72367, 0.517598, 0.00889254)),
      hurdle_lognormal = list(eval = 651.0987, est = c(0.9099315, 0.2513436, -0.01860506, 0.5041683, 0.276, 649.7051),
        mu = c(3.484947, 0.6949459, 0.09514768), var = c(18.64546, 0.7344804, 0.01343696)))
    iter_fams <- c("zero_inflated_poisson", "zero_inflated_negative_binomial", "hurdle_gamma", "hurdle_lognormal")
    for (fam in fams) {
      y <- sim_y(fam, eta_true(fam, gp_rep))
      # likelihood evaluation at given parameters (Cholesky)
      gpm <- GPModel(gp_coords = coords_rep, cov_function = "exponential", gp_approx = "vecchia", num_neighbors = 15L,
                     vecchia_ordering = "none", likelihood = fam, matrix_inversion_method = "cholesky")
      nll <- gpm$neg_log_likelihood(cov_pars = c(1.0, 0.2), y = y, fixed_effects = rep(0.0, n), aux_pars = aux0(fam))
      expect_equal(nll, REP[[fam]]$eval, tolerance = TOL_STRICT)
      # estimation (+ nll at the estimated parameters) and prediction with predictive variances (Cholesky)
      gp <- fitGPModel(gp_coords = coords_rep, cov_function = "exponential", gp_approx = "vecchia", num_neighbors = 15L,
                       vecchia_ordering = "none", likelihood = fam, y = y, X = X, matrix_inversion_method = "cholesky",
                       params = list(maxit = 100, trace = FALSE))
      est <- c(gp$get_cov_pars(), as.numeric(gp$get_coef()), gp$get_aux_pars(), gp$get_current_neg_log_likelihood())
      expect_equal(as.numeric(est), REP[[fam]]$est, tolerance = TOL_MED)
      pd <- predict(gp, gp_coords_pred = coords_u[1:3, ], X_pred = matrix(1.0, 3, 1), predict_var = TRUE, predict_response = TRUE)
      expect_equal(as.numeric(pd$mu), REP[[fam]]$mu, tolerance = TOL_MED)
      expect_equal(as.numeric(pd$var), REP[[fam]]$var, tolerance = TOL_MED)
      # iterative must reproduce the Cholesky estimates where supported (nonnegative Fisher information). The zero-inflated
      # count families default to 'combined' (indefinite observed-Hessian determinant -> no iterative), so their iterative
      # check uses the explicit full Fisher-Laplace variant (append '_fisher_laplace') and compares against a Fisher-Laplace
      # Cholesky reference; the hurdle families use their default observed-Hessian Laplace (positive W, iterative supported).
      if (fam %in% iter_fams) {
        lik_it <- if (is_count(fam)) paste0(fam, "_fisher_laplace") else fam
        gp_ref <- if (is_count(fam)) fitGPModel(gp_coords = coords_rep, cov_function = "exponential", gp_approx = "vecchia",
                                                num_neighbors = 15L, vecchia_ordering = "none", likelihood = lik_it, y = y, X = X,
                                                matrix_inversion_method = "cholesky", params = list(maxit = 100, trace = FALSE)) else gp
        gpi <- fitGPModel(gp_coords = coords_rep, cov_function = "exponential", gp_approx = "vecchia", num_neighbors = 15L,
                          vecchia_ordering = "none", likelihood = lik_it, y = y, X = X, matrix_inversion_method = "iterative",
                          params = c(list(maxit = 100, trace = FALSE), it_par))
        expect_equal(as.numeric(c(gpi$get_cov_pars(), gpi$get_coef(), gpi$get_aux_pars())),
                     as.numeric(c(gp_ref$get_cov_pars(), gp_ref$get_coef(), gp_ref$get_aux_pars())), tolerance = TOL_ITER)
      }
    }
  })

  test_that("undocumented aliases resolve to the canonical likelihood", {
    y <- sim_y("zero_inflated_poisson", eta_true("zero_inflated_poisson", 0.7 * b1[group]))
    gp1 <- fitGPModel(group_data = group, likelihood = "hurdle_poisson", y = y, X = X, params = list(maxit = 5, trace = FALSE))
    expect_equal(gp1$get_likelihood_name(), "zero_inflated_poisson")
    yln <- sim_y("hurdle_lognormal", eta_true("hurdle_lognormal", 0.7 * b1[group]))
    gp2 <- fitGPModel(group_data = group, likelihood = "zero_inflated_lognormal", y = yln, X = X, params = list(maxit = 5, trace = FALSE))
    expect_equal(gp2$get_likelihood_name(), "hurdle_lognormal")
  })

  test_that("conditional densities agree with independent base-R formulas", {
    # A nearly-zero independent random-effect variance makes the Laplace marginal likelihood converge to
    # the conditional likelihood at the supplied fixed effects. This checks the density independently of
    # the fitting and prediction code used to generate the golden values above.
    tiny_var <- 1e-10
    eta <- c(-1, -0.3, 0.2, 0.7, 1)
    mu <- exp(eta)
    yc <- c(0L, 1L, 3L, 0L, 2L)
    p0 <- 0.27
    count_cases <- list(
      zero_inflated_poisson_laplace = list(aux = p0, base = dpois(yc, mu)),
      zero_inflated_negative_binomial_laplace = list(aux = c(1.7, p0), base = dnbinom(yc, size = 1.7, mu = mu)),
      zero_inflated_negative_binomial_1_laplace = list(aux = c(0.6, p0), base = dnbinom(yc, size = mu / 0.6, mu = mu))
    )
    for (fam in names(count_cases)) {
      z <- count_cases[[fam]]
      expected <- -sum(log(ifelse(yc == 0L, p0 + (1 - p0) * z$base, (1 - p0) * z$base)))
      mod <- GPModel(group_data = seq_along(yc), likelihood = fam)
      got <- mod$neg_log_likelihood(cov_pars = tiny_var, y = yc, fixed_effects = eta, aux_pars = z$aux)
      expect_equal(got, expected, tolerance = 1e-8, info = fam)
    }

    zeta <- c(-2, -1, 0, 1, 2)
    pi_i <- plogis(zeta)
    count_reg_cases <- list(
      zero_inflated_regression_poisson_laplace = list(aux = numeric(0), base = dpois(yc, mu)),
      zero_inflated_regression_negative_binomial_laplace = list(
        aux = 1.7, base = dnbinom(yc, size = 1.7, mu = mu)),
      zero_inflated_regression_negative_binomial_1_laplace = list(
        aux = 0.6, base = dnbinom(yc, size = mu / 0.6, mu = mu))
    )
    for (fam in names(count_reg_cases)) {
      z <- count_reg_cases[[fam]]
      expected <- -sum(log(ifelse(yc == 0L, pi_i + (1 - pi_i) * z$base,
                                  (1 - pi_i) * z$base)))
      mod <- GPModel(group_data = seq_along(yc), likelihood = fam)
      got <- mod$neg_log_likelihood(cov_pars = tiny_var, y = yc,
                                    fixed_effects = c(eta, zeta), aux_pars = z$aux)
      expect_equal(got, expected, tolerance = 1e-8, info = fam)
    }

    yh <- c(0, 0.4, 1.2, 0, 2.1)
    hurdle_cases <- list(
      hurdle_gamma = list(aux = c(1.7, p0), regression = FALSE,
                          base = dgamma(yh, shape = 1.7, rate = 1.7 / mu)),
      hurdle_lognormal = list(aux = c(0.6, p0), regression = FALSE,
                              base = dlnorm(yh, meanlog = eta - 0.3, sdlog = sqrt(0.6))),
      hurdle_regression_gamma = list(aux = 1.7, regression = TRUE,
                                     base = dgamma(yh, shape = 1.7, rate = 1.7 / mu)),
      hurdle_regression_lognormal = list(aux = 0.6, regression = TRUE,
                                         base = dlnorm(yh, meanlog = eta - 0.3, sdlog = sqrt(0.6)))
    )
    for (fam in names(hurdle_cases)) {
      z <- hurdle_cases[[fam]]
      pi <- if (z$regression) pi_i else rep(p0, length(yh))
      expected <- -sum(log(ifelse(yh == 0, pi, (1 - pi) * z$base)))
      mod <- GPModel(group_data = seq_along(yh), likelihood = fam)
      fe <- if (z$regression) c(eta, zeta) else eta
      got <- mod$neg_log_likelihood(cov_pars = tiny_var, y = yh, fixed_effects = fe, aux_pars = z$aux)
      expect_equal(got, expected, tolerance = 1e-8, info = fam)
    }
  })

  test_that("explicit Fisher-Laplace and combined aliases are accepted", {
    y <- c(0L, 1L, 0L, 2L, 3L, 0L)
    eta <- rep(0, length(y))
    cases <- list(
      negative_binomial_1_fisher_laplace = 0.6,
      zero_inflated_poisson_fisher_laplace = 0.3,
      zero_inflated_negative_binomial_fisher_laplace = c(1.5, 0.3),
      zero_inflated_negative_binomial_1_fisher_laplace = c(0.6, 0.3)
    )
    for (fam in names(cases)) {
      mod <- GPModel(group_data = seq_along(y), likelihood = fam)
      expect_true(is.finite(mod$neg_log_likelihood(cov_pars = 0.2, y = y,
                                                   fixed_effects = eta, aux_pars = cases[[fam]])), info = fam)
    }
    for (base in c("zero_inflated_poisson", "zero_inflated_negative_binomial", "zero_inflated_negative_binomial_1")) {
      aux <- cases[[paste0(base, "_fisher_laplace")]]
      default <- GPModel(group_data = seq_along(y), likelihood = base)
      combined <- GPModel(group_data = seq_along(y), likelihood = paste0(base, "_fisher_laplace_combined"))
      args <- list(cov_pars = 0.2, y = y, fixed_effects = eta, aux_pars = aux)
      expect_equal(do.call(default$neg_log_likelihood, args), do.call(combined$neg_log_likelihood, args),
                   tolerance = TOL_STRICT, info = base)
    }
    for (fam in c("zero_inflated_regression_poisson_fisher_laplace",
                  "zero_inflated_regression_negative_binomial_fisher_laplace",
                  "zero_inflated_regression_negative_binomial_1_fisher_laplace")) {
      mod_reg <- GPModel(group_data = seq_along(y), likelihood = fam)
      aux <- if (grepl("negative_binomial", fam)) 0.6 else NULL
      expect_true(is.finite(mod_reg$neg_log_likelihood(cov_pars = 0.2, y = y,
                                                       fixed_effects = rep(0, 2 * length(y)),
                                                       aux_pars = aux)), info = fam)
    }
    for (base in c("zero_inflated_regression_poisson", "zero_inflated_regression_negative_binomial",
                   "zero_inflated_regression_negative_binomial_1")) {
      aux <- if (grepl("negative_binomial", base)) 0.6 else NULL
      default <- GPModel(group_data = seq_along(y), likelihood = base)
      combined <- GPModel(group_data = seq_along(y), likelihood = paste0(base, "_fisher_laplace_combined"))
      args <- list(cov_pars = 0.2, y = y, fixed_effects = rep(0, 2 * length(y)), aux_pars = aux)
      expect_equal(do.call(default$neg_log_likelihood, args), do.call(combined$neg_log_likelihood, args),
                   tolerance = TOL_STRICT, info = base)
    }
    fisher_zi_fams <- c(
      "zero_inflated_poisson_fisher_laplace",
      "zero_inflated_negative_binomial_fisher_laplace",
      "zero_inflated_negative_binomial_1_fisher_laplace",
      "zero_inflated_regression_poisson_fisher_laplace",
      "zero_inflated_regression_negative_binomial_fisher_laplace",
      "zero_inflated_regression_negative_binomial_1_fisher_laplace"
    )
    for (fam in fisher_zi_fams) {
      mod_iter <- GPModel(group_data = group_crossed, likelihood = fam,
                          matrix_inversion_method = "iterative")
      expect_true(inherits(mod_iter, "GPModel"), info = fam)
    }
  })

  test_that("combined determinant attempts the mode and recommends full Fisher-Laplace when iterative W is incompatible", {
    # At large eta the ZIP observed information is negative at the zero observations. Very small random-effect variances
    # keep the complete Sigma^-1 + Z^T W Z matrix positive definite, so direct Cholesky remains valid. (A single positive
    # observation is included because an all-zero response is separately rejected up front for zero-inflated counts.)
    y0 <- c(rep(0L, 5), 1L)
    groups0 <- cbind(1:6, rep(1:2, 3))
    eta0 <- rep(log(10), 6)
    direct <- GPModel(group_data = groups0, likelihood = "zero_inflated_poisson",
                      matrix_inversion_method = "cholesky")
    expect_true(is.finite(direct$neg_log_likelihood(cov_pars = c(1e-3, 1e-3), y = y0,
                                                    fixed_effects = eta0, aux_pars = 0.5)))
    cases <- list(
      zero_inflated_poisson = list(aux = 0.5, fixed_effects = eta0),
      zero_inflated_negative_binomial = list(aux = c(1.5, 0.5), fixed_effects = eta0),
      zero_inflated_negative_binomial_1 = list(aux = c(0.6, 0.5), fixed_effects = eta0),
      zero_inflated_regression_poisson = list(aux = NULL, fixed_effects = c(eta0, rep(0, length(y0)))),
      zero_inflated_regression_negative_binomial = list(aux = 1.5, fixed_effects = c(eta0, rep(0, length(y0)))),
      zero_inflated_regression_negative_binomial_1 = list(aux = 0.6, fixed_effects = c(eta0, rep(0, length(y0))))
    )
    for (fam in names(cases)) {
      iterative <- GPModel(group_data = groups0, likelihood = fam,
                           matrix_inversion_method = "iterative")
      args <- list(cov_pars = c(1e-3, 1e-3), y = y0, fixed_effects = cases[[fam]]$fixed_effects)
      if (!is.null(cases[[fam]]$aux)) args$aux_pars <- cases[[fam]]$aux
      expect_error(do.call(iterative$neg_log_likelihood, args),
                   regexp = paste0("FindModePostRandEffCalcMLLGroupedRE: Negative values found in W.*",
                                   fam, "_fisher_laplace"), info = fam)
    }
  })

  test_that("hurdle inputs are finite and no-zero samples have an interior start", {
    mod <- GPModel(group_data = 1:6, likelihood = "hurdle_lognormal")
    expect_error(mod$neg_log_likelihood(cov_pars = 0.2, y = c(0, Inf, 1, 2, 3, 4),
                                        fixed_effects = rep(0, 6), aux_pars = c(0.5, 0.3)), "finite")
    expect_error(mod$neg_log_likelihood(cov_pars = 0.2, y = c(0, NaN, 1, 2, 3, 4),
                                        fixed_effects = rep(0, 6), aux_pars = c(0.5, 0.3)), "finite")
    set.seed(123)
    for (fam in c("hurdle_gamma", "hurdle_lognormal")) {
      y <- if (fam == "hurdle_gamma") rgamma(40, shape = 2) else rlnorm(40)
      # The all-positive samples intentionally have no zeros, so the hurdle model emits an (expected) "No 0's in the
      # response variable" warning; capture it so it does not clutter the test output.
      capture.output(fit <- fitGPModel(group_data = 1:40, likelihood = fam, y = y, X = matrix(1, 40, 1),
                                       params = list(maxit = 5, trace = FALSE)), file = "NUL")
      expect_true(all(is.finite(c(fit$get_cov_pars(), fit$get_coef(), fit$get_aux_pars()))), info = fam)
      expect_gt(tail(fit$get_aux_pars(), 1), 0)
    }
    capture.output(fit_constant <- fitGPModel(group_data = 1:40, likelihood = "hurdle_gamma", y = rep(1, 40),
                               X = matrix(1, 40, 1),
                               params = list(estimate_aux_pars = FALSE, maxit = 1, trace = FALSE)), file = "NUL")
    expect_true(all(is.finite(c(fit_constant$get_cov_pars(), fit_constant$get_coef(),
                                fit_constant$get_aux_pars()))))
  })

  test_that("FITC includes the coupled zero-regression gradient terms", {
    # With almost as many inducing points as observations, FITC is close to the exact GP.
    # Their fitted zero-model coefficients must therefore agree. A direct-score-only FITC
    # implementation can still optimize and pass self-golden tests, but fails this comparison.
    # Both fits use the full Fisher-Laplace variant: the default 'combined' approximation uses the
    # observed Hessian, which is indefinite for this dense zero-inflated count GP (mode finding via
    # Rasmussen-Williams then fails), so the comparison is done with the (nonnegative) Fisher information.
    set.seed(19)
    nf <- 150L
    coords_f <- matrix(runif(2 * nf), nf, 2)
    xf <- runif(nf, -1, 1)
    Xf <- cbind(1, xf)
    eta_f <- -0.3 + 0.4 * xf
    zeta_f <- -1 + 0.5 * xf
    yf <- ifelse(runif(nf) < plogis(zeta_f), 0L, rpois(nf, exp(eta_f)))
    pars <- list(init_cov_pars = c(0.4, 0.25), estimate_cov_par_index = c(0L, 0L),
                 init_coef = c(0, 0, -0.5, 0), init_coef_aux_pars_from_iid_model = FALSE,
                 maxit = 500, delta_rel_conv = 1e-9, trace = FALSE)
    exact <- fitGPModel(gp_coords = coords_f, cov_function = "exponential",
                        likelihood = "zero_inflated_regression_poisson_fisher_laplace", y = yf, X = Xf, params = pars)
    fitc <- fitGPModel(gp_coords = coords_f, cov_function = "exponential", gp_approx = "fitc",
                       num_ind_points = 120L, likelihood = "zero_inflated_regression_poisson_fisher_laplace",
                       y = yf, X = Xf, params = pars)
    expect_equal(as.numeric(fitc$get_coef())[3:4], as.numeric(exact$get_coef())[3:4], tolerance = 0.01)
    expect_equal(fitc$get_current_neg_log_likelihood(), exact$get_current_neg_log_likelihood(), tolerance = 0.05)
  })

}
