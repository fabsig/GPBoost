context("GPModel_weights")

# Cross-likelihood regression tests for the handling of sample 'weights'.
#
# The documented contract is that the conditional log-likelihood contribution of observation i is
# multiplied by weights[i]. For INTEGER weights this is equivalent to replicating observation i
# weights[i] times, which gives a golden-free invariant that can be checked for every likelihood:
#
#     evaluate / fit with weights w   ==   evaluate / fit on the data set with rows replicated by w
#
# The invariant is checked (i) on the approximate marginal log-likelihood at FIXED parameters, which
# is the sharper test since it depends on the mode and therefore on the weighting of the first
# derivative and of the information, and (ii) on a full estimation for a few representative families.
#
# Note: the auxiliary parameters are deliberately perturbed away from their defaults. Several of the
# per-observation constants have the form n * g(aux_pars) and g() vanishes at the default values
# (e.g. lgamma(1) = 0), so a check at the default auxiliary parameters silently passes even when the
# constant is scaled by the number of observations instead of by the sum of the weights.
#
# Excluded on purpose:
#   - "binomial_probit", "binomial_logit" and "beta_binomial", for which 'weights' holds the number of
#     trials n_i rather than a sample weight
#   - "gaussian", for which 'weights' divide the error variance ("nugget") instead of multiplying the
#     log-likelihood contribution. Both give the same posterior mean, but the density normalization
#     differs by the constant sum_{i} (0.5*log(w_i) + (w_i - 1)*0.5*log(2*pi*sigma2)), so the invariant
#     does not apply. The latent Gaussian likelihood ("gaussian_latent") does go through the
#     multiply-the-log-likelihood path and is covered below.
#   - a Vecchia approximation, and hence "gaussian_heteroscedastic_fixed_and_random" which requires
#     one: replicating a row duplicates a coordinate, which changes the Vecchia conditioning sets and
#     therefore the approximation itself. A control with "poisson" (whose weighting is verified exactly
#     by the grouped random effects and the exact GP cases below) shows the same effect, so this is a
#     property of the approximation and not of the weighting.

if (Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS") {

  TOLERANCE_WEIGHTS <- 1e-6
  # The estimation check is a secondary smoke test: the two data sets have a different number of
  # observations, so the optimizer follows a slightly different path and the estimates agree only to
  # about 1e-4 even when the weighting is exactly right. The log-likelihood checks are the sharp ones.
  TOLERANCE_WEIGHTS_ESTIM <- 2e-3

  sim_rand_unif <- function(n, init_c = 0.1) {
    mod_lcg <- 2^32
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for (i in 2:n) sim[i] <- (22695477 * sim[i - 1] + 1) %% mod_lcg
    return(sim / mod_lcg)
  }

  n <- 400L
  m <- 40L
  group <- rep(1:m, each = n / m)
  b <- qnorm(sim_rand_unif(m, init_c = 0.734)) * 0.5
  eta <- b[group]
  mu <- exp(eta)
  u1 <- sim_rand_unif(n, init_c = 0.213)
  u2 <- sim_rand_unif(n, init_c = 0.917)

  # integer weights and the equivalent replicated data set
  weights <- rep(1, n)
  weights[1:100] <- 2
  idx_rep <- c(seq_len(n), which(weights == 2))

  # the second location-parameter block of the two-block likelihoods (fixed effects only)
  fixed_effects_block_1 <- 0.3 + 0.4 * (u1 - 0.5)
  fixed_effects_block_2 <- -0.3 + 0.8 * (u2 - 0.5)

  y_positive <- qexp(u1) * mu                        # positive continuous
  y_zero_positive <- ifelse(u1 < 0.3, 0, qexp(u2) * mu)  # point mass at zero + positive
  y_zero_count <- ifelse(u1 < 0.3, 0, qpois(u2, lambda = mu))
  y_continuous <- eta + qnorm(u1) * 0.5

  sim_y <- function(lik) {
    if (lik %in% c("gpd", "egpd_power", "egpd_beta", "egpd_power_mixture", "egpd_power_beta")) {
      return(y_positive)
    }
    if (grepl("^hurdle_gpd|^hurdle_egpd|^hurdle_regression", lik)) return(y_zero_positive)
    if (grepl("^zero_inflated_regression", lik)) return(y_zero_count)
    if (lik == "gaussian_heteroscedastic") return(y_continuous)
    if (lik == "zero_censored_power_transformed_normal_heteroscedastic") return(pmax(0, y_continuous))
    switch(lik,
      bernoulli_probit = as.numeric(u1 < pnorm(eta)),
      bernoulli_logit  = as.numeric(u1 < 1 / (1 + exp(-eta))),
      poisson             = qpois(u1, lambda = mu),
      negative_binomial   = qnbinom(u1, size = 2, mu = mu),
      negative_binomial_1 = qnbinom(u1, size = 2, mu = mu),
      gamma            = qgamma(u1, shape = 2, rate = 2 / mu),
      lognormal        = qlnorm(u1, meanlog = eta - 0.5, sdlog = 1),
      gaussian_latent  = y_continuous,
      t                = eta + qt(u1, df = 4) * 0.3,
      asymmetric_laplace = y_continuous,
      beta             = qbeta(u1, shape1 = 2 * (1 / (1 + exp(-eta))), shape2 = 2 * (1 - 1 / (1 + exp(-eta)))),
      tweedie          = ifelse(u1 < 0.3, 0, qgamma(u2, shape = 2, rate = 2 / mu)),
      zero_inflated_poisson             = y_zero_count,
      zero_inflated_negative_binomial   = ifelse(u1 < 0.3, 0, qnbinom(u2, size = 2, mu = mu)),
      zero_inflated_negative_binomial_1 = ifelse(u1 < 0.3, 0, qnbinom(u2, size = 2, mu = mu)),
      hurdle_gamma     = ifelse(u1 < 0.3, 0, qgamma(u2, shape = 2, rate = 2 / mu)),
      hurdle_lognormal = ifelse(u1 < 0.3, 0, qlnorm(u2, meanlog = eta, sdlog = 1)),
      zero_censored_power_transformed_normal = pmax(0, y_continuous),
      zoctn            = pmin(pmax(y_continuous, 0), 1),
      # censored at both ends, so that the point masses at 0 and 1 are actually exercised
      zero_one_censored_transformed_beta = pmin(pmax(y_continuous, 0), 1),
      zero_one_censored_shifted_gamma    = pmin(pmax(y_continuous, 0), 1),
      stop("sim_y: unknown likelihood ", lik))
  }

  # one location-parameter block; 'weights' are ordinary sample weights
  likelihoods_single_block <- c(
    "bernoulli_probit", "bernoulli_logit", "poisson", "negative_binomial", "negative_binomial_1",
    "gamma", "lognormal", "gaussian_latent", "t", "beta", "tweedie", "asymmetric_laplace",
    "zero_inflated_poisson", "zero_inflated_negative_binomial", "zero_inflated_negative_binomial_1",
    "hurdle_gamma", "hurdle_lognormal",
    "gpd", "egpd_power", "egpd_beta", "egpd_power_mixture", "egpd_power_beta",
    "hurdle_gpd", "hurdle_egpd_power", "hurdle_egpd_beta", "hurdle_egpd_power_mixture",
    "hurdle_egpd_power_beta",
    "zero_censored_power_transformed_normal", "zoctn",
    "zero_one_censored_transformed_beta", "zero_one_censored_shifted_gamma")

  # two location-parameter blocks; the second one is a fixed-effects-only block
  likelihoods_two_blocks <- c(
    "gaussian_heteroscedastic", "zero_censored_power_transformed_normal_heteroscedastic",
    "hurdle_regression_gamma", "hurdle_regression_lognormal", "hurdle_regression_gpd",
    "hurdle_regression_egpd_power", "hurdle_regression_egpd_beta",
    "hurdle_regression_egpd_power_mixture", "hurdle_regression_egpd_power_beta",
    "zero_inflated_regression_poisson", "zero_inflated_regression_negative_binomial",
    "zero_inflated_regression_negative_binomial_1")

  make_model <- function(lik, group_data, w) {
    args <- list(group_data = group_data, likelihood = lik)
    if (!is.null(w)) args$weights <- w
    if (lik == "asymmetric_laplace") args$likelihood_additional_param <- 0.5
    do.call(GPModel, args)
  }

  # perturb the auxiliary parameters away from their defaults (see the comment at the top)
  perturbed_aux_pars <- function(model) {
    aux_pars <- model$get_aux_pars()
    if (length(aux_pars) == 0 || any(is.na(aux_pars))) return(NULL)
    aux_pars * c(1.7, 1.3, 1.1, 0.9, 1.15)[seq_along(aux_pars)]
  }

  expect_weights_equal_replication <- function(lik, two_blocks) {
    y <- sim_y(lik)
    capture.output(model_weights <- make_model(lik, group, weights), file = "NUL")
    capture.output(model_replicated <- make_model(lik, group[idx_rep], NULL), file = "NUL")
    aux_pars <- perturbed_aux_pars(model_weights)
    if (two_blocks) {
      fe_weights <- c(fixed_effects_block_1, fixed_effects_block_2)
      fe_replicated <- c(fixed_effects_block_1[idx_rep], fixed_effects_block_2[idx_rep])
    } else {
      fe_weights <- NULL
      fe_replicated <- NULL
    }
    nll_weights <- model_weights$neg_log_likelihood(cov_pars = 0.4, y = y, aux_pars = aux_pars,
                                                    fixed_effects = fe_weights)
    nll_replicated <- model_replicated$neg_log_likelihood(cov_pars = 0.4, y = y[idx_rep],
                                                          aux_pars = aux_pars,
                                                          fixed_effects = fe_replicated)
    expect_equal(nll_weights, nll_replicated, tolerance = TOLERANCE_WEIGHTS, info = lik)
  }

  test_that("integer 'weights' equal replication, one location parameter block", {
    for (lik in likelihoods_single_block) {
      expect_weights_equal_replication(lik, two_blocks = FALSE)
    }
  })

  test_that("integer 'weights' equal replication, two location parameter blocks", {
    for (lik in likelihoods_two_blocks) {
      expect_weights_equal_replication(lik, two_blocks = TRUE)
    }
  })

  test_that("integer 'weights' equal replication for a Gaussian process (no approximation)", {
    # the checks above all use grouped random effects; this covers a GP random-effect structure
    n_gp <- 100L
    coords <- cbind(sim_rand_unif(n_gp, init_c = 0.11), sim_rand_unif(n_gp, init_c = 0.37))
    weights_gp <- rep(1, n_gp)
    weights_gp[1:25] <- 2
    idx_gp <- c(seq_len(n_gp), which(weights_gp == 2))
    for (lik in c("poisson", "gamma", "t", "hurdle_gamma")) {
      y_gp <- switch(lik,
        poisson = qpois(sim_rand_unif(n_gp, init_c = 0.213), lambda = 1.5),
        gamma   = qgamma(sim_rand_unif(n_gp, init_c = 0.213), shape = 2, rate = 2),
        t       = qt(sim_rand_unif(n_gp, init_c = 0.213), df = 4) * 0.3,
        hurdle_gamma = ifelse(sim_rand_unif(n_gp, init_c = 0.213) < 0.3, 0,
                              qgamma(sim_rand_unif(n_gp, init_c = 0.917), shape = 2, rate = 2)))
      capture.output(model_weights <- GPModel(gp_coords = coords, cov_function = "exponential",
                                              likelihood = lik, weights = weights_gp), file = "NUL")
      capture.output(model_replicated <- GPModel(gp_coords = coords[idx_gp, ],
                                                 cov_function = "exponential", likelihood = lik), file = "NUL")
      aux_pars <- perturbed_aux_pars(model_weights)
      nll_weights <- model_weights$neg_log_likelihood(cov_pars = c(0.5, 0.2), y = y_gp, aux_pars = aux_pars)
      nll_replicated <- model_replicated$neg_log_likelihood(cov_pars = c(0.5, 0.2), y = y_gp[idx_gp],
                                                            aux_pars = aux_pars)
      expect_equal(nll_weights, nll_replicated, tolerance = TOLERANCE_WEIGHTS, info = lik)
    }
  })

  test_that("integer 'weights' equal replication for the estimated parameters", {
    # a representative subset: one of every auxiliary-parameter layout
    for (lik in c("poisson", "gamma", "hurdle_gamma", "t", "lognormal", "negative_binomial", "beta")) {
      y <- sim_y(lik)
      params <- list(trace = FALSE, maxit = 100)
      capture.output(fit_weights <- fitGPModel(group_data = group, y = y, likelihood = lik,
                                               weights = weights, params = params), file = "NUL")
      capture.output(fit_replicated <- fitGPModel(group_data = group[idx_rep], y = y[idx_rep],
                                                  likelihood = lik, params = params), file = "NUL")
      expect_equal(as.vector(fit_weights$get_cov_pars()), as.vector(fit_replicated$get_cov_pars()),
                   tolerance = TOLERANCE_WEIGHTS_ESTIM, info = lik)
      expect_equal(as.vector(fit_weights$get_aux_pars()), as.vector(fit_replicated$get_aux_pars()),
                   tolerance = TOLERANCE_WEIGHTS_ESTIM, info = lik)
    }
  })

  test_that("a Gaussian likelihood uses precision weights, not frequency weights", {
    # For "gaussian" the error variance of observation i is sigma2 / weights[i]. The quadratic term is
    # then identical to the one of the replicated data set, but the normalizing constant is not, so the
    # two differ by an exactly predictable constant. This pins the documented semantics down.
    y <- y_continuous
    sigma2 <- 0.5
    capture.output(model_weights <- GPModel(group_data = group, likelihood = "gaussian",
                                            weights = weights), file = "NUL")
    capture.output(model_replicated <- GPModel(group_data = group[idx_rep], likelihood = "gaussian"), file = "NUL")
    nll_weights <- model_weights$neg_log_likelihood(cov_pars = c(sigma2, 0.4), y = y)
    nll_replicated <- model_replicated$neg_log_likelihood(cov_pars = c(sigma2, 0.4), y = y[idx_rep])
    n_replicated <- sum(weights == 2)
    expected_difference <- n_replicated * (0.5 * log(2 * pi * sigma2) + 0.5 * log(2))
    expect_equal(nll_replicated - nll_weights, expected_difference, tolerance = TOLERANCE_WEIGHTS)
  })

  test_that("the 'test_neg_log_likelihood' metric works for 't' with both approximation types", {
    # 'CalcDiagInformationLogLikOneSample' (used only by the adaptive Gauss-Hermite quadrature behind
    # the 'test_neg_log_likelihood' metric) used to support 't' for 'fisher_laplace' only, so that the
    # metric aborted for the observed-Hessian Laplace approximation
    y <- sim_y("t")
    X <- matrix(sim_rand_unif(2 * n, init_c = 0.55), ncol = 2)
    dtrain <- gpb.Dataset(data = X, label = y)
    folds <- list(as.integer((n / 2 + 1):n))
    scores <- rep(NA_real_, 2)
    likelihoods_t <- c("t_laplace", "t_fisher_laplace")
    for (i in seq_along(likelihoods_t)) {
      gp_model <- GPModel(group_data = group, likelihood = likelihoods_t[i])
      capture.output(cvbst <- gpb.cv(params = list(objective = "regression", learning_rate = 0.1,
                                                   verbose = -1),
                                     data = dtrain, gp_model = gp_model, nrounds = 3,
                                     metric = "test_neg_log_likelihood",
                                     use_gp_model_for_validation = TRUE, folds = folds,
                                     verbose = -1), file = "NUL")
      scores[i] <- cvbst$best_score
    }
    expect_true(all(is.finite(scores)))
  })

}
