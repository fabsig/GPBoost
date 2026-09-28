context("Joint and varying-dispersion Tweedie likelihoods")

# Avoid being tested on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

  TOLERANCE_LOOSE <- 1E-2
  TOLERANCE_MEDIUM <- 1e-3
  TOLERANCE_STRICT_LOWER <- 1E-5
  TOLERANCE_STRICT <- 1E-6
  # See helper-tolerances.R, which defines this and reports it once per test run
  USE_STRICT_TOLERANCES <- gpb_use_strict_tolerances()
  TOLERANCE_NON_CONVEX <- if (USE_STRICT_TOLERANCES) TOLERANCE_MEDIUM else 0.5
  relax_tolerance_strict <- function(tol) if (USE_STRICT_TOLERANCES) tol else 100 * tol
  OPTIM_PARAMS <- list(optimizer_cov = "lbfgs", optimizer_coef = "lbfgs", maxit = 300, init_coef_aux_pars_from_iid_model = FALSE)

  # Function that simulates uniform random variables (as in test_GPModel_tweedie.R)
  sim_rand_unif <- function(n, init_c=0.1){
    mod_lcg <- 2^32
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for(i in 2:n) sim[i] <- (22695477 * sim[i-1] + 1) %% mod_lcg
    sim / mod_lcg
  }
  # Compound Poisson--Gamma draws via the inverse cdfs: the number of events N and the aggregate response y
  sim_tweedie_yn <- function(mu, phi, p, init_count, init_gamma){
    phi <- rep_len(phi, length(mu))
    counts <- qpois(sim_rand_unif(length(mu), init_count), lambda = mu^(2-p) / (phi * (2-p)))
    y <- numeric(length(mu))
    ind <- counts > 0
    y[ind] <- qgamma(sim_rand_unif(sum(ind), init_gamma), shape = counts[ind] * (2-p) / (p-1), scale = phi[ind] * (p-1) * mu[ind]^(p-1))
    list(y = y, n = counts)
  }
  # Reference implementations, independent of the GPBoost C++ code (eta = log(mu), zeta = log(phi)):
  # the joint log-density of (y, N) and the marginal log-density of y as the sum of the joint density over N
  ll_joint <- function(y, n, eta, zeta, p) {
    len <- max(length(y), length(n)); y <- rep_len(y, len); n <- rep_len(n, len)
    mu <- rep_len(exp(eta), len); phi <- rep_len(exp(zeta), len)
    out <- dpois(n, mu^(2 - p) / (phi * (2 - p)), log = TRUE)
    pos <- n > 0
    out[pos] <- out[pos] + dgamma(y[pos], shape = n[pos] * (2 - p) / (p - 1), scale = phi[pos] * (p - 1) * mu[pos]^(p - 1), log = TRUE)
    out
  }
  ll_marg <- function(y, eta, zeta, p, nmax = 300) {
    eta <- rep_len(eta, length(y)); zeta <- rep_len(zeta, length(y))
    sapply(seq_along(y), function(i) {
      if (y[i] == 0) return(ll_joint(0, 0, eta[i], zeta[i], p))
      lj <- ll_joint(rep(y[i], nmax), 1:nmax, eta[i], zeta[i], p)
      m <- max(lj)
      m + log(sum(exp(lj - m)))
    })
  }

  # Data for the models with a grouped random effect
  n <- 200
  m <- 20
  group <- rep(1:m, each = n / m)
  X <- cbind(1, sim_rand_unif(n, 0.52))
  b_gr <- 0.5 * qnorm(sim_rand_unif(m, 0.33))
  eta_gr <- as.vector(X %*% c(0.3, 0.8)) + b_gr[group]
  zeta_gr <- as.vector(X %*% c(-0.3, 0.9))
  sim_gr <- sim_tweedie_yn(exp(eta_gr), exp(zeta_gr), 1.5, 0.61, 0.27)
  y_gr <- sim_gr$y
  N_gr <- sim_gr$n

  test_that("joint and varying-dispersion Tweedie likelihoods: density and derivatives against independent formulas ", {

    n_d <- 200
    X_d <- cbind(1, sim_rand_unif(n = n_d, init_c = 0.311))
    eta_d <- as.vector(X_d %*% c(0.2, 0.9))
    zeta_d <- as.vector(X_d %*% c(-0.4, 0.8))
    p_d <- 1.45
    sim_d <- sim_tweedie_yn(exp(eta_d), exp(zeta_d), p_d, 0.428, 0.173)
    y_d <- sim_d$y
    N_d <- sim_d$n
    expect_equal(sum(y_d == 0), 19L)
    # An iid model (no random effects) has no Laplace approximation: its negative log-likelihood is the sum of the
    # per-observation log-densities (the 'cov_pars' argument is required by the interface but not used)
    gp_iid_m <- GPModel(num_data = n_d, likelihood = "tweedie")
    gp_iid_j <- GPModel(num_data = n_d, likelihood = "tweedie_joint", additional_likelihood_data = N_d)
    gp_iid_vd <- GPModel(num_data = n_d, likelihood = "tweedie_varying_dispersion")
    gp_iid_jvd <- GPModel(num_data = n_d, likelihood = "tweedie_joint_varying_dispersion", additional_likelihood_data = N_d)
    gp_iid_jvd_fixed <- GPModel(num_data = n_d, likelihood = "tweedie_joint_varying_dispersion_fixed_p", additional_likelihood_data = N_d,
                                likelihood_additional_param = p_d)
    expect_equal(as.vector(gp_iid_jvd_fixed$get_aux_pars()), p_d)

    ###################
    ## 1) The C++ log-likelihoods against the reference densities, at parameters that are not the maximizer
    ###################
    phi_c <- 0.9
    nll_m <- gp_iid_m$neg_log_likelihood(cov_pars = 1, y = y_d, fixed_effects = eta_d, aux_pars = c(phi_c, p_d))
    expect_lt(abs(nll_m + sum(ll_marg(y_d, eta_d, log(phi_c), p_d))), relax_tolerance_strict(TOLERANCE_STRICT))
    nll_j <- gp_iid_j$neg_log_likelihood(cov_pars = 1, y = y_d, fixed_effects = eta_d, aux_pars = c(phi_c, p_d))
    expect_lt(abs(nll_j + sum(ll_joint(y_d, N_d, eta_d, log(phi_c), p_d))), relax_tolerance_strict(TOLERANCE_STRICT))
    nll_vd <- gp_iid_vd$neg_log_likelihood(cov_pars = 1, y = y_d, fixed_effects = c(eta_d, zeta_d), aux_pars = p_d)
    expect_lt(abs(nll_vd + sum(ll_marg(y_d, eta_d, zeta_d, p_d))), relax_tolerance_strict(TOLERANCE_STRICT))
    nll_jvd <- gp_iid_jvd$neg_log_likelihood(cov_pars = 1, y = y_d, fixed_effects = c(eta_d, zeta_d), aux_pars = p_d)
    expect_lt(abs(nll_jvd + sum(ll_joint(y_d, N_d, eta_d, zeta_d, p_d))), relax_tolerance_strict(TOLERANCE_STRICT))
    nll_jvd_fixed <- gp_iid_jvd_fixed$neg_log_likelihood(cov_pars = 1, y = y_d, fixed_effects = c(eta_d, zeta_d), aux_pars = p_d)
    expect_lt(abs(nll_jvd_fixed - nll_jvd), relax_tolerance_strict(TOLERANCE_STRICT))
    expect_lt(abs(nll_jvd - 585.15804891), TOLERANCE_MEDIUM)

    ###################
    ## 2) The analytical per-observation derivatives against central finite differences of the reference densities.
    ## With A = mu^(2-p) / phi, B = y * mu^(1-p) / phi, and lambda = A / (2-p), these are, for both the marginal and the
    ## joint density: l_eta = B - A, J = -l_etaeta = (2-p) A + (p-1) B, l_eta_zeta = -l_eta, and dJ/dzeta = -J. For the
    ## joint density: l_zeta = lambda + (B - N) / (p-1) and the derivative wrt p, and for both the derivatives of l_eta
    ## and J wrt p, which enter the Laplace approximation (y = 0 is included)
    ###################
    fd1 <- function(f, x, h = 1e-5) (f(x + h) - f(x - h)) / (2 * h)
    fd2 <- function(f, x, h = 1e-3) (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    fd_dxdy <- function(f, x, y, h = 1e-3) (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h)
    fd_dx2dy <- function(f, x, y, h = 3e-3) (f(x + h, y + h) - 2 * f(x, y + h) + f(x - h, y + h) -
                                               f(x + h, y - h) + 2 * f(x, y - h) - f(x - h, y - h)) / (2 * h^3)
    rel_err <- function(analytical, numerical) abs(analytical - numerical) / max(abs(analytical), 1)
    err <- c(l_eta = 0, J = 0, l_eta_zeta = 0, dJ_dzeta = 0, l_zeta = 0, l_p = 0, dl_eta_dp = 0, dJ_dp = 0)
    for (y in c(0, 0.2, 1.3, 4)) for (N in if (y == 0) 0 else c(1, 2, 5)) for (eta in c(-0.8, 0.3, 1.2)) for (zeta in c(-0.7, 0.2)) for (p in c(1.2, 1.5, 1.8)) {
      A <- exp((2 - p) * eta - zeta)
      B <- y * exp((1 - p) * eta - zeta)
      lambda <- A / (2 - p)
      s <- B - A
      J <- (2 - p) * A + (p - 1) * B
      l_p <- N / (2 - p) + lambda * (eta - 1 / (2 - p)) + (eta / (p - 1) + 1 / (p - 1)^2) * B +
        (if (N > 0) N / (p - 1)^2 * (zeta - log(y) + log(p - 1) - (2 - p) + digamma(N * (2 - p) / (p - 1))) else 0)
      for (joint in c(TRUE, FALSE)) {
        f <- if (joint) function(e, z) ll_joint(y, N, e, z, p) else function(e, z) ll_marg(y, e, z, p)
        err["l_eta"] <- max(err["l_eta"], rel_err(s, fd1(function(e) f(e, zeta), eta)))
        err["J"] <- max(err["J"], rel_err(J, -fd2(function(e) f(e, zeta), eta)))
        err["l_eta_zeta"] <- max(err["l_eta_zeta"], rel_err(-s, fd_dxdy(f, eta, zeta)))
        err["dJ_dzeta"] <- max(err["dJ_dzeta"], rel_err(-J, -fd_dx2dy(f, eta, zeta)))
      }
      err["l_zeta"] <- max(err["l_zeta"], rel_err(lambda + (B - N) / (p - 1), fd1(function(z) ll_joint(y, N, eta, z, p), zeta)))
      err["l_p"] <- max(err["l_p"], rel_err(l_p, fd1(function(pp) ll_joint(y, N, eta, zeta, pp), p)))
      err["dl_eta_dp"] <- max(err["dl_eta_dp"], rel_err(-eta * s, fd1(function(pp) y * exp((1 - pp) * eta - zeta) - exp((2 - pp) * eta - zeta), p)))
      err["dJ_dp"] <- max(err["dJ_dp"], rel_err(s - eta * J, fd1(function(pp) (2 - pp) * exp((2 - pp) * eta - zeta) + (pp - 1) * y * exp((1 - pp) * eta - zeta), p)))
    }
    expect_lt(max(err[c("l_eta", "l_zeta", "l_p", "dl_eta_dp", "dJ_dp")]), 1e-6)
    expect_lt(max(err[c("J", "l_eta_zeta")]), 1e-5)
    expect_lt(err["dJ_dzeta"], 1e-4)

    ###################
    ## 3) The C++ scores against the analytical formulas, via finite differences of the iid negative log-likelihood wrt the
    ## regression coefficients of both blocks. The gradient wrt the coefficients of a block is -X^T (score of that block)
    ###################
    fd_coef_grad <- function(model, coefs, num_blocks, aux_pars) {
      eval_nll <- function(cf) {
        fixed_effects <- as.vector(sapply(1:num_blocks, function(k) X_d %*% cf[(k - 1) * 2 + 1:2]))
        model$neg_log_likelihood(cov_pars = 1, y = y_d, fixed_effects = fixed_effects, aux_pars = aux_pars)
      }
      h <- 1e-5
      sapply(seq_along(coefs), function(j) {
        cp <- cm <- coefs; cp[j] <- cp[j] + h; cm[j] <- cm[j] - h
        (eval_nll(cp) - eval_nll(cm)) / (2 * h)
      })
    }
    A_d <- exp((2 - p_d) * eta_d - zeta_d)
    B_d <- y_d * exp((1 - p_d) * eta_d - zeta_d)
    score_eta <- B_d - A_d
    score_zeta_joint <- A_d / (2 - p_d) + (B_d - N_d) / (p_d - 1)
    # For the marginal density, the zeta score contains the derivative of the series normalizer, which is obtained from the reference
    score_zeta_marg <- sapply(seq_len(n_d), function(i) fd1(function(z) ll_marg(y_d[i], eta_d[i], z, p_d), zeta_d[i]))
    coefs_d <- c(0.2, 0.9, -0.4, 0.8)
    grad_fd_jvd <- fd_coef_grad(gp_iid_jvd, coefs_d, 2, p_d)
    grad_an_jvd <- c(-as.vector(t(X_d) %*% score_eta), -as.vector(t(X_d) %*% score_zeta_joint))
    expect_lt(max(abs(grad_fd_jvd - grad_an_jvd)) / max(abs(grad_an_jvd)), TOLERANCE_STRICT_LOWER)
    grad_fd_vd <- fd_coef_grad(gp_iid_vd, coefs_d, 2, p_d)
    grad_an_vd <- c(-as.vector(t(X_d) %*% score_eta), -as.vector(t(X_d) %*% score_zeta_marg))
    expect_lt(max(abs(grad_fd_vd - grad_an_vd)) / max(abs(grad_an_vd)), TOLERANCE_STRICT_LOWER)
    # Given (phi, p), the marginal and the joint likelihood have the same score wrt the mean predictor
    expect_lt(max(abs(grad_fd_jvd[1:2] - grad_fd_vd[1:2])), TOLERANCE_STRICT_LOWER)
    grad_fd_m <- fd_coef_grad(gp_iid_m, coefs_d[1:2], 1, c(phi_c, p_d))
    grad_fd_j <- fd_coef_grad(gp_iid_j, coefs_d[1:2], 1, c(phi_c, p_d))
    expect_lt(max(abs(grad_fd_m - grad_fd_j)), TOLERANCE_STRICT_LOWER)

    ###################
    ## 4) Maximum likelihood estimation with the C++ gradients (including those wrt the power p and the dispersion) against
    ## the maximizer of the reference likelihood (obtained with R's optim() on the reference densities above)
    ###################
    capture.output(fit_jvd <- fitGPModel(likelihood = "tweedie_joint_varying_dispersion", additional_likelihood_data = N_d, y = y_d, X = X_d,
                                         params = c(OPTIM_PARAMS, list(delta_rel_conv = 1e-12))), file = "NUL")
    expect_lt(sum(abs(c(fit_jvd$get_coef(), fit_jvd$get_aux_pars()) - c(0.1266292, 0.7769147, -0.3459874, 0.6537557, 1.4449531))), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(fit_jvd$get_current_neg_log_likelihood() - 581.6610326), TOLERANCE_STRICT_LOWER)
    expect_equal(colnames(fit_jvd$get_coef(std_err = TRUE)), c("Covariate_1", "Covariate_2", "Covariate_1_dispersion", "Covariate_2_dispersion"))
    capture.output(fit_j <- fitGPModel(likelihood = "tweedie_joint", additional_likelihood_data = N_d, y = y_d, X = X_d,
                                       params = c(OPTIM_PARAMS, list(delta_rel_conv = 1e-12))), file = "NUL")
    expect_lt(sum(abs(c(fit_j$get_coef(), log(fit_j$get_aux_pars()[1]), fit_j$get_aux_pars()[2]) - c(0.1598896, 0.7110888, 0.0475792, 1.4913891))), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(fit_j$get_current_neg_log_likelihood() - 599.1493188), TOLERANCE_STRICT_LOWER)
    capture.output(fit_vd <- fitGPModel(likelihood = "tweedie_varying_dispersion", y = y_d, X = X_d,
                                        params = c(OPTIM_PARAMS, list(delta_rel_conv = 1e-12))), file = "NUL")
    expect_lt(sum(abs(c(fit_vd$get_coef(), fit_vd$get_aux_pars()) - c(0.1170347, 0.7972553, -0.3248616, 0.8131398, 1.4878823))), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(fit_vd$get_current_neg_log_likelihood() - 348.2735926), TOLERANCE_STRICT_LOWER)
  })

  test_that("joint and varying-dispersion Tweedie likelihoods: Laplace approximation and equivalences ", {

    ###################
    ## With an intercept-only design matrix, the log-dispersion predictor is constant and the varying-dispersion models
    ## are exactly the constant-dispersion models (the log-dispersion intercept equals log(phi))
    ###################
    X_int <- X[, 1, drop = FALSE]
    capture.output(fit_c <- fitGPModel(group_data = group, likelihood = "tweedie", y = y_gr, X = X_int, params = OPTIM_PARAMS), file = "NUL")
    capture.output(fit_v <- fitGPModel(group_data = group, likelihood = "tweedie_varying_dispersion", y = y_gr, X = X_int, params = OPTIM_PARAMS), file = "NUL")
    expect_lt(abs(fit_c$get_current_neg_log_likelihood() - 391.81228920), TOLERANCE_MEDIUM)
    expect_lt(abs(fit_v$get_current_neg_log_likelihood() - fit_c$get_current_neg_log_likelihood()), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(as.vector(fit_v$get_coef())[1] - as.vector(fit_c$get_coef())), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(as.vector(fit_v$get_coef())[2] - log(fit_c$get_aux_pars()[1])), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(fit_v$get_aux_pars() - fit_c$get_aux_pars()[2]), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(fit_v$get_cov_pars() - fit_c$get_cov_pars()), TOLERANCE_STRICT_LOWER)
    capture.output(fit_jc <- fitGPModel(group_data = group, likelihood = "tweedie_joint", additional_likelihood_data = N_gr, y = y_gr, X = X_int,
                                        params = OPTIM_PARAMS), file = "NUL")
    capture.output(fit_jv <- fitGPModel(group_data = group, likelihood = "tweedie_joint_varying_dispersion", additional_likelihood_data = N_gr,
                                        y = y_gr, X = X_int, params = OPTIM_PARAMS), file = "NUL")
    expect_lt(abs(fit_jc$get_current_neg_log_likelihood() - 651.08652494), TOLERANCE_MEDIUM)
    expect_lt(abs(fit_jv$get_current_neg_log_likelihood() - fit_jc$get_current_neg_log_likelihood()), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(as.vector(fit_jv$get_coef())[2] - log(fit_jc$get_aux_pars()[1])), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(fit_jv$get_aux_pars() - fit_jc$get_aux_pars()[2]), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(fit_jv$get_cov_pars() - fit_jc$get_cov_pars()), TOLERANCE_STRICT_LOWER)

    ###################
    ## Given (phi, p), the joint and the marginal likelihood have the same score and curvature wrt the mean predictor, so
    ## their Laplace approximations differ by the normalizers only, independently of the covariance parameters
    ###################
    nll_diff <- function(cov_pars) {
      GPModel(group_data = group, likelihood = "tweedie_joint", additional_likelihood_data = N_gr)$neg_log_likelihood(
        cov_pars = cov_pars, y = y_gr, fixed_effects = eta_gr, aux_pars = c(0.8, 1.4)) -
        GPModel(group_data = group, likelihood = "tweedie")$neg_log_likelihood(cov_pars = cov_pars, y = y_gr, fixed_effects = eta_gr, aux_pars = c(0.8, 1.4))
    }
    diff_ref <- -sum(ll_joint(y_gr, N_gr, eta_gr, log(0.8), 1.4) - ll_marg(y_gr, eta_gr, log(0.8), 1.4))
    expect_lt(abs(nll_diff(0.3) - diff_ref), TOLERANCE_STRICT_LOWER)
    expect_lt(abs(nll_diff(1.7) - diff_ref), TOLERANCE_STRICT_LOWER)

    ###################
    ## Full Laplace gradients: the finite-difference gradient of the approximate marginal likelihood (a new model for every
    ## evaluation) vanishes at the optimum found with the analytical gradients wrt the covariance parameter, the coefficients
    ## of both blocks, and the power, and the gradient-free Nelder-Mead optimizer finds the same optimum
    ###################
    for (lik in c("tweedie_joint_varying_dispersion", "tweedie_varying_dispersion")) {
      ald <- if (lik == "tweedie_joint_varying_dispersion") N_gr else NULL
      params_tight <- c(OPTIM_PARAMS, list(maxit = 2000, delta_rel_conv = 1e-13))
      capture.output(fit <- fitGPModel(group_data = group, likelihood = lik, additional_likelihood_data = ald, y = y_gr, X = X, params = params_tight), file = "NUL")
      nll_at <- function(th) {
        GPModel(group_data = group, likelihood = lik, additional_likelihood_data = ald)$neg_log_likelihood(
          cov_pars = exp(th[1]), y = y_gr, fixed_effects = c(X %*% th[2:3], X %*% th[4:5]), aux_pars = 1.01 + 0.98 * plogis(th[6]))
      }
      th_opt <- c(log(fit$get_cov_pars()), fit$get_coef(), qlogis((fit$get_aux_pars() - 1.01) / 0.98))
      expect_lt(abs(nll_at(th_opt) - fit$get_current_neg_log_likelihood()), TOLERANCE_STRICT_LOWER)
      grad_fd <- sapply(seq_along(th_opt), function(j) {
        tp <- tm <- th_opt; tp[j] <- tp[j] + 1e-5; tm[j] <- tm[j] - 1e-5
        (nll_at(tp) - nll_at(tm)) / 2e-5
      })
      expect_lt(max(abs(grad_fd)), 1e-3)
      capture.output(fit_nm <- fitGPModel(group_data = group, likelihood = lik, additional_likelihood_data = ald, y = y_gr, X = X,
                                          params = list(optimizer_cov = "nelder_mead", optimizer_coef = "nelder_mead", maxit = 20000,
                                                        delta_rel_conv = 1e-12, init_coef_aux_pars_from_iid_model = FALSE)), file = "NUL")
      expect_lt(abs(fit_nm$get_current_neg_log_likelihood() - fit$get_current_neg_log_likelihood()), TOLERANCE_STRICT_LOWER)
      expect_lt(max(abs(c(fit_nm$get_cov_pars(), fit_nm$get_coef(), fit_nm$get_aux_pars()) - c(fit$get_cov_pars(), fit$get_coef(), fit$get_aux_pars()))), TOLERANCE_MEDIUM)
    }
  })

  test_that("joint and varying-dispersion Tweedie likelihoods with a grouped random effect: estimation and prediction ", {

    expected <- list(
      tweedie_joint = list(coef = c(0.43707010, 0.44734734), se = c(0.17184369, 0.26274926), aux = c(1.26042049, 1.57823667), cov = 0.14017306,
                           nll = 649.67420586, mu = c(2.03090121, 2.18605387, 1.96217712), var = c(4.19841752, 4.71255557, 4.47251138)),
      tweedie_varying_dispersion = list(coef = c(0.45298523, 0.42844479, -0.21426196, 0.90178436), se = c(0.15570694, 0.26113846, 0.19173968, 0.33890967),
                                        aux = 1.55459989, cov = 0.11882683, nll = 386.88253670,
                                        mu = c(2.12371120, 2.10920503, 1.95862838), var = c(4.47722668, 4.70594229, 3.86606256)),
      tweedie_joint_varying_dispersion = list(coef = c(0.44716239, 0.43411258, -0.08799397, 0.58187316), se = c(0.15948176, 0.25680348, 0.06734313, 0.12416569),
                                              aux = 1.55649181, cov = 0.12779029, nll = 638.98482095,
                                              mu = c(2.08949060, 2.13266904, 1.96014386), var = c(4.21978536, 4.52979667, 3.95176944)),
      tweedie_joint_varying_dispersion_fixed_p = list(coef = c(0.44485515, 0.42850600, -0.18751939, 0.61772781), se = c(0.15440690, 0.24240949, 0.06369068, 0.11744282),
                                                      aux = 1.5, cov = 0.13822941, nll = 642.08160922,
                                                      mu = c(2.08973818, 2.12916715, 1.96175300), var = c(3.73631205, 4.00502902, 3.59134461)))
    for (lik in names(expected)) {
      ald <- if (grepl("joint", lik)) N_gr else NULL
      lap <- if (grepl("fixed_p", lik)) 1.5 else NULL
      capture.output(fit <- fitGPModel(group_data = group, likelihood = lik, additional_likelihood_data = ald, likelihood_additional_param = lap,
                                       y = y_gr, X = X, params = OPTIM_PARAMS), file = "NUL")
      ref <- expected[[lik]]
      expect_lt(sum(abs(as.vector(fit$get_coef()) - ref$coef)), TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(fit$get_coef(std_err = TRUE)[2, ]) - ref$se)), TOLERANCE_MEDIUM)
      expect_lt(sum(abs(as.vector(fit$get_aux_pars()) - ref$aux)), TOLERANCE_MEDIUM)
      expect_lt(abs(as.vector(fit$get_cov_pars()) - ref$cov), TOLERANCE_MEDIUM)
      expect_lt(abs(fit$get_current_neg_log_likelihood() - ref$nll), TOLERANCE_MEDIUM)
      pred <- predict(fit, group_data_pred = c(1, 5, 25), X_pred = X[c(1, 60, 120), ], predict_var = TRUE, predict_response = TRUE)
      expect_lt(sum(abs(pred$mu - ref$mu)), TOLERANCE_MEDIUM)
      expect_lt(sum(abs(pred$var - ref$var)), TOLERANCE_MEDIUM)
    }
    # Predicting a varying-dispersion model requires covariate data for both linear predictors
    expect_error(predict(fit, group_data_pred = c(1, 5, 25), predict_var = TRUE, predict_response = TRUE))

    ###################
    ## Save and load
    ###################
    capture.output(fit <- fitGPModel(group_data = group, likelihood = "tweedie_joint_varying_dispersion", additional_likelihood_data = N_gr,
                                     y = y_gr, X = X, params = OPTIM_PARAMS), file = "NUL")
    pred <- predict(fit, group_data_pred = c(1, 5, 25), X_pred = X[1:3, ], predict_var = TRUE, predict_response = TRUE)
    filename <- tempfile(fileext = ".json")
    saveGPModel(fit, filename = filename)
    fit_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(fit_loaded, group_data_pred = c(1, 5, 25), X_pred = X[1:3, ], predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu - pred_loaded$mu)) + sum(abs(pred$var - pred_loaded$var)), TOLERANCE_STRICT)
    expect_lt(abs(fit_loaded$get_current_neg_log_likelihood() - fit$get_current_neg_log_likelihood()), TOLERANCE_STRICT)
    expect_equal(fit_loaded$get_additional_likelihood_data(), matrix(as.numeric(N_gr), ncol = 1))
  })

  test_that("joint and varying-dispersion Tweedie likelihoods for Gaussian processes, approximations, and crossed random effects ", {

    n_gp <- 150
    X_gp <- cbind(1, sim_rand_unif(n_gp, 0.193))
    coords <- matrix(sim_rand_unif(2 * n_gp, 0.749), ncol = 2)
    Sigma <- 0.5 * exp(-as.matrix(dist(coords)) / 0.15) + diag(1e-10, n_gp)
    b_gp <- as.vector(t(chol(Sigma)) %*% qnorm(sim_rand_unif(n_gp, 0.836)))
    eta_gp <- as.vector(X_gp %*% c(0.3, 0.8)) + b_gp
    zeta_gp <- as.vector(X_gp %*% c(-0.3, 0.9))
    sim_gp <- sim_tweedie_yn(exp(eta_gp), exp(zeta_gp), 1.5, 0.44, 0.91)
    y_gp <- sim_gp$y
    N_gp <- sim_gp$n
    lik <- "tweedie_joint_varying_dispersion"
    params_iter <- c(OPTIM_PARAMS, list(num_rand_vec_trace = 200, seed_rand_vec_trace = 1, cg_preconditioner_type = "vadu"))

    # Dense GP ("Stable")
    capture.output(fit_dense <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = lik, additional_likelihood_data = N_gp,
                                           y = y_gp, X = X_gp, params = OPTIM_PARAMS), file = "NUL")
    expected_dense <- c(0.63245040, 0.73194310, -0.26531262, 0.83845113, 1.50578503, 0.18419567, 0.10348776)
    expect_lt(sum(abs(c(fit_dense$get_coef(), fit_dense$get_aux_pars(), fit_dense$get_cov_pars()) - expected_dense)), TOLERANCE_MEDIUM)
    expect_lt(abs(fit_dense$get_current_neg_log_likelihood() - 528.45142619), TOLERANCE_MEDIUM)
    pred <- predict(fit_dense, gp_coords_pred = coords[1:3, ] + 1e-3, X_pred = X_gp[1:3, ], predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu - c(2.95638950, 1.97500314, 2.23651177))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var - c(5.82048915, 2.85942004, 4.35733154))), TOLERANCE_MEDIUM)
    # Vecchia approximation with num_neighbors = n - 1 is exact and must match the dense GP, also for a random ordering,
    # which permutes the data internally (the number of events has to be permuted in the same way as the response)
    for (ordering in c("none", "random")) {
      capture.output(fit_vecchia <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = lik, additional_likelihood_data = N_gp,
                                               gp_approx = "vecchia", num_neighbors = n_gp - 1, vecchia_ordering = ordering,
                                               matrix_inversion_method = "cholesky", y = y_gp, X = X_gp, params = OPTIM_PARAMS), file = "NUL")
      expect_lt(sum(abs(c(fit_vecchia$get_coef(), fit_vecchia$get_aux_pars(), fit_vecchia$get_cov_pars()) - expected_dense)), TOLERANCE_MEDIUM)
      expect_lt(abs(fit_vecchia$get_current_neg_log_likelihood() - 528.45142619), TOLERANCE_MEDIUM)
    }
    # Vecchia approximation, Cholesky and iterative
    capture.output(fit_vecchia <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = lik, additional_likelihood_data = N_gp,
                                             gp_approx = "vecchia", num_neighbors = 20, vecchia_ordering = "none", matrix_inversion_method = "cholesky",
                                             y = y_gp, X = X_gp, params = OPTIM_PARAMS), file = "NUL")
    expected_vecchia <- c(0.67917691, 0.73691878, -0.26052167, 0.83883048, 1.50859022, 0.17220641, 0.14764583)
    expect_lt(sum(abs(c(fit_vecchia$get_coef(), fit_vecchia$get_aux_pars(), fit_vecchia$get_cov_pars()) - expected_vecchia)), TOLERANCE_MEDIUM)
    expect_lt(abs(fit_vecchia$get_current_neg_log_likelihood() - 528.41017749), TOLERANCE_MEDIUM)
    pred <- predict(fit_vecchia, gp_coords_pred = coords[1:3, ] + 1e-3, X_pred = X_gp[1:3, ], predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu - c(2.81348623, 1.94954497, 2.31200778))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var - c(5.27313990, 2.71690034, 4.46470334))), TOLERANCE_MEDIUM)
    capture.output(fit_vecchia_it <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = lik, additional_likelihood_data = N_gp,
                                                gp_approx = "vecchia", num_neighbors = 20, vecchia_ordering = "none", matrix_inversion_method = "iterative",
                                                y = y_gp, X = X_gp, params = params_iter), file = "NUL")
    expected_vecchia_it <- c(0.63383448, 0.73282785, -0.26673570, 0.83879646, 1.50475414, 0.19541929, 0.10620782)
    expect_lt(sum(abs(c(fit_vecchia_it$get_coef(), fit_vecchia_it$get_aux_pars(), fit_vecchia_it$get_cov_pars()) - expected_vecchia_it)), TOLERANCE_NON_CONVEX)
    expect_lt(abs(fit_vecchia_it$get_current_neg_log_likelihood() - 528.21503551), TOLERANCE_NON_CONVEX)
    # FITC approximation
    capture.output(fit_fitc <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = lik, additional_likelihood_data = N_gp,
                                          gp_approx = "fitc", num_ind_points = 50, y = y_gp, X = X_gp, params = OPTIM_PARAMS), file = "NUL")
    expected_fitc <- c(0.64343015, 0.74334861, -0.26610752, 0.84015935, 1.50613619, 0.18861284, 0.11858142)
    expect_lt(sum(abs(c(fit_fitc$get_coef(), fit_fitc$get_aux_pars(), fit_fitc$get_cov_pars()) - expected_fitc)), TOLERANCE_MEDIUM)
    expect_lt(abs(fit_fitc$get_current_neg_log_likelihood() - 528.47046119), TOLERANCE_MEDIUM)
    # Vecchia approximation with multiple observations at the same locations
    coords_u <- matrix(sim_rand_unif(2 * 50, 0.19), ncol = 2)
    capture.output(fit_rep <- fitGPModel(gp_coords = coords_u[rep(1:50, length.out = n_gp), ], cov_function = "exponential", likelihood = lik,
                                         additional_likelihood_data = N_gp, gp_approx = "vecchia", num_neighbors = 15, vecchia_ordering = "none",
                                         matrix_inversion_method = "cholesky", y = y_gp, X = X_gp, params = OPTIM_PARAMS), file = "NUL")
    expected_rep <- c(0.57351078, 0.71176669, -0.26808144, 0.85188058, 1.50805793, 0.17816592, 0.04713350)
    expect_lt(sum(abs(c(fit_rep$get_coef(), fit_rep$get_aux_pars(), fit_rep$get_cov_pars()) - expected_rep)), TOLERANCE_MEDIUM)
    expect_lt(abs(fit_rep$get_current_neg_log_likelihood() - 527.62095755), TOLERANCE_MEDIUM)
    # Y-only varying-dispersion model with a Vecchia approximation
    capture.output(fit_vd <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "tweedie_varying_dispersion", gp_approx = "vecchia",
                                        num_neighbors = 20, vecchia_ordering = "none", matrix_inversion_method = "cholesky", y = y_gp, X = X_gp,
                                        params = OPTIM_PARAMS), file = "NUL")
    expected_vd <- c(0.69171207, 0.75176155, 0.02543444, 0.64574750, 1.49029146, 0.09198423, 0.19362058)
    expect_lt(sum(abs(c(fit_vd$get_coef(), fit_vd$get_aux_pars(), fit_vd$get_cov_pars()) - expected_vd)), TOLERANCE_MEDIUM)
    expect_lt(abs(fit_vd$get_current_neg_log_likelihood() - 336.85729907), TOLERANCE_MEDIUM)

    # Crossed grouped random effects, Cholesky and iterative
    g1 <- rep(1:15, each = 10)
    g2 <- rep(1:10, times = 15)
    params_iter_cr <- c(OPTIM_PARAMS, list(num_rand_vec_trace = 200, seed_rand_vec_trace = 1))
    expected_cr <- list(tweedie_joint = list(cholesky = c(0.62880113, 0.70378071, 1.26817702, 1.56385548, 0.10351720, 0.00002636, 544.14910729),
                                             iterative = c(0.62828807, 0.70399634, 1.26806550, 1.56389193, 0.10357109, 0.00002174, 544.14897769)),
                        tweedie_joint_varying_dispersion = list(cholesky = c(0.63778940, 0.68942094, -0.23553215, 0.81267110, 1.51469420, 0.09619381, 0.00001994, 526.79475638),
                                                                iterative = c(0.63559497, 0.69237351, -0.23816489, 0.81678556, 1.51446100, 0.09751821, 0.00026800, 526.79760434)))
    for (lik_cr in names(expected_cr)) for (method in c("cholesky", "iterative")) {
      capture.output(fit_cr <- fitGPModel(group_data = cbind(g1, g2), likelihood = lik_cr, additional_likelihood_data = N_gp, y = y_gp, X = X_gp,
                                          matrix_inversion_method = method, params = if (method == "cholesky") OPTIM_PARAMS else params_iter_cr), file = "NUL")
      res <- c(fit_cr$get_coef(), fit_cr$get_aux_pars(), fit_cr$get_cov_pars(), fit_cr$get_current_neg_log_likelihood())
      expect_lt(sum(abs(res - expected_cr[[lik_cr]][[method]])), if (method == "cholesky") TOLERANCE_MEDIUM else TOLERANCE_NON_CONVEX)
    }

    ###################
    ## The number of events has to stay aligned with the response under the internal reordering by clusters
    ###################
    cl <- rep(1:3, length.out = n_gp)
    perm <- c(seq(2, n_gp, by = 2), seq(1, n_gp, by = 2))
    nll_cl <- GPModel(group_data = g1, cluster_ids = cl, likelihood = lik, additional_likelihood_data = N_gp)$neg_log_likelihood(
      cov_pars = 0.4, y = y_gp, fixed_effects = c(eta_gp, zeta_gp), aux_pars = 1.45)
    nll_cl_perm <- GPModel(group_data = g1[perm], cluster_ids = cl[perm], likelihood = lik, additional_likelihood_data = N_gp[perm])$neg_log_likelihood(
      cov_pars = 0.4, y = y_gp[perm], fixed_effects = c(eta_gp[perm], zeta_gp[perm]), aux_pars = 1.45)
    expect_lt(abs(nll_cl - 518.90525677), TOLERANCE_MEDIUM)
    expect_lt(abs(nll_cl_perm - nll_cl), TOLERANCE_STRICT)
  })

  test_that("joint and varying-dispersion Tweedie likelihoods for the GPBoost algorithm and cross-validation ", {

    params_boost <- list(learning_rate = 0.1, max_depth = 2, min_data_in_leaf = 5, verbose = 0, deterministic = TRUE)
    gp_model <- GPModel(group_data = group, likelihood = "tweedie_joint_varying_dispersion", additional_likelihood_data = N_gr)
    gp_model$set_optim_params(params = OPTIM_PARAMS)
    bst <- gpb.train(data = gpb.Dataset(data = X[, 2, drop = FALSE], label = y_gr), gp_model = gp_model, nrounds = 20, params = params_boost)
    pred <- predict(bst, data = X[1:3, 2, drop = FALSE], group_data_pred = c(1, 5, 25), predict_var = TRUE, pred_latent = FALSE)
    expect_lt(abs(gp_model$get_cov_pars() - 0.11598325), TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_aux_pars() - 1.53670563), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$response_mean - c(1.49963226, 2.04623764, 2.43172853))), TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$response_var - c(2.82505772, 2.49368651, 6.14823292))), TOLERANCE_MEDIUM)
    # Save and load of the booster
    filename <- tempfile(fileext = ".json")
    gpb.save(bst, filename = filename)
    bst_loaded <- gpb.load(filename = filename)
    pred_loaded <- predict(bst_loaded, data = X[1:3, 2, drop = FALSE], group_data_pred = c(1, 5, 25), predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(pred$response_mean - pred_loaded$response_mean)) + sum(abs(pred$response_var - pred_loaded$response_var)), TOLERANCE_STRICT_LOWER)

    # Cross-validation: the number of events is subset together with the other data (a misaligned subset would violate
    # 'N = 0 if and only if y = 0'). The default metric uses the marginal density of y for the joint likelihood
    folds <- list(seq(1, n, by = 2), seq(2, n, by = 2))
    gp_model <- GPModel(group_data = group, likelihood = "tweedie_joint_varying_dispersion", additional_likelihood_data = N_gr)
    gp_model$set_optim_params(params = OPTIM_PARAMS)
    cvbst <- gpb.cv(data = gpb.Dataset(data = X[, 2, drop = FALSE], label = y_gr), gp_model = gp_model, nrounds = 10, params = params_boost,
                    folds = folds, metric = "mse")
    expect_equal(cvbst$best_iter, 3)
    expect_lt(abs(cvbst$best_score - 5.03521930), TOLERANCE_MEDIUM)
    gp_model <- GPModel(group_data = group, likelihood = "tweedie_joint", additional_likelihood_data = N_gr)
    gp_model$set_optim_params(params = OPTIM_PARAMS)
    cvbst <- gpb.cv(data = gpb.Dataset(data = X[, 2, drop = FALSE], label = y_gr), gp_model = gp_model, nrounds = 10, params = params_boost, folds = folds)
    expect_equal(cvbst$best_iter, 1)
    expect_lt(abs(cvbst$best_score - 1.96781409), TOLERANCE_MEDIUM)
  })

  test_that("joint and varying-dispersion Tweedie likelihoods: input validation ", {
    expect_error(GPModel(group_data = group, likelihood = "tweedie_joint"), "additional_likelihood_data")
    expect_error(GPModel(group_data = group, likelihood = "tweedie_joint_varying_dispersion_fixed_p", additional_likelihood_data = N_gr),
                 "No value was provided for 'likelihood_additional_param'", fixed = TRUE)
    expect_error(GPModel(group_data = group, likelihood = "tweedie_joint", additional_likelihood_data = N_gr[-1]), "does not match")
    N_wrong <- N_gr
    N_wrong[which(y_gr > 0)[1]] <- 0
    expect_error(fitGPModel(group_data = group, likelihood = "tweedie_joint", additional_likelihood_data = N_wrong, y = y_gr), "0 if and only if")
    N_wrong <- N_gr + 0.5
    expect_error(fitGPModel(group_data = group, likelihood = "tweedie_joint", additional_likelihood_data = N_wrong, y = y_gr), "nonnegative integer")
  })

}
