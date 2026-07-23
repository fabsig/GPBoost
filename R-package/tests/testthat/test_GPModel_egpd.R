context("GPD and EGPD likelihoods")

if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

sim_rand_unif_egpd <- function(n, init_c=0.1){
  mod_lcg <- 2^32
  sim <- rep(NA, n)
  sim[1] <- floor(init_c * mod_lcg)
  for(i in 2:n) sim[i] <- (22695477 * sim[i-1] + 1) %% mod_lcg
  sim / mod_lcg
}

sim_gpd_lcg <- function(eta, shape, init_c){
  u <- sim_rand_unif_egpd(length(eta), init_c)
  z <- if(shape == 0) -log1p(-u) else expm1(-shape * log1p(-u)) / shape
  exp(eta) * z
}

test_that("all continuous carriers estimate, evaluate, and predict", {
  n <- 80
  group <- rep(1:16, each=5)
  x <- 2 * sim_rand_unif_egpd(n, 0.17) - 1
  b <- 0.25 * qnorm(sim_rand_unif_egpd(16, 0.31))
  eta <- 0.1 + 0.3 * x + b[group]
  kappa <- 1.4
  u <- sim_rand_unif_egpd(n, 0.71)^(1 / kappa)
  shape <- 0.1
  y <- exp(eta) * expm1(-shape * log1p(-u)) / shape
  initial <- list(gpd=c(0.05), egpd_power=c(0.05, 1.2), egpd_beta=c(0.05, 1),
                  egpd_power_beta=c(0.05, 1, 1.2), egpd_power_mixture=c(0.05, 0.9, 0.6, 0.6))
  expected <- list(
    gpd=list(aux=-0.1904527582, coef=c(0.4118840337, 0.5137677124), cov=0.04699721933, nll=97.73954415,
             given=100.13358045, mu=c(0.8522038025, 0.7835895604), var=c(0.5759926659, 0.4869757118)),
    egpd_power=list(aux=c(-0.08742481805, 1.1517620975), coef=c(0.2475385011, 0.5370884373), cov=0.02808931417,
                    nll=97.49848773, given=98.71975502, mu=c(0.8713290923, 0.7981281827), var=c(0.6003439312, 0.5037104818)),
    egpd_beta=list(aux=c(0.2405998405, 1.7151732379), coef=c(-0.3452957342, 0.5867795921), cov=0.007032903292,
                   nll=100.40833525, given=108.62519397, mu=c(0.8833455357, 0.8025926386), var=c(1.0165839065, 0.8392132597)),
    egpd_power_beta=list(aux=c(-0.08668289867, 1.0469984245, 1.1559812290), coef=c(0.2527227024, 0.5378556834),
                         cov=0.02642681026, nll=97.49633508, given=98.71975502, mu=c(0.8730075086, 0.7995653591),
                         var=c(0.6031604222, 0.5059467598)),
    egpd_power_mixture=list(aux=c(-0.05164320454, 1.0751327224, 0.3677128479, 0.6463666057), coef=c(0.1827105737, 0.5408923322),
                            cov=0.01536002914, nll=97.59735768, given=99.48223799, mu=c(0.8767495404, 0.8025942982),
                            var=c(0.6161613395, 0.5163396687)))
  expected_names <- list(gpd="shape", egpd_power=c("shape", "kappa"), egpd_beta=c("shape", "delta"),
                         egpd_power_beta=c("shape", "delta", "kappa"),
                         egpd_power_mixture=c("shape", "kappa1", "delta_kappa", "p"))
  for(likelihood in names(expected)){
    fit <- fitGPModel(group_data=group, y=y, X=cbind(1, x), likelihood=likelihood,
                      params=list(maxit=15, delta_rel_conv=1e-5, init_aux_pars=initial[[likelihood]], init_coef_aux_pars_from_iid_model=FALSE))
    actual_aux <- fit$get_aux_pars()
    reference <- expected[[likelihood]]
    expect_identical(names(actual_aux), expected_names[[likelihood]])
    expect_equal(unname(actual_aux), reference$aux, tolerance=1e-4)
    expect_equal(unname(fit$get_coef()), reference$coef, tolerance=1e-4)
    expect_equal(unname(fit$get_cov_pars()), reference$cov, tolerance=1e-4)
    expect_equal(fit$get_current_neg_log_likelihood(), reference$nll, tolerance=1e-4)
    evaluated <- fit$neg_log_likelihood(unname(fit$get_cov_pars()), y, fixed_effects=drop(cbind(1, x) %*% fit$get_coef()), aux_pars=unname(actual_aux))
    expect_equal(evaluated, reference$nll, tolerance=1e-4)
    pred <- predict(fit, group_data_pred=group[1:2], X_pred=cbind(1, x[1:2]), predict_response=TRUE, predict_var=TRUE)
    expect_equal(unname(pred$mu), reference$mu, tolerance=1e-4)
    expect_equal(unname(pred$var), reference$var, tolerance=1e-4)
    expect_equal(fit$neg_log_likelihood(0.04, y, fixed_effects=0.1 + 0.3 * x, aux_pars=initial[[likelihood]]), reference$given, tolerance=1e-4)
  }
})

test_that("GPD covers grouped, crossed, Vecchia, and combined latent models", {
  n <- 80
  x <- 2 * sim_rand_unif_egpd(n, 0.17) - 1
  coords <- cbind(sim_rand_unif_egpd(n, 0.31), sim_rand_unif_egpd(n, 0.67))
  group1 <- rep(seq_len(16), each=5)
  group2 <- rep(seq_len(10), times=8)
  b1 <- 0.3 * qnorm(sim_rand_unif_egpd(16, 0.73))
  b2 <- 0.2 * qnorm(sim_rand_unif_egpd(10, 0.29))
  gp_cov <- 0.12 * exp(-as.matrix(dist(coords)) / 0.25) + diag(1e-10, n)
  gp_effect <- drop(t(chol(gp_cov)) %*% qnorm(sim_rand_unif_egpd(n, 0.83)))
  params_chol <- list(maxit=25, delta_rel_conv=1e-5, init_aux_pars=c(0.05), init_coef_aux_pars_from_iid_model=FALSE)
  params_iter <- list(maxit=20, delta_rel_conv=1e-4, init_aux_pars=c(0.05), cg_preconditioner_type="ssor", num_rand_vec_trace=200,
                      cg_max_num_it=300, cg_max_num_it_tridiag=300, cg_delta_conv=1e-7, init_coef_aux_pars_from_iid_model=FALSE)

  eta_group <- 0.2 + 0.35 * x + b1[group1]
  y_group <- sim_gpd_lcg(eta_group, 0.1, 0.41)
  fit_group <- fitGPModel(group_data=group1, y=y_group, X=cbind(1, x), likelihood="gpd", params=params_chol)
  expect_equal(unname(fit_group$get_aux_pars()), -0.03752757247, tolerance=1e-4)
  expect_equal(unname(fit_group$get_coef()), c(0.1767480618, 0.03703002668), tolerance=1e-4)
  expect_equal(unname(fit_group$get_cov_pars()), 0.2032628084, tolerance=1e-4)
  expect_equal(fit_group$get_current_neg_log_likelihood(), 96.60467923, tolerance=1e-4)
  expect_equal(fit_group$neg_log_likelihood(unname(fit_group$get_cov_pars()), y_group, fixed_effects=drop(cbind(1, x) %*% fit_group$get_coef()),
                                            aux_pars=unname(fit_group$get_aux_pars())), 96.60467923, tolerance=1e-4)
  pred_group <- predict(fit_group, group_data_pred=group1[1:3], X_pred=cbind(1, x[1:3]), predict_response=TRUE, predict_var=TRUE)
  expect_equal(unname(pred_group$mu), c(1.4913611673, 1.4823656181, 1.4870656611), tolerance=1e-4)
  expect_equal(unname(pred_group$var), c(2.4511914239, 2.4217105525, 2.4370916272), tolerance=1e-4)

  eta_crossed <- 0.1 + b1[group1] + b2[group2]
  y_crossed <- sim_gpd_lcg(eta_crossed, 0.1, 0.47)
  expected_crossed <- list(
    cholesky=list(aux=-0.04344375872, cov=c(0.07955212538, 0.1122116685), nll=91.28575879, eval=91.28575879,
                  mu=c(0.8475744992, 0.818141704, 0.654439765), var=c(0.8326477476, 0.7846484255, 0.5058407081)),
    iterative=list(aux=-0.04219820068, cov=c(0.08797824184, 0.1042743589), nll=91.31085537, eval=91.31085537,
                   mu=c(0.8405726542, 0.8100799401, 0.658135969), var=c(0.8244578115, 0.7747829283, 0.5144882703)))
  for(method in c("cholesky", "iterative")){
    fit <- fitGPModel(group_data=cbind(group1, group2), y=y_crossed, likelihood="gpd", matrix_inversion_method=method,
                      params=if(method == "cholesky") params_chol else params_iter)
    reference <- expected_crossed[[method]]
    tolerance <- if(method == "cholesky") 1e-4 else 1e-2
    expect_equal(unname(fit$get_aux_pars()), reference$aux, tolerance=tolerance)
    expect_equal(unname(fit$get_cov_pars()), reference$cov, tolerance=tolerance)
    expect_equal(fit$get_current_neg_log_likelihood(), reference$nll, tolerance=tolerance)
    expect_equal(fit$neg_log_likelihood(unname(fit$get_cov_pars()), y_crossed, aux_pars=unname(fit$get_aux_pars())), reference$eval, tolerance=tolerance)
    pred <- predict(fit, group_data_pred=cbind(group1[1:3], group2[1:3]), predict_response=TRUE, predict_var=TRUE)
    expect_equal(unname(pred$mu), reference$mu, tolerance=tolerance)
    expect_equal(unname(pred$var), reference$var, tolerance=tolerance * 5)
  }

  eta_gp <- 0.15 + 0.3 * x + gp_effect
  y_gp <- sim_gpd_lcg(eta_gp, 0.1, 0.53)
  expected_vecchia <- list(
    cholesky=list(aux=0.1280153215, coef=c(0.2330672807, 0.2956505845), cov=c(0.01006370593, 0.006742792854), nll=108.5825156,
                  eval=108.5825156, mu=c(1.197245466, 1.154883065, 1.161833531), var=c(1.963507856, 1.82677478, 1.849281107)),
    iterative=list(aux=0.08480078684, coef=c(0.2625809123, 0.2747986863), cov=c(0.06770450802, 0.02546282803), nll=108.6881116,
                   eval=108.6881116, mu=c(1.188597295, 1.259126496, 1.173604482), var=c(1.913616409, 2.132313753, 1.881919813)))
  for(method in c("cholesky", "iterative")){
    params <- if(method == "cholesky") params_chol else within(params_iter, cg_preconditioner_type <- "vadu")
    fit <- fitGPModel(gp_coords=coords, gp_approx="vecchia", num_neighbors=12, matrix_inversion_method=method, y=y_gp, X=cbind(1, x),
                      likelihood="gpd", params=params)
    reference <- expected_vecchia[[method]]
    tolerance <- 1e-2
    expect_equal(unname(fit$get_aux_pars()), reference$aux, tolerance=tolerance)
    expect_equal(unname(fit$get_coef()), reference$coef, tolerance=tolerance)
    expect_equal(unname(fit$get_cov_pars()), reference$cov, tolerance=tolerance)
    expect_equal(fit$get_current_neg_log_likelihood(), reference$nll, tolerance=tolerance)
    evaluated <- fit$neg_log_likelihood(unname(fit$get_cov_pars()), y_gp, fixed_effects=drop(cbind(1, x) %*% fit$get_coef()), aux_pars=unname(fit$get_aux_pars()))
    expect_equal(evaluated, reference$eval, tolerance=tolerance)
    pred <- predict(fit, gp_coords_pred=coords[1:3, ], X_pred=cbind(1, x[1:3]), predict_response=TRUE, predict_var=TRUE)
    expect_equal(unname(pred$mu), reference$mu, tolerance=tolerance)
    expect_equal(unname(pred$var), reference$var, tolerance=tolerance * 5)
  }

  eta_combined <- 0.1 + 0.25 * x + b1[group1] + gp_effect
  y_combined <- sim_gpd_lcg(eta_combined, 0.3, 0.59)
  fit_combined <- fitGPModel(group_data=group1, gp_coords=coords, gp_approx="none", y=y_combined, X=cbind(1, x), likelihood="gpd", params=params_chol)
  expect_equal(unname(fit_combined$get_aux_pars()), 0.2086584656, tolerance=1e-4)
  expect_equal(unname(fit_combined$get_coef()), c(0.2348555118, 0.6550444871), tolerance=1e-4)
  expect_equal(unname(fit_combined$get_cov_pars()), c(7.193903706e-06, 0.0005339096720, 0.005044869741), tolerance=1e-4)
  expect_equal(fit_combined$get_current_neg_log_likelihood(), 112.08649465, tolerance=1e-4)
  evaluated <- fit_combined$neg_log_likelihood(unname(fit_combined$get_cov_pars()), y_combined,
                                               fixed_effects=drop(cbind(1, x) %*% fit_combined$get_coef()), aux_pars=unname(fit_combined$get_aux_pars()))
  expect_equal(evaluated, 112.08649465, tolerance=1e-4)
  pred_combined <- predict(fit_combined, group_data_pred=group1[1:3], gp_coords_pred=coords[1:3, ], X_pred=cbind(1, x[1:3]),
                           predict_response=TRUE, predict_var=TRUE)
  expect_equal(unname(pred_combined$mu), c(1.0381680386, 0.9326345468, 0.9862718573), tolerance=1e-4)
  expect_equal(unname(pred_combined$var), c(1.8530712810, 1.4954773708, 1.6724384879), tolerance=1e-4)
})

test_that("EGPD carriers with multiple observations at the same location (Vecchia, Cholesky and iterative)", {
  # Many observations share the same coordinates -> the use_random_effects_indices_of_data_ Vecchia path. Covered carriers:
  # gpd (aux: shape) and egpd_beta (aux: shape, delta). GPD-distributed responses are simulated and both carriers are fitted.
  n <- 80
  x <- 2 * sim_rand_unif_egpd(n, 0.17) - 1
  nu_rep <- 30L
  coords_rep_u <- cbind(sim_rand_unif_egpd(nu_rep, 0.23), sim_rand_unif_egpd(nu_rep, 0.61))
  rep_idx <- rep(seq_len(nu_rep), length.out = n)
  coords_rep <- coords_rep_u[rep_idx, ]
  gp_cov_rep <- 0.1 * exp(-as.matrix(dist(coords_rep_u)) / 0.3) + diag(1e-10, nu_rep)
  gp_eff_rep <- drop(t(chol(gp_cov_rep)) %*% qnorm(sim_rand_unif_egpd(nu_rep, 0.79)))[rep_idx]
  eta_rep <- 0.15 + 0.3 * x + gp_eff_rep
  y_rep <- sim_gpd_lcg(eta_rep, 0.1, 0.47)
  params_chol <- list(maxit = 25, delta_rel_conv = 1e-5, init_coef_aux_pars_from_iid_model = FALSE)
  params_iter <- list(maxit = 20, delta_rel_conv = 1e-4, cg_preconditioner_type = "vadu", num_rand_vec_trace = 200,
                      cg_max_num_it = 300, cg_max_num_it_tridiag = 300, cg_delta_conv = 1e-7, init_coef_aux_pars_from_iid_model = FALSE)
  init_aux <- list(gpd = c(0.05), egpd_beta = c(0.05, 1))
  # The Vecchia GP covariance-parameter likelihood is very flat (a marginal-variance / range ridge),
  # so the fitted parameters and derived predictions differ noticeably across compilers (e.g. gcc vs
  # MSVC) even though the log-likelihood agrees. Use a loose tolerance for those parameter and
  # prediction checks while keeping the negative log-likelihood checks tight.
  tolerance_vecchia <- 0.15
  expected <- list(
    gpd = list(
      cholesky = list(aux = -0.1755019, coef = c(0.52490354, 0.03620733), cov = c(0.23144346, 0.02691741), nll = 116.4854,
                      mu = c(1.194444, 1.213371, 1.291308), var = c(1.452638, 1.495390, 1.625194)),
      iterative = list(aux = -0.1787143, coef = c(0.52956126, 0.01912643), cov = c(0.23614061, 0.03923343), nll = 116.4977,
                       mu = c(1.202196, 1.235888, 1.261303), var = c(1.488869, 1.572646, 1.516421))),
    egpd_beta = list(
      cholesky = list(aux = c(0.3092383, 1.2882979), coef = c(-0.3577017, 0.1644255), cov = c(0.19462495, 0.03643538), nll = 128.3454,
                      mu = c(1.275096, 1.282584, 1.231061), var = c(3.348404, 3.360593, 3.130387)),
      iterative = list(aux = c(0.3274186, 1.2220556), coef = c(-0.3850626, 0.1604101), cov = c(0.17193140, 0.06874916), nll = 128.4917,
                       mu = c(1.325761, 1.322082, 1.256015), var = c(3.911032, 3.872978, 3.422747))))
  for (lik in c("gpd", "egpd_beta")) {
    for (method in c("cholesky", "iterative")) {
      params <- if (method == "cholesky") params_chol else params_iter
      params$init_aux_pars <- init_aux[[lik]]
      fit <- fitGPModel(gp_coords = coords_rep, gp_approx = "vecchia", num_neighbors = 12, matrix_inversion_method = method,
                        y = y_rep, X = cbind(1, x), likelihood = lik, params = params)
      reference <- expected[[lik]][[method]]
      tolerance <- if (method == "cholesky") 1e-3 else 1e-2
      expect_equal(unname(fit$get_aux_pars()), reference$aux, tolerance = tolerance_vecchia)
      expect_equal(unname(fit$get_coef()), reference$coef, tolerance = tolerance_vecchia)
      expect_equal(unname(fit$get_cov_pars()), reference$cov, tolerance = tolerance_vecchia)
      expect_equal(fit$get_current_neg_log_likelihood(), reference$nll, tolerance = tolerance)
      evaluated <- fit$neg_log_likelihood(unname(fit$get_cov_pars()), y_rep, fixed_effects = drop(cbind(1, x) %*% fit$get_coef()), aux_pars = unname(fit$get_aux_pars()))
      expect_equal(evaluated, reference$nll, tolerance = tolerance)
      pred <- predict(fit, gp_coords_pred = coords_rep_u[1:3, ], X_pred = cbind(1, x[1:3]), predict_response = TRUE, predict_var = TRUE)
      expect_equal(unname(pred$mu), reference$mu, tolerance = tolerance_vecchia)
      expect_equal(unname(pred$var), reference$var, tolerance = tolerance_vecchia)
    }
  }
})

test_that("EGPD response and auxiliary parameter validation is explicit", {
  expect_error(fitGPModel(group_data=1:3, y=c(1, 0, 2), likelihood="gpd"), "y > 0", fixed=TRUE)
  expect_error(fitGPModel(group_data=1:3, y=c(1, Inf, 2), likelihood="egpd_power"), "NaN or Inf")
  expect_error(fitGPModel(group_data=1:3, y=c(1, 2, 3), likelihood="gpd", params=list(init_aux_pars=-0.5)), "larger than -0.5")
  expect_error(fitGPModel(group_data=1:3, y=c(1, 2, 3), likelihood="egpd_power", params=list(init_aux_pars=c(0, 0))), "larger than 0")
  expect_error(fitGPModel(group_data=1:3, y=c(1, 2, 3), likelihood="egpd_power_mixture", params=list(init_aux_pars=c(0, 1, 1, 1))),
               "strictly between 0 and 1")
})

test_that("EGPD carriers reduce to their special cases and match a closed-form GPD density", {
  # These checks are deliberately non-self-referential: the expected values are
  # either exact analytic identities between carriers or a hand-computed density,
  # so they fail if the density code is wrong even though every other test in
  # this file only reproduces values generated by the implementation itself.
  n <- 60
  group <- rep(seq_len(12), each=5)
  x <- 2 * sim_rand_unif_egpd(n, 0.19) - 1
  eta <- 0.2 + 0.4 * x
  y <- sim_gpd_lcg(eta, 0.15, 0.37)
  cov <- 0.25
  nll <- function(likelihood, aux, y_use=y, eta_use=eta, cov_use=cov) {
    GPModel(group_data=group, likelihood=likelihood)$neg_log_likelihood(cov_use, y_use, fixed_effects=eta_use, aux_pars=aux)
  }

  # (1) The power carrier G(u) = u^kappa is the identity at kappa = 1, so
  #     'egpd_power' with kappa = 1 is exactly the GPD.
  expect_equal(nll("egpd_power", c(0.15, 1)), nll("gpd", c(0.15)), tolerance=1e-5)

  # (2) With delta = 1 the beta building blocks collapse to B(u) = u^2 and
  #     B'(u) = 2u, so 'egpd_power_beta' reduces to 'egpd_power' with the same
  #     kappa (density 0.5*kappa*B^(kappa/2-1)*B' = kappa*u^(kappa-1)).
  expect_equal(nll("egpd_power_beta", c(0.15, 1, 1.3)), nll("egpd_power", c(0.15, 1.3)), tolerance=1e-5)

  # (3) A two-component mixture with identical powers is that single power, for
  #     any mixing weight p. delta_kappa must be > 0, so use a negligibly small
  #     gap and a correspondingly loose tolerance.
  expect_equal(nll("egpd_power_mixture", c(0.15, 1.3, 1e-7, 0.4)), nll("egpd_power", c(0.15, 1.3)), tolerance=1e-3)

  # (4) Closed-form anchor: the GPD with shape = 0 is the exponential, for which
  #     -log f(y | eta) = eta + y * exp(-eta). With a vanishing random-effect
  #     variance the marginal collapses onto the conditional sum.
  eta0 <- 0.1 + 0.3 * x
  y_exp <- sim_gpd_lcg(eta0, 0, 0.51)
  analytic <- sum(eta0 + y_exp * exp(-eta0))
  expect_equal(nll("gpd", c(0), y_use=y_exp, eta_use=eta0, cov_use=1e-4), analytic, tolerance=5e-3)
})

}
