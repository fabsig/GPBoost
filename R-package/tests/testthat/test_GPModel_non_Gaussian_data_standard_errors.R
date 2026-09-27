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
  # Separate helper for the very strict tolerances (1e-6) of comparisons whose expected values cannot be
  # handed to 'relax_tolerance', so that its lower bound cannot be tied to their magnitude. Deviations of
  # a few 1e-6 occur under valgrind in particular, which does not reproduce floating point arithmetic
  # bit-wise (it rounds the 80 bit intermediate results of x87 to 64 bit and its libm differs)
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
    cov_pars_v <- c(0.0578304572593316, 0.0330021964490000, 0.715705182971271,
                    0.081226357385000, 0.0483181738202239, 0.00771178320850000)
    coef_v <- c(0.962932883927947, 0.110839163937625, 1.10211572960383, 0.0925165508666763)
    nll_v <- 537.536566769249
    expect_lt(sum(abs(as.vector(gp_model_gauss_v$get_cov_pars(std_err = TRUE))-cov_pars_v)),relax_tolerance(TOLERANCE_STRICT, cov_pars_v))
    expect_lt(sum(abs(as.vector(gp_model_gauss_v$get_coef(std_err = TRUE))-coef_v)),relax_tolerance(TOLERANCE_STRICT, coef_v))
    expect_lt(abs(gp_model_gauss_v$get_current_neg_log_likelihood()-nll_v),relax_tolerance(TOLERANCE_MEDIUM, nll_v))
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
    expect_lt(sum(abs(cov_pars_t_v_result[c(1,3)]-cov_pars_t_v[c(1,3)])),relax_tolerance(TOLERANCE_STRICT, cov_pars_t_v[c(1,3)]))# estimates
    # The standard errors of covariance and auxiliary parameters are obtained from a Hessian that is approximated
    #   with finite differences of a gradient which itself relies on an iterative mode finding algorithm (see
    #   'CalcHessianCovParAuxPars'). Within one build they are reproducible to about 1e-6, also with a different
    #   number of threads, but the optimizer of this model reaches a different stationary point on another
    #   compiler (see the note on the negative log-likelihood below), and the standard errors there differ by far
    #   more than the estimates do: 0.23 in the sum below has been measured with clang and libc++ in the
    #   sanitizer container of R-hub, for standard errors of the order of 0.01 - 0.11. They are therefore
    #   compared with 'tol_vecchia_cov_pars_se', the tolerance that the comparisons further below use for the
    #   standard errors of the same covariance parameters
    expect_lt(sum(abs(cov_pars_t_v_result[c(2,4)]-cov_pars_t_v[c(2,4)])),tol_vecchia_cov_pars_se)# standard errors
    expect_lt(sum(abs(aux_pars_t_v_result[1]-aux_pars_t_v[1])),relax_tolerance(TOLERANCE_STRICT, aux_pars_t_v[1]))# estimate
    expect_lt(sum(abs(aux_pars_t_v_result[2:3]-aux_pars_t_v[2:3])),relax_tolerance(TOLERANCE_MEDIUM, aux_pars_t_v[2:3]))# standard error and fixed df
    expect_true(is.nan(aux_pars_t_v_result[4])) # no standard error for the fixed (not estimated) degrees-of-freedom parameter
    expect_lt(sum(abs(coef_t_v_result[c(1,3)]-coef_t_v[c(1,3)])),relax_tolerance(TOLERANCE_STRICT, coef_t_v[c(1,3)]))
    if (coef_se_available(coef_t_v_result)) expect_lt(sum(abs(coef_t_v_result[c(2,4)]-coef_t_v[c(2,4)])),relax_tolerance(TOLERANCE_STRICT, coef_t_v[c(2,4)]))
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

}
