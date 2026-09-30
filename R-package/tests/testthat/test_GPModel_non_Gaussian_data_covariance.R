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

  test_that("linear covariance ", {

    params <- OPTIM_PARAMS_BFGS

    d_lin <- 50 # dimension of GP locations
    coords_lin <- matrix(sim_rand_unif(n=n*d_lin, init_c=0.1156), ncol=d_lin)
    beta_lin <- qnorm(sim_rand_unif(n=d_lin, init_c=0.1234),sd=1)
    lp_lin <- coords_lin %*% beta_lin + X %*% beta
    y <- lp_lin + qnorm(sim_rand_unif(n=n, init_c=0.2224), sd=0.1)
    coord_test <- matrix(sim_rand_unif(n=3*d_lin, init_c=0.19156), ncol=d_lin)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))

    likelihood <- "gaussian"
    for (cov_function in c("linear", "linear_no_woodbury")) {
      if (cov_function == "linear") {
        matrix_inversion_method_loop <- c("cholesky", "iterative")
      } else {
        matrix_inversion_method_loop <- c("cholesky")
      }
      for (matrix_inversion_method in matrix_inversion_method_loop) {
        if(matrix_inversion_method == "iterative") {
          tolerance_loc_1 <- 2
          tolerance_loc_2 <- 0.05
          tolerance_loc_3 <- 6
        } else {
          tolerance_loc_1 <- TOLERANCE_STRICT
          tolerance_loc_2 <- TOLERANCE_STRICT
          tolerance_loc_3 <- TOLERANCE_STRICT
          tolerance_loc_4 <- 0.002
        }

        # Evaluate negative log-likelihood
        gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                            matrix_inversion_method = matrix_inversion_method, cov_function = cov_function)
        # gp_model$set_optim_params(params = list(num_rand_vec_trace=500, init_coef_aux_pars_from_iid_model = FALSE))
        nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
        nll_exp <- 268.6641569
        expect_lt(abs(nll-nll_exp),tolerance_loc_1)
        # Estimation
        capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                               matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params) , file='NUL')
        cov_pars_exp <- c(0.01428942126, 0.92806146725)
        coef_exp <- c(0.08076221412, 1.97947766605)
        nll_opt_exp <- 81.26251299
        num_it <- 17
        expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tolerance_loc_2)
        expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),2*tolerance_loc_2)
        expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tolerance_loc_1)
        if (matrix_inversion_method == "cholesky") expect_equal(gp_model$get_num_optim_iter(), num_it)
        # Prediction
        pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                        predict_var=TRUE, predict_response = FALSE)
        expected_mu <- c(4.671312214, 3.029084877, 7.400864491)
        expected_var <- c(0.01524446, 0.01621295, 0.01564379)
        expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(tolerance_loc_3, expected_mu))
        expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_4)

        # X_testd <- X
        # X_testd[,2] <- 0
        # X_testd[,1] <- 0
        # pred <- predict(gp_model, y=y, gp_coords_pred = coords_lin, X_pred = X_testd,
        #                 predict_var=TRUE, predict_response = FALSE)
        # b <- coords_lin %*% beta_lin
        # plot(b,pred$mu)
        # mean((b-pred$mu)^2) # 0.006641768

        ## Vecchia approximation
        if (matrix_inversion_method == "cholesky") {
          gp_approx <- "vecchia"
          num_neighbors <- n - 1
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors,
                              vecchia_ordering = "none")
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),tolerance_loc_1)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors,
                                                 vecchia_ordering = "none") , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tolerance_loc_2)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tolerance_loc_2)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tolerance_loc_1)
          expect_equal(gp_model$get_num_optim_iter(), num_it)
          gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(tolerance_loc_3, expected_mu))
          expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_2)

          num_neighbors <- 50
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors, vecchia_ordering = "none")
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),relax_tolerance_nll(5))
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors, vecchia_ordering = "none") , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.2)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.3)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),45)
          expect_equal(gp_model$get_num_optim_iter(), 15)
          gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.05, expected_mu))
          expect_lt(sum(abs(pred$var-expected_var)),0.05)

          gp_approx <- "fitc"
          ind_points_selection <- "random"
          num_ind_points <- n-1
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),tolerance_loc_4)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),relax_tolerance(tolerance_loc_4, cov_pars_exp))
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tolerance_loc_4)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tolerance_loc_4)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_4)
          expect_lt(sum(abs(pred$var-expected_var)),0.1)

          num_ind_points <- 50
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),1.2)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.02)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.05)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),2.5)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),0.15)
          expect_lt(sum(abs(pred$var-expected_var)),0.1)

          # VIF approximation
          gp_approx <- "vif"
          ind_points_selection <- "random"
          num_ind_points <- n-1
          num_neighbors <- 20
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                              ind_points_selection = ind_points_selection)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),tolerance_loc_4)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors,
                                                 num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),relax_tolerance(tolerance_loc_4, cov_pars_exp))
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tolerance_loc_4)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tolerance_loc_4)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),tolerance_loc_4)
          expect_lt(sum(abs(pred$var-expected_var)),tolerance_loc_4)

          num_ind_points <- 50
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                              ind_points_selection = ind_points_selection)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5, 0.9),y=y)
          expect_lt(abs(nll-nll_exp),relax_tolerance_nll(0.1))
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors,
                                                 num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),relax_tolerance(tolerance_loc_4, cov_pars_exp))
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),relax_tolerance(0.02, coef_exp))
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),relax_tolerance_nll(0.2))
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.05, expected_mu))
          expect_lt(sum(abs(pred$var-expected_var)),0.05)
        }

        ## GPBoost algorithm
        if (matrix_inversion_method == "cholesky") {
          dtrain <- gpb.Dataset(data = X, label = y)
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function)
          gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
          bst <- gpboost(data = dtrain, gp_model = gp_model,
                         nrounds = 30, learning_rate = 0.1, max_depth = 6,
                         min_data_in_leaf = 5, verbose = 0)
          expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-c(0.03919405941, 0.91870507429 ))),TOLERANCE_MEDIUM)
          # Prediction
          pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
                          predict_var = TRUE, pred_latent = TRUE)
          expect_lt(sum(abs(tail(pred$fixed_effect, n=3)-c(1.654867987, 2.755278195, 3.513302218))),TOLERANCE_MEDIUM)
          # Predict response
          pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
                          predict_var = TRUE, pred_latent = FALSE)
          expect_lt(sum(abs(tail(pred$response_mean, n=3)-c( 4.498812041, 2.449730254, 7.779354333))),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(tail(pred$response_var, n=3)-c(0.08045027559, 0.08266782334, 0.08156700720))), TOLERANCE_MEDIUM)
        }
      }
    }

    likelihood <- "t_fix_df"
    for (cov_function in c("linear", "linear_no_woodbury")) {
      if (cov_function == "linear") {
        matrix_inversion_method_loop <- c("cholesky", "iterative")
      } else {
        matrix_inversion_method_loop <- c("cholesky")
      }
      for (matrix_inversion_method in matrix_inversion_method_loop) {
        if(matrix_inversion_method == "iterative") {
          tolerance_loc_1 <- 2
          tolerance_loc_2 <- 0.05
          tolerance_loc_3 <- 4
        } else {
          tolerance_loc_1 <- TOLERANCE_STRICT
          tolerance_loc_2 <- TOLERANCE_STRICT
          tolerance_loc_3 <- TOLERANCE_STRICT
          tolerance_loc_4 <- 0.002
        }

        # Evaluate negative log-likelihood
        gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                            matrix_inversion_method = matrix_inversion_method, cov_function = cov_function)
        # gp_model$set_optim_params(params = list(num_rand_vec_trace=500, init_coef_aux_pars_from_iid_model = FALSE))
        nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5),y=y)
        nll_exp <- 227.5314805
        expect_lt(abs(nll-nll_exp),tolerance_loc_1)

        # Estimation
        if(matrix_inversion_method == "choklesky") {
          # Estimation is very slow for iterative methods
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,
                                                 matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                                 X=X, y = y, params = params) , file='NUL')
          cov_pars_exp <- c(0.9357944695)
          aux_par_exp <- c(0.09651268839, 2.00000000000)
          coef_exp <- c(0.1011884891, 1.9905600506)
          nll_opt_exp <- 82.49996414
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_par_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_MEDIUM)
          # Prediction
          pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                          predict_var=TRUE, predict_response = TRUE)
          expected_mu <- c(4.600315578, 3.029201064, 7.466329615)
          expected_var <- c(0.02586692444, 0.02691118187, 0.02630117411)
          expect_lt(sum(abs(pred$mu-expected_mu)),0.1)
          expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_MEDIUM)

          ## Vecchia approximation
          gp_approx <- "vecchia"
          num_neighbors <- n - 1
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5),y=y)
          expect_lt(abs(nll-nll_exp),tolerance_loc_1)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,
                                                 matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                                 X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_par_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_MEDIUM)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = TRUE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_MEDIUM)

          ## FITC approximation
          gp_approx <- "fitc"
          ind_points_selection <- "random"
          num_ind_points <- n-1
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, ind_points_selection = ind_points_selection, num_ind_points=num_ind_points)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5),y=y)
          expect_lt(abs(nll-nll_exp),1e-5)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,
                                                 matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                                 X=X, y = y, params = params,
                                                 gp_approx = gp_approx, ind_points_selection = ind_points_selection, num_ind_points=num_ind_points) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.5)
          expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-aux_par_exp)),0.01)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.5)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),3)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = TRUE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),3)
          expect_lt(sum(abs(pred$var-expected_var)),0.1)

          # VIF approximation
          gp_approx <- "vif"
          ind_points_selection <- "kmeans++"
          num_ind_points <- 10
          num_neighbors <- 80
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                              gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                              ind_points_selection = ind_points_selection)
          nll <- gp_model$neg_log_likelihood(cov_pars=c(0.5),y=y)
          expect_lt(abs(nll-nll_exp),tolerance_loc_4)
          capture.output( gp_model <- fitGPModel(gp_coords = coords_lin, likelihood = likelihood,  cov_function = cov_function,
                                                 matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                                 gp_approx = gp_approx, num_neighbors = num_neighbors,
                                                 num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
          expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),3)
          expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),4)
          expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),120)
          capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                          predict_var=TRUE, predict_response = FALSE) , file='NUL')
          expect_lt(sum(abs(pred$mu-expected_mu)),2)
          expect_lt(sum(abs(pred$var-expected_var)),7)

          ## GPBoost algorithm
          dtrain <- gpb.Dataset(data = X, label = y)
          gp_model <- GPModel(gp_coords = coords_lin, likelihood = likelihood,
                              matrix_inversion_method = matrix_inversion_method, cov_function = cov_function)
          gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
          bst <- gpboost(data = dtrain, gp_model = gp_model,
                         nrounds = 30, learning_rate = 0.1, max_depth = 6,
                         min_data_in_leaf = 5, verbose = 0)
          expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-c(0.9269398031  ))),TOLERANCE_MEDIUM)
          expect_lt(sum(abs(as.vector(gp_model$get_aux_pars())-c(0.1895315932, 2.0000000000))),0.01)
          # Predict response
          pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
                          predict_var = TRUE, pred_latent = FALSE)
          expect_lt(sum(abs(tail(pred$response_mean, n=3)-c(4.510024051, 3.325082390, 7.658811482))),0.1)
          expect_lt(sum(abs(tail(pred$response_var, n=3)-c( 0.0982840077, 0.1011744166, 0.1000118224))), 0.01)
        }
      }
    }

  }) # end linear covariance

  test_that("hurst covariance ", {

    hurst_cov <- function(t, pars) {
      sigma2 <- pars[1]
      H <- pars[2]
      r  <- rowSums(t^2)
      rH <- r^H
      D2 <- as.matrix(dist(t))^2
      A <- outer(rH, rH, "+")
      K <- 0.5 * sigma2 * (A - D2^H)
      K
    }
    simulate_hurst <- function(t, pars, jitter = 1e-8) {
      K <- hurst_cov(t, pars)
      n  <- dim(t)[1]
      K <- K + jitter * diag(n)  # small jitter for numerical stability
      L <- chol(K)
      z <- qnorm(sim_rand_unif(n=n, init_c=0.1346), sd=0.1)
      y <- drop(L %*% z)
      y
    }

    params <- OPTIM_PARAMS_BFGS

    H_true <- 0.5
    sigma2_true <- 1
    pars <- c(sigma2_true, H_true)
    b <- simulate_hurst(coords, pars)
    y <- X %*% beta + b + qnorm(sim_rand_unif(n=n, init_c=0.1354), sd=sqrt(0.01))

    coord_test <- matrix(sim_rand_unif(n=3*2, init_c=0.19156), ncol=2)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))

    likelihood <- "gaussian"
    cov_function <- "hurst"
    matrix_inversion_method <- "cholesky"

    # Evaluate negative log-likelihood
    cov_pars_eval = c(0.01, sigma2_true, H_true)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    nll_exp <- 2508.161111
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params) , file='NUL')
    cov_pars_exp <- c(2.430011710e-02, 1.417072813e-07, 9.571564920e-01)
    coef_exp <- c(0.06807413795, 2.01626778203)
    nll_opt_exp <- -43.96963741
    num_it <- 26
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    if (matrix_inversion_method == "cholesky") expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-0.9400622610, 0.4713289372, 0.8745803091)
    expected_var <- c(1.416871849e-07, 1.416920045e-07, 1.417021983e-07)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## Vecchia approximation
    gp_approx <- "vecchia"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 50
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors,
                                        vecchia_ordering = "none") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),2)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           vecchia_ordering = "none") , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.3)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.01)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),0.1)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01, expected_mu))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    gp_approx <- "vecchia_correlation"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 50
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-2512.097),10)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.3)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.01)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),0.1)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01, expected_mu))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    gp_approx <- "fitc"
    ind_points_selection <- "random"
    num_ind_points <- n-1
    tol_fitc <- 0.001
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),tol_fitc)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_fitc)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01, expected_mu))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    num_ind_points <- 50
    tol_fitc2 <- 0.15
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),300)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc2)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),1)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01, expected_mu))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    # VIF approximation
    gp_approx <- "vif"
    ind_points_selection <- "random"
    num_ind_points <- n-1
    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                                        ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),1e-5)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_ind_points <- 50
    tol_vif <- 0.05
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                                        ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),relax_tolerance_nll(1))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_vif)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_vif)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_vif)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01, expected_mu))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function) , file='NUL')
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model,
                   nrounds = 30, learning_rate = 0.1, max_depth = 6,
                   min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-c(2.828394259e-02, 1.947589183e-10, 3.121883864e-01 ))),TOLERANCE_MEDIUM)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=3)-c(-0.6202239136, 0.3687055841, 0.7950425174))),TOLERANCE_MEDIUM)
    # Predict response
    pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=3)-c(-0.6202239139, 0.3687055879, 0.7950425167))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(tail(pred$response_var, n=3)-c(0.02828394279, 0.02828394279, 0.02828394279))), TOLERANCE_MEDIUM)

    ####################
    ## non-Gaussian likelihood
    ####################
    likelihood <- "t_fix_df"

    # Evaluate negative log-likelihood
    cov_pars_eval = c(sigma2_true, H_true)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    nll_exp <- 196.6342458
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Estimation
    # Note: the optimizer ends up in a very flat region of the log-likelihood (the Hurst parameter goes to almost 0).
    #   Which point in this region it stops at depends on the order in which floating point numbers are summed, i.e.,
    #   on the number of OpenMP threads: measured against the values below (which were recorded with all threads of
    #   this machine), the deviations over 1, 2, 4 and 8 threads are up to 2e-4 for the covariance parameters, 7e-3
    #   for the coefficients, 2e-3 for the negative log-likelihood and 6e-2 for the predicted means, and the number
    #   of iterations varies between 59 and 69. The tolerances below have to accommodate this. Note: this must not be
    #   solved by pinning the number of threads of the model, since that would only hide the dependence on the number
    #   of threads instead of covering it
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params) , file='NUL')
    cov_pars_exp <- c(0.01850628942, 0.008229165631)
    coef_exp <- c(0.087477519806 ,2.0203797574)
    nll_opt_exp <- -76.64280779
    tol_flat_region <- 0.05
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_flat_region)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_flat_region)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_flat_region)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-0.9401364534, 0.4826689913, 0.8811585466)
    expected_var <- c(0.0092433831081, 0.0091525152933, 0.0091867058615)
    expect_lt(sum(abs(pred$mu-expected_mu)),0.15)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_LOOSE)

    ## Vecchia approximation
    gp_approx <- "vecchia"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # slow
    # capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
    #                                        matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
    #                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    # expect_equal(gp_model$get_num_optim_iter(), num_it)
    # gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    # capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
    #                                 predict_var=TRUE, predict_response = FALSE) , file='NUL')
    # expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    # expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors,
                                        vecchia_ordering = "none") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),0.5)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),0.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.1)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),10)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),0.2)
    expect_lt(sum(abs(pred$var-expected_var)),0.1)

    gp_approx <- "fitc"
    ind_points_selection <- "random"
    num_ind_points <- n-1
    tol_fitc <- 0.001
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),tol_fitc)
    #slow
    # capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
    #                                        matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
    #                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc)
    # expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc)
    # expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_fitc)
    # capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
    #                                 predict_var=TRUE, predict_response = FALSE) , file='NUL')
    # expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01, expected_mu))
    # expect_lt(sum(abs(pred$var-expected_var)),0.01)

    num_ind_points <- 20
    # Note: with only 20 inducing points, this is a very crude approximation and the fit is only compared to the
    #   exact one above to check that it is in the same ballpark. The exact fit ends up in a very flat region of
    #   the log-likelihood (see the note there), which the approximation does not reproduce, and the tolerances
    #   below thus have to be loose
    tol_fitc2 <- 2.5
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),2)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc2)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),50)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),1.)
    expect_lt(sum(abs(pred$var-expected_var)),0.1)

    # VIF approximation
    gp_approx <- "vif"
    ind_points_selection <- "kmeans++"
    num_ind_points <- 10
    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                                        ind_points_selection = ind_points_selection) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),0.05)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),1.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.3)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),200)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),0.2)
    expect_lt(sum(abs(pred$var-expected_var)),2)

    # slow
    # ## GPBoost algorithm
    # dtrain <- gpb.Dataset(data = X, label = y)
    # capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
    #                                     matrix_inversion_method = matrix_inversion_method, cov_function = cov_function) , file='NUL')
    # gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    # bst <- gpboost(data = dtrain, gp_model = gp_model,
    #                nrounds = 2, learning_rate = 0.1, max_depth = 6,
    #                min_data_in_leaf = 5, verbose = 0)
    # expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-c(2.828394259e-02, 1.947589183e-10, 3.121883864e-01 ))),TOLERANCE_MEDIUM)
    # # Prediction
    # pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
    #                 predict_var = TRUE, pred_latent = TRUE)
    # expect_lt(sum(abs(tail(pred$fixed_effect, n=3)-c(-0.6202239136, 0.3687055841, 0.7950425174))),TOLERANCE_MEDIUM)
    # # Predict response
    # pred <- predict(bst, data = X_test, gp_coords_pred = coord_test,
    #                 predict_var = TRUE, pred_latent = FALSE)
    # expect_lt(sum(abs(tail(pred$response_mean, n=3)-c(-0.6202239139, 0.3687055879, 0.7950425167))),TOLERANCE_MEDIUM)
    # expect_lt(sum(abs(tail(pred$response_var, n=3)-c(0.02828394279, 0.02828394279, 0.02828394279))), TOLERANCE_MEDIUM)

    ####################
    ## Hurst ARD
    ####################
    likelihood <- "gaussian"
    cov_function <- "hurst_ard"
    matrix_inversion_method <- "cholesky"

    # Evaluate negative log-likelihood
    cov_pars_eval = c(0.01, sigma2_true, H_true, 1.5)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    nll_exp <- 2817.257021
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params) , file='NUL')
    cov_pars_exp <- c(2.430010340e-02, 1.159703688e-07, 9.612311400e-01, 8.006726088e-01)
    coef_exp <- c(0.06798478941, 2.01626275734)
    nll_opt_exp <- -43.96966225
    num_it <- 27
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    if (matrix_inversion_method == "cholesky") expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                    predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-0.9401500043, 0.4712381007, 0.8744885264)
    expected_var <- c(8.895122085e-08, 4.318616931e-08, 2.009318253e-08)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ## Vecchia approximation
    gp_approx <- "vecchia"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors,
                                        vecchia_ordering = "none") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           vecchia_ordering = "none") , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),relax_tolerance_nll(5))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),3)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.01)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),1)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),0.025)
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    gp_approx <- "vecchia_correlation"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    # expect_lt(abs(nll-627.4222728),10) # some randomness
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors) , file='NUL')
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),1)  some randomness
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),0.5)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),15)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),0.3)
    expect_lt(sum(abs(pred$var-expected_var)),0.3)

    gp_approx <- "fitc"
    ind_points_selection <- "random"
    num_ind_points <- n-1
    tol_fitc <- 0.005
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y) , file='NUL')
    expect_lt(abs(nll-nll_exp),tol_fitc)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_fitc)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01, expected_mu))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    num_ind_points <- 50
    tol_fitc2 <- 6
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y) , file='NUL')
    expect_lt(abs(nll-nll_exp),600)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_fitc2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_fitc2)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),1)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01, expected_mu))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

    # VIF approximation
    gp_approx <- "vif"
    ind_points_selection <- "random"
    num_ind_points <- n-1
    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                                        ind_points_selection = ind_points_selection) , file='NUL')
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y) , file='NUL')
    expect_lt(abs(nll-nll_exp),5e-5)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_ind_points <- 50
    tol_vif <- 0.06
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood,
                                        matrix_inversion_method = matrix_inversion_method, cov_function = cov_function,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, num_ind_points = num_ind_points,
                                        ind_points_selection = ind_points_selection) , file='NUL')
    capture.output( nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y) , file='NUL')
    expect_lt(abs(nll-nll_exp),relax_tolerance_nll(5))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood,  cov_function = cov_function,
                                           matrix_inversion_method = matrix_inversion_method, X=X, y = y, params = params,
                                           gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           num_ind_points = num_ind_points, ind_points_selection = ind_points_selection) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_vif)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_vif)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_vif)
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),relax_tolerance(0.01, expected_mu))
    expect_lt(sum(abs(pred$var-expected_var)),0.01)

  }) # end hurst covariance

  test_that("hurst covariance of order 2 ", {

    # Anchored second-order Hurst covariance (1 < H < 2); 'ranges' are the ARD ranges of the coordinates 2, ..., d
    hurst2_cov <- function(t, pars, t2 = t, ranges = NULL) {
      force(t2)
      if (!is.null(ranges)) {
        t <- t %*% diag(c(1, 1 / ranges), ncol(t))
        t2 <- t2 %*% diag(c(1, 1 / ranges), ncol(t2))
      }
      H <- pars[2]
      r <- rowSums(t^2)
      r2 <- rowSums(t2^2)
      D2 <- 0
      for (k in 1:ncol(t)) D2 <- D2 + outer(t[, k], t2[, k], "-")^2
      pars[1] / (2 * (2 * H - 1)) * (D2^H - outer(r^H, r2^H, "+") + 2 * H * (t %*% t(t2)) * outer(r^(H - 1), r2^(H - 1), "+"))
    }
    gauss_nll <- function(y, Sigma) {
      0.5 * sum(y * solve(Sigma, y)) + 0.5 * as.numeric(determinant(Sigma)$modulus) + 0.5 * length(y) * log(2 * pi)
    }

    params <- OPTIM_PARAMS_BFGS

    H_true <- 1.5
    sigma2_true <- 1
    K <- hurst2_cov(coords, c(sigma2_true, H_true))
    b <- drop(t(chol(K + 1e-10 * diag(n))) %*% qnorm(sim_rand_unif(n=n, init_c=0.2461)))
    y <- drop(X %*% beta) + b + qnorm(sim_rand_unif(n=n, init_c=0.3317), sd=0.1)

    coord_test <- matrix(sim_rand_unif(n=3*2, init_c=0.19156), ncol=2)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))

    likelihood <- "gaussian"
    cov_function <- "hurst"
    cov_fct_order <- 2

    # Evaluate negative log-likelihood
    cov_pars_eval <- c(0.01, sigma2_true, H_true)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    nll_exp <- 8328.797471
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    expect_lt(abs(nll-gauss_nll(y, hurst2_cov(coords, cov_pars_eval[-1]) + diag(cov_pars_eval[1], n))),TOLERANCE_STRICT)
    capture.output( gp_model_int <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = 2L) , file='NUL')
    expect_equal(gp_model_int$neg_log_likelihood(cov_pars=cov_pars_eval,y=y), nll)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function,
                                           cov_fct_order = cov_fct_order, X=X, y = y, params = params) , file='NUL')
    cov_pars_exp <- c(0.01433277887, 0.9375252319, 1.457563386)
    cov_pars_std_err_exp <- c(0.002306404878, 0.5120739393, 0.2985738105)
    coef_exp <- c(0.06880575111, 1.990641292)
    nll_opt_exp <- -51.06899076
    num_it <- 25
    cov_pars <- gp_model$get_cov_pars(std_err = TRUE)
    expect_lt(sum(abs(as.vector(cov_pars[1,])-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(cov_pars[2,])-cov_pars_std_err_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test, predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-1.045440734, 0.1936336084, 0.7465159477)
    expected_var <- c(0.003768442958, 0.001592648545, 0.001669263371)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)
    # The predictions equal the kriging formulas with the covariance function above
    Sigma_opt <- hurst2_cov(coords, cov_pars_exp[-1]) + diag(cov_pars_exp[1], n)
    K_cross <- hurst2_cov(coord_test, cov_pars_exp[-1], coords)
    mu_R <- drop(X_test %*% coef_exp) + drop(K_cross %*% solve(Sigma_opt, y - drop(X %*% coef_exp)))
    var_R <- diag(hurst2_cov(coord_test, cov_pars_exp[-1])) - rowSums(K_cross * t(solve(Sigma_opt, t(K_cross))))
    expect_lt(sum(abs(pred$mu-mu_R)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-var_R)),TOLERANCE_STRICT)
    # Saving and loading
    filename <- tempfile(fileext = ".json")
    on.exit(unlink(filename), add = TRUE)
    saveGPModel(gp_model, filename = filename)
    capture.output( gp_model_loaded <- loadGPModel(filename) , file='NUL')
    pred_loaded <- predict(gp_model_loaded, y=y, gp_coords_pred = coord_test, X_pred = X_test, predict_var=TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_loaded$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_loaded$var-expected_var)),TOLERANCE_STRICT)
    # A model saved without 'cov_fct_order' (i.e., with an older version) is loaded with order 1
    capture.output( gp_model_order1 <- fitGPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function,
                                                  X=X, y = y, params = params) , file='NUL')
    model_list <- gp_model_order1$model_to_list()
    expect_equal(model_list[["cov_fct_order"]], 1L)
    model_list[["cov_fct_order"]] <- NULL
    writeLines(RJSONIO::toJSON(model_list, digits = 17), filename)
    capture.output( gp_model_loaded <- loadGPModel(filename) , file='NUL')
    pred_order1 <- predict(gp_model_order1, y=y, gp_coords_pred = coord_test, X_pred = X_test, predict_var=TRUE, predict_response = FALSE)
    pred_loaded <- predict(gp_model_loaded, y=y, gp_coords_pred = coord_test, X_pred = X_test, predict_var=TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred_loaded$mu-pred_order1$mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred_loaded$var-pred_order1$var)),TOLERANCE_STRICT)
    expect_gt(sum(abs(pred_order1$mu-expected_mu)),0.01)

    ## Vecchia approximation
    gp_approx <- "vecchia"
    num_neighbors <- n - 1
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, vecchia_ordering = "none") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                           X=X, y = y, params = params, gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           vecchia_ordering = "none") , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    num_neighbors <- 20
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                        gp_approx = gp_approx, num_neighbors = num_neighbors, vecchia_ordering = "none") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-8251.906099),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                           X=X, y = y, params = params, gp_approx = gp_approx, num_neighbors = num_neighbors,
                                           vecchia_ordering = "none") , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(0.01406317641, 1.106301664, 1.558216601))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(0.07376208843, 1.9898108))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-(-51.53614832))),TOLERANCE_MEDIUM)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-c(-1.046957336, 0.1970476608, 0.7471312879))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-c(0.00359821399, 0.001538512908, 0.001584914559))),TOLERANCE_MEDIUM)

    ## FITC and VIF approximations with n - 1 inducing points (and n - 1 neighbors for VIF, which makes it exact)
    # Note: FITC depends on which point is not an inducing point. This point is chosen randomly, and the random numbers differ
    #   between standard libraries (e.g., MSVC and libstdc++). Over 30 different choices (seeds), the maximal absolute differences
    #   to the exact values below were 5.8e-2 (FITC) and 4.3e-6 (VIF) for the negative log-likelihood at 'cov_pars_eval', and
    #   1.2e-3 (FITC) and 2.5e-4 (VIF) for the estimates and predictions. For VIF, the latter is due to the optimizer stopping at
    #   slightly different points of a flat likelihood (the negative log-likelihoods at the optimum differed by at most 1.2e-6)
    ind_points_selection <- "random"
    num_ind_points <- n - 1
    for (gp_approx in c("fitc", "vif")) {
      tol_approx <- if (gp_approx == "fitc") 3e-3 else 1e-3
      tol_nll_eval <- if (gp_approx == "fitc") 0.1 else 1e-4
      capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                          gp_approx = gp_approx, num_ind_points = num_ind_points, ind_points_selection = ind_points_selection,
                                          num_neighbors = n - 1, vecchia_ordering = "none") , file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
      expect_lt(abs(nll-nll_exp),tol_nll_eval)
      capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                             X=X, y = y, params = params, gp_approx = gp_approx, num_ind_points = num_ind_points,
                                             ind_points_selection = ind_points_selection, num_neighbors = n - 1, vecchia_ordering = "none") , file='NUL')
      expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),tol_approx)
      expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),tol_approx)
      expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),tol_approx)
      capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                      predict_var=TRUE, predict_response = FALSE) , file='NUL')
      expect_lt(sum(abs(pred$mu-expected_mu)),tol_approx)
      expect_lt(sum(abs(pred$var-expected_var)),tol_approx)
    }

    ## GPBoost algorithm
    dtrain <- gpb.Dataset(data = X, label = y)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order) , file='NUL')
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    bst <- gpboost(data = dtrain, gp_model = gp_model, nrounds = 20, learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5, verbose = 0)
    expect_lt(sum(abs(gp_model$get_cov_pars(std_err = FALSE)-c(0.08593392492, 0.8276047013, 1.567443162))),TOLERANCE_MEDIUM)
    pred <- predict(bst, data = X_test, gp_coords_pred = coord_test, predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(pred$fixed_effect-c(-0.4292684618, 0.2094116307, 0.5277758807))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$random_effect_mean-c(-0.08877789705, -0.2558258985, -0.04872671587))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$random_effect_cov-c(0.007648948665, 0.00450959984, 0.00355998048))),TOLERANCE_MEDIUM)
    # Cross-validation (the order has to be passed on to the models of the folds; the order 1 model has best_score = 0.2075509972)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order) , file='NUL')
    gp_model$set_optim_params(params=OPTIM_PARAMS_BFGS)
    capture.output( cvbst <- gpb.cv(params = params_cv, data = dtrain, gp_model = gp_model, nrounds = 20, early_stopping_rounds = 5,
                                    folds = folds, verbose = 0, eval = "l2") , file='NUL')
    expect_lt(abs(cvbst$best_score-0.1503614213),TOLERANCE_MEDIUM)
    expect_equal(cvbst$best_iter, 14)

    ## GP random coefficients: all components use the same order
    pars_rc <- c(0.01, 1, 1.5, 0.7, 1.2, 0.5, 1.8)
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, likelihood = likelihood,
                                        cov_function = cov_function, cov_fct_order = cov_fct_order) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=pars_rc,y=y)
    Sigma_rc <- hurst2_cov(coords, pars_rc[2:3]) + diag(Z_SVC[,1]) %*% hurst2_cov(coords, pars_rc[4:5]) %*% diag(Z_SVC[,1]) +
      diag(Z_SVC[,2]) %*% hurst2_cov(coords, pars_rc[6:7]) %*% diag(Z_SVC[,2]) + diag(pars_rc[1], n)
    expect_lt(abs(nll-gauss_nll(y, Sigma_rc)),TOLERANCE_STRICT)
    expect_lt(abs(nll-7303.702863),TOLERANCE_STRICT)

    ####################
    ## non-Gaussian likelihood
    ####################
    likelihood <- "poisson"
    y_pois <- qpois(sim_rand_unif(n=n, init_c=0.7713), lambda = exp(2 + sqrt(3) * b))
    # Evaluate negative log-likelihood
    cov_pars_eval_pois <- c(3, H_true)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                        matrix_inversion_method = "cholesky") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval_pois,y=y_pois)
    nll_exp_pois <- 288.1632286
    expect_lt(abs(nll-nll_exp_pois),TOLERANCE_STRICT)
    # Laplace approximation calculated directly (Newton's method for the mode, Rasmussen and Williams, 2006, Algorithm 3.1)
    K_pois <- hurst2_cov(coords, cov_pars_eval_pois)
    b_mode <- rep(0, n)
    for (it in 1:100) {
      mu <- exp(b_mode)
      L <- t(chol(diag(n) + outer(sqrt(mu), sqrt(mu)) * K_pois))
      a_vec <- mu * b_mode + (y_pois - mu)
      b_mode <- drop(K_pois %*% (a_vec - sqrt(mu) * backsolve(t(L), forwardsolve(L, sqrt(mu) * drop(K_pois %*% a_vec)))))
    }
    mu <- exp(b_mode)
    L <- t(chol(diag(n) + outer(sqrt(mu), sqrt(mu)) * K_pois))
    nll_laplace <- -sum(dpois(y_pois, mu, log = TRUE)) + 0.5 * sum(b_mode * (y_pois - mu)) + sum(log(diag(L)))
    expect_lt(abs(nll-nll_laplace),TOLERANCE_STRICT)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                        matrix_inversion_method = "cholesky", gp_approx = "vecchia", num_neighbors = n - 1,
                                        vecchia_ordering = "none") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval_pois,y=y_pois)
    expect_lt(abs(nll-nll_exp_pois),TOLERANCE_STRICT)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                           matrix_inversion_method = "cholesky", X=X, y = y_pois, params = params) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(2.186339104, 1.347510797))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(1.730161538, -0.01931586557))),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-251.408017),TOLERANCE_MEDIUM)
    # Prediction
    pred <- predict(gp_model, y=y_pois, gp_coords_pred = coord_test, X_pred = X_test, predict_var=TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-c(1.798566508, 1.466171955, 1.608439515))),TOLERANCE_MEDIUM)
    expect_lt(sum(abs(pred$var-c(0.02043631575, 0.0136666458, 0.01215659935))),TOLERANCE_MEDIUM)

    ####################
    ## Hurst ARD
    ####################
    likelihood <- "gaussian"
    cov_function <- "hurst_ard"
    cov_pars_eval <- c(0.01, sigma2_true, H_true, 1.5)
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    nll_exp <- 8633.532102
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    expect_lt(abs(nll-gauss_nll(y, hurst2_cov(coords, cov_pars_eval[2:3], ranges = cov_pars_eval[4]) + diag(cov_pars_eval[1], n))),TOLERANCE_STRICT)
    # Estimation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function,
                                           cov_fct_order = cov_fct_order, X=X, y = y, params = params) , file='NUL')
    cov_pars_exp <- c(0.01423901997, 0.8784232162, 1.459892413, 0.9278775066)
    cov_pars_std_err_exp <- c(0.002296094898, 0.6337320655, 0.2958239064, 0.3048287672)
    coef_exp <- c(0.06736140402, 1.990622554)
    nll_opt_exp <- -51.08995827
    cov_pars <- gp_model$get_cov_pars(std_err = TRUE)
    expect_lt(sum(abs(as.vector(cov_pars[1,])-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(cov_pars[2,])-cov_pars_std_err_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(gp_model$get_current_neg_log_likelihood()-nll_opt_exp)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 33)
    # Prediction
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test, predict_var=TRUE, predict_response = FALSE)
    expected_mu <- c(-1.041885666, 0.1968946523, 0.745204728)
    expected_var <- c(0.003946186509, 0.001622511011, 0.001715037408)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)
    ## Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                        gp_approx = "vecchia", num_neighbors = n - 1, vecchia_ordering = "none") , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y)
    expect_lt(abs(nll-nll_exp),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, likelihood = likelihood, cov_function = cov_function, cov_fct_order = cov_fct_order,
                                           X=X, y = y, params = params, gp_approx = "vecchia", num_neighbors = n - 1,
                                           vecchia_ordering = "none") , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-cov_pars_exp)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-coef_exp)),TOLERANCE_STRICT)
    gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_all")
    capture.output( pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                                    predict_var=TRUE, predict_response = FALSE) , file='NUL')
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE_STRICT)

    ####################
    ## Continuous-time RW1 and RW2 priors (H fixed to 0.5 and 1.5)
    ####################
    cov_function <- "hurst"
    time_mat <- matrix(time)
    a_min <- outer(time, time, pmin)
    a_max <- outer(time, time, pmax)
    # RW2: integrated Wiener process with covariance q / 6 * min^2 * (3 * max - min) and q = 3 * sigma2
    capture.output( gp_model <- GPModel(gp_coords = time_mat, likelihood = likelihood, cov_function = cov_function, cov_fct_order = 2) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.01, 0.8, 1.5),y=y)
    expect_lt(abs(nll-gauss_nll(y, 3 * 0.8 / 6 * a_min^2 * (3 * a_max - a_min) + diag(0.01, n))),TOLERANCE_STRICT)
    expect_lt(abs(nll-9600.29288),TOLERANCE_STRICT)
    # RW1: Brownian motion with covariance q * min and q = sigma2
    capture.output( gp_model <- GPModel(gp_coords = time_mat, likelihood = likelihood, cov_function = cov_function, cov_fct_order = 1) , file='NUL')
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.01, 0.8, 0.5),y=y)
    expect_lt(abs(nll-gauss_nll(y, 0.8 * a_min + diag(0.01, n))),TOLERANCE_STRICT)
    expect_lt(abs(nll-6317.787058),TOLERANCE_STRICT)
    # Estimation with H fixed, data simulated from an integrated Wiener process (q = 30) and a Brownian motion (q = 5)
    z_t <- qnorm(sim_rand_unif(n=n, init_c=0.5173))
    eps_t <- qnorm(sim_rand_unif(n=n, init_c=0.2291), sd=0.1)
    y_iwp <- 1 + 2 * time + drop(t(chol(30 / 6 * a_min^2 * (3 * a_max - a_min) + diag(1e-10, n))) %*% z_t) + eps_t
    y_bm <- 1 + drop(t(chol(5 * a_min)) %*% z_t) + eps_t
    capture.output( gp_model <- fitGPModel(gp_coords = time_mat, likelihood = likelihood, cov_function = cov_function, cov_fct_order = 2,
                                           X = cbind(1, time), y = y_iwp,
                                           params = c(params, list(init_cov_pars = c(0.1, 1, 1.5), estimate_cov_par_index = c(1, 1, 0)))) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(0.01100721703, 0.7023021607, 1.5))),TOLERANCE_STRICT)
    expect_equal(as.numeric(gp_model$get_cov_pars(std_err = FALSE)[3]), 1.5)
    expect_lt(sum(abs(as.vector(gp_model$get_coef(std_err = FALSE))-c(0.9706791991, 2.252732195))),TOLERANCE_STRICT)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-(-76.24786882)),TOLERANCE_STRICT)
    capture.output( gp_model <- fitGPModel(gp_coords = time_mat, likelihood = likelihood, cov_function = cov_function, cov_fct_order = 1,
                                           X = matrix(1, n), y = y_bm,
                                           params = c(params, list(init_cov_pars = c(0.1, 1, 0.5), estimate_cov_par_index = c(1, 1, 0)))) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars(std_err = FALSE))-c(0.01821958853, 4.424321581, 0.5))),TOLERANCE_STRICT)
    expect_equal(as.numeric(gp_model$get_cov_pars(std_err = FALSE)[3]), 0.5)
    expect_lt(abs(gp_model$get_current_neg_log_likelihood()-13.15477627),TOLERANCE_STRICT)

    ####################
    ## Validation of the order and of H
    ####################
    expect_error(capture.output( GPModel(gp_coords = coords, cov_function = "hurst", cov_fct_order = 3) , file='NUL'))
    expect_error(capture.output( GPModel(gp_coords = coords, cov_function = "hurst", cov_fct_order = 0) , file='NUL'))
    expect_error(capture.output( GPModel(gp_coords = coords, cov_function = "hurst", cov_fct_order = 1.5) , file='NUL'))
    expect_error(capture.output( GPModel(gp_coords = coords, cov_function = "hurst", cov_fct_order = c(1, 2)) , file='NUL'))
    expect_error(capture.output( GPModel(gp_coords = coords, cov_function = "matern", cov_fct_order = 2) , file='NUL'))
    expect_error(capture.output( GPModel(gp_coords = coords, cov_function = "gaussian_ard", cov_fct_order = 2) , file='NUL'))
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "hurst", cov_fct_order = 2) , file='NUL')
    expect_error(gp_model$neg_log_likelihood(cov_pars=c(0.01, 1, 0.5),y=y))
    expect_error(gp_model$neg_log_likelihood(cov_pars=c(0.01, 1, 2),y=y))
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "hurst", cov_fct_order = 1) , file='NUL')
    expect_error(gp_model$neg_log_likelihood(cov_pars=c(0.01, 1, 1.5),y=y))
    expect_error(capture.output( fitGPModel(gp_coords = coords, cov_function = "hurst", cov_fct_order = 2, y = y,
                                            params = list(init_cov_pars = c(0.01, 1, 0.5))) , file='NUL'))

  }) # end hurst covariance of order 2

  test_that("hurst covariance: observations at the anchor and standard errors ", {

    # Anchored Hurst covariance of order 1 or 2; 'ranges' are the ARD ranges of the coordinates 2, ..., d
    hurst_cov <- function(t, pars, order, ranges = NULL) {
      if (!is.null(ranges)) t <- t %*% diag(c(1, 1 / ranges), ncol(t))
      H <- pars[2]
      r <- rowSums(t^2)
      D2 <- 0
      for (k in 1:ncol(t)) D2 <- D2 + outer(t[, k], t[, k], "-")^2
      if (order == 1) return(pars[1] / 2 * (outer(r^H, r^H, "+") - D2^H))
      pars[1] / (2 * (2 * H - 1)) * (D2^H - outer(r^H, r^H, "+") + 2 * H * (t %*% t(t)) * outer(r^(H - 1), r^(H - 1), "+"))
    }
    gauss_nll <- function(y, Sigma) {
      0.5 * sum(y * solve(Sigma, y)) + 0.5 * as.numeric(determinant(Sigma)$modulus) + 0.5 * length(y) * log(2 * pi)
    }

    ## Observations exactly at the anchor (the origin), where the Gaussian process is zero
    time_0 <- matrix(c(0, time[-n]))
    a_min <- outer(time_0[, 1], time_0[, 1], pmin)
    a_max <- outer(time_0[, 1], time_0[, 1], pmax)
    b_0 <- drop(t(chol(30 / 6 * a_min^2 * (3 * a_max - a_min) + diag(1e-10, n))) %*% qnorm(sim_rand_unif(n=n, init_c=0.5173)))
    y_0 <- 1 + b_0 + qnorm(sim_rand_unif(n=n, init_c=0.2291), sd=0.1)
    y_pois_0 <- qpois(sim_rand_unif(n=n, init_c=0.7713), lambda = exp(1 + b_0))
    y_bin_0 <- as.numeric(sim_rand_unif(n=n, init_c=0.4127) < plogis(b_0))
    y_beta_0 <- plogis(b_0 + qnorm(sim_rand_unif(n=n, init_c=0.3319), sd=0.3))
    for (order in 1:2) {
      cov_pars_eval <- c(0.01, 0.8, order - 0.5)
      capture.output( gp_model <- GPModel(gp_coords = time_0, cov_function = "hurst", cov_fct_order = order) , file='NUL')
      nll_exact <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y_0)
      expect_lt(abs(nll_exact-gauss_nll(y_0, hurst_cov(time_0, cov_pars_eval[2:3], order) + diag(cov_pars_eval[1], n))),TOLERANCE_STRICT)
      # The predictive distribution at the anchor is a point mass at zero
      pred <- predict(gp_model, y=y_0, gp_coords_pred = matrix(c(0, 0.5)), cov_pars = cov_pars_eval, predict_var = TRUE, predict_response = FALSE)
      expect_identical(pred$mu[1], 0)
      expect_identical(pred$var[1], 0)
      # The predictive distribution of the response at the anchor is the conditional distribution given a latent value of zero
      for (likelihood in c("bernoulli_logit", "beta")) {
        capture.output( gp_model <- GPModel(gp_coords = time_0, cov_function = "hurst", cov_fct_order = order, likelihood = likelihood) , file='NUL')
        pred <- predict(gp_model, y = if (likelihood == "beta") y_beta_0 else y_bin_0, gp_coords_pred = matrix(c(0, 0.5)),
                        cov_pars = cov_pars_eval[2:3], predict_var = TRUE, predict_response = TRUE)
        expect_true(all(is.finite(c(pred$mu, pred$var))))
        expect_lt(abs(pred$mu[1] - 0.5), TOLERANCE_STRICT)
        var_expected <- if (likelihood == "beta") 0.25 / (1 + gp_model$get_aux_pars()[1]) else 0.25
        expect_lt(abs(pred$var[1] - var_expected), TOLERANCE_STRICT)
      }
      # The anchor is not used as an inducing point, since it would make the covariance matrix of the inducing points singular.
      #   All other n - 1 points are inducing points, and the approximations are exact
      for (gp_approx in c("fitc", "vif")) {
        capture.output( gp_model <- GPModel(gp_coords = time_0, cov_function = "hurst", cov_fct_order = order, gp_approx = gp_approx,
                                            num_ind_points = n - 1, ind_points_selection = "random", num_neighbors = n - 1,
                                            vecchia_ordering = "none") , file='NUL')
        expect_lt(abs(gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y_0)-nll_exact),TOLERANCE_MEDIUM)
      }
      # Poisson likelihood
      capture.output( gp_model <- GPModel(gp_coords = time_0, cov_function = "hurst", cov_fct_order = order, likelihood = "poisson",
                                          matrix_inversion_method = "cholesky") , file='NUL')
      nll_exact <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval[2:3],y=y_pois_0)
      capture.output( gp_model <- GPModel(gp_coords = time_0, cov_function = "hurst", cov_fct_order = order, likelihood = "poisson",
                                          matrix_inversion_method = "cholesky", gp_approx = "fitc", num_ind_points = n - 1,
                                          ind_points_selection = "random") , file='NUL')
      expect_lt(abs(gp_model$neg_log_likelihood(cov_pars=cov_pars_eval[2:3],y=y_pois_0)-nll_exact),TOLERANCE_MEDIUM)
      # A Vecchia approximation of the latent process cannot handle the zero conditional variance at the anchor
      capture.output( gp_model <- GPModel(gp_coords = time_0, cov_function = "hurst", cov_fct_order = order, likelihood = "poisson",
                                          gp_approx = "vecchia", num_neighbors = 10) , file='NUL')
      expect_error(gp_model$neg_log_likelihood(cov_pars=cov_pars_eval[2:3],y=y_pois_0), "conditional variance of zero")
    }
    # The "test_neg_log_likelihood" metric of the GPBoost algorithm with a validation point at the anchor, where the latent
    #   predictive variance is zero and the density is the conditional density at the predictive mean, and a second one
    #   elsewhere, where the latent variable is integrated out. A likelihood without and one with an additional predictor
    x_0 <- matrix(sim_rand_unif(n=n, init_c=0.4411), ncol = 1)
    y_metric <- list(poisson = qpois(sim_rand_unif(n=n, init_c=0.6917), lambda = exp(0.5 + x_0[, 1] + b_0)),
                     gaussian_heteroscedastic = 0.5 + x_0[, 1] + b_0 + qnorm(sim_rand_unif(n=n, init_c=0.6029)) * exp(0.5 * (-1 + x_0[, 1])))
    log_dens <- list(poisson = function(y, e, z) dpois(y, exp(e), log = TRUE),
                     gaussian_heteroscedastic = function(y, e, z) dnorm(y, e, exp(0.5 * z), log = TRUE))
    va <- c(1, n %/% 2)
    tr <- setdiff(1:n, va)
    for (likelihood in names(y_metric)) {
      y_m <- y_metric[[likelihood]]
      dtrain <- gpb.Dataset(data = x_0[tr, , drop = FALSE], label = y_m[tr])
      dvalid <- gpb.Dataset.create.valid(dtrain, data = x_0[va, , drop = FALSE], label = y_m[va])
      capture.output( gp_model <- GPModel(gp_coords = time_0[tr, , drop = FALSE], cov_function = "hurst", cov_fct_order = 2, likelihood = likelihood) , file='NUL')
      gp_model$set_optim_params(params = list(optimizer_cov = "lbfgs", maxit = 300, init_coef_aux_pars_from_iid_model = FALSE))
      gp_model$set_prediction_data(gp_coords_pred = time_0[va, , drop = FALSE])
      capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 2, learning_rate = 0.1, max_depth = 2, min_data_in_leaf = 5,
                                       valids = list(valid = dvalid), verbose = 0, deterministic = TRUE) , file='NUL')
      metric <- unlist(bst$record_evals$valid$test_neg_log_likelihood$eval)[2]
      pred <- predict(bst, data = x_0[va, , drop = FALSE], gp_coords_pred = time_0[va, , drop = FALSE], predict_var = TRUE, pred_latent = TRUE, num_iteration = 2)
      raw <- predict(bst, data = x_0[va, , drop = FALSE], ignore_gp_model = TRUE, pred_latent = TRUE, num_iteration = 2)
      z <- if (likelihood == "poisson") c(0, 0) else raw[c(2, 4)]# the tree values of the second predictor are interleaved per observation
      mean_eta <- pred$fixed_effect + pred$random_effect_mean
      expect_identical(pred$random_effect_cov[1], 0)
      reference <- mean(sapply(1:2, function(i) {
        if (i == 1) return(-log_dens[[likelihood]](y_m[va][i], mean_eta[i], z[i]))
        sd_eta <- sqrt(pred$random_effect_cov[i])
        integrand <- function(e) exp(log_dens[[likelihood]](y_m[va][i], e, z[i])) * dnorm(e, mean_eta[i], sd_eta)
        -log(integrate(integrand, mean_eta[i] - 12 * sd_eta, mean_eta[i] + 12 * sd_eta, rel.tol = 1e-10)$value)
      }))
      expect_true(is.finite(metric))
      expect_lt(abs(metric - reference), 1e-6)
    }
    # Estimation with FITC, which is exact here, exercises the gradients at the anchor. Note: for order 1, the exact fit stops
    #   prematurely for these data (unsuccessful line search) and cannot serve as a reference
    capture.output( gp_model <- fitGPModel(gp_coords = time_0, cov_function = "hurst", cov_fct_order = 2, likelihood = "poisson", y = y_pois_0,
                                           matrix_inversion_method = "cholesky") , file='NUL')
    capture.output( gp_model_fitc <- fitGPModel(gp_coords = time_0, cov_function = "hurst", cov_fct_order = 2, likelihood = "poisson", y = y_pois_0,
                                                matrix_inversion_method = "cholesky", gp_approx = "fitc", num_ind_points = n - 1,
                                                ind_points_selection = "random") , file='NUL')
    expect_lt(sum(abs(gp_model_fitc$get_cov_pars()-gp_model$get_cov_pars())),TOLERANCE_MEDIUM)
    expect_lt(abs(gp_model_fitc$get_current_neg_log_likelihood()-gp_model$get_current_neg_log_likelihood()),TOLERANCE_MEDIUM)

    # Methods that use centroids ('kmeans++', 'cover_tree', 'space_time_kmeans++') can place an inducing point at the origin also if no
    #   location is there. Such an inducing point is replaced by the closest location, which is -1 and (-1, -1), respectively, in the
    #   following examples with a single inducing point
    fitc_nll <- function(y, coords, z, cov_pars, order) {
      num_data <- nrow(coords)
      C <- hurst_cov(rbind(coords, z), cov_pars[2:3], order)
      k <- C[1:num_data, num_data + 1, drop = FALSE]
      Q <- k %*% t(k) / C[num_data + 1, num_data + 1]
      gauss_nll(y, Q + diag(diag(C)[1:num_data] - diag(Q)) + diag(cov_pars[1], num_data))
    }
    coords_1d <- matrix(c(-2, -1, 3))
    y_1d <- c(0.3, -0.2, 1.1)
    coords_2d <- as.matrix(expand.grid(c(-2, -1, 3), c(-2, -1, 3)))
    y_2d <- sin(1:9)
    for (order in 1:2) {
      cov_pars_eval <- c(0.1, 1, order - 0.5)
      for (ind_points_selection in c("kmeans++", "cover_tree")) {
        capture.output( gp_model <- GPModel(gp_coords = coords_1d, cov_function = "hurst", cov_fct_order = order, gp_approx = "fitc",
                                            num_ind_points = 1, ind_points_selection = ind_points_selection, cover_tree_radius = 100) , file='NUL')
        nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y_1d)
        expect_lt(abs(nll-fitc_nll(y_1d, coords_1d, -1, cov_pars_eval, order)),1e-4)
      }
      capture.output( gp_model <- GPModel(gp_coords = coords_2d, cov_function = "hurst", cov_fct_order = order, gp_approx = "fitc",
                                          num_ind_points = 1, ind_points_selection = "space_time_kmeans++") , file='NUL')
      nll <- gp_model$neg_log_likelihood(cov_pars=cov_pars_eval,y=y_2d)
      expect_lt(abs(nll-fitc_nll(y_2d, coords_2d, c(-1, -1), cov_pars_eval, order)),1e-4)
    }
    # No inducing points can be chosen if all locations are at the origin
    expect_error(capture.output( GPModel(gp_coords = matrix(0, 10, 1), cov_function = "hurst", gp_approx = "fitc", num_ind_points = 1,
                                         ind_points_selection = "cover_tree") , file='NUL'), "all locations are at the origin")

    ## Standard errors: inverse Fisher information of the estimated parameters, with points close to the origin
    # Note: the points close to the origin check the derivatives wrt the ARD ranges for tiny squared norms (below 1e-10),
    #   which were set to zero for order 1 instead of their small but, for small H, not negligible values
    # Standard errors from the Fisher information 0.5 * tr(Sigma^-1 dSigma_a Sigma^-1 dSigma_b), where the derivatives of
    #   the covariance matrix are obtained by central finite differences
    se_fisher <- function(cov_pars, coords, order, estimated) {
      Sigma_fct <- function(p) hurst_cov(coords, p[2:3], order, ranges = p[-(1:3)]) + diag(p[1], nrow(coords))
      Sigma_inv <- solve(Sigma_fct(cov_pars))
      Sigma_inv_dSigma <- lapply(which(estimated), function(i) {
        h <- 1e-6 * cov_pars[i]
        p_up <- p_down <- cov_pars
        p_up[i] <- cov_pars[i] + h
        p_down[i] <- cov_pars[i] - h
        Sigma_inv %*% (Sigma_fct(p_up) - Sigma_fct(p_down)) / (2 * h)
      })
      ind <- seq_along(Sigma_inv_dSigma)
      FI <- outer(ind, ind, Vectorize(function(a, b) 0.5 * sum(Sigma_inv_dSigma[[a]] * t(Sigma_inv_dSigma[[b]]))))
      se <- rep(NaN, length(cov_pars))
      se[estimated] <- sqrt(diag(solve(FI)))
      se
    }
    coords_near <- coords
    coords_near[1:5, ] <- coords[1:5, ] * 1e-6
    for (order in 1:2) {
      H_true <- if (order == 1) 0.1 else 1.5
      K <- hurst_cov(coords_near, c(1, H_true), order, ranges = 0.5)
      y_near <- drop(t(chol(K + diag(1e-8, n))) %*% qnorm(sim_rand_unif(n=n, init_c=0.2461))) + qnorm(sim_rand_unif(n=n, init_c=0.3317), sd=0.1)
      for (fix_H in c(FALSE, TRUE)) {
        # all parameters estimated, and H fixed, which has no standard error then
        estimated <- c(TRUE, TRUE, !fix_H, TRUE)
        init_cov_pars <- c(0.5, 1, if (fix_H) H_true else order - 0.7, 1)
        capture.output( gp_model <- fitGPModel(gp_coords = coords_near, cov_function = "hurst_ard", cov_fct_order = order, y = y_near,
                                               params = list(init_cov_pars = init_cov_pars, estimate_cov_par_index = as.integer(estimated))) , file='NUL')
        cov_pars <- gp_model$get_cov_pars(std_err = TRUE)
        expect_equal(unname(cov_pars[2, ]), se_fisher(cov_pars[1, ], coords_near, order, estimated), tolerance = 1e-5)
      }
    }

  }) # end hurst covariance: observations at the anchor and standard errors

}
