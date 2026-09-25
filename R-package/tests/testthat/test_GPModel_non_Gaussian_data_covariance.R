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

}
