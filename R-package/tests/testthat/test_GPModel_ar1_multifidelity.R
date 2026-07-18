context("GPModel_ar1_multifidelity")

if (Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS") {

  sim_rand_unif <- function(n, init_c=0.1){
    mod_lcg <- 2^32 # modulus for linear congruential generator (random0 used)
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for(i in 2:n) sim[i] <- (22695477 * sim[i-1] + 1) %% mod_lcg
    return(sim / mod_lcg)
  }

  ar1_mf_exponential_covariance <- function(gp_coords, cov_pars) {
    fidelity <- gp_coords[, ncol(gp_coords)]
    coords <- gp_coords[, -ncol(gp_coords), drop = FALSE]
    distance <- as.matrix(dist(coords))
    covariance_low <- cov_pars[1] * exp(-distance / cov_pars[2])
    covariance_discrepancy <- cov_pars[3] * exp(-distance / cov_pars[4])
    loading <- ifelse(fidelity == 0, 1, cov_pars[5])
    covariance_low * outer(loading, loading) + covariance_discrepancy * outer(fidelity, fidelity)
  }

  gaussian_nll_ar1_mf <- function(y, gp_coords, cov_pars) {
    covariance <- ar1_mf_exponential_covariance(gp_coords, cov_pars[-1])
    covariance <- covariance + cov_pars[1] * diag(length(y))
    L <- t(chol(covariance))
    y_tilde <- forwardsolve(L, y)
    as.numeric(0.5 * crossprod(y_tilde) + sum(log(diag(L))) + length(y) / 2 * log(2 * pi))
  }

  simulate_ar1_mf_test_data <- function() {
    x_low <- seq(0.02, 0.98, length.out = 18)
    x_high <- seq(0.04, 0.96, length.out = 14) + 0.001
    stopifnot(length(intersect(x_low, x_high)) == 0)
    gp_coords <- rbind(cbind(x_low, 0), cbind(x_high, 1))
    cov_pars <- c(error_var = 0.08, low_var = 1.1, low_range = 0.25,
                  discrepancy_var = 0.5, discrepancy_range = 0.12, rho = -0.6)
    covariance <- ar1_mf_exponential_covariance(gp_coords, cov_pars[-1])
    latent <- drop(t(chol(covariance + 1e-10 * diag(nrow(gp_coords)))) %*% qnorm(sim_rand_unif(nrow(gp_coords), init_c = 0.8)))
    y_gaussian <- latent + sqrt(cov_pars[1]) * qnorm(sim_rand_unif(length(latent), init_c = 0.1))
    y_binary <- as.numeric(sim_rand_unif(length(latent), init_c = 0.2341) < pnorm(0.2 + latent))
    list(gp_coords = gp_coords, cov_pars = cov_pars, latent = latent,
         y_gaussian = y_gaussian, y_binary = y_binary)
  }

  test_that("exact Gaussian AR1 multifidelity likelihood agrees with R", {
    data <- simulate_ar1_mf_test_data()
    expected <- gaussian_nll_ar1_mf(data$y_gaussian, data$gp_coords, data$cov_pars)
    gp_model <- GPModel(
      gp_coords = data$gp_coords, cov_function = "ar1_mf_exponential", likelihood = "gaussian",
      gp_approx = "none", matrix_inversion_method = "cholesky"
    )
    actual <- gp_model$neg_log_likelihood(y = data$y_gaussian, cov_pars = data$cov_pars)
    expect_equal(actual, expected, tolerance = 1e-8)
    expect_equal(actual, 32.151882541105891, tolerance = 1e-8)

    invisible(capture.output(
      fit(gp_model, y = data$y_gaussian, params = list(
        init_cov_pars = data$cov_pars, optimizer_cov = "lbfgs", maxit = 100, trace = FALSE
      ))
    ))
    estimated_cov_pars <- as.numeric(gp_model$get_cov_pars())
    expected_cov_pars <- c(0.08322014059140172, 1.5703531702502687, 0.7399583067633473, 0.29026274917533995, 0.07650147001772063, -0.34035408255962485)
    expect_equal(estimated_cov_pars, expected_cov_pars, tolerance = 2e-4)
    expect_equal(as.numeric(gp_model$get_current_neg_log_likelihood()), 29.648872793557288, tolerance = 1e-5)

    prediction <- predict(
      gp_model, gp_coords_pred = matrix(c(0.333, 0, 0.777, 1), ncol = 2, byrow = TRUE),
      predict_var = TRUE
    )
    expect_equal(prediction$mu, c(1.2562002371164092, -0.6291054571606833), tolerance = 2e-4)
    expect_equal(prediction$var, c(0.17368271785874975, 0.24035170660934424), tolerance = 2e-4)
  })

  test_that("AR1 multifidelity supports covariance variants and both Vecchia selections", {
    data <- simulate_ar1_mf_test_data()

    model_ard <- GPModel(
      gp_coords = cbind(data$gp_coords[, 1], data$gp_coords[, 1]^2, data$gp_coords[, 2]),
      cov_function = "ar1_mf_matern_ard_estimate_shape", likelihood = "gaussian"
    )
    ard_cov_pars <- c(0.08, 1.1, 0.25, 0.4, 1.5, 0.5, 0.12, 0.3, 2.5, -0.6)
    expect_equal(model_ard$neg_log_likelihood(y = data$y_gaussian, cov_pars = ard_cov_pars), 33.334283736830095, tolerance = 1e-8)

    expected_nll <- c(vecchia = 32.401799696083145, vecchia_euclidean = 32.227109558939020)
    for (gp_approx in names(expected_nll)) {
      gp_model <- GPModel(
        gp_coords = data$gp_coords, cov_function = "ar1_mf_exponential", likelihood = "gaussian",
        gp_approx = gp_approx, num_neighbors = 6, vecchia_ordering = "none",
        matrix_inversion_method = "cholesky"
      )
      nll <- gp_model$neg_log_likelihood(y = data$y_gaussian, cov_pars = data$cov_pars)
      expect_equal(nll, expected_nll[[gp_approx]], tolerance = 1e-8)
    }
  })

  test_that("AR1 multifidelity rejects invalid fidelity indicators", {
    data <- simulate_ar1_mf_test_data()
    invalid_coords <- data$gp_coords
    invalid_coords[5, ncol(invalid_coords)] <- 2
    expect_error(
      GPModel(gp_coords = invalid_coords, cov_function = "ar1_mf_exponential", likelihood = "gaussian"),
      "must contain only 0 \\(low fidelity\\) and 1 \\(high fidelity\\)"
    )

    valid_model <- GPModel(
      gp_coords = data$gp_coords, cov_function = "ar1_mf_exponential", likelihood = "gaussian"
    )
    expect_error(
      predict(valid_model, y = data$y_gaussian, cov_pars = data$cov_pars,
              gp_coords_pred = matrix(c(0.5, -1), nrow = 1), predict_var = TRUE),
      "must contain only 0 \\(low fidelity\\) and 1 \\(high fidelity\\), found -1"
    )
  })

  test_that("non-Gaussian AR1 multifidelity works exactly and with Vecchia-Laplace", {
    data <- simulate_ar1_mf_test_data()
    process_cov_pars <- data$cov_pars[-1]

    exact_model <- GPModel(
      gp_coords = data$gp_coords, cov_function = "ar1_mf_exponential", likelihood = "bernoulli_probit",
      gp_approx = "none", matrix_inversion_method = "cholesky"
    )
    expect_equal(exact_model$neg_log_likelihood(y = data$y_binary, cov_pars = process_cov_pars), 20.102559552391824, tolerance = 1e-8)
    invisible(capture.output(
      fit(exact_model, y = data$y_binary, params = list(
        init_cov_pars = process_cov_pars, optimizer_cov = "lbfgs", maxit = 2,
        init_coef_aux_pars_from_iid_model = FALSE, trace = FALSE
      ))
    ))
    expect_equal(as.numeric(exact_model$get_cov_pars()), c(1.1993161569145947, 0.42481600526343305, 0.5515040869950153, 0.12864558193679757, 0.5711717180678637), tolerance = 1e-8)
    expect_equal(as.numeric(exact_model$get_current_neg_log_likelihood()), 16.364965896631873, tolerance = 1e-8)

    for (inversion_method in c("cholesky", "iterative")) {
      gp_model <- GPModel(
        gp_coords = data$gp_coords, cov_function = "ar1_mf_exponential", likelihood = "bernoulli_probit",
        gp_approx = "vecchia", num_neighbors = 6, vecchia_ordering = "none",
        matrix_inversion_method = inversion_method
      )
      optim_params <- list(
        init_cov_pars = process_cov_pars, optimizer_cov = "gradient_descent", lr_cov = 0.05,
        maxit = 2, init_coef_aux_pars_from_iid_model = FALSE, trace = FALSE
      )
      tol_pred <- 1e-8
      if (inversion_method == "iterative") {
        optim_params$num_rand_vec_trace <- 20
        optim_params$seed_rand_vec_trace <- 1
        optim_params$cg_delta_conv <- 1e-4
        optim_params$cg_max_num_it <- 100
        optim_params$cg_preconditioner_type <- "piv_chol_on_Sigma"
        optim_params$fitc_piv_chol_preconditioner_rank <- 10
        tol_pred <- 1e-3
      }
      if (inversion_method == "cholesky") {
        expect_equal(gp_model$neg_log_likelihood(y = data$y_binary, cov_pars = process_cov_pars), 20.118953180464363, tolerance = 1e-8)
      }
      invisible(capture.output(fit(gp_model, y = data$y_binary, params = optim_params)))
      expected_fit <- list(
        cholesky = list(cov_pars = c(1.1208846731049009, 0.2837165041795852, 0.5265774974772421, 0.1241828718058514, -0.1234287872320286),
                        nll = 17.743260933163558, mu = c(0.8232655588081717, 0.6753089735531903), var = c(0.1454993784884405, 0.2192667637917268)),
        iterative = list(cov_pars = c(1.1055593547491722, 0.2824705662808080, 0.5300671780510321, 0.1250140324130210, -0.1027052545776411),
                         nll = 17.577849124499139, mu = c(0.8237742637310388, 0.6794675882713714), var = c(0.1451702261454237, 0.2177913847600575)))
      expected <- expected_fit[[inversion_method]]
      expect_equal(as.numeric(gp_model$get_cov_pars()), expected$cov_pars, tolerance = 1e-8)
      expect_equal(as.numeric(gp_model$get_current_neg_log_likelihood()), expected$nll, tolerance = 1e-8)
      prediction <- predict(gp_model, gp_coords_pred = data$gp_coords[c(4, 20), , drop = FALSE], predict_var = TRUE, predict_response = TRUE)
      expect_equal(prediction$mu, expected$mu, tolerance = tol_pred)
      expect_equal(prediction$var, expected$var, tolerance = tol_pred)
    }
  })

  test_that("AR1 multifidelity can be used by the GPBoost algorithm", {
    data <- simulate_ar1_mf_test_data()
    features <- cbind(x = data$gp_coords[, 1], nonlinear = sin(4 * data$gp_coords[, 1]))
    dtrain <- gpb.Dataset(data = features, label = data$y_gaussian + 5 * data$gp_coords[, 2])
    gp_model <- GPModel(
      gp_coords = data$gp_coords, cov_function = "ar1_mf_exponential", likelihood = "gaussian",
      gp_approx = "vecchia", num_neighbors = 6, vecchia_ordering = "none"
    )
    gp_model$set_optim_params(params = list(
      init_cov_pars = data$cov_pars, init_coef_aux_pars_from_iid_model = FALSE
    ))

    booster <- gpb.train(
      data = dtrain, gp_model = gp_model, train_gp_model_cov_pars = FALSE, nrounds = 2,
      learning_rate = 0.1, max_depth = 2, min_data_in_leaf = 4, objective = "regression_l2", verbose = 0
    )
    expect_equal(dtrain$dim()[2], 3L)
    expect_equal(tail(dtrain$get_colnames(), 1), "AR1_MF_fidelity")
    prediction <- predict(booster, data = features[c(2, 20), , drop = FALSE], gp_coords_pred = data$gp_coords[c(2, 20), , drop = FALSE], predict_var = TRUE)
    expect_equal(prediction$response_mean, c(0.8817623475013590, 4.9223685785147211), tolerance = 1e-8)
    expect_equal(prediction$response_var, c(0.14145808652762754, 0.14515537122937899), tolerance = 1e-8)
    same_x_features <- features[c(5, 5), , drop = FALSE]
    same_x_coords <- rbind(c(data$gp_coords[5, 1], 0), c(data$gp_coords[5, 1], 1))
    tree_mean <- predict(booster, data = same_x_features, gp_coords_pred = same_x_coords, ignore_gp_model = TRUE)
    expect_equal(tree_mean, c(2.8165058065167723, 2.9922163527923269), tolerance = 1e-8)
  })

  test_that("AR1 multifidelity has independent low- and high-fidelity linear means", {
    data <- simulate_ar1_mf_test_data()
    fidelity <- data$gp_coords[, 2]
    X <- cbind(intercept = 1, x = data$gp_coords[, 1])
    y <- data$y_gaussian + ifelse(fidelity == 0, 1 + 0.5 * X[, "x"], -2 + 2 * X[, "x"])
    optim_params <- list(init_cov_pars = data$cov_pars, optimizer_cov = "lbfgs", maxit = 30, trace = FALSE)

    automatic <- GPModel(
      gp_coords = data$gp_coords, cov_function = "ar1_mf_exponential", likelihood = "gaussian"
    )
    manual <- GPModel(
      gp_coords = data$gp_coords, cov_function = "ar1_mf_exponential", likelihood = "gaussian",
      fidelity_specific_mean = FALSE
    )
    X_manual <- cbind(X * (1 - fidelity), X * fidelity)
    colnames(X_manual) <- c("intercept_low", "x_low", "intercept_high", "x_high")
    invisible(capture.output(fit(automatic, y = y, X = X, params = optim_params)))
    invisible(capture.output(fit(manual, y = y, X = X_manual, params = optim_params)))

    expect_equal(names(automatic$get_coef()), colnames(X_manual))
    expect_equal(as.numeric(automatic$get_coef()), as.numeric(manual$get_coef()), tolerance = 1e-8)
    expect_equal(automatic$get_cov_pars(), manual$get_cov_pars(), tolerance = 1e-8)
    expect_equal(automatic$get_current_neg_log_likelihood(), manual$get_current_neg_log_likelihood(), tolerance = 1e-8)
    expect_equal(as.numeric(automatic$get_coef()), c(1.8609293931525670, 1.4014499261271571, -2.5684875919979087, 2.4236913287346837), tolerance = 5e-4)
    expected_cov_pars <- c(0.1580295186369034, 0.1777984356023217, 0.1137185099659451, 0.0264454996525318, 0.0742742232476946, -1.2523925132081939)
    expect_equal(as.numeric(automatic$get_cov_pars()), expected_cov_pars, tolerance = 5e-4)
    expect_equal(as.numeric(automatic$get_current_neg_log_likelihood()), 26.250580774970221, tolerance = 1e-5)

    coords_pred <- matrix(c(0.25, 0, 0.25, 1), ncol = 2, byrow = TRUE)
    X_pred <- cbind(intercept = 1, x = coords_pred[, 1])
    X_pred_manual <- cbind(X_pred * (1 - coords_pred[, 2]), X_pred * coords_pred[, 2])
    pred_automatic <- predict(automatic, gp_coords_pred = coords_pred, X_pred = X_pred, predict_var = TRUE)
    pred_manual <- predict(manual, gp_coords_pred = coords_pred, X_pred = X_pred_manual, predict_var = TRUE)
    expect_equal(pred_automatic$mu, pred_manual$mu, tolerance = 1e-8)
    expect_equal(pred_automatic$var, pred_manual$var, tolerance = 1e-8)
    expect_equal(pred_automatic$mu, c(2.7069412065042058, -2.6291424253336420), tolerance = 5e-4)
    expect_equal(pred_automatic$var, c(0.20446944838870407, 0.23560931737560456), tolerance = 5e-4)
  })

  test_that("AR1 multifidelity likelihood matches R reference for rho outside (-1, 1)", {
    ar1_mf_combine <- function(gp_coords, cov_low, cov_discrepancy, rho) {
      fidelity <- gp_coords[, ncol(gp_coords)]
      loading <- ifelse(fidelity == 0, 1, rho)
      cov_low * outer(loading, loading) + cov_discrepancy * outer(fidelity, fidelity)
    }
    gaussian_nll_from_cov <- function(y, error_var, covariance) {
      covariance <- covariance + error_var * diag(length(y))
      L <- t(chol(covariance))
      y_tilde <- forwardsolve(L, y)
      as.numeric(0.5 * crossprod(y_tilde) + sum(log(diag(L))) + length(y) / 2 * log(2 * pi))
    }
    x_low <- seq(0.02, 0.98, length.out = 18)
    x_high <- seq(0.04, 0.96, length.out = 14) + 0.001
    gp_coords <- rbind(cbind(x_low, 0), cbind(x_high, 1))
    gp_model <- GPModel(
      gp_coords = gp_coords, cov_function = "ar1_mf_exponential", likelihood = "gaussian",
      gp_approx = "none", matrix_inversion_method = "cholesky"
    )

    # rho > 1: high fidelity amplifies the low-fidelity signal
    cov_pars_pos <- c(error_var = 0.05, low_var = 0.9, low_range = 0.3,
                       discrepancy_var = 0.4, discrepancy_range = 0.15, rho = 3.2)
    covariance_pos <- ar1_mf_exponential_covariance(gp_coords, cov_pars_pos[-1])
    latent_pos <- drop(t(chol(covariance_pos + 1e-10 * diag(nrow(gp_coords)))) %*% qnorm(sim_rand_unif(nrow(gp_coords), init_c = 0.67)))
    y_pos <- latent_pos + sqrt(cov_pars_pos[1]) * qnorm(sim_rand_unif(length(latent_pos), init_c = 0.31))
    expect_equal(gp_model$neg_log_likelihood(y = y_pos, cov_pars = cov_pars_pos), 34.088192143602271, tolerance = 1e-8)

    # rho << -1: high fidelity strongly anti-correlated with and amplified relative to low fidelity
    cov_pars_neg <- cov_pars_pos
    cov_pars_neg["rho"] <- -4.1
    covariance_neg <- ar1_mf_exponential_covariance(gp_coords, cov_pars_neg[-1])
    latent_neg <- drop(t(chol(covariance_neg + 1e-10 * diag(nrow(gp_coords)))) %*% qnorm(sim_rand_unif(nrow(gp_coords), init_c = 0.87)))
    y_neg <- latent_neg + sqrt(cov_pars_neg[1]) * qnorm(sim_rand_unif(length(latent_neg), init_c = 0.51))
    expected_neg <- gaussian_nll_from_cov(y_neg, cov_pars_neg[["error_var"]],
                                           ar1_mf_exponential_covariance(gp_coords, cov_pars_neg[-1]))
    actual_neg <- gp_model$neg_log_likelihood(y = y_neg, cov_pars = cov_pars_neg)
    expect_equal(actual_neg, expected_neg, tolerance = 1e-8)
    expect_equal(actual_neg, 60.287345447859956, tolerance = 1e-8)
  })

  test_that("AR1 multifidelity predictions are correct across low, high, and mixed fidelity queries", {
    data <- simulate_ar1_mf_test_data()
    gp_model <- GPModel(
      gp_coords = data$gp_coords, cov_function = "ar1_mf_exponential", likelihood = "gaussian",
      gp_approx = "none", matrix_inversion_method = "cholesky"
    )
    prediction <- predict(
      gp_model, y = data$y_gaussian, cov_pars = data$cov_pars,
      gp_coords_pred = matrix(
        c(0.10, 0, 0.50, 0, 0.10, 1, 0.50, 1, 0.90, 1), ncol = 2, byrow = TRUE
      ),
      predict_var = TRUE
    )
    expected_mu <- c(0.6832304306195364, 1.2397267461527930, -0.0886583175845477, -1.0299825257429802, 0.1133074174557247)
    expected_var <- c(0.226632553591379, 0.237126742823349, 0.234005258708113,
                       0.307218946490873, 0.221657615333974)
    expect_equal(prediction$mu, expected_mu, tolerance = 1e-6)
    expect_equal(prediction$var, expected_var, tolerance = 1e-6)
  })

  test_that("AR1 multifidelity supports linear and hurst base covariances", {
    ar1_mf_combine <- function(gp_coords, cov_low, cov_discrepancy, rho) {
      fidelity <- gp_coords[, ncol(gp_coords)]
      loading <- ifelse(fidelity == 0, 1, rho)
      cov_low * outer(loading, loading) + cov_discrepancy * outer(fidelity, fidelity)
    }
    gaussian_nll_from_cov <- function(y, error_var, covariance) {
      covariance <- covariance + error_var * diag(length(y))
      L <- t(chol(covariance))
      y_tilde <- forwardsolve(L, y)
      as.numeric(0.5 * crossprod(y_tilde) + sum(log(diag(L))) + length(y) / 2 * log(2 * pi))
    }
    data <- simulate_ar1_mf_test_data()
    x <- data$gp_coords[, 1, drop = FALSE]

    # linear base: 1 parameter (variance) per block, plus rho
    linear_cov_pars <- c(error_var = 0.08, low_var = 1.0, discrepancy_var = 0.6, rho = -0.6)
    cov_low_lin <- linear_cov_pars[["low_var"]] * (x %*% t(x))
    cov_disc_lin <- linear_cov_pars[["discrepancy_var"]] * (x %*% t(x))
    Sigma_lin <- ar1_mf_combine(data$gp_coords, cov_low_lin, cov_disc_lin, linear_cov_pars[["rho"]])
    expected_nll_linear <- gaussian_nll_from_cov(data$y_gaussian, linear_cov_pars[["error_var"]], Sigma_lin)
    gp_model_linear <- GPModel(
      gp_coords = data$gp_coords, cov_function = "ar1_mf_linear", likelihood = "gaussian",
      gp_approx = "none", matrix_inversion_method = "cholesky"
    )
    actual_nll_linear <- gp_model_linear$neg_log_likelihood(y = data$y_gaussian, cov_pars = linear_cov_pars)
    expect_equal(actual_nll_linear, expected_nll_linear, tolerance = 1e-8)
    expect_equal(actual_nll_linear, 98.314896711405126, tolerance = 1e-8)

    # hurst (fractional Brownian motion) base: variance and Hurst exponent per block, plus rho
    hurst_cov_matrix <- function(x, var, H) {
      n <- length(x)
      sqrd <- x^2
      var / 2 * (outer(sqrd^H, rep(1, n)) + outer(rep(1, n), sqrd^H) -
                   abs(outer(x, x, "-"))^(2 * H))
    }
    hurst_cov_pars <- c(error_var = 0.08, low_var = 1.0, low_H = 0.3,
                         discrepancy_var = 0.5, discrepancy_H = 0.6, rho = -0.6)
    cov_low_hurst <- hurst_cov_matrix(x[, 1], hurst_cov_pars[["low_var"]], hurst_cov_pars[["low_H"]])
    cov_disc_hurst <- hurst_cov_matrix(x[, 1], hurst_cov_pars[["discrepancy_var"]], hurst_cov_pars[["discrepancy_H"]])
    Sigma_hurst <- ar1_mf_combine(data$gp_coords, cov_low_hurst, cov_disc_hurst, hurst_cov_pars[["rho"]])
    expected_nll_hurst <- gaussian_nll_from_cov(data$y_gaussian, hurst_cov_pars[["error_var"]], Sigma_hurst)
    gp_model_hurst <- GPModel(
      gp_coords = data$gp_coords, cov_function = "ar1_mf_hurst", likelihood = "gaussian",
      gp_approx = "none", matrix_inversion_method = "cholesky"
    )
    actual_nll_hurst <- gp_model_hurst$neg_log_likelihood(y = data$y_gaussian, cov_pars = hurst_cov_pars)
    expect_equal(actual_nll_hurst, expected_nll_hurst, tolerance = 1e-8)
    expect_equal(actual_nll_hurst, 34.029297710299844, tolerance = 1e-8)
  })
}
