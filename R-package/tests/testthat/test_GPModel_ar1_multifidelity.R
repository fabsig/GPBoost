context("GPModel_ar1_multifidelity")

if (Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS") {

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
    set.seed(123)
    x_low <- seq(0.02, 0.98, length.out = 18)
    x_high <- seq(0.04, 0.96, length.out = 14) + 0.001
    stopifnot(length(intersect(x_low, x_high)) == 0)
    gp_coords <- rbind(cbind(x_low, 0), cbind(x_high, 1))
    cov_pars <- c(error_var = 0.08, low_var = 1.1, low_range = 0.25,
                  discrepancy_var = 0.5, discrepancy_range = 0.12, rho = -0.6)
    covariance <- ar1_mf_exponential_covariance(gp_coords, cov_pars[-1])
    latent <- drop(t(chol(covariance + 1e-10 * diag(nrow(gp_coords)))) %*% rnorm(nrow(gp_coords)))
    y_gaussian <- latent + sqrt(cov_pars[1]) * rnorm(length(latent))
    y_binary <- as.numeric(runif(length(latent)) < pnorm(0.2 + latent))
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
    expect_equal(actual, 32.843208523135765, tolerance = 1e-8)

    invisible(capture.output(
      fit(gp_model, y = data$y_gaussian, params = list(
        init_cov_pars = data$cov_pars, optimizer_cov = "lbfgs", maxit = 100, trace = FALSE
      ))
    ))
    estimated_cov_pars <- as.numeric(gp_model$get_cov_pars())
    expected_cov_pars <- c(1.6226420647333633e-05, 0.64422745553918837, 0.096317186820072093,
                            0.51478018722899233, 0.12669344832918286, -0.44936100175738442)
    expect_lt(max(abs(estimated_cov_pars - expected_cov_pars)), 2e-4)
    expect_equal(as.numeric(gp_model$get_current_neg_log_likelihood()), 31.721852291669705, tolerance = 1e-5)

    prediction <- predict(
      gp_model, gp_coords_pred = matrix(c(0.333, 0, 0.777, 1), ncol = 2, byrow = TRUE),
      predict_var = TRUE
    )
    expect_equal(prediction$mu, c(1.4843761294148727, -0.53011299586035843), tolerance = 2e-4)
    expect_equal(prediction$var, c(0.17147309962836466, 0.17759532068270897), tolerance = 2e-4)
  })

  test_that("AR1 multifidelity supports covariance variants and both Vecchia selections", {
    data <- simulate_ar1_mf_test_data()

    model_ard <- GPModel(
      gp_coords = cbind(data$gp_coords[, 1], data$gp_coords[, 1]^2, data$gp_coords[, 2]),
      cov_function = "ar1_mf_matern_ard_estimate_shape", likelihood = "gaussian"
    )
    ard_cov_pars <- c(0.08, 1.1, 0.25, 0.4, 1.5, 0.5, 0.12, 0.3, 2.5, -0.6)
    expect_true(is.finite(model_ard$neg_log_likelihood(y = data$y_gaussian, cov_pars = ard_cov_pars)))

    expected_nll <- c(vecchia = 32.917232164339424, vecchia_euclidean = 32.942553605415199)
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
    expect_equal(exact_model$neg_log_likelihood(y = data$y_binary, cov_pars = process_cov_pars),
                 21.554398898146594, tolerance = 1e-8)
    invisible(capture.output(
      fit(exact_model, y = data$y_binary, params = list(
        init_cov_pars = process_cov_pars, optimizer_cov = "lbfgs", maxit = 2,
        init_coef_aux_pars_from_iid_model = FALSE, trace = FALSE
      ))
    ))
    expect_true(all(is.finite(exact_model$get_cov_pars())))

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
      if (inversion_method == "iterative") {
        optim_params$num_rand_vec_trace <- 20
        optim_params$seed_rand_vec_trace <- 1
        optim_params$cg_delta_conv <- 1e-4
        optim_params$cg_max_num_it <- 100
        optim_params$cg_preconditioner_type <- "piv_chol_on_Sigma"
        optim_params$fitc_piv_chol_preconditioner_rank <- 10
      }
      if (inversion_method == "cholesky") {
        expect_equal(gp_model$neg_log_likelihood(y = data$y_binary, cov_pars = process_cov_pars),
                     21.54373959495182, tolerance = 1e-8)
      }
      invisible(capture.output(fit(gp_model, y = data$y_binary, params = optim_params)))
      expect_true(all(is.finite(gp_model$get_cov_pars())))
      expect_true(is.finite(gp_model$get_current_neg_log_likelihood()))
      prediction <- predict(
        gp_model, gp_coords_pred = data$gp_coords[c(4, 20), , drop = FALSE],
        predict_var = TRUE, predict_response = TRUE
      )
      expect_true(all(is.finite(prediction$mu)))
      expect_true(all(is.finite(prediction$var)))
    }
  })

  test_that("AR1 multifidelity can be used by the GPBoost algorithm", {
    data <- simulate_ar1_mf_test_data()
    features <- cbind(x = data$gp_coords[, 1], nonlinear = sin(4 * data$gp_coords[, 1]))
    dtrain <- gpb.Dataset(data = features, label = data$y_gaussian)
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
    prediction <- predict(
      booster, data = features[c(2, 18), , drop = FALSE],
      gp_coords_pred = data$gp_coords[c(2, 18), , drop = FALSE], predict_var = TRUE
    )
    expect_true(all(is.finite(prediction$response_mean)))
    expect_true(all(is.finite(prediction$response_var)))
  })

  test_that("AR1 multifidelity ARD fit converges to a hard-wired stationary point", {
    gp_coords <- structure(c(
      0.955893761711195, 0.937285519437864, 0.238220451399684, 0.255073635373265, 0.390511967940256,
      0.341179859358817, 0.452380596427247, 0.289932820713148, 0.45067322277464, 0.806595719885081,
      0.60572106949985, 0.362841087626293, 0.766338474582881, 0.0450869468040764, 0.592171875061467,
      0.201632188400254, 0.632738025626168, 0.403589994646609, 0.290503408294171, 0.640459010610357,
      0.635967755690217, 0.989878454012796, 0.931174009572715, 0.485504753654823, 0.574689255096018,
      0.751238248776644, 0.992722376249731, 0.430864384630695, 0.124365017516539, 0.594260291894898,
      0.0629573746118695, 0.0644475566223264, 0.661190658109263, 0.104733102722093, 0.737024179426953,
      0.0752035917248577, 0.890810017939657, 0.889870930695906, 0.43858266249299, 0.095246899407357,
      0.545594914350659, 0.400917568709701, 0.0129982747603208, 0.926055597607046, 0.949058262864128,
      0.785141906002536, 0.436708099208772, 0.905955827562138, 0.511275585507974, 0.197399229044095,
      0.062189121497795, 0.132820880273357, 0.531148725189269, 0.978559362702072, 0.598859366727993,
      0.310912406072021, 0.210504578659311, 0.710253312950954, 0.67582987062633, 0.992905060993508,
      0.662878743140027, 0.438706010812894, 0.718227664474398, 0.025583088863641, 0.672329963650554,
      0.540953136282042, 0.820495622931048, 0.276118888054043, 0.457370079820976, 0.86474969657138,
      0.838505795458332, 0.342684569535777, 0.141806389205158, 0.905463349539787, 0.224220921751112,
      0.176713337190449, 0.537842873018235, 0.280659310054034, 0.00987849663943052, 0.0968077331781387,
      0.661849477794021, 0.677335348911583, 0.131287127966061, 0.262729513226077, 0.317505860934034,
      0.676745156291872, 0.596418351167813, 0.791821458842605, 0.788068760652095, 0.651753669138998,
      0.63360236515291, 0.165338897379115, 0.60207345453091, 0.956934875342995, 0.829927456565201,
      0.52693268051371, 0.872108481591567, 0.913770709419623, 0.162486956687644, 0.494465354364365,
      0.181502339197323, 0.085233764257282, 0.186976805329323, 0.996793107828125, 0.401548811234534,
      0.945521813584492, 0.625838312087581, 0.692089836578816, 0.252907125279307, 0.41514985030517,
      0.15650732954964, 0.263633264228702,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ), dim = c(56L, 3L))
    y <- c(
      -0.613184061549911, -0.256904032816329, -1.35974595229133, -2.05304752507736, -0.970038857498489,
      -0.808795283365785, -0.415318339819361, -0.624193103343754, -0.360988371733806, -0.479669828852826,
      -0.571053994479074, -1.10157420741037, -1.16096509818449, -1.81939496940961, -0.462806020065931,
      -1.06681234585014, -1.2391635549264, -0.890681068068505, -0.742792829263907, -1.20502330341894,
      -1.00703168765364, -0.442602681076671, -0.860841314759546, -0.591205925952818, -0.420243259316959,
      -0.445913181920937, -0.653222989102103, -1.14254196818598, -0.676702039418635, -0.544373330909535,
      -2.30386069437313, -2.62268574187269, 0.163172787712591, -2.91894598714203, -0.210870387840498,
      -0.520676538187388, -0.401864762070673, 1.19684169901942, -1.15350431859263, -2.12330728385027,
      -0.782903987926529, -1.70362960439862, 0.270369146774122, -1.23465913927951, -0.736043558043401,
      -1.11958902915861, -2.9547194547813, 1.42654875765076, -2.32805509962523, -2.58946693956982,
      -2.14021269849876, -3.11528497153199, -2.65426851818763, -0.813719628897388, -2.3732690140449,
      -2.01890238169795
    )
    gp_model <- GPModel(
      gp_coords = gp_coords, cov_function = "ar1_mf_matern_ard", cov_fct_shape = 1.5,
      likelihood = "gaussian", gp_approx = "none"
    )
    init_cov_pars <- c(0.02, 1.0, 0.4, 0.6, 1.2, 0.3, 0.5, 0.6)
    invisible(capture.output(
      fit(gp_model, y = y, params = list(
        init_cov_pars = init_cov_pars, optimizer_cov = "lbfgs", maxit = 1000,
        delta_rel_conv = 1e-10, trace = FALSE
      ))
    ))
    estimated_cov_pars <- as.numeric(gp_model$get_cov_pars())
    expected_cov_pars <- c(0.025309051614217, 0.917459549944310, 0.496494013463690, 0.777494649812674,
                            0.374030755190022, 0.190590368990344, 0.396667804765932, 1.526349408593240)
    expect_lt(max(abs(estimated_cov_pars - expected_cov_pars)), 2e-3)
    expect_equal(as.numeric(gp_model$get_current_neg_log_likelihood()), 26.665631966241800, tolerance = 1e-5)
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
    set.seed(99)
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
    latent_pos <- drop(t(chol(covariance_pos + 1e-10 * diag(nrow(gp_coords)))) %*% rnorm(nrow(gp_coords)))
    y_pos <- latent_pos + sqrt(cov_pars_pos[1]) * rnorm(length(latent_pos))
    expect_equal(gp_model$neg_log_likelihood(y = y_pos, cov_pars = cov_pars_pos),
                 39.027809432467400, tolerance = 1e-8)

    # rho << -1: high fidelity strongly anti-correlated with and amplified relative to low fidelity
    cov_pars_neg <- cov_pars_pos
    cov_pars_neg["rho"] <- -4.1
    covariance_neg <- ar1_mf_exponential_covariance(gp_coords, cov_pars_neg[-1])
    latent_neg <- drop(t(chol(covariance_neg + 1e-10 * diag(nrow(gp_coords)))) %*% rnorm(nrow(gp_coords)))
    y_neg <- latent_neg + sqrt(cov_pars_neg[1]) * rnorm(length(latent_neg))
    expected_neg <- gaussian_nll_from_cov(y_neg, cov_pars_neg[["error_var"]],
                                           ar1_mf_exponential_covariance(gp_coords, cov_pars_neg[-1]))
    actual_neg <- gp_model$neg_log_likelihood(y = y_neg, cov_pars = cov_pars_neg)
    expect_equal(actual_neg, expected_neg, tolerance = 1e-8)
    expect_equal(actual_neg, 36.187324153896100, tolerance = 1e-8)
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
    expected_mu <- c(0.094604114348692, -0.260689478905627, 0.093127238112009,
                      -0.949093221793138, -0.541655403108458)
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
    expect_equal(actual_nll_linear, 84.233731345027000, tolerance = 1e-8)

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
    expect_equal(actual_nll_hurst, 38.745523614557300, tolerance = 1e-8)
  })
}
