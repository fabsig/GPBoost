if(Sys.getenv("NO_GPBOOST_ALGO_TESTS") != "NO_GPBOOST_ALGO_TESTS"){

  context("GPBoost_tuning_bayesian_optimization")

  # Avoid that long tests get executed on CRAN
  if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

    # The order in which OpenMP adds up the partial sums of the threads can change the last digits of the scores
    TOLERANCE <- 1E-8
    # Function that simulates uniform random variables
    sim_rand_unif <- function(n, init_c=0.1){
      mod_lcg <- 134456 # modulus for linear congruential generator (random0 used)
      sim <- rep(NA, n)
      sim[1] <- floor(init_c * mod_lcg)
      for(i in 2:n) sim[i] <- (8121 * sim[i-1] + 28411) %% mod_lcg
      return(sim / mod_lcg)
    }
    n <- 200
    m <- 20
    X <- matrix(sim_rand_unif(2*n, init_c=0.3726154), ncol=2)
    group <- rep(1:m, each=n/m)
    b <- 0.5 * qnorm(sim_rand_unif(m, init_c=0.8153342))
    y <- sin(4*X[,1]) + X[,2] + b[group] + 0.3 * qnorm(sim_rand_unif(n, init_c=0.5281734))
    y_binary <- as.numeric(y > median(y))
    folds <- list(seq(1, n, by=2), seq(2, n, by=2))
    search_space <- list("learning_rate" = c(0.01, 1), "lambda_l2" = c(0, 100), "min_data_in_leaf" = c(1, 50),
                         "max_depth" = c(-1, -1), "line_search_step_length" = c(TRUE, FALSE))
    # bagging_fraction > 1 is invalid, so that some trials fail
    search_space_failures <- list("learning_rate" = c(0.01, 1), "bagging_fraction" = c(0.5, 1.5))
    params_binary <- list(objective = "binary", bagging_freq = 1, verbose = -1)
    cv_best_score <- function(params, label, metric, gp_model = NULL) {
      params$verbose <- -1
      # The same parameters for the construction of the histograms as in the tuning functions
      params$force_col_wise <- TRUE
      params$deterministic <- TRUE
      cvbst <- gpb.cv(params = params, data = gpb.Dataset(X, label = label),
                      gp_model = gp_model, nrounds = 50, early_stopping_rounds = 5, folds = folds,
                      metric = metric, verbose = 0)
      return(cvbst$best_score)
    }
    check_best_params <- function(best_params) {
      expect_equal(names(best_params), names(search_space))
      expect_gte(best_params$learning_rate, 0.01)
      expect_lte(best_params$learning_rate, 1)
      expect_gte(best_params$lambda_l2, 1e-8)
      expect_lte(best_params$lambda_l2, 100)
      expect_true(is.integer(best_params$min_data_in_leaf))
      expect_gte(best_params$min_data_in_leaf, 1)
      expect_lte(best_params$min_data_in_leaf, 50)
      expect_equal(best_params$max_depth, -1)
      expect_true(is.logical(best_params$line_search_step_length))
    }
    # Spatial data for choosing parameters of the GP model
    coords <- matrix(sim_rand_unif(2*n, init_c=0.6342817), ncol=2)
    b_spatial <- t(chol(exp(-as.matrix(dist(coords)) / 0.2) + 1e-10 * diag(n))) %*%
      qnorm(sim_rand_unif(n, init_c=0.2716403))
    y_spatial <- as.numeric(sin(4*X[,1]) + X[,2] + b_spatial + 0.3 * qnorm(sim_rand_unif(n, init_c=0.7419375)))
    spatial_gp_model <- function(cov_function = "matern", cov_fct_shape = 2.5) {
      gp_model <- GPModel(gp_coords = coords, cov_function = cov_function, cov_fct_shape = cov_fct_shape,
                          gp_approx = "vecchia", num_neighbors = 10, vecchia_ordering = "random", seed = 1)
      gp_model$set_prediction_data(vecchia_pred_type = "order_obs_first_cond_obs_only", num_neighbors_pred = 10)
      return(gp_model)
    }
    # Score of gpb.cv after 'best_iter' iterations with the parameters of the booster and the GP model chosen by a
    # tuning function
    cv_score_of_best_params <- function(best_params, best_iter) {
      is_gp_model_param <- names(best_params) %in% c("cov_fct_shape", "cov_function")
      params <- c(best_params[!is_gp_model_param], list(verbose = -1, force_col_wise = TRUE, deterministic = TRUE))
      cvbst <- gpb.cv(params = params, data = gpb.Dataset(X, label = y_spatial),
                      gp_model = do.call(spatial_gp_model, best_params[is_gp_model_param]), nrounds = best_iter,
                      folds = folds, metric = "test_neg_log_likelihood", verbose = 0)
      return(cvbst$record_evals$valid$test_neg_log_likelihood$eval[[best_iter]])
    }
    gp_model_field <- function(gp_model, name) {
      return(gp_model$.__enclos_env__$private[[name]])
    }

    test_that("Search spaces and parameters of the tuning functions", {

      parsed <- gpboost:::gpb.tuning.parse.search.space(list("min_gain_to_split" = c(0, 1), "min_data_in_leaf" = c(0, 50),
                                                             "lambda_l1" = c(0, 10), "num_leaves" = c(2, 1024)),
                                                        fun_name = "test")
      expect_equal(parsed$min_gain_to_split, list(kind = "float", lower = 1e-8, upper = 1, log = TRUE))
      expect_equal(parsed$min_data_in_leaf, list(kind = "int", lower = 1, upper = 50, log = TRUE))
      expect_equal(parsed$lambda_l1, list(kind = "float", lower = 1e-8, upper = 10, log = TRUE))
      expect_equal(parsed$num_leaves, list(kind = "int", lower = 2, upper = 1024, log = TRUE))
      expect_error(gpboost:::gpb.tuning.parse.search.space(list("min_data_in_leaf" = c(-1, 50)), fun_name = "test"),
                   "lower bound >= 0")
      expect_error(gpboost:::gpb.tuning.parse.search.space(list("num_leaves" = c(0, 1024)), fun_name = "test"),
                   "positive bounds")
      expect_error(gpboost:::gpb.tuning.parse.search.space(list("feature_fraction" = c(1, 0.5)), fun_name = "test"),
                   "not larger than the upper bound")
      # Sets of values, which are sorted if they are numbers, and parameters of the GP model
      parse_search_space <- function(search_space) {
        return(gpboost:::gpb.tuning.parse.search.space(search_space, fun_name = "test"))
      }
      parsed <- parse_search_space(list("cov_fct_shape" = list(values = c(2.5, 0.5, 1.5)),
                                        "cov_function" = list(values = c("matern", "gaussian")),
                                        "line_search_step_length" = c(TRUE, FALSE),
                                        "max_depth" = list(values = c(5, -1)), "cov_fct_order" = c(1, 2)))
      expect_equal(parsed$cov_fct_shape, list(kind = "ordinal", values = c(0.5, 1.5, 2.5)))
      expect_equal(parsed$cov_function, list(kind = "categorical", values = c("matern", "gaussian")))
      expect_equal(parsed$line_search_step_length, list(kind = "logical", values = c(TRUE, FALSE)))
      expect_equal(parsed$max_depth, list(kind = "ordinal", values = c(-1, 5)))
      expect_equal(parsed$cov_fct_order, list(kind = "int", lower = 1, upper = 2, log = FALSE))
      # A vector of two numbers is a range
      expect_equal(parse_search_space(list("cov_fct_shape" = c(0.5, 2.5)))$cov_fct_shape,
                   list(kind = "float", lower = 0.5, upper = 2.5, log = FALSE))
      expect_error(parse_search_space(list("cov_fct_shape" = list(value = c(0.5, 1.5)))),
                   "must be a vector c\\(lower, upper\\) or a list list\\(values")
      expect_error(parse_search_space(list("cov_fct_shape" = list(values = numeric(0)))), "at least one value")
      expect_error(parse_search_space(list("cov_fct_shape" = list(values = c(0.5, 1.5, 0.5)))),
                   "must not contain a value twice")
      expect_error(parse_search_space(list("cov_function" = c("matern", "gaussian"))),
                   "search_space\\['cov_function'\\] must be a set of values")
      expect_error(parse_search_space(list("num_neighbors" = list(values = c(10, 20)))),
                   "Unknown parameter 'num_neighbors'")
      # Parameters of the GP model require a GP model
      expect_error(gpb.grid.search.tune.parameters(param_grid = list("cov_fct_shape" = c(0.5, 1.5)),
                                                   data = gpb.Dataset(X, label = y)),
                   "'cov_fct_shape' is a parameter of the GP model and can only be tuned with a 'gp_model'")

      deterministic_params <- gpboost:::gpb.tuning.deterministic.params
      expect_equal(deterministic_params(list(), list(list())), list(force_col_wise = TRUE, deterministic = TRUE))
      # Parameters given by the user are not changed
      expect_equal(deterministic_params(list(force_row_wise = TRUE), list(list())),
                   list(force_row_wise = TRUE, deterministic = TRUE))
      expect_equal(deterministic_params(list(), list(list(force_col_wise = FALSE, deterministic = FALSE))), list())
      expect_equal(deterministic_params(list(), list(list(force_row_wise = c(TRUE, FALSE)), list())),
                   list(deterministic = TRUE))
    })

    test_that("Direction of the metric in 'gpb.grid.search.tune.parameters'", {

      # Whether higher scores are better is taken from the first metric of the cross-validation. A metric given in
      #   'params' was minimized when higher values are better, and so was a custom metric with 'higher_better' = TRUE
      #   when the built-in metric was disabled
      params_auc <- list(objective = "binary", verbose = -1)
      param_grid <- list("learning_rate" = c(0.01, 0.5))
      grid_search <- function(params, ...) {
        gpb.grid.search.tune.parameters(param_grid = param_grid, data = gpb.Dataset(X, label = y_binary), params = params,
                                        nrounds = 50, early_stopping_rounds = 5, folds = folds, verbose_eval = 0, ...)
      }
      auc_scores <- sapply(param_grid$learning_rate, function(learning_rate) {
        cv_best_score(c(params_auc, list(learning_rate = learning_rate)), label = y_binary, metric = "auc")
      })
      opt_params_metric <- grid_search(params_auc, metric = "auc")
      expect_lt(abs(opt_params_metric$best_score - max(auc_scores)), TOLERANCE)
      expect_equal(grid_search(c(params_auc, list(metric = "auc"))), opt_params_metric)
      agreement <- function(preds, dtrain) {
        return(list(name = "agreement", value = mean((preds > 0.5) == getinfo(dtrain, "label")), higher_better = TRUE))
      }
      agreement_scores <- sapply(param_grid$learning_rate, function(learning_rate) {
        cvbst <- gpb.cv(params = c(params_auc, list(learning_rate = learning_rate, force_col_wise = TRUE, deterministic = TRUE)),
                        data = gpb.Dataset(X, label = y_binary), nrounds = 50, early_stopping_rounds = 5, folds = folds,
                        metric = "None", eval = agreement, verbose = 0)
        max(unlist(cvbst$record_evals$valid$agreement$eval))
      })
      opt_params_eval <- grid_search(params_auc, metric = "None", eval = agreement)
      expect_lt(abs(opt_params_eval$best_score - max(agreement_scores)), TOLERANCE)
    })

    test_that("Parameters of the GP model in 'gpb.grid.search.tune.parameters'", {

      # The smoothness of 'gp_model' is not in the grid, so that the scores show whether the GP models of the folds use
      #   the smoothness of the parameter combinations
      gp_model <- spatial_gp_model(cov_fct_shape = 2.5)
      opt_params <- gpb.grid.search.tune.parameters(param_grid = list("cov_fct_shape" = c(0.5, 1.5),
                                                                      "cov_function" = c("matern", "exponential")),
                                                    data = gpb.Dataset(X, label = y_spatial), gp_model = gp_model,
                                                    params = list(learning_rate = 0.1, verbose = -1), nrounds = 50,
                                                    early_stopping_rounds = 5, folds = folds,
                                                    metric = "test_neg_log_likelihood", verbose_eval = 0,
                                                    return_all_combinations = TRUE)
      expect_equal(gp_model_field(gp_model, "cov_fct_shape"), 2.5)
      expect_equal(gp_model_field(gp_model, "cov_function"), "matern")
      expect_length(opt_params$all_combinations, 4L)
      for (combination in opt_params$all_combinations) {
        score <- cv_score_of_best_params(c(combination$params, list(learning_rate = 0.1)), combination$nrounds)
        expect_lt(abs(combination$score - score), TOLERANCE)
      }
      expect_true(opt_params$best_params$cov_fct_shape %in% c(0.5, 1.5))
      # The exponential covariance function does not depend on the smoothness
      scores_exponential <- sapply(Filter(function(combination) combination$params$cov_function == "exponential",
                                          opt_params$all_combinations), function(combination) combination$score)
      expect_lt(abs(scores_exponential[1] - scores_exponential[2]), TOLERANCE)
    })

    test_that("Other covariance functions and failed combinations in 'gpb.grid.search.tune.parameters'", {

      # Covariance functions with more and with fewer covariance parameters than the one of 'gp_model'
      for (setting in list(list(cov_function = "matern", cov_functions = c("matern", "matern_ard")),
                           list(cov_function = "matern_ard", cov_functions = "matern"))) {
        gp_model <- spatial_gp_model(cov_function = setting$cov_function)
        num_cov_pars <- gp_model_field(gp_model, "num_cov_pars")
        opt_params <- gpb.grid.search.tune.parameters(param_grid = list("cov_function" = setting$cov_functions),
                                                      data = gpb.Dataset(X, label = y_spatial), gp_model = gp_model,
                                                      params = list(learning_rate = 0.1, verbose = -1), nrounds = 50,
                                                      early_stopping_rounds = 5, folds = folds,
                                                      metric = "test_neg_log_likelihood", verbose_eval = 0,
                                                      return_all_combinations = TRUE)
        expect_equal(gp_model_field(gp_model, "cov_function"), setting$cov_function)
        expect_equal(gp_model_field(gp_model, "num_cov_pars"), num_cov_pars)
        for (combination in opt_params$all_combinations) {
          score <- cv_score_of_best_params(c(combination$params, list(learning_rate = 0.1)), combination$nrounds)
          expect_lt(abs(combination$score - score), TOLERANCE)
        }
      }
      # A mask that estimates all covariance parameters is the same as no mask
      grid_scores <- function(gp_model) {
        opt_params <- gpb.grid.search.tune.parameters(param_grid = list("cov_function" = c("matern", "matern_ard")),
                                                      data = gpb.Dataset(X, label = y_spatial), gp_model = gp_model,
                                                      params = list(learning_rate = 0.1, verbose = -1), nrounds = 50,
                                                      early_stopping_rounds = 5, folds = folds,
                                                      metric = "test_neg_log_likelihood", verbose_eval = 0,
                                                      return_all_combinations = TRUE)
        return(sapply(opt_params$all_combinations, function(combination) combination$score))
      }
      gp_model <- spatial_gp_model()
      gp_model$set_optim_params(params = list(estimate_cov_par_index = c(1, 1, 1)))
      expect_lt(max(abs(grid_scores(gp_model) - grid_scores(spatial_gp_model()))), TOLERANCE)
      # A combination whose GP models cannot be constructed has no score, and the search continues
      time <- matrix(seq_len(n) / n, ncol = 1)
      capture.output(expect_message(
        opt_params <- gpb.grid.search.tune.parameters(param_grid = list("cov_function" = c("matern", "hurst"),
                                                                        "cov_fct_order" = c(1, 2)),
                                                      data = gpb.Dataset(X, label = y_spatial),
                                                      gp_model = GPModel(gp_coords = time, cov_function = "hurst"),
                                                      params = list(learning_rate = 0.1, verbose = -1), nrounds = 3,
                                                      folds = folds, metric = "test_neg_log_likelihood",
                                                      verbose_eval = 0, return_all_combinations = TRUE),
        "'cov_fct_order' = 2 is not supported for the 'matern' covariance function"), file = "NUL")
      expect_length(opt_params$all_combinations, 4L)
      for (combination in opt_params$all_combinations) {
        if (combination$params$cov_function == "matern" && combination$params$cov_fct_order == 2) {
          expect_true(is.na(combination$score))
        } else {
          expect_true(is.finite(combination$score))
        }
      }
    })

    test_that("Parameters of the GP model that cannot be tuned with 'gp_model'", {

      grid_search <- function(param_grid, gp_model, label = y_spatial) {
        gpb.grid.search.tune.parameters(param_grid = param_grid, data = gpb.Dataset(X, label = label),
                                        gp_model = gp_model, params = list(learning_rate = 0.1, verbose = -1),
                                        nrounds = 10, folds = folds, metric = "test_neg_log_likelihood",
                                        verbose_eval = 0, return_all_combinations = TRUE)
      }
      expect_error(grid_search(list("cov_function" = c("matern", "exponential")), GPModel(group_data = group)),
                   "can only be tuned with a 'gp_model' that has a Gaussian process")
      # The number and the meaning of the covariance parameters depend on 'cov_function' and 'cov_fct_order'
      gp_model <- spatial_gp_model()
      gp_model$set_optim_params(params = list(init_cov_pars = c(0.1, 1, 0.2)))
      expect_error(grid_search(list("cov_function" = c("matern", "exponential")), gp_model),
                   "cannot be tuned with initial values 'init_cov_pars'")
      gp_model <- spatial_gp_model()
      gp_model$set_optim_params(params = list(estimate_cov_par_index = c(1, 0, 1)))
      expect_error(grid_search(list("cov_fct_order" = c(1, 2)), gp_model),
                   "or with covariance parameters that are not estimated")
      expect_error(grid_search(list("cov_function" = c(1, 2)), spatial_gp_model()),
                   "must be names of covariance functions")
      # With a fidelity-specific mean, the trees have the fidelity indicator as an additional feature, and covariance
      #   functions with and without it cannot be compared in one search
      fidelity <- rep(c(0, 1), n / 2)
      coords_mf <- cbind(coords, fidelity)
      y_mf <- y_spatial + 5 * fidelity
      expect_error(grid_search(list("cov_function" = c("ar1_mf_matern", "matern_ard")),
                               GPModel(gp_coords = coords_mf, cov_function = "ar1_mf_matern"), label = y_mf),
                   "all values of 'cov_function' must then be AR1 multifidelity")
      expect_error(grid_search(list("cov_function" = "ar1_mf_matern"),
                               GPModel(gp_coords = coords_mf, cov_function = "matern_ard"), label = y_mf),
                   "is an AR1 multifidelity covariance function and can only be tuned")
      opt_params <- grid_search(list("cov_function" = c("ar1_mf_matern", "ar1_mf_exponential")),
                                GPModel(gp_coords = coords_mf, cov_function = "ar1_mf_matern"), label = y_mf)
      # A search with a fidelity-specific mean adds the fidelity indicator to the gpb.Dataset as a feature, and a
      #   later search without it must not use this gpb.Dataset
      data_mf <- gpb.Dataset(X, label = y_mf)
      gpb.grid.search.tune.parameters(param_grid = list("cov_function" = "ar1_mf_matern"), data = data_mf,
                                      gp_model = GPModel(gp_coords = coords_mf, cov_function = "ar1_mf_matern"),
                                      params = list(learning_rate = 0.1, verbose = -1), nrounds = 2, folds = folds,
                                      metric = "test_neg_log_likelihood", verbose_eval = 0)
      expect_error(gpb.grid.search.tune.parameters(param_grid = list("cov_function" = "matern_ard"), data = data_mf,
                                                   gp_model = GPModel(gp_coords = coords_mf,
                                                                      cov_function = "matern_ard"),
                                                   params = list(learning_rate = 0.1, verbose = -1), nrounds = 2,
                                                   folds = folds, metric = "test_neg_log_likelihood",
                                                   verbose_eval = 0),
                   "Use a new gpb.Dataset for a model without a fidelity-specific mean")
      for (combination in opt_params$all_combinations) {
        cvbst <- gpb.cv(params = list(learning_rate = 0.1, verbose = -1, force_col_wise = TRUE, deterministic = TRUE),
                        data = gpb.Dataset(X, label = y_mf),
                        gp_model = GPModel(gp_coords = coords_mf, cov_function = combination$params$cov_function),
                        nrounds = 10, folds = folds, metric = "test_neg_log_likelihood", verbose = 0)
        expect_lt(abs(combination$score - cvbst$best_score), TOLERANCE)
      }
    })

    test_that("Choosing tuning parameters with the approximation of SMAC in 'gpb.tune.pars.rf.bayesian.optimization'", {

      # 'mlr3' and 'mlr' (used by 'gpb.tune.pars.bayesian.optimization') warn when they are loaded in the same session
      suppressWarnings({ requireNamespace("mlr3mbo", quietly = TRUE); requireNamespace("mlr3learners", quietly = TRUE) })
      skip_if_not_installed("mlr3mbo")
      skip_if_not_installed("mlr3learners")
      skip_if_not_installed("ranger")
      logger_thresholds <- function() {
        index <- lgr::logger_index()
        return(index[index$name %in% c("mlr3/bbotk", "mlr3/core"), c("name", "threshold", "threshold_inherited")])
      }
      thresholds_before <- logger_thresholds()
      gp_model <- GPModel(group_data = group)
      expect_silent(
        opt_params <- gpb.tune.pars.rf.bayesian.optimization(search_space = search_space, n_trials = 12,
                                                             data = gpb.Dataset(X, label = y), gp_model = gp_model,
                                                             params = list(verbose = -1), nrounds = 50,
                                                             early_stopping_rounds = 5, folds = folds, metric = "mse",
                                                             cv_seed = 1, verbose_eval = 0)
      )
      expect_equal(logger_thresholds(), thresholds_before)
      check_best_params(opt_params$best_params)
      expect_gte(opt_params$best_iter, 1)
      best_score <- cv_best_score(opt_params$best_params, label = y, metric = "mse",
                                  gp_model = GPModel(group_data = group))
      expect_lt(abs(opt_params$best_score - best_score), TOLERANCE)
      # The same 'cv_seed' gives the same result, also with a matrix and a label instead of a gpb.Dataset
      opt_params_2 <- gpb.tune.pars.rf.bayesian.optimization(search_space = search_space, n_trials = 12,
                                                             data = X, label = y, gp_model = gp_model,
                                                             params = list(verbose = -1), nrounds = 50,
                                                             early_stopping_rounds = 5, folds = folds, metric = "mse",
                                                             cv_seed = 1, verbose_eval = 0)
      expect_equal(opt_params_2, opt_params)

      # Metric for which higher is better, and failed trials
      capture.output(
        messages <- capture_messages(
          opt_params <- gpb.tune.pars.rf.bayesian.optimization(search_space = search_space_failures, n_trials = 15,
                                                               data = gpb.Dataset(X, label = y_binary),
                                                               params = params_binary, nrounds = 50,
                                                               early_stopping_rounds = 5, folds = folds,
                                                               metric = "auc", cv_seed = 2, verbose_eval = 1))
        , file = "NUL")
      expect_true(any(grepl("Error in trial number", messages)))
      expect_lte(opt_params$best_params$bagging_fraction, 1)
      best_score <- cv_best_score(c(params_binary, opt_params$best_params), label = y_binary, metric = "auc")
      expect_lt(abs(opt_params$best_score - best_score), TOLERANCE)
      # A metric given in 'params' gives the same result. It was minimized when higher values are better
      tune_auc <- function(params, ...) {
        gpb.tune.pars.rf.bayesian.optimization(search_space = list("learning_rate" = c(0.01, 1)), n_trials = 4,
                                               data = gpb.Dataset(X, label = y_binary), params = params, nrounds = 50,
                                               early_stopping_rounds = 5, folds = folds, cv_seed = 2, verbose_eval = 0, ...)
      }
      params_auc <- list(objective = "binary", verbose = -1)
      opt_params <- tune_auc(params_auc, metric = "auc")
      expect_equal(tune_auc(c(params_auc, list(metric = "auc"))), opt_params)
      best_score <- cv_best_score(c(params_auc, opt_params$best_params), label = y_binary, metric = "auc")
      expect_lt(abs(opt_params$best_score - best_score), TOLERANCE)
      # No valid trial at all
      capture.output(
        expect_warning(
          opt_params <- gpb.tune.pars.rf.bayesian.optimization(search_space = list("learning_rate" = c(0.01, 1),
                                                                                   "bagging_fraction" = c(1.5, 3)),
                                                               n_trials = 3, data = gpb.Dataset(X, label = y_binary),
                                                               params = params_binary, nrounds = 10, folds = folds,
                                                               metric = "auc", verbose_eval = 0),
          "Found no parameter combination")
        , file = "NUL")
      expect_true(is.na(opt_params$best_score))

      # Parameters of the GP model. The smoothness of 'gp_model' is not in the search space, so that the scores show
      #   whether the GP models of the folds use the smoothness of the trials
      gp_model <- spatial_gp_model(cov_fct_shape = 2.5)
      tune_spatial <- function(search_space) {
        # GPBoost warns about an almost constant GP in trials whose trees fit the data closely
        capture.output(
          opt_params <- gpb.tune.pars.rf.bayesian.optimization(search_space = search_space, n_trials = 6,
                                                               data = gpb.Dataset(X, label = y_spatial),
                                                               gp_model = gp_model, params = list(verbose = -1),
                                                               nrounds = 50, early_stopping_rounds = 5,
                                                               folds = folds, metric = "test_neg_log_likelihood",
                                                               cv_seed = 1, verbose_eval = 0)
          , file = "NUL")
        return(opt_params)
      }
      search_space_spatial <- list("learning_rate" = c(0.01, 1), "min_data_in_leaf" = c(1, 50))
      opt_params <- tune_spatial(c(search_space_spatial, list("cov_fct_shape" = list(values = c(0.5, 1.5)))))
      expect_true(opt_params$best_params$cov_fct_shape %in% c(0.5, 1.5))
      expect_equal(gp_model_field(gp_model, "cov_fct_shape"), 2.5)
      best_score <- cv_score_of_best_params(opt_params$best_params, opt_params$best_iter)
      expect_lt(abs(opt_params$best_score - best_score), TOLERANCE)
      # A parameter of the GP model with the single value of 'gp_model' is held fixed and gives the same trials as
      #   without it
      opt_params <- tune_spatial(search_space_spatial)
      opt_params_fixed <- tune_spatial(c(search_space_spatial, list("cov_fct_shape" = list(values = 2.5))))
      expect_equal(opt_params_fixed, list(best_params = c(opt_params$best_params, list(cov_fct_shape = 2.5)),
                                          best_iter = opt_params$best_iter, best_score = opt_params$best_score))

      # Invalid search spaces
      expect_error(gpb.tune.pars.rf.bayesian.optimization(search_space = list("not_a_parameter" = c(0, 1)),
                                                          n_trials = 3, data = gpb.Dataset(X, label = y)),
                   "Unknown parameter")
      expect_error(gpb.tune.pars.rf.bayesian.optimization(search_space = list("lambda_l1" = c(-1, 1)),
                                                          n_trials = 3, data = gpb.Dataset(X, label = y)),
                   "lower bound >= 0")
      expect_error(gpb.tune.pars.rf.bayesian.optimization(search_space = list("learning_rate" = c(0, 1)),
                                                          n_trials = 3, data = gpb.Dataset(X, label = y)),
                   "positive bounds")
      expect_error(gpb.tune.pars.rf.bayesian.optimization(search_space = list("max_depth" = c(-1, -1)),
                                                          n_trials = 3, data = gpb.Dataset(X, label = y)),
                   "no parameter with two different values")
    })

    test_that("Choosing tuning parameters with 'mlrMBO' in 'gpb.tune.pars.bayesian.optimization'", {

      # 'mlr' and 'mlr3' (used by 'gpb.tune.pars.rf.bayesian.optimization') warn when they are loaded in the same session
      suppressWarnings(requireNamespace("mlrMBO", quietly = TRUE))
      skip_if_not_installed("mlrMBO")
      skip_if_not_installed("DiceKriging")
      skip_if_not_installed("rgenoud")
      skip_if_not_installed("randomForest")
      # If 'paradox' (used by 'gpb.tune.pars.rf.bayesian.optimization') is loaded after 'ParamHelpers', its S3 method
      # c.ParamSet replaces the one of 'ParamHelpers'
      if (suppressMessages(suppressWarnings(requireNamespace("paradox", quietly = TRUE)))) {
        c_param_set <- getS3method("c", "ParamSet")
        on.exit(registerS3method("c", "ParamSet", c_param_set, envir = environment(c_param_set)), add = TRUE)
        registerS3method("c", "ParamSet", get("c.ParamSet", envir = asNamespace("paradox")),
                         envir = asNamespace("paradox"))
      }
      gp_model <- GPModel(group_data = group)
      expect_silent(
        opt_params <- gpb.tune.pars.bayesian.optimization(search_space = search_space, n_iter = 2,
                                                          data = gpb.Dataset(X, label = y), gp_model = gp_model,
                                                          params = list(verbose = -1), nrounds = 50,
                                                          early_stopping_rounds = 5, folds = folds, metric = "mse",
                                                          cv_seed = 1, verbose_eval = 0)
      )
      check_best_params(opt_params$best_params)
      expect_gte(opt_params$best_iter, 1)
      best_score <- cv_best_score(opt_params$best_params, label = y, metric = "mse",
                                  gp_model = GPModel(group_data = group))
      expect_lt(abs(opt_params$best_score - best_score), TOLERANCE)
      # The same 'cv_seed' gives the same result, also with a matrix and a label instead of a gpb.Dataset
      opt_params_2 <- gpb.tune.pars.bayesian.optimization(search_space = search_space, n_iter = 2,
                                                          data = X, label = y, gp_model = gp_model,
                                                          params = list(verbose = -1), nrounds = 50,
                                                          early_stopping_rounds = 5, folds = folds, metric = "mse",
                                                          cv_seed = 1, verbose_eval = 0)
      expect_equal(opt_params_2, opt_params)

      # Numeric search space (Gaussian process as surrogate model), metric for which higher is better, and failed trials
      capture.output(
        expect_warning(
          opt_params <- gpb.tune.pars.bayesian.optimization(search_space = search_space_failures, n_iter = 4,
                                                            data = gpb.Dataset(X, label = y_binary),
                                                            params = params_binary, nrounds = 50,
                                                            early_stopping_rounds = 5, folds = folds, metric = "auc",
                                                            cv_seed = 1, verbose_eval = 0),
          "Found NA or Inf score")
        , file = "NUL")
      expect_lte(opt_params$best_params$bagging_fraction, 1)
      best_score <- cv_best_score(c(params_binary, opt_params$best_params), label = y_binary, metric = "auc")
      expect_lt(abs(opt_params$best_score - best_score), TOLERANCE)
      # A metric given in 'params' gives the same result. It was minimized when higher values are better
      tune_auc <- function(params, ...) {
        gpb.tune.pars.bayesian.optimization(search_space = list("learning_rate" = c(0.01, 1)), n_iter = 1,
                                            data = gpb.Dataset(X, label = y_binary), params = params, nrounds = 50,
                                            early_stopping_rounds = 5, folds = folds, cv_seed = 1, verbose_eval = 0, ...)
      }
      params_auc <- list(objective = "binary", verbose = -1)
      opt_params <- tune_auc(params_auc, metric = "auc")
      expect_equal(tune_auc(c(params_auc, list(metric = "auc"))), opt_params)
      best_score <- cv_best_score(c(params_auc, opt_params$best_params), label = y_binary, metric = "auc")
      expect_lt(abs(opt_params$best_score - best_score), TOLERANCE)

      # A parameter with a single value is held fixed and not part of the initial design. The Gaussian process
      # failed with such a parameter after the initial design ("Domains[,1] must be less than or equal to
      # Domains[,2]")
      capture.output(
        messages <- capture_messages(
          opt_params <- gpb.tune.pars.bayesian.optimization(search_space = list("learning_rate" = c(0.01, 1),
                                                                                "max_depth" = c(-1, -1),
                                                                                "min_data_in_leaf" = c(1, 50)),
                                                            n_iter = 2, data = gpb.Dataset(X, label = y),
                                                            params = list(verbose = -1), nrounds = 50,
                                                            early_stopping_rounds = 5, folds = folds, metric = "mse",
                                                            cv_seed = 1, verbose_eval = 1))
        , file = "NUL")
      expect_equal(sum(grepl("^Trial number", messages)), 4 * 2 + 2)
      expect_equal(names(opt_params$best_params), c("learning_rate", "max_depth", "min_data_in_leaf"))
      expect_equal(opt_params$best_params$max_depth, -1)
      best_score <- cv_best_score(opt_params$best_params, label = y, metric = "mse")
      expect_lt(abs(opt_params$best_score - best_score), TOLERANCE)
      expect_error(gpb.tune.pars.bayesian.optimization(search_space = list("max_depth" = c(-1, -1)), n_iter = 2,
                                                       data = gpb.Dataset(X, label = y)),
                   "no parameter with two different values")

      # Parameters of the GP model. A set of numbers is searched as the index of a value, so that the surrogate model
      #   is a Gaussian process. The smoothness of 'gp_model' is not in the search space, so that the scores show
      #   whether the GP models of the folds use the smoothness of the trials
      gp_model <- spatial_gp_model(cov_fct_shape = 2.5)
      opt_params <- gpb.tune.pars.bayesian.optimization(search_space = list("learning_rate" = c(0.01, 1),
                                                                            "cov_fct_shape" = list(values = c(0.5, 1.5))),
                                                        n_iter = 2, data = gpb.Dataset(X, label = y_spatial),
                                                        gp_model = gp_model, params = list(verbose = -1),
                                                        nrounds = 50, early_stopping_rounds = 5, folds = folds,
                                                        metric = "test_neg_log_likelihood", cv_seed = 1,
                                                        verbose_eval = 0)
      expect_true(opt_params$best_params$cov_fct_shape %in% c(0.5, 1.5))
      expect_equal(gp_model_field(gp_model, "cov_fct_shape"), 2.5)
      best_score <- cv_score_of_best_params(opt_params$best_params, opt_params$best_iter)
      expect_lt(abs(opt_params$best_score - best_score), TOLERANCE)
    })

  }

}
