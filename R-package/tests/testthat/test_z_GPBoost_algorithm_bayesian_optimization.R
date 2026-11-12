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

      deterministic_params <- gpboost:::gpb.tuning.deterministic.params
      expect_equal(deterministic_params(list(), list(list())), list(force_col_wise = TRUE, deterministic = TRUE))
      # Parameters given by the user are not changed
      expect_equal(deterministic_params(list(force_row_wise = TRUE), list(list())),
                   list(force_row_wise = TRUE, deterministic = TRUE))
      expect_equal(deterministic_params(list(), list(list(force_col_wise = FALSE, deterministic = FALSE))), list())
      expect_equal(deterministic_params(list(), list(list(force_row_wise = c(TRUE, FALSE)), list())),
                   list(deterministic = TRUE))
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
      # The same 'cv_seed' gives the same result
      opt_params_2 <- gpb.tune.pars.rf.bayesian.optimization(search_space = search_space, n_trials = 12,
                                                             data = gpb.Dataset(X, label = y), gp_model = gp_model,
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
      # The same 'cv_seed' gives the same result
      opt_params_2 <- gpb.tune.pars.bayesian.optimization(search_space = search_space, n_iter = 2,
                                                          data = gpb.Dataset(X, label = y), gp_model = gp_model,
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
    })

  }

}
