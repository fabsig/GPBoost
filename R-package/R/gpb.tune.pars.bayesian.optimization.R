# Functions for choosing tuning parameters with Bayesian optimization

# Parameters that can be tuned, grouped by their type and scale
gpb.tuning.params.float.log <- c("learning_rate", "shrinkage_rate")
# Parameters on a logarithmic scale for which 0 is a valid value. Since a logarithmic scale excludes 0, a lower
# bound of 0 is replaced by 1e-8, which practically has the same effect
gpb.tuning.params.float.log.zero <- c("lambda_l2", "reg_lambda", "lambda",
                                      "lambda_l1", "reg_alpha",
                                      "min_gain_to_split", "min_split_gain",
                                      "min_sum_hessian_in_leaf", "min_sum_hessian_per_leaf", "min_sum_hessian",
                                      "min_hessian", "min_child_weight")
gpb.tuning.params.float <- c("bagging_fraction", "sub_row", "subsample", "bagging",
                             "feature_fraction", "sub_feature", "colsample_bytree",
                             "cat_l2",
                             "cat_smooth")
gpb.tuning.params.int.log <- c("num_leaves", "num_leaf", "max_leaves", "max_leaf",
                               "max_bin")
# Integer parameters on a logarithmic scale for which 0 is a valid value. A lower bound of 0 is replaced by 1
gpb.tuning.params.int.log.zero <- c("min_data_in_leaf", "min_data_per_leaf", "min_data", "min_child_samples")
gpb.tuning.params.int <- c("max_depth")
gpb.tuning.params.logical <- c("line_search_step_length")

# Determines the type, the bounds, and the scale of every parameter in a search space.
# Returns a list with an entry list(kind, lower, upper, log) for every parameter, where kind is
# "float", "int", or "logical"
gpb.tuning.parse.search.space <- function(search_space, fun_name) {
  if (!is.list(search_space) || is.null(names(search_space)) || any(names(search_space) == "")) {
    stop(fun_name, ": 'search_space' needs to be a named list")
  }
  parsed <- list()
  for (param in names(search_space)) {
    bounds <- search_space[[param]]
    if (length(bounds) != 2L) {
      stop(sprintf("%s: search_space['%s'] must have length 2", fun_name, param))
    }
    lower <- bounds[1L]
    upper <- bounds[2L]
    if (param %in% c(gpb.tuning.params.float.log.zero, gpb.tuning.params.int.log.zero)) {
      if (lower < 0 || upper <= 0) {
        stop(sprintf("%s: search_space['%s'] must have a lower bound >= 0 and an upper bound > 0", fun_name, param))
      }
      if (param %in% gpb.tuning.params.float.log.zero) {
        parsed[[param]] <- list(kind = "float", lower = max(lower, 1e-8), upper = upper, log = TRUE)
      } else {
        parsed[[param]] <- list(kind = "int", lower = max(lower, 1), upper = upper, log = TRUE)
      }
    } else if (param %in% c(gpb.tuning.params.float.log, gpb.tuning.params.int.log)) {
      if (lower <= 0 || upper <= 0) {
        stop(sprintf("%s: search_space['%s'] must have positive bounds since '%s' is searched on a logarithmic scale",
                     fun_name, param, param))
      }
      kind <- if (param %in% gpb.tuning.params.float.log) "float" else "int"
      parsed[[param]] <- list(kind = kind, lower = lower, upper = upper, log = TRUE)
    } else if (param %in% gpb.tuning.params.float) {
      parsed[[param]] <- list(kind = "float", lower = lower, upper = upper, log = FALSE)
    } else if (param %in% gpb.tuning.params.int) {
      parsed[[param]] <- list(kind = "int", lower = lower, upper = upper, log = FALSE)
    } else if (param %in% gpb.tuning.params.logical) {
      parsed[[param]] <- list(kind = "logical", lower = as.logical(lower), upper = as.logical(upper), log = FALSE)
    } else {
      stop(sprintf("%s: Unknown parameter '%s'", fun_name, param))
    }
    if (parsed[[param]]$kind != "logical" && parsed[[param]]$lower > parsed[[param]]$upper) {
      stop(sprintf("%s: search_space['%s'] must have a lower bound that is not larger than the upper bound",
                   fun_name, param))
    }
  }
  return(parsed)
}

# Adds the parameters that make the score of a parameter combination reproducible, unless they are given by the user
# in 'params' or in the other lists of parameters in 'other_params'. Without 'force_col_wise' or 'force_row_wise',
# the histograms of the trees are constructed column-wise or row-wise, depending on which is faster in a time
# measurement; the two give slightly different scores. The column-wise construction is also independent of the
# number of threads. 'deterministic' sums the initial score of the boosting sequentially
gpb.tuning.deterministic.params <- function(params, other_params = list()) {
  given <- c(names(params), unlist(lapply(other_params, names)))
  if (!any(c("force_col_wise", "force_row_wise") %in% given)) {
    params[["force_col_wise"]] <- TRUE
  }
  if (!("deterministic" %in% given)) {
    params[["deterministic"]] <- TRUE
  }
  return(params)
}

# Whether higher values of the metric are better
gpb.tuning.higher.better <- function(metric, eval) {
  higher_better <- FALSE
  if (is.null(metric)) {
    if (!is.null(eval)) {
      if (!is.list(eval)) {
        eval <- list(eval)
      }
      if (is.function(eval[[1L]])) {
        dummy_data <- gpb.Dataset(data = matrix(c(0, 1)), label = c(0, 1))
        dummy_data$construct()
        higher_better <- eval[[1L]](0, dummy_data)$higher_better
      } else if (any(startsWith(eval[[1L]], c("auc", "ndcg@", "map@", "average_precision")))) {
        higher_better <- TRUE
      }
    }
  } else if (any(startsWith(metric, c("auc", "ndcg@", "map@", "average_precision")))) {
    higher_better <- TRUE
  }
  return(higher_better)
}

# Runs cross-validation for one parameter combination and returns the best score, the corresponding
# number of boosting iterations, and the name of the metric. If 'cv_seed' is given, the folds are generated
# with this seed in every trial, and the state of the random number generator is restored afterwards so
# that the random numbers of the optimizer are not affected
gpb.tuning.cv.score <- function(params, data, gp_model, nrounds, early_stopping_rounds, folds, nfold,
                                metric, eval, cv_seed, verbose_cv, use_gp_model_for_validation,
                                train_gp_model_cov_pars, categorical_feature, ...) {
  if (!is.null(cv_seed)) {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      saved_seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
      on.exit(assign(".Random.seed", saved_seed, envir = globalenv()), add = TRUE)
    }
    set.seed(cv_seed)
  }
  cvbst <- gpb.cv(data = data, gp_model = gp_model, params = params, nrounds = nrounds,
                  early_stopping_rounds = early_stopping_rounds, folds = folds, nfold = nfold,
                  metric = metric, eval = eval, verbose = verbose_cv,
                  use_gp_model_for_validation = use_gp_model_for_validation,
                  train_gp_model_cov_pars = train_gp_model_cov_pars,
                  categorical_feature = categorical_feature, delete_boosters_folds = TRUE, ...)
  return(list(score = cvbst$best_score, best_iter = cvbst$best_iter,
              metric_name = names(cvbst$record_evals$valid)))
}

gpb.tuning.params.to.string <- function(params) {
  return(paste0(names(params), ": ", unlist(lapply(params, format)), collapse = ", "))
}

# Loads the namespaces of the packages and stops if one of them is not installed. The packages of 'mlr' (used by
# 'mlrMBO') and 'mlr3' (used by 'mlr3mbo') warn when both are loaded in one session, since 'ParamHelpers' and
# 'paradox' both register the S3 method c.ParamSet. These warnings and the message about the overwritten method are
# muffled, since the tuning functions do not call c() on parameter sets
gpb.tuning.check.packages <- function(packages, fun_name) {
  is_conflict_warning <- function(w) {
    msg <- conditionMessage(w)
    return(grepl("Packages 'paradox' and 'ParamHelpers' are conflicting", msg, fixed = TRUE) ||
             grepl("Packages\\s+'mlr3'\\s+and\\s+'mlr'\\s+are\\s+conflicting", msg))
  }
  is_c_paramset_message <- function(m) {
    msg <- conditionMessage(m)
    return(grepl("^Registered S3 method overwritten by '(paradox|ParamHelpers)'", msg) &&
             grepl("c.ParamSet", msg, fixed = TRUE))
  }
  is_loaded <- withCallingHandlers(
    vapply(packages, requireNamespace, logical(1L), quietly = TRUE),
    warning = function(w) if (is_conflict_warning(w)) invokeRestart("muffleWarning"),
    message = function(m) if (is_c_paramset_message(m)) invokeRestart("muffleMessage"))
  missing <- packages[!is_loaded]
  if (length(missing) > 0L) {
    stop(fun_name, " requires the package(s) ", paste0("'", missing, "'", collapse = ", "),
         ". They can be installed with install.packages(c(", paste0("\"", missing, "\"", collapse = ", "), "))")
  }
}

#' @name gpb.tune.pars.bayesian.optimization
#' @title Function for choosing tuning parameters using Bayesian optimization with the 'mlrMBO' package
#' @description Function for choosing tuning parameters using Bayesian optimization (model-based optimization)
#' as implemented in the \code{mbo} function of the 'mlrMBO' package, using cross-validation or validation data sets.
#' With the default settings of 'mlrMBO', the surrogate model is a Gaussian process (kriging) if all parameters
#' in the search space are numeric and a random forest if the search space contains the logical parameter
#' \code{line_search_step_length}, and the next parameter combination is chosen with a (lower) confidence bound.
#' This function requires the packages 'mlrMBO', 'ParamHelpers', and 'smoof', and, for the default surrogate models
#' of 'mlrMBO', the packages 'DiceKriging' and 'rgenoud' (Gaussian process) and 'randomForest'
#' (random forest). For an approximation of SMAC, see \code{\link{gpb.tune.pars.rf.bayesian.optimization}}.
#' @param search_space \code{list} with ranges for every parameter over which a search is done.
#' The format for every entry of the list must be "parameter_name" = c(lower, upper).
#' The parameters 'learning_rate', 'min_gain_to_split', 'min_sum_hessian_in_leaf', 'lambda_l1', 'lambda_l2',
#' 'num_leaves', 'min_data_in_leaf', and 'max_bin' (and their aliases) are searched on a logarithmic scale,
#' all other parameters on a linear scale. A lower bound of 0 is replaced by 1e-8 for 'min_gain_to_split', 'min_sum_hessian_in_leaf', 'lambda_l1', and 'lambda_l2' and by 1 for 'min_data_in_leaf'.
#' A parameter with lower = upper is held fixed at this value.
#' See https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#tuning-parameters--hyperparameters-for-the-tree-boosting-part
#' @param n_iter Number of sequential Bayesian optimization steps after the initial design. The initial design is
#' the default of 'mlrMBO', a Latin hypercube design with 4 d parameter combinations, where d is the number of
#' parameters that are searched
#' @param params \code{list} with other parameters not included in \code{search_space}
#' @param crit \code{MBOInfillCrit}. How should infill points be rated.
#' See infillcrits in the 'mlrMBO' package for an overview of available infill criteria or implement a custom one via makeMBOInfillCrit.
#' Default is ``(lower) confidence bound" (see makeMBOInfillCritCB).
#' @inheritParams gpb_shared_params
#' @param ... other parameters, see Parameters.rst for more information.
#' @inheritSection gpb_shared_params Early Stopping
#' @return         A \code{list} with the best parameter combination and score
#' The list has the following format:
#'  list("best_params" = best_params, "best_iter" = best_iter, "best_score" = best_score)
#' @details The random numbers of the optimization are drawn from the random number generator of R.
#' If \code{cv_seed} is given, \code{set.seed(cv_seed)} is called at the beginning, and the cross-validation folds
#' are the same in every trial. A parameter combination for which the cross-validation fails or gives a score that
#' is NA or infinite gets a score that is worse than all valid scores.
#'
#' The parameters \code{force_col_wise = TRUE} and \code{deterministic = TRUE} are used unless \code{params} or
#' \code{...} contain \code{force_col_wise}, \code{force_row_wise}, or \code{deterministic}, respectively.
#' Without \code{force_col_wise} and \code{force_row_wise}, the histograms of the trees are constructed column-wise
#' or row-wise depending on a time measurement, so that the score of a parameter combination can differ slightly
#' between runs, and the Bayesian optimization can then choose other parameter combinations.
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' \donttest{
#' if (requireNamespace("mlrMBO", quietly = TRUE) && requireNamespace("DiceKriging", quietly = TRUE) &&
#'     requireNamespace("rgenoud", quietly = TRUE) && requireNamespace("randomForest", quietly = TRUE)) {
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' n <- length(y)
#' # Define search space
#' # Note: if the best combination found below is close to the bounday for a paramter,
#' #   you might want to extend the corresponding range
#' search_space <- list("learning_rate" = c(0.001, 10),
#'                      "min_data_in_leaf" = c(1, 1000),
#'                      "max_depth" = c(-1, -1),
#'                      "num_leaves" = c(2, 2^10),
#'                      "lambda_l2" = c(0, 100),
#'                      "max_bin" = c(63, min(n,10000)),
#'                      "line_search_step_length" = c(TRUE, FALSE))
#' # Note: "max_depth" = c(-1, -1) means no depth limit as we tune 'num_leaves'.
#' #    Can also additionally tune 'max_depth', e.g., "max_depth" = c(-1, 10)
#' metric = "mse" # Define metric
#' # Note: can also use metric = "test_neg_log_likelihood".
#' # See https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
#' gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
#' data_train <- gpb.Dataset(data = X, label = y)
#' # Run parameter optimization using Bayesian optimization and 5-fold CV
#' opt_params <- gpb.tune.pars.bayesian.optimization(search_space = search_space, n_iter = 100,
#'                                                   data = data_train, gp_model = gp_model,
#'                                                   nfold = 5, nrounds = 1000, early_stopping_rounds = 20,
#'                                                   metric = metric, cv_seed = 4, verbose_eval = 1)
#' print(paste0("Best parameters: ",
#'              paste0(unlist(lapply(seq_along(opt_params$best_params),
#'                                   function(y, n, i) { paste0(n[[i]],": ", y[[i]]) },
#'                                   y=opt_params$best_params,
#'                                   n=names(opt_params$best_params))), collapse=", ")))
#' print(paste0("Best number of iterations: ", opt_params$best_iter))
#' print(paste0("Best score: ", round(opt_params$best_score, digits=3)))
#' }
#' }
#' @author Fabio Sigrist
#' @export
gpb.tune.pars.bayesian.optimization <- function(search_space
                                                , n_iter
                                                , data
                                                , gp_model = NULL
                                                , params = list()
                                                , nrounds = 1000L
                                                , early_stopping_rounds = NULL
                                                , folds = NULL
                                                , nfold = 5L
                                                , metric = NULL
                                                , verbose_eval = 1L
                                                , cv_seed = NULL
                                                , crit = NULL
                                                , use_gp_model_for_validation = TRUE
                                                , train_gp_model_cov_pars = TRUE
                                                , eval = NULL
                                                , categorical_feature = NULL
                                                , ...
) {
  fun_name <- "gpb.tune.pars.bayesian.optimization"
  gpb.tuning.check.packages(c("mlrMBO", "ParamHelpers", "smoof"), fun_name = fun_name)
  parsed_search_space <- gpb.tuning.parse.search.space(search_space, fun_name = fun_name)
  if (!is.numeric(n_iter) || length(n_iter) != 1L || n_iter <= 0) {
    stop(fun_name, ": 'n_iter' needs to be a positive integer")
  }
  verbose_eval <- if (is.null(verbose_eval)) 0L else as.integer(verbose_eval)
  verbose_cv <- if (verbose_eval < 2L) 0L else 1L
  if (is.null(params)) {
    params <- list()
  }
  params <- gpb.tuning.deterministic.params(params, other_params = list(list(...)))
  if (!is.null(cv_seed)) {
    set.seed(cv_seed)
  }
  higher_better <- gpb.tuning.higher.better(metric = metric, eval = eval)

  # Parameter set of 'ParamHelpers'. Parameters with a single value are held fixed and not searched: the
  # Gaussian process of 'mlrMBO' does not support a parameter without range, and it would enlarge the initial
  # design. Parameters on a logarithmic scale are searched as their logarithm with the prefix 'log_'. The parameters
  # are not combined with c(), since the S3 method c.ParamSet of 'paradox' replaces the one of 'ParamHelpers'
  # when 'paradox' is loaded afterwards
  fixed_params <- list()
  par_list <- list()
  for (param in names(parsed_search_space)) {
    spec <- parsed_search_space[[param]]
    if (spec$lower == spec$upper) {
      fixed_params[[param]] <- spec$lower
      next
    }
    if (spec$kind == "logical") {
      new_param <- ParamHelpers::makeLogicalParam(param)
    } else if (spec$log) {
      new_param <- ParamHelpers::makeNumericParam(paste0("log_", param), lower = log(spec$lower),
                                                  upper = log(spec$upper))
    } else if (spec$kind == "int") {
      new_param <- ParamHelpers::makeIntegerParam(param, lower = spec$lower, upper = spec$upper)
    } else {
      new_param <- ParamHelpers::makeNumericParam(param, lower = spec$lower, upper = spec$upper)
    }
    par_list[[length(par_list) + 1L]] <- new_param
  }
  if (length(par_list) == 0L) {
    stop(fun_name, ": 'search_space' contains no parameter with two different values")
  }
  par_set <- ParamHelpers::makeParamSet(params = par_list)

  best_score <- if (higher_better) -1E99 else 1E99
  worst_score_init <- best_score
  worst_score <- worst_score_init
  has_found_nonNA_score <- FALSE
  has_found_two_nonNA_scores <- FALSE
  best_params <- list()
  best_iter <- nrounds
  counter_num_comb <- 1L

  objective_fun <- function(params_trial) {
    params_comb <- list()
    for (param in names(params_trial)) {
      if (startsWith(param, "log_")) {
        param_name <- substr(param, 5L, nchar(param))
        params_comb[[param_name]] <- exp(params_trial[[param]])
        if (parsed_search_space[[param_name]]$kind == "int") {
          params_comb[[param_name]] <- as.integer(ceiling(params_comb[[param_name]]))
        }
      } else if (parsed_search_space[[param]]$kind == "logical") {
        params_comb[[param]] <- as.logical(params_trial[[param]])
      } else {
        params_comb[[param]] <- params_trial[[param]]
      }
    }
    params_comb <- c(params_comb, fixed_params)[names(parsed_search_space)]
    params_loc <- params
    params_loc[names(params_comb)] <- params_comb
    if (verbose_eval >= 1L) {
      message(paste0("Trial number ", counter_num_comb, " with the parameters: ",
                     gpb.tuning.params.to.string(params_comb)))
    }
    cv_result <- tryCatch(
      gpb.tuning.cv.score(params = params_loc, data = data, gp_model = gp_model, nrounds = nrounds,
                          early_stopping_rounds = early_stopping_rounds, folds = folds, nfold = nfold,
                          metric = metric, eval = eval, cv_seed = cv_seed, verbose_cv = verbose_cv,
                          use_gp_model_for_validation = use_gp_model_for_validation,
                          train_gp_model_cov_pars = train_gp_model_cov_pars,
                          categorical_feature = categorical_feature, ...),
      error = function(err) {
        if (verbose_eval >= 1L) {
          message(paste0("Error in trial number ", counter_num_comb, ": ", conditionMessage(err)))
        }
        NULL
      })
    score <- if (is.null(cv_result)) NA_real_ else cv_result$score
    current_score_is_better <- FALSE
    if (is.na(score) || is.infinite(score)) {
      # Impute the score of a failed trial with a score that is worse than all valid scores
      score <- worst_score_init
      if (has_found_two_nonNA_scores) {
        score <- if (higher_better) worst_score - (best_score - worst_score) else worst_score + (worst_score - best_score)
      } else if (has_found_nonNA_score) {
        score <- if (higher_better) worst_score - abs(worst_score) else worst_score + abs(worst_score)
      }
      if (has_found_nonNA_score) {
        warning(paste0("Found NA or Inf score in trial ", counter_num_comb, ". This was imputed with ", score))
      }
    } else {
      score_capped <- FALSE
      if (higher_better) {
        current_score_is_better <- score > best_score
        # Cap very bad scores so that they do not dominate the surrogate model
        if (has_found_two_nonNA_scores) {
          if (score < (worst_score - 10 * (best_score - worst_score))) {
            score <- worst_score - 10 * (best_score - worst_score)
            score_capped <- TRUE
          }
        } else if (has_found_nonNA_score) {
          if (score < (worst_score - 10 * abs(worst_score))) {
            score <- worst_score - 10 * abs(worst_score)
            score_capped <- TRUE
          }
        }
        if (score < worst_score || worst_score == worst_score_init) {
          worst_score <<- score
        }
      } else {
        current_score_is_better <- score < best_score
        if (has_found_two_nonNA_scores) {
          if (score > (worst_score + 10 * (worst_score - best_score))) {
            score <- worst_score + 10 * (worst_score - best_score)
            score_capped <- TRUE
          }
        } else if (has_found_nonNA_score) {
          if (score > (worst_score + 10 * abs(worst_score))) {
            score <- worst_score + 10 * abs(worst_score)
            score_capped <- TRUE
          }
        }
        if (score > worst_score || worst_score == worst_score_init) {
          worst_score <<- score
        }
      }
      if (score_capped) {
        warning(paste0("Found very bad score in trial ", counter_num_comb, ". This was replaced with ", score))
      }
      if (has_found_nonNA_score) {
        has_found_two_nonNA_scores <<- TRUE
      }
      has_found_nonNA_score <<- TRUE
    }
    if (current_score_is_better) {
      best_score <<- score
      best_iter <<- cv_result$best_iter
      best_params <<- params_comb
      if (verbose_eval >= 1L) {
        message(paste0("***** New best test score (", cv_result$metric_name, " = ", score,
                       ") found for the following parameter combination: ",
                       gpb.tuning.params.to.string(c(params_comb, list(nrounds = cv_result$best_iter)))))
      }
    }
    counter_num_comb <<- counter_num_comb + 1L
    return(score)
  }

  obj_fun <- smoof::makeSingleObjectiveFunction(name = "gpboost_cv", fn = objective_fun, par.set = par_set,
                                                minimize = !higher_better)
  control <- mlrMBO::makeMBOControl()
  control <- mlrMBO::setMBOControlTermination(control, iters = n_iter)
  if (is.null(crit)) {
    crit <- mlrMBO::makeMBOInfillCritCB()
  }
  control <- mlrMBO::setMBOControlInfill(control, crit = crit)
  mlrMBO::mbo(obj_fun, control = control, show.info = FALSE)
  if (!has_found_nonNA_score) {
    warning("Found no parameter combination with a score that is not NA or Inf ")
  }
  return(list(best_params = best_params, best_iter = best_iter, best_score = best_score))
}

#' @name gpb.tune.pars.rf.bayesian.optimization
#' @title Function for choosing tuning parameters using an approximation of SMAC with the 'mlr3mbo' package
#' @description Function for choosing tuning parameters using an approximation of SMAC (Sequential Model-based
#' Algorithm Configuration, Hutter et al., 2011) implemented with the 'mlr3mbo' package, using cross-validation
#' or validation data sets. SMAC is a Bayesian optimization method with a random forest as surrogate model.
#' This function is not the SMAC3 software (Lindauer et al., 2022), which is available in the Python package
#' as \code{tune_pars_SMAC}, but an approximation of it with the components of SMAC3 for hyperparameter optimization:
#' \itemize{
#' \item a Sobol initial design with \code{max(1, min(10 * d, n_trials / 4))} parameter combinations,
#' where d is the number of parameters that are searched,
#' \item a random forest for the logarithm of the score as surrogate model,
#' \item expected improvement as acquisition function, which is maximized with a local search,
#' \item a randomly chosen parameter combination in every fifth trial after the initial design.
#' }
#' The random forest is the default random forest of 'mlr3mbo' ('ranger' package with 500 trees and the law
#' of total variance for the uncertainty), not the random forest of SMAC3 with 10 trees.
#' In a comparison of tuning methods for tree-boosting (Koster and Sigrist, 2026), SMAC performed best on average.
#' This function requires the packages 'mlr3mbo', 'mlr3learners', 'ranger', 'bbotk', 'paradox', and 'lgr'.
#' @param search_space \code{list} with ranges for every parameter over which a search is done.
#' The format for every entry of the list must be "parameter_name" = c(lower, upper).
#' The parameters 'learning_rate', 'min_gain_to_split', 'min_sum_hessian_in_leaf', 'lambda_l1', 'lambda_l2',
#' 'num_leaves', 'min_data_in_leaf', and 'max_bin' (and their aliases) are searched on a logarithmic scale,
#' all other parameters on a linear scale. A lower bound of 0 is replaced by 1e-8 for 'min_gain_to_split', 'min_sum_hessian_in_leaf', 'lambda_l1', and 'lambda_l2' and by 1 for 'min_data_in_leaf'.
#' A parameter with lower = upper is held fixed at this value.
#' See https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#tuning-parameters--hyperparameters-for-the-tree-boosting-part
#' @param n_trials Number of parameter combinations that are evaluated, including the initial design
#' @param params \code{list} with other parameters not included in \code{search_space}
#' @param verbose_eval \code{integer}. Whether to display information on the progress of tuning parameter choice.
#' If NULL or 0, verbose is of. If = 1, summary progress information is displayed for every trial
#' and warnings of 'mlr3mbo' are shown. If = 2, detailed progress is displayed at every boosting stage
#' for every trial.
#' @inheritParams gpb_shared_params
#' @param ... other parameters, see Parameters.rst for more information.
#' @inheritSection gpb_shared_params Early Stopping
#' @return         A \code{list} with the best parameter combination and score
#' The list has the following format:
#'  list("best_params" = best_params, "best_iter" = best_iter, "best_score" = best_score)
#' @details The random numbers of the optimization are drawn from the random number generator of R.
#' If \code{cv_seed} is given, \code{set.seed(cv_seed)} is called at the beginning, and the cross-validation folds
#' are the same in every trial. A parameter combination for which the cross-validation fails or gives a score that
#' is NA or infinite gets a score for the surrogate model that is worse than all valid scores, so that similar
#' parameter combinations are avoided.
#'
#' The parameters \code{force_col_wise = TRUE} and \code{deterministic = TRUE} are used unless \code{params} or
#' \code{...} contain \code{force_col_wise}, \code{force_row_wise}, or \code{deterministic}, respectively.
#' Without \code{force_col_wise} and \code{force_row_wise}, the histograms of the trees are constructed column-wise
#' or row-wise depending on a time measurement, so that the score of a parameter combination can differ slightly
#' between runs, and the Bayesian optimization can then choose other parameter combinations.
#' @references
#' Hutter, F., Hoos, H. H., and Leyton-Brown, K. (2011). Sequential model-based optimization for general
#' algorithm configuration. In Learning and Intelligent Optimization (LION 5), pages 507-523.
#'
#' Lindauer, M., Eggensperger, K., Feurer, M., Biedenkapp, A., Deng, D., Benjamins, C., Ruhkopf, T., Sass, R.,
#' and Hutter, F. (2022). SMAC3: A versatile Bayesian optimization package for hyperparameter optimization.
#' Journal of Machine Learning Research, 23(54):1-9.
#'
#' Koster, F. J. and Sigrist, F. (2026). Selecting hyperparameters for tree-boosting. arXiv:2602.05786.
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' \donttest{
#' if (requireNamespace("mlr3mbo", quietly = TRUE) && requireNamespace("mlr3learners", quietly = TRUE) &&
#'     requireNamespace("ranger", quietly = TRUE)) {
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' n <- length(y)
#' # Define search space
#' # Note: if the best combination found below is close to the bounday for a paramter,
#' #   you might want to extend the corresponding range
#' search_space <- list("learning_rate" = c(0.001, 10),
#'                      "min_data_in_leaf" = c(1, 1000),
#'                      "max_depth" = c(-1, -1),
#'                      "num_leaves" = c(2, 2^10),
#'                      "lambda_l2" = c(0, 100),
#'                      "max_bin" = c(63, min(n,10000)),
#'                      "line_search_step_length" = c(TRUE, FALSE))
#' # Note: "max_depth" = c(-1, -1) means no depth limit as we tune 'num_leaves'.
#' #    Can also additionally tune 'max_depth', e.g., "max_depth" = c(-1, 10)
#' metric = "mse" # Define metric
#' # Note: can also use metric = "test_neg_log_likelihood".
#' # See https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
#' gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
#' data_train <- gpb.Dataset(data = X, label = y)
#' # Run parameter optimization using the approximation of SMAC and 5-fold CV
#' opt_params <- gpb.tune.pars.rf.bayesian.optimization(search_space = search_space, n_trials = 100,
#'                                                      data = data_train, gp_model = gp_model,
#'                                                      nfold = 5, nrounds = 1000, early_stopping_rounds = 20,
#'                                                      metric = metric, cv_seed = 4, verbose_eval = 1)
#' print(paste0("Best parameters: ",
#'              paste0(unlist(lapply(seq_along(opt_params$best_params),
#'                                   function(y, n, i) { paste0(n[[i]],": ", y[[i]]) },
#'                                   y=opt_params$best_params,
#'                                   n=names(opt_params$best_params))), collapse=", ")))
#' print(paste0("Best number of iterations: ", opt_params$best_iter))
#' print(paste0("Best score: ", round(opt_params$best_score, digits=3)))
#' }
#' }
#' @author Fabio Sigrist
#' @export
gpb.tune.pars.rf.bayesian.optimization <- function(search_space
                                                   , n_trials
                                                   , data
                                                   , gp_model = NULL
                                                   , params = list()
                                                   , nrounds = 1000L
                                                   , early_stopping_rounds = NULL
                                                   , folds = NULL
                                                   , nfold = 5L
                                                   , metric = NULL
                                                   , verbose_eval = 1L
                                                   , cv_seed = NULL
                                                   , use_gp_model_for_validation = TRUE
                                                   , train_gp_model_cov_pars = TRUE
                                                   , eval = NULL
                                                   , categorical_feature = NULL
                                                   , ...
) {
  fun_name <- "gpb.tune.pars.rf.bayesian.optimization"
  gpb.tuning.check.packages(c("mlr3mbo", "mlr3learners", "ranger", "bbotk", "paradox", "lgr"),
                            fun_name = fun_name)
  parsed_search_space <- gpb.tuning.parse.search.space(search_space, fun_name = fun_name)
  if (!is.numeric(n_trials) || length(n_trials) != 1L || n_trials < 1) {
    stop(fun_name, ": 'n_trials' needs to be a positive integer")
  }
  n_trials <- as.integer(n_trials)
  verbose_eval <- if (is.null(verbose_eval)) 0L else as.integer(verbose_eval)
  verbose_cv <- if (verbose_eval < 2L) 0L else 1L
  if (is.null(params)) {
    params <- list()
  }
  params <- gpb.tuning.deterministic.params(params, other_params = list(list(...)))
  if (!is.null(cv_seed)) {
    set.seed(cv_seed)
  }
  higher_better <- gpb.tuning.higher.better(metric = metric, eval = eval)

  # Search space of 'paradox'. Parameters with a single value are held fixed and not searched
  fixed_params <- list()
  domain_params <- list()
  for (param in names(parsed_search_space)) {
    spec <- parsed_search_space[[param]]
    if (spec$lower == spec$upper) {
      fixed_params[[param]] <- spec$lower
    } else if (spec$kind == "float") {
      domain_params[[param]] <- paradox::p_dbl(lower = spec$lower, upper = spec$upper, logscale = spec$log)
    } else if (spec$kind == "int") {
      domain_params[[param]] <- paradox::p_int(lower = spec$lower, upper = spec$upper, logscale = spec$log)
    } else {
      domain_params[[param]] <- paradox::p_lgl()
    }
  }
  if (length(domain_params) == 0L) {
    stop(fun_name, ": 'search_space' contains no parameter with two different values")
  }
  domain <- do.call(paradox::ps, domain_params)
  codomain <- paradox::ps(score_for_optimizer = paradox::p_dbl(tags = if (higher_better) "maximize" else "minimize"))

  best_score <- NA_real_
  counter_num_comb <- 1L
  objective_fun <- function(xs) {
    params_comb <- c(xs, fixed_params)[names(parsed_search_space)]
    params_loc <- params
    params_loc[names(params_comb)] <- params_comb
    if (verbose_eval >= 1L) {
      message(paste0("Trial number ", counter_num_comb, " of ", n_trials, " with the parameters: ",
                     gpb.tuning.params.to.string(params_comb)))
    }
    cv_result <- tryCatch(
      gpb.tuning.cv.score(params = params_loc, data = data, gp_model = gp_model, nrounds = nrounds,
                          early_stopping_rounds = early_stopping_rounds, folds = folds, nfold = nfold,
                          metric = metric, eval = eval, cv_seed = cv_seed, verbose_cv = verbose_cv,
                          use_gp_model_for_validation = use_gp_model_for_validation,
                          train_gp_model_cov_pars = train_gp_model_cov_pars,
                          categorical_feature = categorical_feature, ...),
      error = function(err) {
        if (verbose_eval >= 1L) {
          message(paste0("Error in trial number ", counter_num_comb, ": ", conditionMessage(err)))
        }
        NULL
      })
    counter_num_comb <<- counter_num_comb + 1L
    if (is.null(cv_result) || !is.finite(cv_result$score)) {
      # Replaced by a score that is worse than all valid scores in 'impute_callback' below
      return(list(score_for_optimizer = if (higher_better) -Inf else Inf, score = NA_real_, best_iter = NA_real_))
    }
    if (is.na(best_score) || (higher_better && cv_result$score > best_score) ||
        (!higher_better && cv_result$score < best_score)) {
      best_score <<- cv_result$score
      if (verbose_eval >= 1L) {
        message(paste0("***** New best test score (", cv_result$metric_name, " = ", cv_result$score,
                       ") found for the following parameter combination: ",
                       gpb.tuning.params.to.string(c(params_comb, list(nrounds = cv_result$best_iter)))))
      }
    }
    return(list(score_for_optimizer = cv_result$score, score = cv_result$score,
                best_iter = as.numeric(cv_result$best_iter)))
  }
  objective <- bbotk::ObjectiveRFun$new(fun = objective_fun, domain = domain, codomain = codomain,
                                        check_values = FALSE)

  # Failed trials get a score that is worse than all valid scores. This is recalculated after every evaluation,
  # since the range of the valid scores changes. As long as there is no valid score, the failed trials get
  # the placeholder 0, since 'bbotk' requires a finite result
  impute_callback <- bbotk::callback_batch(
    "gpboost.impute_failed_trials",
    on_optimizer_after_eval = function(callback, context) {
      archive_data <- context$instance$archive$data
      valid <- is.finite(archive_data$score)
      if (!any(valid)) {
        data.table::set(archive_data, j = "score_for_optimizer", value = 0)
      } else if (!all(valid)) {
        valid_scores <- archive_data$score[valid]
        worst <- if (higher_better) min(valid_scores) else max(valid_scores)
        best <- if (higher_better) max(valid_scores) else min(valid_scores)
        gap <- abs(worst - best)
        if (gap == 0) {
          gap <- max(abs(worst), 1)
        }
        data.table::set(archive_data, i = which(!valid), j = "score_for_optimizer",
                        value = if (higher_better) worst - gap else worst + gap)
      }
    })
  instance <- bbotk::oi(objective = objective, terminator = bbotk::trm("evals", n_evals = n_trials),
                        callbacks = impute_callback)

  # Messages of 'mlr3mbo' and 'bbotk' are shown according to 'verbose_eval', and the thresholds of their
  # loggers are restored afterwards (NULL means that the threshold is inherited from the parent logger)
  logger_index <- lgr::logger_index()
  restore_logger_thresholds <- lapply(c("mlr3/bbotk", "mlr3/core"), function(logger_name) {
    logger <- lgr::get_logger(logger_name)
    index_row <- match(logger_name, logger_index$name)
    old_threshold <- if (is.na(index_row) || logger_index$threshold_inherited[index_row]) NULL else logger$threshold
    logger$set_threshold(if (verbose_eval >= 1L) "warn" else "error")
    return(function() logger$set_threshold(old_threshold))
  })
  on.exit(for (restore in restore_logger_thresholds) restore(), add = TRUE)

  # Initial design as in SMAC3
  n_init <- as.integer(max(1L, min(10L * length(domain_params), floor(n_trials / 4))))
  instance$eval_batch(paradox::generate_design_sobol(instance$search_space, n = n_init)$data)
  if (!instance$is_terminated) {
    surrogate <- mlr3mbo::default_surrogate(instance, force_random_forest = TRUE)
    acq_function <- mlr3mbo::acqf("ei")
    acq_optimizer <- mlr3mbo::default_acqoptimizer(acq_function = acq_function, instance = instance)
    optimizer <- bbotk::opt("mbo", loop_function = mlr3mbo::bayesopt_ego, surrogate = surrogate,
                            acq_function = acq_function, acq_optimizer = acq_optimizer,
                            args = list(random_interleave_iter = 5L))
    optimizer$optimize(instance)
  }

  archive_data <- instance$archive$data
  valid <- which(is.finite(archive_data$score))
  if (length(valid) == 0L) {
    warning("Found no parameter combination with a score that is not NA or Inf ")
    return(list(best_params = list(), best_iter = nrounds, best_score = NA_real_))
  }
  best_row <- valid[if (higher_better) which.max(archive_data$score[valid]) else which.min(archive_data$score[valid])]
  best_params <- c(archive_data$x_domain[[best_row]], fixed_params)[names(parsed_search_space)]
  return(list(best_params = best_params, best_iter = as.integer(archive_data$best_iter[best_row]),
              best_score = archive_data$score[best_row]))
}
