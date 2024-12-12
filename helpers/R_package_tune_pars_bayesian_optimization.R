#' @name tune.pars.bayesian.optimization
#' @title Function for choosing tuning parameters using Bayesian optimization. This is essentially a wrapper of the 'mbo' function in the 'mlrMBO' package
#' @description Function that allows for choosing tuning parameters from a grid in a determinstic or random way using cross validation or validation data sets.
#' @param search_space \code{list} with ranges for every parameter over which a search is done. 
#' The format for every entry of the list must be "parameter_name" = c(lower, upper).
#' See https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#tuning-parameters--hyperparameters-for-the-tree-boosting-part
#' @param n_iter Number of sequential Bayesian optimization steps
#' @param params \code{list} with other parameters not included in \code{param_grid}
#' @param crit \code{MBOInfillCrit}. How should infill points be rated. 
#' See infillcrits in the 'mlrMBO' package for an overview of available infill criteria or implement a custom one via makeMBOInfillCrit. 
#' Default is ``(lower) confidence bound" (see makeMBOInfillCritCB).
#' @inheritParams gpb_shared_params
#' @param ... other parameters, see Parameters.rst for more information.
#' @inheritSection gpb_shared_params Early Stopping
#' @return         A \code{list} with the best parameter combination and score
#' The list has the following format:
#'  list("best_params" = best_params, "best_iter" = best_iter, "best_score" = best_score)
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' \donttest{
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' library(mlrMBO)
#' library(DiceKriging)
#' library(rgenoud)
#' # Define search space
#' # Note: if the best combination found below is close to the bounday for a paramter, you might want to extend the corresponding range
#' search_space <- list("learning_rate" = c(0.001, 10), 
#'                      "min_data_in_leaf" = c(1, 1000),
#'                      "max_depth" = c(-1, -1), # -1 means no depth limit as we tune 'num_leaves'. Can also additionally tune 'max_depth', e.g., "max_depth" = c(-1, 1, 2, 3, 5, 10)
#'                      "num_leaves" = c(2, 2^10),
#'                      "lambda_l2" = c(0, 100),
#'                      "max_bin" = c(63, min(n,10000)),
#'                      "line_search_step_length" = c(TRUE, FALSE))
#' metric = "mse" # Define metric
#' # Note: can also use metric = "test_neg_log_likelihood". For more options, see https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
#' gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
#' data_train <- gpb.Dataset(data = X, label = y)
#' # Run parameter optimization using Bayesian optimization and k-fold CV 
#' opt_params <- tune.pars.bayesian.optimization(search_space = search_space, n_iter = 100,
#'                                               data = dataset, gp_model = gp_model,
#'                                               nfold = 5, nrounds = 1000, early_stopping_rounds = 20,
#'                                               metric = metric, cv_seed = 4, verbose_eval = 1)
#' print(paste0("Best parameters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), 
#'                                                        function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, 
#'                                                        n=names(opt_params$best_params))), collapse=", ")))
#' print(paste0("Best number of iterations: ", opt_params$best_iter))
#' print(paste0("Best score: ", round(opt_params$best_score, digits=3)))
#' 
#' # Alternatively and faster: using manually defined validation data instead of cross-validation
#' valid_tune_idx <- sample.int(length(y), as.integer(0.2*length(y))) # use 20% of the data as validation data
#' folds <- list(valid_tune_idx)
#' opt_params <- tune.pars.bayesian.optimization(search_space = search_space, n_iter = 100,
#'                                               data = dataset, gp_model = gp_model,
#'                                               folds = folds, nrounds = 1000, early_stopping_rounds = 20,
#'                                               metric = metric, cv_seed = 4, verbose_eval = 1)
#' 
#' }
#' @author Fabio Sigrist
#' @export
tune.pars.bayesian.optimization <- function(search_space
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
                                            , eval = NULL
                                            , categorical_feature = NULL
                                            , ...
) {
  # Check format
  if (!is.list(search_space)) {
    stop("tune.pars.bayesian.optimization: 'search_space' needs to be a list")
  }
  if (n_iter <= 0) {
    stop("tune.pars.bayesian.optimization: 'n_iter' needs to be a positive integer")
  }
  if (is.null(verbose_eval)) {
    verbose_eval = 0
  } else {
    verbose_eval = as.integer(verbose_eval)
  }
  if (is.null(params)) {
    params <- list()
  }
  # Higher better?
  higher_better = FALSE
  if (is.null(metric)) {
    eval_copy <- eval
    if (!is.null(eval_copy)) {
      if (!is.list(eval_copy)) {
        eval_copy <- list(eval_copy)
      }
      if (is.function(eval_copy[[1]])) {
        dummy_data <- gpb.Dataset(data=matrix(c(0,1)),label=c(0,1))
        dummy_data$construct()
        higher_better <- eval_copy[[1]](0,dummy_data)$higher_better
      } else {
        if (any(startsWith(eval_copy[[1]], c('auc', 'ndcg@', 'map@', 'average_precision')))){
          higher_better <- TRUE
        }
      }
    } 
  } else {
    if (any(startsWith(metric, c('auc', 'ndcg@', 'map@', 'average_precision')))){
      higher_better <- TRUE
    }
  }
  if (verbose_eval < 2) {
    verbose_cv <- 0L
  } else {
    verbose_cv <- 1L
  }
  par.set = makeParamSet()
  for (param in names(search_space)) {
    if (length(search_space[[param]]) != 2) {
      stop(sprintf("search_space['%s'] must have length 2", param))
    }
    
    if (param %in% c('learning_rate', 'shrinkage_rate',
                     'min_gain_to_split', 'min_split_gain',
                     'min_sum_hessian_in_leaf', 'min_sum_hessian_per_leaf', 'min_sum_hessian', 'min_hessian', 'min_child_weight')) {
      # log-transform values
      if (search_space[[param]][1] <= 0 || search_space[[param]][2] <= 0) {
        stop(sprintf("found non-positive values in search_space['%s'], need to use values > 0 ", param))
      }
      par.set <- c(par.set, makeParamSet(makeNumericParam(paste0("log_",param), lower = log(search_space[[param]][1]), 
                                                          upper = log(search_space[[param]][2]))))# trafo = trafoLog(base=exp(1)) is not helpful
    } else if (param %in% c('lambda_l2', 'reg_lambda', 'lambda',
                            'lambda_l1', 'reg_alpha',
                            'bagging_fraction', 'sub_row', 'subsample', 'bagging',
                            'feature_fraction', 'sub_feature', 'colsample_bytree',
                            'cat_l2',
                            'cat_smooth')) {
      par.set <- c(par.set, makeParamSet(makeNumericParam(param, lower = search_space[[param]][1], 
                                                          upper = search_space[[param]][2])))
    } else if (param %in% c('num_leaves', 'num_leaf', 'max_leaves', 'max_leaf',
                            'min_data_in_leaf', 'min_data_per_leaf', 'min_data', 'min_child_samples',
                            'max_bin')) {
      # log-transform values
      if (search_space[[param]][1] <= 0 || search_space[[param]][2] <= 0) {
        stop(sprintf("found non-positive values in search_space['%s'], need to use values > 0 ", param))
      }
      par.set <- c(par.set, makeParamSet(makeNumericParam(paste0("log_",param), lower = log(search_space[[param]][1]), 
                                                          upper = log(search_space[[param]][2]))))
    } else if (param %in% c('max_depth')) {
      par.set <- c(par.set, makeParamSet(makeIntegerParam(param, lower = search_space[[param]][1], 
                                                          upper = search_space[[param]][2])))
    } else if (param %in% c('line_search_step_length')) {
      par.set <- c(par.set, makeParamSet(makeLogicalParam(param)))
    } else {
      stop(sprintf("Unknown parameter '%s'", param))
    }
  }
  
  best_score <- 1E99
  if (higher_better) best_score <- -1E99
  worst_score <- best_score
  worst_score_init <- worst_score
  has_found_nonNA_score <- FALSE
  has_found_two_nonNA_scores <- FALSE
  best_params <- list()
  best_iter <- nrounds
  counter_num_comb <- 1
  
  obj.fun <- makeSingleObjectiveFunction(
    name = "gpboost_cv",
    
    fn = function(params_trial) {
      params_loc <- params
      for (param in names(params_trial)) {
        param_name <- param
        if (substr(param, 1, 4) == "log_"){
          param_name <- substr(param, 5, nchar(param))
        }
        if (param_name %in% c('learning_rate', 'shrinkage_rate',
                              'min_gain_to_split', 'min_split_gain',
                              'min_sum_hessian_in_leaf', 'min_sum_hessian_per_leaf', 'min_sum_hessian', 'min_hessian', 'min_child_weight')) {
          params_loc[[param_name]] <- exp(params_trial[[param]])
          
        } else if (param_name %in% c('lambda_l2', 'reg_lambda', 'lambda',
                                     'lambda_l1', 'reg_alpha',
                                     'bagging_fraction', 'sub_row', 'subsample', 'bagging',
                                     'feature_fraction', 'sub_feature', 'colsample_bytree',
                                     'cat_l2',
                                     'cat_smooth')) {
          params_loc[[param_name]] <- params_trial[[param]]
        } else if (param_name %in% c('num_leaves', 'num_leaf', 'max_leaves', 'max_leaf',
                                     'min_data_in_leaf', 'min_data_per_leaf', 'min_data', 'min_child_samples',
                                     'max_bin')) {
          params_loc[[param_name]] <- ceiling(exp(params_trial[[param]]))
          
        } else if (param_name %in% c('max_depth')) {
          params_loc[[param_name]] <- params_trial[[param]]
        } else if (param_name %in% c('line_search_step_length')) {
          params_loc[[param_name]] <- as.logical(params_trial[[param]])
        } else {
          stop(sprintf("Unknown parameter '%s'", param_name))
        }
      }
      
      if (!is.null(cv_seed)) {
        set.seed(cv_seed)
      }
      param_comb_print <- params_loc
      param_comb_print[["verbose"]] <- NULL
      param_comb_str <- lapply(seq_along(param_comb_print), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=param_comb_print, n=names(param_comb_print))
      param_comb_str <- paste0(unlist(param_comb_str), collapse=", ")
      if (verbose_eval >= 1L) {
        message(paste0("Trial number ", counter_num_comb, " with the parameters: ", param_comb_str))
      }
      tryCatch({
        cvbst <- gpb.cv(data = data, gp_model = gp_model,
                        params = params_loc, nrounds = nrounds, early_stopping_rounds = early_stopping_rounds, 
                        folds = folds, nfold = nfold, metric = metric,
                        verbose = verbose_cv, use_gp_model_for_validation = use_gp_model_for_validation,
                        ...)
      } ,
      error = function(err) {
        cvbst <<- list(best_score = worst_score_init)
      }
      ) #end tryCatch
      current_score_is_better <- FALSE
      if (is.na(cvbst$best_score) || is.infinite(cvbst$best_score) || cvbst$best_score == worst_score_init) {
        # impute current score in case there are NA's or Inf's
        if (higher_better) {
          if (has_found_two_nonNA_scores) {
            cvbst$best_score <- worst_score - 1 * (best_score - worst_score)
          } else if (has_found_nonNA_score) {
            cvbst$best_score <- worst_score - abs(worst_score)
          }
        } else {
          if (has_found_two_nonNA_scores) {
            cvbst$best_score <- worst_score + 1 * (worst_score - best_score)
          } else if (has_found_nonNA_score) {
            cvbst$best_score <- worst_score + 1 * abs(worst_score)
          }
        }
        if (has_found_two_nonNA_scores || has_found_nonNA_score) {
          warning(paste0("Found NA or Inf score in trial ", counter_num_comb, ". This was imputed with ", cvbst$best_score))
        }
      } else {# not NA and not Inf
        score_capped <- FALSE
        if (higher_better) {
          if (cvbst$best_score > best_score) {
            current_score_is_better <- TRUE
          }
          # cap current score to avoid very bad values
          if (has_found_two_nonNA_scores) {
            if (cvbst$best_score < (worst_score - 10 * (best_score - worst_score))) {
              cvbst$best_score <- worst_score - 10 * (best_score - worst_score)
              score_capped <- TRUE
            }
          } else if (has_found_nonNA_score) {
            if (cvbst$best_score < (worst_score - 10 * abs(worst_score))) {
              cvbst$best_score <- worst_score - 10 * abs(worst_score)
              score_capped <- TRUE
            }
          }
          # update worst_score
          if (cvbst$best_score < worst_score || worst_score == worst_score_init) {
            worst_score <<- cvbst$best_score 
          }
        } else {# !higher_better
          if (cvbst$best_score < best_score) {
            current_score_is_better <- TRUE
          }
          # cap current score to avoid very bad values
          if (has_found_two_nonNA_scores) {
            if (cvbst$best_score > (worst_score + 10 * (worst_score - best_score))) {
              cvbst$best_score <- worst_score + 10 * (worst_score - best_score)
              score_capped <- TRUE
            }
          } else if (has_found_nonNA_score) {
            if (cvbst$best_score > (worst_score + 10 * abs(worst_score))) {
              cvbst$best_score <- worst_score + 10 * abs(worst_score)
              score_capped <- TRUE
            }
          }
          # update worst_score
          if (cvbst$best_score > worst_score || worst_score == worst_score_init) {
            worst_score <<- cvbst$best_score
          }
        }# end !higher_better
        if (score_capped) {
          warning(paste0("Found very bad score in trial ", counter_num_comb, ". This was replaced with ", cvbst$best_score))
        }
        if (has_found_nonNA_score){
          has_found_two_nonNA_scores <<- TRUE
        }
        has_found_nonNA_score <<- TRUE
      }# end not NA and not Inf
      if (current_score_is_better) {
        best_score <<- cvbst$best_score
        best_iter <<- cvbst$best_iter
        best_params <<- params_loc
        if (verbose_eval >= 1L) {
          metric_name <- names(cvbst$record_evals$valid)
          param_comb_print <- params_loc
          param_comb_print[["verbose"]] <- NULL
          param_comb_print[["nrounds"]] <- best_iter
          str <- lapply(seq_along(param_comb_print), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=param_comb_print, n=names(param_comb_print))
          message(paste0("***** New best test score (",metric_name, " = ", 
                         best_score,  ") found for the following parameter combination: ", 
                         paste0(unlist(str), collapse=", ")))
        }
      }
      counter_num_comb <<- counter_num_comb + 1
      return(cvbst$best_score)
    },
    
    par.set = par.set,
    
    minimize = !higher_better
  )
  
  control = makeMBOControl()
  control = setMBOControlTermination(control, iters = n_iter)
  if (is.null(crit)) {
    crit = makeMBOInfillCritCB() # makeMBOInfillCritEI()
  }
  control = setMBOControlInfill(control, crit = crit)
  set.seed(cv_seed)
  run = mbo(obj.fun, control = control, show.info = FALSE)
  best_params[["verbose"]] <- NULL
  
  return(list(best_params=best_params, best_iter=best_iter, best_score=run$y))
  
  # tried to use 'impute.y.fun' to solve crash problems, but it did not work (07.12.2024)
  # if (higher_better) {
  #   impute.y.fun = function(x, y, opt.path) {
  #     ys <- getOptPathY(opt.path)
  #     min(ys) - 0.2 * (max(ys) - min(ys))
  #   }
  # } else {
  #   impute.y.fun = function(x, y, opt.path) {
  #     ys <- getOptPathY(opt.path)
  #     max(ys) + 0.2 * (max(ys) - min(ys))
  #   }
  # }
  # impute.y.fun = function(x, y, opt.path) 1 
  # control = makeMBOControl(impute.y.fun = impute.y.fun)
  
}
