# Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
# Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
# Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.


#' @name gpb.train
#' @title Main training logic for GBPoost
#' @description Logic to train with GBPoost
#' @inheritParams gpb_shared_params
#' @param callbacks List of callback functions that are applied at each iteration.
#' @param reset_data Boolean, setting it to TRUE (not the default value) will transform the
#'                   booster model into a predictor model which frees up memory and the
#'                   original datasets
#' @param ... other parameters, see \href{https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst}{the parameter documentation} for more information. 
#' @inheritSection gpb_shared_params Early Stopping
#' @return a trained booster model \code{gpb.Booster}.
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' #--------------------Combine tree-boosting and grouped random effects model----------------
#' # Create random effects model
#' gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
#' # The default optimizer for covariance parameters (hyperparameters) is 
#' # Nesterov-accelerated gradient descent.
#' # This can be changed to, e.g., Nelder-Mead as follows:
#' # re_params <- list(optimizer_cov = "nelder_mead")
#' # gp_model$set_optim_params(params=re_params)
#' # Use trace = TRUE to monitor convergence:
#' # re_params <- list(trace = TRUE)
#' # gp_model$set_optim_params(params=re_params)
#' dtrain <- gpb.Dataset(data = X, label = y)
#' # Train model
#' bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 16,
#'                  learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
#'                  objective = "regression_l2", verbose = 0)
#' # Estimated random effects model
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
#'                 predict_var= TRUE)
#' pred$random_effect_mean # Predicted mean
#' pred$random_effect_cov # Predicted variances
#' pred$fixed_effect # Predicted fixed effect from tree ensemble
#' # Sum them up to otbain a single prediction
#' pred$random_effect_mean + pred$fixed_effect
#'
#' \donttest{
#' #--------------------Combine tree-boosting and Gaussian process model----------------
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
#'                     likelihood = "gaussian")
#' # Train model
#' dtrain <- gpb.Dataset(data = X, label = y)
#' bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 16,
#'                  learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
#'                  objective = "regression_l2", verbose = 0)
#' # Estimated random effects model
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
#'                 predict_cov_mat =TRUE)
#' pred$random_effect_mean # Predicted (posterior) mean of GP
#' pred$random_effect_cov # Predicted (posterior) covariance matrix of GP
#' pred$fixed_effect # Predicted fixed effect from tree ensemble
#' # Sum them up to otbain a single prediction
#' pred$random_effect_mean + pred$fixed_effect
#'
#'
#' #--------------------Using validation data-------------------------
#' set.seed(1)
#' train_ind <- sample.int(length(y),size=250)
#' dtrain <- gpb.Dataset(data = X[train_ind,], label = y[train_ind])
#' dtest <- gpb.Dataset.create.valid(dtrain, data = X[-train_ind,], label = y[-train_ind])
#' valids <- list(test = dtest)
#' gp_model <- GPModel(group_data = group_data[train_ind,1], likelihood="gaussian")
#' # Need to set prediction data for gp_model
#' gp_model$set_prediction_data(group_data_pred = group_data[-train_ind,1])
#' # Training with validation data and use_gp_model_for_validation = TRUE
#' bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 100,
#'                  learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
#'                  objective = "regression_l2", verbose = 1, valids = valids,
#'                  early_stopping_rounds = 10, use_gp_model_for_validation = TRUE)
#' print(paste0("Optimal number of iterations: ", bst$best_iter,
#'              ", best test error: ", bst$best_score))
#' # Plot validation error
#' val_error <- unlist(bst$record_evals$test$l2$eval)
#' plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
#'      xlab="iteration", ylab="Validation error", main="Validation error vs. boosting iteration")
#'
#'
#' #--------------------Do Newton updates for tree leaves---------------
#' # Note: run the above examples first
#' bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 100,
#'                  learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
#'                  objective = "regression_l2", verbose = 1, valids = valids,
#'                  early_stopping_rounds = 5, use_gp_model_for_validation = FALSE,
#'                  leaves_newton_update = TRUE)
#' print(paste0("Optimal number of iterations: ", bst$best_iter,
#'              ", best test error: ", bst$best_score))
#' # Plot validation error
#' val_error <- unlist(bst$record_evals$test$l2$eval)
#' plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
#'      xlab="iteration", ylab="Validation error", main="Validation error vs. boosting iteration")
#'
#'
#' #--------------------GPBoostOOS algorithm: GP parameters estimated out-of-sample----------------
#' # Create random effects model and dataset
#' gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
#' dtrain <- gpb.Dataset(X, label = y)
#' params <- list(learning_rate = 0.05,
#'                max_depth = 6,
#'                min_data_in_leaf = 5,
#'                objective = "regression_l2")
#' # Stage 1: run cross-validation to (i) determine to optimal number of iterations
#' #           and (ii) to estimate the GPModel on the out-of-sample data
#' cvbst <- gpb.cv(params = params,
#'                 data = dtrain,
#'                 gp_model = gp_model,
#'                 nrounds = 100,
#'                 nfold = 4,
#'                 eval = "l2",
#'                 early_stopping_rounds = 5,
#'                 use_gp_model_for_validation = TRUE,
#'                 fit_GP_cov_pars_OOS = TRUE)
#' print(paste0("Optimal number of iterations: ", cvbst$best_iter))
#' # Estimated random effects model
#' # Note: ideally, one would have to find the optimal combination of
#' #               other tuning parameters such as the learning rate, tree depth, etc.)
#' summary(gp_model)
#' # Stage 2: Train tree-boosting model while holding the GPModel fix
#' bst <- gpb.train(data = dtrain,
#'                  gp_model = gp_model,
#'                  nrounds = cvbst$best_iter,
#'                  learning_rate = 0.05,
#'                  max_depth = 6,
#'                  min_data_in_leaf = 5,
#'                  objective = "regression_l2",
#'                  verbose = 0,
#'                  train_gp_model_cov_pars = FALSE)
#' # The GPModel has not changed:
#' summary(gp_model)
#' }
#' @author Fabio Sigrist, authors of the LightGBM R package
#' @export
gpb.train <- function(params = list(),
                      data,
                      nrounds = 100L,
                      gp_model = NULL,
                      use_gp_model_for_validation = TRUE,
                      train_gp_model_cov_pars = TRUE,
                      valids = list(),
                      obj = NULL,
                      eval = NULL,
                      verbose = 1L,
                      record = TRUE,
                      eval_freq = 1L,
                      init_model = NULL,
                      colnames = NULL,
                      categorical_feature = NULL,
                      early_stopping_rounds = NULL,
                      callbacks = list(),
                      reset_data = FALSE,
                      ...) {

  # validate inputs early to avoid unnecessary computation
  if (nrounds <= 0L) {
    stop("nrounds should be greater than zero")
  }
  if (!gpb.is.Dataset(x = data)) {
    stop("gpb.train: data must be an gpb.Dataset instance")
  }
  if (length(valids) > 0L) {
    if (!identical(class(valids), "list") || !all(vapply(valids, gpb.is.Dataset, logical(1L)))) {
      stop("gpb.train: valids must be a list of gpb.Dataset elements")
    }
    evnames <- names(valids)
    if (is.null(evnames) || !all(nzchar(evnames))) {
      stop("gpb.train: each element of valids must have a name")
    }
  }

  # Setup temporary variables
  additional_params <- list(...)
  params <- append(params, additional_params)
  params$verbose <- verbose
  params <- gpb.check.obj(params = params, obj = obj)
  params <- gpb.check.eval(params = params, eval = eval)
  fobj <- NULL
  eval_functions <- list(NULL)
  
  params$train_gp_model_cov_pars <- train_gp_model_cov_pars

  # set some parameters, resolving the way they were passed in with other parameters
  # in `params`.
  # this ensures that the model stored with Booster$save() correctly represents
  # what was passed in
  params <- gpb.check.wrapper_param(
    main_param_name = "num_iterations"
    , params = params
    , alternative_kwarg_value = nrounds
  )
  params <- gpb.check.wrapper_param(
    main_param_name = "early_stopping_round"
    , params = params
    , alternative_kwarg_value = early_stopping_rounds
  )
  early_stopping_rounds <- params[["early_stopping_round"]]

  # Check for objective (function or not)
  if (is.function(params$objective)) {
    fobj <- params$objective
    params$objective <- "NONE"
  }

  # If eval is a single function, store it as a 1-element list
  # (for backwards compatibility). If it is a list of functions, store
  # all of them. This makes it possible to pass any mix of strings like "auc"
  # and custom functions to eval
  if (is.function(eval)) {
    eval_functions <- list(eval)
  }
  if (methods::is(eval, "list")) {
    eval_functions <- Filter(
      f = is.function
      , x = eval
    )
  }

  # Init predictor to empty
  predictor <- NULL

  # Check for boosting from a trained model
  if (is.character(init_model)) {
    predictor <- Predictor$new(modelfile = init_model)
  } else if (gpb.is.Booster(x = init_model)) {
    predictor <- init_model$to_predictor()
  }

  # Set the iteration to start from / end to (and check for boosting from a trained model, again)
  begin_iteration <- 1L
  if (!is.null(predictor)) {
    begin_iteration <- predictor$current_iter() + 1L
  }
  end_iteration <- begin_iteration + params[["num_iterations"]] - 1L

  # Construct datasets, if needed
  data$update_params(params = params)
  data$construct()

  # Check interaction constraints
  cnames <- NULL
  if (!is.null(colnames)) {
    cnames <- colnames
  } else if (!is.null(data$get_colnames())) {
    cnames <- data$get_colnames()
  }
  params[["interaction_constraints"]] <- gpb.check_interaction_constraints(
    params = params
    , column_names = cnames
  )

  # Update parameters with parsed parameters
  data$update_params(params)

  # Create the predictor set
  data$.__enclos_env__$private$set_predictor(predictor)

  # Write column names
  if (!is.null(colnames)) {
    data$set_colnames(colnames)
  }

  # Write categorical features
  if (!is.null(categorical_feature)) {
    data$set_categorical_feature(categorical_feature)
  }

  valid_contain_train <- FALSE
  train_data_name <- "train"
  reduced_valid_sets <- list()

  # Parse validation datasets
  if (length(valids) > 0L) {

    # Loop through all validation datasets using name
    for (key in names(valids)) {

      # Use names to get validation datasets
      valid_data <- valids[[key]]

      # Check for duplicate train/validation dataset
      if (identical(data, valid_data)) {
        valid_contain_train <- TRUE
        train_data_name <- key
        next
      }

      # Update parameters, data
      valid_data$update_params(params)
      valid_data$set_reference(data)
      reduced_valid_sets[[key]] <- valid_data

    }

  }
  
  if (!is.null(gp_model)) {
    if (is.function(eval) & use_gp_model_for_validation) {
      # Note: if this option should be added, it can be done similarly as in gpb.cv using booster$add_valid(..., valid_set_gp = valid_set_gp, ...)
      stop("use_gp_model_for_validation=TRUE is currently not supported for custom validation functions.
           If you need this feature, contact the developer of this package or open a GitHub issue.")
    }
    if (length(reduced_valid_sets) > 1 & use_gp_model_for_validation) {
      stop("Can use only one validation set when use_gp_model_for_validation = TRUE")
    }
    if (!valid_contain_train & use_gp_model_for_validation & length(reduced_valid_sets)>0 && is.null(gp_model$.__enclos_env__$private$num_data_pred)) {
      stop(paste0("Prediction data for ", sQuote("gp_model"), " has not been set. 
       This needs to be set prior to trainig when having a validation set and ", sQuote("use_gp_model_for_validation=TRUE"), ". 
       Either call ", sQuote("set_prediction_data(gp_model, ...)"), " first or use ", sQuote("use_gp_model_for_validation=FALSE"),"."))
    }
    # Set the default metric to the (approximate marginal) negative log-likelihood if only the training loss should be calculated
    if (valid_contain_train & length(reduced_valid_sets) == 0 & length(params$metric)==0) {
      if (gp_model$get_likelihood_name() != "gaussian") {
        params$metric <- append(params$metric, "approx_neg_marginal_log_likelihood")
      }
      else {
        params$metric <- append(params$metric, "neg_log_likelihood")
      }
    }
  }
  
  params$use_gp_model_for_validation <- use_gp_model_for_validation

  # Add printing log callback
  if (verbose > 0L && eval_freq > 0L) {
    callbacks <- add.cb(cb_list = callbacks, cb = cb.print.evaluation(period = eval_freq))
  }

  # Add evaluation log callback
  if (record && length(valids) > 0L) {
    callbacks <- add.cb(cb_list = callbacks, cb = cb.record.evaluation())
  }

  # Did user pass parameters that indicate they want to use early stopping?
  using_early_stopping <- !is.null(early_stopping_rounds) && early_stopping_rounds > 0L

  boosting_param_names <- .PARAMETER_ALIASES()[["boosting"]]
  using_dart <- any(
    sapply(
      X = boosting_param_names
      , FUN = function(param) {
        identical(params[[param]], "dart")
      }
    )
  )

  # Cannot use early stopping with 'dart' boosting
  if (using_dart) {
    warning("Early stopping is not available in 'dart' mode.")
    using_early_stopping <- FALSE

    # Remove the cb.early.stop() function if it was passed in to callbacks
    callbacks <- Filter(
      f = function(cb_func) {
        !identical(attr(cb_func, "name"), "cb.early.stop")
      }
      , x = callbacks
    )
  }

  # If user supplied early_stopping_rounds, add the early stopping callback
  if (using_early_stopping) {
    callbacks <- add.cb(
      cb_list = callbacks
      , cb = cb.early.stop(
        stopping_rounds = early_stopping_rounds
        , first_metric_only = isTRUE(params[["first_metric_only"]])
        , verbose = verbose
      )
    )
  }

  cb <- categorize.callbacks(cb_list = callbacks)

  # Construct booster with datasets
  booster <- Booster$new(params = params, train_set = data, gp_model = gp_model)
  if (valid_contain_train) {
    booster$set_train_data_name(name = train_data_name)
  }
  for (key in names(reduced_valid_sets)) {
    booster$add_valid(reduced_valid_sets[[key]], key, use_gp_model_for_validation=use_gp_model_for_validation)
  }

  # Callback env
  env <- CB_ENV$new()
  env$model <- booster
  env$begin_iteration <- begin_iteration
  env$end_iteration <- end_iteration

  # Start training model using number of iterations to start and end with
  for (i in seq.int(from = begin_iteration, to = end_iteration)) {

    # Overwrite iteration in environment
    env$iteration <- i
    env$eval_list <- list()

    # Loop through "pre_iter" element
    for (f in cb$pre_iter) {
      f(env)
    }

    # Update one boosting iteration
    booster$update(fobj = fobj)

    # Prepare collection of evaluation results
    eval_list <- list()

    # Collection: Has validation dataset?
    if (length(valids) > 0L) {

      # Get evaluation results with passed-in functions
      for (eval_function in eval_functions) {

        # Validation has training dataset?
        if (valid_contain_train) {
          eval_list <- append(eval_list, booster$eval_train(feval = eval_function))
        }

        eval_list <- append(eval_list, booster$eval_valid(feval = eval_function))
      }

      # Calling booster$eval_valid() will get
      # evaluation results with the metrics in params$metric by calling LGBM_BoosterGetEval_R",
      # so need to be sure that gets called, which it wouldn't be above if no functions
      # were passed in
      if (length(eval_functions) == 0L) {
        if (valid_contain_train) {
          eval_list <- append(eval_list, booster$eval_train(feval = eval_function))
        }
        eval_list <- append(eval_list, booster$eval_valid(feval = eval_function))
      }

    }

    # Write evaluation result in environment
    env$eval_list <- eval_list

    # Loop through env
    for (f in cb$post_iter) {
      f(env)
    }

    # Check for early stopping and break if needed
    if (env$met_early_stop) break

  }

  # check if any valids were given other than the training data
  non_train_valid_names <- names(valids)[!(names(valids) == train_data_name)]
  first_valid_name <- non_train_valid_names[1L]

  # When early stopping is not activated, we compute the best iteration / score ourselves by
  # selecting the first metric and the first dataset
  if (record && length(non_train_valid_names) > 0L && is.na(env$best_score)) {

    # when using a custom eval function, the metric name is returned from the
    # function, so figure it out from record_evals
    if (!is.null(eval_functions[1L])) {
      first_metric <- names(booster$record_evals[[first_valid_name]])[1L]
    } else {
      first_metric <- booster$.__enclos_env__$private$eval_names[1L]
    }

    .find_best <- which.min
    if (isTRUE(env$eval_list[[1L]]$higher_better[1L])) {
      .find_best <- which.max
    }
    booster$best_iter <- unname(
      .find_best(
        unlist(
          booster$record_evals[[first_valid_name]][[first_metric]][[.EVAL_KEY()]]
        )
      )
    )
    booster$best_score <- booster$record_evals[[first_valid_name]][[first_metric]][[.EVAL_KEY()]][[booster$best_iter]]
  }

  # Check for booster model conversion to predictor model
  if (reset_data) {

    # Store temporarily model data elsewhere
    booster_old <- list(
      best_iter = booster$best_iter
      , best_score = booster$best_score
      , record_evals = booster$record_evals
    )

    # Reload model
    booster <- gpb.load(model_str = booster$save_model_to_string())
    booster$best_iter <- booster_old$best_iter
    booster$best_score <- booster_old$best_score
    booster$record_evals <- booster_old$record_evals

  }

  return(booster)

}
