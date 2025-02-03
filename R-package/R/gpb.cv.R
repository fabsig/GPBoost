# Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
# Modified work Copyright (c) 2020 - 2024 Fabio Sigrist. All rights reserved.
# Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.

#' @importFrom R6 R6Class
CVBooster <- R6::R6Class(
  classname = "gpb.CVBooster",
  cloneable = FALSE,
  public = list(
    best_iter = -1L,
    best_score = NA,
    record_evals = list(),
    boosters = list(),
    initialize = function(x) {
      self$boosters <- x
      return(invisible(NULL))
    },
    reset_parameter = function(new_params) {
      for (x in boosters) { x$reset_parameter(new_params) }
      return(invisible(self))
    }
  )
)

#' @name gpb.cv
#' @title CV function for number of boosting iterations
#' @description Cross validation function for determining number of boosting iterations
#' @inheritParams gpb_shared_params
#' @param label Vector of labels, used if \code{data} is not an \code{\link{gpb.Dataset}}
#' @param weight vector of response values. If not NULL, will set to dataset
#' @param record Boolean, TRUE will record iteration message to \code{booster$record_evals}
#' @param showsd \code{boolean}, whether to show standard deviation of cross validation.
#'               This parameter defaults to \code{TRUE}.
#' @param stratified a \code{boolean} indicating whether sampling of folds should be stratified
#'                   by the values of outcome labels.
#' @param fit_GP_cov_pars_OOS Boolean (default = FALSE). If TRUE, the covariance parameters of the 
#'            \code{gp_model} model are estimated using the out-of-sample (OOS) predictions 
#'            on the validation data using the optimal number of iterations (after performing the CV). 
#'            This corresponds to the GPBoostOOS algorithm.
#' @param colnames feature names, if not null, will use this to overwrite the names in dataset
#' @param callbacks List of callback functions that are applied at each iteration.
#' @param reset_data Boolean, setting it to TRUE (not the default value) will transform the booster model
#'                   into a predictor model which frees up memory and the original datasets
#' @param delete_boosters_folds Boolean, setting it to TRUE (not the default value) will delete the boosters of the individual folds
#' @param ... other parameters, see Parameters.rst for more information.
#' @inheritSection gpb_shared_params Early Stopping
#' @return a trained model \code{gpb.CVBooster}.
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' \donttest{
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' # Create random effects model and dataset
#' gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
#' dtrain <- gpb.Dataset(X, label = y)
#' params <- list(learning_rate = 0.05,
#'                max_depth = 6,
#'                min_data_in_leaf = 5)
#' # Run CV
#' cvbst <- gpb.cv(params = params,
#'                 data = dtrain,
#'                 gp_model = gp_model,
#'                 nrounds = 100,
#'                 nfold = 4,
#'                 eval = "l2",
#'                 early_stopping_rounds = 5,
#'                 use_gp_model_for_validation = TRUE)
#' print(paste0("Optimal number of iterations: ", cvbst$best_iter,
#'              ", best test error: ", cvbst$best_score))
#' }
#' @importFrom data.table data.table setorderv
#' @author Authors of the LightGBM R package, Fabio Sigrist
#' @export
gpb.cv <- function(params = list()
                   , data
                   , gp_model = NULL
                   , nrounds = 1000L
                   , early_stopping_rounds = NULL
                   , folds = NULL
                   , nfold = 5L
                   , metric = NULL
                   , verbose = 1L
                   , use_gp_model_for_validation = TRUE
                   , fit_GP_cov_pars_OOS = FALSE
                   , train_gp_model_cov_pars = TRUE
                   , label = NULL
                   , weight = NULL
                   , obj = NULL
                   , eval = NULL
                   , record = TRUE
                   , eval_freq = 1L
                   , showsd = FALSE
                   , stratified = TRUE
                   , init_model = NULL
                   , colnames = NULL
                   , categorical_feature = NULL
                   , callbacks = list()
                   , reset_data = FALSE
                   , delete_boosters_folds = FALSE
                   , ...
) {
  
  if (nrounds <= 0L) {
    stop("nrounds should be greater than zero")
  }
  
  # If 'data' is not an gpb.Dataset, try to construct one using 'label'
  if (!gpb.is.Dataset(x = data)) {
    if (is.null(label)) {
      stop("'label' must be provided for gpb.cv if 'data' is not an 'gpb.Dataset'")
    }
    data <- gpb.Dataset(data = data, label = label)
  }
  
  if (data$.__enclos_env__$private$free_raw_data) {
    warning("For true out-of-sample (cross-) validation, it is recommended to set free_raw_data = False when constructing the Dataset")
  }
  
  # Setup temporary variables
  if (!is.null(metric)) {
    params <- append(params, list(metric = metric))
  }
  params <- append(params, list(...))
  params$verbose <- verbose
  params <- gpb.check.obj(params = params, obj = obj)
  params <- gpb.check.eval(params = params, eval = eval)
  fobj <- NULL
  eval_functions <- list(NULL)
  has_custom_eval_functions <- FALSE
  
  params$use_gp_model_for_validation <- use_gp_model_for_validation
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
    has_custom_eval_functions <- TRUE
  }
  if (methods::is(eval, "list")) {
    eval_functions <- Filter(
      f = is.function
      , x = eval
    )
    if (length(eval_functions)>0) {
      has_custom_eval_functions <- TRUE
    }
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
  if (data$.__enclos_env__$private$free_raw_data) {
    data$construct()
  } else if (is.character(data$.__enclos_env__$private$raw_data)) {
    data$construct()
  }
  
  # Check interaction constraints
  cnames <- NULL
  if (!is.null(colnames)) {
    cnames <- colnames
  } else if (!is.null(data$get_colnames())) {
    cnames <- data$get_colnames()
  }
  params[["interaction_constraints"]] <- gpb.check_interaction_constraints(params = params, column_names = cnames)
  
  if (!is.null(gp_model)) {
    if (has_custom_eval_functions & use_gp_model_for_validation) {
      # Note: if this option should be added, it can be done similarly as in gpb.cv using booster$add_valid(..., valid_set_gp = valid_set_gp, ...)
      stop("use_gp_model_for_validation=TRUE is currently not supported for custom validation functions. If you need this feature, contact the developer of this package or open a GitHub issue.")
    }
    if (gp_model$get_num_data() != data$dim()[1]) {
      stop("Different number of samples in data and gp_model")
    }
  }
  
  # Check for weights
  if (!is.null(weight)) {
    data$setinfo(name = "weight", info = weight)
  }
  
  # Update parameters with parsed parameters
  data$update_params(params = params)
  
  # Create the predictor set
  data$.__enclos_env__$private$set_predictor(predictor = predictor)
  
  # Write column names
  if (!is.null(colnames)) {
    data$set_colnames(colnames = colnames)
  }
  
  # Write categorical features
  if (!is.null(categorical_feature)) {
    data$set_categorical_feature(categorical_feature = categorical_feature)
  }
  
  # Check for folds
  if (!is.null(folds)) {
    
    # Check for list of folds
    if (!identical(class(folds), "list")) {
      stop(sQuote("folds"), " must be a list with vectors of indices for each CV-fold")
    }
    
    # Set number of folds
    nfold <- length(folds)
    
  } else {
    
    # Check fold value
    if (nfold <= 1L) {
      stop(sQuote("nfold"), " must be > 1")
    }
    
    # Create folds
    if (data$.__enclos_env__$private$free_raw_data){
      folds <- generate.cv.folds(
        nfold = nfold
        , nrows = nrow(data)
        , stratified = stratified
        , label = getinfo(dataset = data, name = "label")
        , group = getinfo(dataset = data, name = "group")
        , params = params
      )
    } else {
      folds <- generate.cv.folds(
        nfold = nfold
        , nrows = nrow(data)
        , stratified = stratified
        , label = data$.__enclos_env__$private$info$label
        , group = data$.__enclos_env__$private$info$group
        , params = params
      )
    }
    
  }
  
  # Add printing log callback
  if (verbose > 0L && eval_freq > 0L) {
    callbacks <- add.cb(cb_list = callbacks, cb = cb.print.evaluation(period = eval_freq))
  }
  
  # Add evaluation log callback
  if (record) {
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
  
  # Construct booster for each fold. The data.table() code below is used to
  # guarantee that indices are sorted while keeping init_score and weight together
  # with the correct indices. Note that it takes advantage of the fact that
  # someDT$some_column returns NULL is 'some_column' does not exist in the data.table
  bst_folds <- lapply(
    X = seq_along(folds)
    , FUN = function(k) {
      
      # For learning-to-rank, each fold is a named list with two elements:
      #   * `fold` = an integer vector of row indices
      #   * `group` = an integer vector describing which groups are in the fold
      # For classification or regression tasks, it will just be an integer
      # vector of row indices
      folds_have_group <- "group" %in% names(folds[[k]])
      if (folds_have_group) {
        test_indices <- folds[[k]]$fold
        test_group_indices <- folds[[k]]$group
        if (data$.__enclos_env__$private$free_raw_data){
          test_groups <- getinfo(dataset = data, name = "group")[test_group_indices]
          train_groups <- getinfo(dataset = data, name = "group")[-test_group_indices]
        }else{
          test_groups <- data$.__enclos_env__$private$info$group[test_group_indices]
          train_groups <- data$.__enclos_env__$private$info$group[-test_group_indices]
        }
      } else {
        test_indices <- folds[[k]]
      }
      train_indices <- seq_len(nrow(data))[-test_indices]
      
      # set up test and train Datasets
      if (data$.__enclos_env__$private$free_raw_data){# free_raw_data
        
        # set up indices for train and test data
        test_indexDT <- data.table::data.table(
          indices = test_indices
          , weight = getinfo(dataset = data, name = "weight")[test_indices]
          , init_score = getinfo(dataset = data, name = "init_score")[test_indices]
        )
        data.table::setorderv(x = test_indexDT, cols = "indices", order = 1L)
        train_indexDT <- data.table::data.table(
          indices = train_indices
          , weight = getinfo(dataset = data, name = "weight")[train_indices]
          , init_score = getinfo(dataset = data, name = "init_score")[train_indices]
        )
        data.table::setorderv(x = train_indexDT, cols = "indices", order = 1L)
        
        dtest <- slice(data, test_indexDT$indices)
        setinfo(dataset = dtest, name = "weight", info = test_indexDT$weight)
        setinfo(dataset = dtest, name = "init_score", info = test_indexDT$init_score)
        
        dtrain <- slice(data, train_indexDT$indices)
        setinfo(dataset = dtrain, name = "weight", info = train_indexDT$weight)
        setinfo(dataset = dtrain, name = "init_score", info = train_indexDT$init_score)
        
      }
      else {# not free_raw_data
        
        # set up indices for train and test data
        test_indexDT <- data.table::data.table(indices = test_indices)
        data.table::setorderv(x = test_indexDT, cols = "indices", order = 1L)
        train_indexDT <- data.table::data.table(indices = train_indices)
        data.table::setorderv(x = train_indexDT, cols = "indices", order = 1L)
        
        weight_train = NULL
        if (!is.null(data$.__enclos_env__$private$info$weight)) {
          weight_train = data$.__enclos_env__$private$info$weight[train_indexDT$indices]
        }
        init_score_train = NULL
        if (!is.null(data$.__enclos_env__$private$info$init_score)) {
          init_score_train = data$.__enclos_env__$private$info$init_score[train_indexDT$indices]
        }
        dtrain <- gpb.Dataset(data = as.matrix(data$.__enclos_env__$private$raw_data[train_indexDT$indices,]),
                              label = data$.__enclos_env__$private$info$label[train_indexDT$indices],
                              weight = weight_train,
                              init_score = init_score_train,
                              colnames = data$.__enclos_env__$private$colnames,
                              categorical_feature = data$.__enclos_env__$private$categorical_feature,
                              params = data$.__enclos_env__$private$params,
                              free_raw_data = data$.__enclos_env__$private$free_raw_data)
        
        weight_test = NULL
        if (!is.null(data$.__enclos_env__$private$info$weight)) {
          weight_test = data$.__enclos_env__$private$info$weight[test_indexDT$indices]
        }
        init_score_test = NULL
        if (!is.null(data$.__enclos_env__$private$info$init_score)) {
          init_score_test = data$.__enclos_env__$private$info$init_score[test_indexDT$indices]
        }
        dtest <- gpb.Dataset(data = as.matrix(data$.__enclos_env__$private$raw_data[test_indexDT$indices,]), 
                             label = data$.__enclos_env__$private$info$label[test_indexDT$indices],
                             weight = weight_test,
                             init_score = init_score_test,
                             reference = dtrain,
                             colnames = data$.__enclos_env__$private$colnames,
                             categorical_feature = data$.__enclos_env__$private$categorical_feature,
                             params = data$.__enclos_env__$private$params,
                             free_raw_data = data$.__enclos_env__$private$free_raw_data)
        
      }# end not free_raw_data
      
      if (folds_have_group) {
        setinfo(dataset = dtest, name = "group", info = test_groups)
        setinfo(dataset = dtrain, name = "group", info = train_groups)
      }
      
      if (!is.null(gp_model)) {
        
        group_data_pred <- NULL
        group_data <- gp_model$get_group_data()
        if (!is.null(group_data)) {
          group_data_pred <- group_data[test_indexDT$indices,]
          group_data <- group_data[train_indexDT$indices,]
        }
        
        group_rand_coef_data_pred <- NULL
        group_rand_coef_data <- gp_model$get_group_rand_coef_data()
        if (!is.null(group_rand_coef_data)) {
          group_rand_coef_data_pred <- group_rand_coef_data[test_indexDT$indices,]
          group_rand_coef_data <- group_rand_coef_data[train_indexDT$indices,]
        }
        
        gp_coords_pred <- NULL
        gp_coords <- gp_model$get_gp_coords()
        if (!is.null(gp_coords)) {
          gp_coords_pred <- gp_coords[test_indexDT$indices,]
          gp_coords <- gp_coords[train_indexDT$indices,]
        }
        
        gp_rand_coef_data_pred <- NULL
        gp_rand_coef_data <- gp_model$get_gp_rand_coef_data()
        if (!is.null(gp_rand_coef_data)) {
          gp_rand_coef_data_pred <- gp_rand_coef_data[test_indexDT$indices,]
          gp_rand_coef_data <- gp_rand_coef_data[train_indexDT$indices,]
        }
        
        cluster_ids_pred <- NULL
        cluster_ids <- gp_model$get_cluster_ids()
        if (!is.null(cluster_ids)) {
          cluster_ids_pred <- cluster_ids[test_indexDT$indices]
          cluster_ids <- cluster_ids[train_indexDT$indices]
        }
        
        gp_model_train <- gpb.GPModel$new(likelihood = gp_model$get_likelihood_name()
                                          , group_data = group_data
                                          , group_rand_coef_data = group_rand_coef_data
                                          , ind_effect_group_rand_coef = gp_model$.__enclos_env__$private$ind_effect_group_rand_coef
                                          , drop_intercept_group_rand_effect = gp_model$.__enclos_env__$private$drop_intercept_group_rand_effect
                                          , gp_coords = gp_coords
                                          , gp_rand_coef_data = gp_rand_coef_data
                                          , cov_function = gp_model$.__enclos_env__$private$cov_function
                                          , cov_fct_shape = gp_model$.__enclos_env__$private$cov_fct_shape
                                          , gp_approx = gp_model$.__enclos_env__$private$gp_approx
                                          , num_parallel_threads = gp_model$.__enclos_env__$private$num_parallel_threads
                                          , cov_fct_taper_range = gp_model$.__enclos_env__$private$cov_fct_taper_range
                                          , cov_fct_taper_shape = gp_model$.__enclos_env__$private$cov_fct_taper_shape
                                          , num_neighbors = gp_model$.__enclos_env__$private$num_neighbors
                                          , vecchia_ordering = gp_model$.__enclos_env__$private$vecchia_ordering
                                          , ind_points_selection = gp_model$.__enclos_env__$private$ind_points_selection
                                          , num_ind_points = gp_model$.__enclos_env__$private$num_ind_points
                                          , cover_tree_radius = gp_model$.__enclos_env__$private$cover_tree_radius
                                          , matrix_inversion_method = gp_model$.__enclos_env__$private$matrix_inversion_method
                                          , seed = gp_model$.__enclos_env__$private$seed
                                          , cluster_ids = cluster_ids
                                          , likelihood_additional_param = gp_model$.__enclos_env__$private$likelihood_additional_param
                                          , free_raw_data = TRUE)
        valid_set_gp <- NULL
        if (use_gp_model_for_validation) {
          gp_model_train$set_prediction_data(group_data_pred = group_data_pred
                                             , group_rand_coef_data_pred = group_rand_coef_data_pred
                                             , gp_coords_pred = gp_coords_pred
                                             , gp_rand_coef_data_pred = gp_rand_coef_data_pred
                                             , cluster_ids_pred = cluster_ids_pred
                                             , vecchia_pred_type = gp_model$.__enclos_env__$private$vecchia_pred_type
                                             , num_neighbors_pred = gp_model$.__enclos_env__$private$num_neighbors_pred
                                             , cg_delta_conv_pred = gp_model$.__enclos_env__$private$cg_delta_conv_pred
                                             , nsim_var_pred = gp_model$.__enclos_env__$private$nsim_var_pred
                                             , rank_pred_approx_matrix_lanczos = gp_model$.__enclos_env__$private$rank_pred_approx_matrix_lanczos)
          if (has_custom_eval_functions) {
            # Note: Validation using the GP model is only done in R if there are custom evaluation functions in eval_functions, 
            #        otherwise it is directly done in C++. See the function Eval() in regression_metric.hpp
            valid_set_gp <- list(group_data_pred = group_data_pred,
                                 group_rand_coef_data_pred = group_rand_coef_data_pred,
                                 gp_coords_pred = gp_coords_pred,
                                 gp_rand_coef_data_pred = gp_rand_coef_data_pred,
                                 cluster_ids_pred = cluster_ids_pred)
          }
          
        }
        
        booster <- Booster$new(params = params, train_set = dtrain, gp_model = gp_model_train)
        gp_model$set_likelihood(gp_model_train$get_likelihood_name()) ## potentially change likelihood in case this was done in the booster to reflect implied changes in the default optimizer for different likelihoods
        gp_model_train$set_optim_params(params = gp_model$get_optim_params())
        
      } else {
        booster <- Booster$new(params = params, train_set = dtrain)
      }
      
      booster$add_valid(data = dtest, name = "valid", valid_set_gp = valid_set_gp,
                        use_gp_model_for_validation = use_gp_model_for_validation)
      
      return(
        list(booster = booster)
      )
    }
  )
  
  # Create new booster
  cv_booster <- CVBooster$new(x = bst_folds)
  
  # Callback env
  env <- CB_ENV$new()
  env$model <- cv_booster
  env$begin_iteration <- begin_iteration
  env$end_iteration <- end_iteration
  error_in_first_iteration <- FALSE
  
  # Start training model using number of iterations to start and end with
  for (i in seq.int(from = begin_iteration, to = end_iteration)) {
    
    # Overwrite iteration in environment
    env$iteration <- i
    env$eval_list <- list()
    
    for (f in cb$pre_iter) {
      f(env)
    }
    
    # Update one boosting iteration
    tryCatch({
      
      msg <- lapply(cv_booster$boosters, function(fd) {
        fd$booster$update(fobj = fobj)
        out <- list()
        for (eval_function in eval_functions) {
          out <- append(out, fd$booster$eval_valid(feval = eval_function))
        }
        
        return(out)
      })
      
    },
    error = function(err) { 
      message(paste0("Error in boosting iteration ", i,":"))
      message(err)
      env$met_early_stop <- TRUE
      if (env$iteration == 1) {
        error_in_first_iteration <<- TRUE
      }
      
    })# end tryCatch
    
    # Check for early stopping and break if needed
    if (error_in_first_iteration) {
      cv_booster$best_score <- NA
      return(cv_booster)
    } else {
      
      tryCatch({
        
        # Prepare collection of evaluation results
        merged_msg <- gpb.merge.cv.result(
          msg = msg
          , showsd = showsd
        )
        
        # Write evaluation result in environment
        env$eval_list <- merged_msg$eval_list
        
        # Check for standard deviation requirement
        if (showsd) {
          env$eval_err_list <- merged_msg$eval_err_list
        }
        
        # Loop through env
        for (f in cb$post_iter) {
          f(env)
        }
        
      },
      error = function(err) {
        env$met_early_stop <- TRUE
      })# end tryCatch
      
    }
    
    if (env$met_early_stop) break
    
  }
  
  # When early stopping is not activated, we compute the best iteration / score ourselves
  # based on the first first metric
  if (record && is.na(env$best_score)) {
    # when using a custom eval function, the metric name is returned from the
    # function, so figure it out from record_evals
    if (!is.null(eval_functions[1L])) {
      first_metric <- names(cv_booster$record_evals[["valid"]])[1L]
    } else {
      first_metric <- cv_booster$.__enclos_env__$private$eval_names[1L]
    }
    .find_best <- which.min
    if (isTRUE(env$eval_list[[1L]]$higher_better[1L])) {
      .find_best <- which.max
    }
    cv_booster$best_iter <- unname(
      .find_best(
        unlist(
          cv_booster$record_evals[["valid"]][[first_metric]][[.EVAL_KEY()]]
        )
      )
    )
    cv_booster$best_score <- cv_booster$record_evals[["valid"]][[first_metric]][[.EVAL_KEY()]][[cv_booster$best_iter]]
  }
  
  if (!is.null(gp_model) & fit_GP_cov_pars_OOS) {
    pred_fixed_effect_OOS <- rep(NA,nrow(data))
    for (k in 1:nfold) {
      fd <- cv_booster$boosters[[k]]
      ##Predict on OOS data
      predictor <- Predictor$new(fd$booster$.__enclos_env__$private$handle)
      pred_fixed_effect_OOS[folds[[k]]] = predictor$predict( data = data$.__enclos_env__$private$raw_data[folds[[k]],]
                                                             , start_iteration = 0L
                                                             , num_iteration = cv_booster$best_iter
                                                             , rawscore = TRUE
                                                             , predleaf = FALSE
                                                             , predcontrib = FALSE
                                                             , header = FALSE
                                                             , reshape = FALSE )
    }
    
    # message("Fitting GPModel on out-of-sample data...") # message removed in version 0.7.8
    if(gp_model$get_likelihood_name() == "gaussian"){
      gp_model$fit(y = data$.__enclos_env__$private$info$label - pred_fixed_effect_OOS)
    }
    else{
      gp_model$fit(y = data$.__enclos_env__$private$info$label, offset = pred_fixed_effect_OOS)
    }
    
  }
  
  if (reset_data) {
    lapply(cv_booster$boosters, function(fd) {
      # Store temporarily model data elsewhere
      booster_old <- list(
        best_iter = fd$booster$best_iter
        , best_score = fd$booster$best_score
        , record_evals = fd$booster$record_evals
      )
      # Reload model
      fd$booster <- gpb.load(model_str = fd$booster$save_model_to_string())
      fd$booster$best_iter <- booster_old$best_iter
      fd$booster$best_score <- booster_old$best_score
      fd$booster$record_evals <- booster_old$record_evals
    })
  }
  if (delete_boosters_folds) {
    lapply(cv_booster$boosters, function(fd) {
      fd$booster$finalize()
    })
    cv_booster$boosters = NULL
  }
  
  return(cv_booster)
  
}

# Generates random (stratified if needed) CV folds
generate.cv.folds <- function(nfold, nrows, stratified, label, group, params) {
  
  # Check for group existence
  if (is.null(group)) {
    
    # Shuffle
    rnd_idx <- sample.int(nrows)
    
    # Request stratified folds
    stratified_folds <- FALSE
    if (!is.null(params$objective)) {
      if (isTRUE(stratified) && params$objective %in% c("binary", "multiclass") && length(label) == length(rnd_idx)) {
        stratified_folds <- TRUE
      }
    }
    if (stratified_folds) {
      
      y <- label[rnd_idx]
      y <- as.factor(y)
      folds <- gpb.stratified.folds(y = y, k = nfold)
      
    } else {
      
      # Make simple non-stratified folds
      folds <- list()
      
      # Loop through each fold
      for (i in seq_len(nfold)) {
        kstep <- length(rnd_idx) %/% (nfold - i + 1L)
        folds[[i]] <- rnd_idx[seq_len(kstep)]
        rnd_idx <- rnd_idx[-seq_len(kstep)]
      }
      
    }
    
  } else {
    
    # When doing group, stratified is not possible (only random selection)
    if (nfold > length(group)) {
      stop("\nYou requested too many folds for the number of available groups.\n")
    }
    
    # Degroup the groups
    ungrouped <- inverse.rle(list(lengths = group, values = seq_along(group)))
    
    # Can't stratify, shuffle
    rnd_idx <- sample.int(length(group))
    
    # Make simple non-stratified folds
    folds <- list()
    
    # Loop through each fold
    for (i in seq_len(nfold)) {
      kstep <- length(rnd_idx) %/% (nfold - i + 1L)
      folds[[i]] <- list(
        fold = which(ungrouped %in% rnd_idx[seq_len(kstep)])
        , group = rnd_idx[seq_len(kstep)]
      )
      rnd_idx <- rnd_idx[-seq_len(kstep)]
    }
    
  }
  
  return(folds)
  
}

# Creates CV folds stratified by the values of y.
# It was borrowed from caret::createFolds and simplified
# by always returning an unnamed list of fold indices.
#' @importFrom stats quantile
gpb.stratified.folds <- function(y, k) {
  
  ## Group the numeric data based on their magnitudes
  ## and sample within those groups.
  ## When the number of samples is low, we may have
  ## issues further slicing the numeric data into
  ## groups. The number of groups will depend on the
  ## ratio of the number of folds to the sample size.
  ## At most, we will use quantiles. If the sample
  ## is too small, we just do regular unstratified CV
  if (is.numeric(y)) {
    
    cuts <- length(y) %/% k
    if (cuts < 2L) {
      cuts <- 2L
    }
    if (cuts > 5L) {
      cuts <- 5L
    }
    y <- cut(
      y
      , unique(stats::quantile(y, probs = seq.int(0.0, 1.0, length.out = cuts)))
      , include.lowest = TRUE
    )
    
  }
  
  if (k < length(y)) {
    
    ## Reset levels so that the possible levels and
    ## the levels in the vector are the same
    y <- as.factor(as.character(y))
    numInClass <- table(y)
    foldVector <- vector(mode = "integer", length(y))
    
    ## For each class, balance the fold allocation as far
    ## as possible, then resample the remainder.
    ## The final assignment of folds is also randomized.
    
    for (i in seq_along(numInClass)) {
      
      ## Create a vector of integers from 1:k as many times as possible without
      ## going over the number of samples in the class. Note that if the number
      ## of samples in a class is less than k, nothing is producd here.
      seqVector <- rep(seq_len(k), numInClass[i] %/% k)
      
      ## Add enough random integers to get  length(seqVector) == numInClass[i]
      if (numInClass[i] %% k > 0L) {
        seqVector <- c(seqVector, sample.int(k, numInClass[i] %% k))
      }
      
      ## Shuffle the integers for fold assignment and assign to this classes's data
      foldVector[y == dimnames(numInClass)$y[i]] <- sample(seqVector)
      
    }
    
  } else {
    
    foldVector <- seq(along = y)
    
  }
  
  out <- split(seq(along = y), foldVector)
  names(out) <- NULL
  return(out)
}

gpb.merge.cv.result <- function(msg, showsd) {
  
  # Get CV message length
  if (length(msg) == 0L) {
    stop("gpb.cv: size of cv result error")
  }
  
  # Get evaluation message length
  eval_len <- length(msg[[1L]])
  
  # Is evaluation message empty?
  if (eval_len == 0L) {
    stop("gpb.cv: should provide at least one metric for CV")
  }
  
  # Get evaluation results using a list apply
  eval_result <- lapply(seq_len(eval_len), function(j) {
    as.numeric(lapply(seq_along(msg), function(i) {
      msg[[i]][[j]]$value }))
  })
  
  # Get evaluation. Just taking the first element here to
  # get structure (name, higher_better, data_name)
  ret_eval <- msg[[1L]]
  
  # Go through evaluation length items
  for (j in seq_len(eval_len)) {
    ret_eval[[j]]$value <- mean(eval_result[[j]])
  }
  
  ret_eval_err <- NULL
  
  # Check for standard deviation
  if (showsd) {
    
    # Parse standard deviation
    for (j in seq_len(eval_len)) {
      ret_eval_err <- c(
        ret_eval_err
        , sqrt(mean(eval_result[[j]] ^ 2L) - mean(eval_result[[j]]) ^ 2L)
      )
    }
    
    # Convert to list
    ret_eval_err <- as.list(ret_eval_err)
    
  }
  
  # Return errors
  return(
    list(
      eval_list = ret_eval
      , eval_err_list = ret_eval_err
    )
  )
  
}

get.grid.size <- function(param_grid) {
  # Determine total number of parameter combinations on a grid
  # Author: Fabio Sigrist
  grid_size = 1
  for (param in param_grid) {
    grid_size = grid_size * length(param)
  }
  return(grid_size)
}

get.param.combination <- function(param_comb_number, param_grid) {
  # Select parameter combination from a grid of parameters
  # param_comb_number: Index number of parameter combination on parameter grid that should be returned (counting starts at 0)
  # Author: Fabio Sigrist
  param_comb = list()
  nk = param_comb_number
  for (param_name in names(param_grid)) {
    ind_p = nk %% length(param_grid[[param_name]])
    param_comb[[param_name]] = param_grid[[param_name]][ind_p + 1]
    nk = (nk - ind_p) / length(param_grid[[param_name]])
  }
  return(param_comb)
}

#' @name gpb.grid.search.tune.parameters
#' @title Function for choosing tuning parameters
#' @description Function that allows for choosing tuning parameters from a grid in a determinstic or random way using cross validation or validation data sets.
#' @param param_grid \code{list} with candidate parameters defining the grid over which a search is done
#' @param params \code{list} with other parameters not included in \code{param_grid}
#' @param num_try_random \code{integer} with number of random trial on parameter grid. If NULL, a deterministic search is done
#' @param label Vector of labels, used if \code{data} is not an \code{\link{gpb.Dataset}}
#' @param weight vector of response values. If not NULL, will set to dataset
#' @param stratified a \code{boolean} indicating whether sampling of folds should be stratified
#'                   by the values of outcome labels.
#' @param colnames feature names, if not null, will use this to overwrite the names in dataset
#' @param callbacks List of callback functions that are applied at each iteration.
#' @param return_all_combinations a \code{boolean} indicating whether all tried 
#' parameter combinations are returned
#' @inheritParams gpb_shared_params
#' @param ... other parameters, see Parameters.rst for more information.
#' @inheritSection gpb_shared_params Early Stopping
#' @return         A \code{list} with the best parameter combination and score
#' The list has the following format:
#'  list("best_params" = best_params, "best_iter" = best_iter, "best_score" = best_score)
#'  If return_all_combinations is TRUE, then the list contains an additional entry 'all_combinations'
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' \donttest{
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' n <- length(y)
#' param_grid <- list("learning_rate" = c(0.001, 0.01, 0.1, 1, 10), 
#'                    "min_data_in_leaf" = c(1, 10, 100, 1000),
#'                    "max_depth" = c(-1), 
#'                    "num_leaves" = 2^(1:10),
#'                    "lambda_l2" = c(0, 1, 10, 100),
#'                    "max_bin" = c(250, 500, 1000, min(n,10000)),
#'                    "line_search_step_length" = c(TRUE, FALSE))
#' # Note: "max_depth" = c(-1) means no depth limit as we tune 'num_leaves'. 
#' #    Can also additionally tune 'max_depth', e.g., "max_depth" = c(-1, 1, 2, 3, 5, 10)
#' metric = "mse" # Define metric
#' # Note: can also use metric = "test_neg_log_likelihood". 
#' # See https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
#' gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
#' data_train <- gpb.Dataset(data = X, label = y)
#' set.seed(1)
#' opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid,
#'                                               data = data_train, gp_model = gp_model,
#'                                               num_try_random = 100, nfold = 5,
#'                                               nrounds = 1000, early_stopping_rounds = 20,
#'                                               verbose_eval = 1, metric = metric, cv_seed = 4)
#' print(paste0("Best parameters: ",
#'              paste0(unlist(lapply(seq_along(opt_params$best_params), 
#'                                   function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, 
#'                                   y=opt_params$best_params, 
#'                                   n=names(opt_params$best_params))), collapse=", ")))
#' print(paste0("Best number of iterations: ", opt_params$best_iter))
#' print(paste0("Best score: ", round(opt_params$best_score, digits=3)))

#' # Alternatively and faster: using manually defined validation data instead of cross-validation
#' # use 20% of the data as validation data
#' valid_tune_idx <- sample.int(length(y), as.integer(0.2*length(y))) 
#' folds <- list(valid_tune_idx)
#' opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid,
#'                                               data = data_train, gp_model = gp_model,
#'                                               num_try_random = 100, folds = folds,
#'                                               nrounds = 1000, early_stopping_rounds = 20,
#'                                               verbose_eval = 1, metric = metric, cv_seed = 4)
#' 
#' }
#' @author Fabio Sigrist
#' @export
gpb.grid.search.tune.parameters <- function(param_grid
                                            , num_try_random = NULL
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
                                            , label = NULL
                                            , weight = NULL
                                            , obj = NULL
                                            , eval = NULL
                                            , stratified = TRUE
                                            , init_model = NULL
                                            , colnames = NULL
                                            , categorical_feature = NULL
                                            , callbacks = list()
                                            , return_all_combinations = FALSE
                                            , ...
) {
  
  if (!is.null(cv_seed)) {
    set.seed(cv_seed)
  }
  # Check format
  if (!is.list(param_grid)) {
    stop("gpb.grid.search.tune.parameters: param_grid needs to be a list")
  }
  if (is.null(verbose_eval)) {
    verbose_eval = 0
  } else {
    verbose_eval = as.integer(verbose_eval)
  }
  if (is.null(params)) {
    params <- list()
  }
  for (param in names(param_grid)) {
    if (!is.vector(param_grid[[param]])) {
      stop(paste0("gpb.grid.search.tune.parameters: Candidate parameters in param_grid need to be given as vectors for every parameter. Found other format for ", param))
    }
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
  # Determine combinations of parameter values that should be tried out
  grid_size <- get.grid.size(param_grid)
  if (is.null(num_try_random)) {
    try_param_combs <- 0:(grid_size - 1L)
    if (verbose_eval >= 1L) {
      message(paste0("gpb.grid.search.tune.parameters: Starting deterministic grid search with ", grid_size, " parameter combinations "))
    }
  } else {
    if(num_try_random > grid_size){
      stop("gpb.grid.search.tune.parameters: num_try_random is larger than the number of all possible combinations of parameters in param_grid ")
    }
    try_param_combs <- sample.int(n = grid_size, size = num_try_random, replace = FALSE) - 1L
    if (verbose_eval >= 1L) {
      message(paste0("gpb.grid.search.tune.parameters: Starting random grid search with ", num_try_random, " trials out of ", grid_size, " parameter combinations "))
    }
  }
  if (verbose_eval < 2) {
    verbose_cv <- 0L
  } else {
    verbose_cv <- 1L
  }
  if (return_all_combinations) {
    all_combinations <- list()
  }
  best_score <- 1E99
  if (higher_better) best_score <- -1E99
  best_params <- list()
  best_iter <- nrounds
  counter_num_comb <- 1
  has_found_better_score <- FALSE
  for (param_comb_number in try_param_combs) {
    param_comb = get.param.combination(param_comb_number=param_comb_number,
                                       param_grid=param_grid)
    for (param in names(param_comb)) {
      params[[param]] <- param_comb[[param]]
    }
    param_comb_str <- lapply(seq_along(param_comb), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=param_comb, n=names(param_comb))
    param_comb_str <- paste0(unlist(param_comb_str), collapse=", ")
    if (verbose_eval >= 1L) {
      message(paste0("Trying parameter combination ", counter_num_comb, 
                     " of ", length(try_param_combs), ": ", param_comb_str))
    }
    current_score_is_better <- FALSE
    if (!is.null(cv_seed)) {
      set.seed(cv_seed)
    }
    cvbst <- gpb.cv(params = params
                    , data = data
                    , gp_model = gp_model
                    , nrounds = nrounds
                    , early_stopping_rounds = early_stopping_rounds
                    , folds = folds
                    , nfold = nfold
                    , metric = metric
                    , use_gp_model_for_validation = use_gp_model_for_validation
                    , train_gp_model_cov_pars = train_gp_model_cov_pars
                    , label = label
                    , weight = weight
                    , obj = obj
                    , eval = eval
                    , verbose = verbose_cv
                    , record = TRUE
                    , eval_freq = 1L
                    , showsd = FALSE
                    , stratified = stratified
                    , init_model = init_model
                    , colnames = colnames
                    , categorical_feature = categorical_feature
                    , callbacks = callbacks
                    , delete_boosters_folds = TRUE
                    , ...
    )
    if (!is.na(cvbst$best_score)) {
      if (higher_better) {
        if (cvbst$best_score > best_score) {
          current_score_is_better <- TRUE
          has_found_better_score <- TRUE
        }
      } else {
        if (cvbst$best_score < best_score) {
          current_score_is_better <- TRUE
          has_found_better_score <- TRUE
        }
      }
    }
    if (current_score_is_better) {
      best_score <- cvbst$best_score
      best_iter <- cvbst$best_iter
      best_params <- param_comb
      if (verbose_eval >= 1L) {
        metric_name <- names(cvbst$record_evals$valid)
        param_comb_print <- param_comb
        param_comb_print[["nrounds"]] <- best_iter
        str <- lapply(seq_along(param_comb_print), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=param_comb_print, n=names(param_comb_print))
        message(paste0("***** New best test score (",metric_name, " = ", 
                       best_score,  ") found for the following parameter combination: ", 
                       paste0(unlist(str), collapse=", ")))
      }
    }
    if (return_all_combinations) {
      all_combinations[[counter_num_comb]] <- list(params=param_comb, nrounds=cvbst$best_iter, score=cvbst$best_score)
    }
    counter_num_comb <- counter_num_comb + 1L
  }
  if (!has_found_better_score) {
    warning("Found no parameter combination with a score that is not NA or Inf ")
  }
  if (return_all_combinations) {
    return(list(best_params=best_params, best_iter=best_iter, best_score=best_score, all_combinations=all_combinations))
  } else {
    return(list(best_params=best_params, best_iter=best_iter, best_score=best_score)) 
  }
  
}

