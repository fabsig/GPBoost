# Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
# Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
# Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.

#' @importFrom R6 R6Class
Booster <- R6::R6Class(
  # Class for boosting model in GPBoost
  # Authors: Authors of the LightGBM R package, Fabio Sigrist
  classname = "gpb.Booster",
  cloneable = FALSE,
  public = list(
    
    best_iter = -1L,
    best_score = NA_real_,
    params = list(),
    record_evals = list(),
    
    # Finalize will free up the handles
    finalize = function() {
      .Call(
        LGBM_BoosterFree_R
        , private$handle
      )
      private$handle <- NULL
      return(invisible(NULL))
    },
    
    # Initialize will create a starter booster
    initialize = function(params = list(),
                          train_set = NULL,
                          modelfile = NULL,
                          model_str = NULL,
                          gp_model = NULL,
                          ...) {
      
      # Create parameters and handle
      params <- append(params, list(...))
      handle <- NULL
      
      if (!is.null(gp_model)) { # has gp_model
        # Check if gp_model is gpb.Dataset or not
        if (!gpb.check.r6.class(gp_model, "GPModel")) {
          stop("gpb.Booster: Can only use GPModel as gp_model")
        }
        
        if (is.null(train_set)) {
          stop("gpb.Booster: You need to provide a training dataset ('train_set') for the GPBoost 
               algorithm. Boosting from a a file or a string is currently not supported.")
        }
        
      }
      
      # Create a handle for the dataset
      # Note: this was previously in a try({}) statement
      
      # Check if training dataset is not null
      if (!is.null(train_set)) {
        # Check if training dataset is gpb.Dataset or not
        if (!gpb.check.r6.class(object = train_set, name = "gpb.Dataset")) {
          stop("gpb.Booster: Can only use gpb.Dataset as training data")
        }
        train_set_handle <- train_set$.__enclos_env__$private$get_handle()
        params <- modifyList(params, train_set$get_params())
        params_str <- gpb.params2str(params = params)
        
        if (!is.null(gp_model)) { # has gp_model
          
          train_set$construct()
          
          if (gp_model$get_num_data() != train_set$dim()[1]) {
            stop("gpb.Booster: Number of data points in gp_model and train_set are not equal")
          }
          
          private$has_gp_model <- TRUE
          private$gp_model <- gp_model
          # Store booster handle
          handle <- .Call(
            LGBM_GPBoosterCreate_R
            , train_set_handle
            , params_str
            , gp_model$.__enclos_env__$private$get_handle()
          )
          
        } else { # No gp_model
          # Store booster handle
          handle <- .Call(
            LGBM_BoosterCreate_R
            , train_set_handle
            , params_str
          )
        }
        
        # Create private booster information
        private$train_set <- train_set
        private$train_set_version <- train_set$.__enclos_env__$private$version
        private$num_dataset <- 1L
        private$init_predictor <- train_set$.__enclos_env__$private$predictor
        
        # Check if predictor is existing
        if (!is.null(private$init_predictor)) {
          
          # Merge booster
          .Call(
            LGBM_BoosterMerge_R
            , handle
            , private$init_predictor$.__enclos_env__$private$handle
          )
          
        }
        
        # Check current iteration
        private$is_predicted_cur_iter <- c(private$is_predicted_cur_iter, FALSE)
        
      } else if (!is.null(modelfile)) {
        
        # Do we have a model file as character?
        if (!is.character(modelfile)) {
          stop("gpb.Booster: Can only use a string as model file path")
        }
        
        ## Does it have a gp_model?
        con <- file(modelfile)
        has_gp_model <- read.table(con,skip=1,nrow=1)
        if (paste0(as.vector(has_gp_model),collapse = "")=="has_gp_model:1,") {
          
          private$has_gp_model = TRUE
          save_data = RJSONIO::fromJSON(content=modelfile)
          # Create booster from string
          handle <- .Call(
            LGBM_BoosterLoadModelFromString_R
            , save_data[["booster_str"]]
          )
          # create gp_model from list
          private$gp_model <- gpb.GPModel$new(model_list = save_data[["gp_model_str"]])
          if (!is.null(save_data[["raw_data"]])) {
            
            private$train_set <- gpb.Dataset(
              data = matrix(unlist(save_data[["raw_data"]]$data),
                            nrow = length(save_data[["raw_data"]]$data),
                            byrow = TRUE),
              label = save_data[["raw_data"]]$label)
            
          } else {
            
            if (private$gp_model$get_likelihood_name() == "gaussian") {
              private$residual_loaded_from_file <- save_data[["residual"]]
            } else {
              private$fixed_effect_train_loaded_from_file <- save_data[["fixed_effect_train"]]
              private$label_loaded_from_file <- save_data[["label"]]
            }
            private$gp_model_prediction_data_loaded_from_file <- TRUE
            
          }
          
          
        } else { # has no gp_model
          
          # Create booster from model
          handle <- .Call(
            LGBM_BoosterCreateFromModelfile_R
            , modelfile
          )
          
        }
        
      } else if (!is.null(model_str)) {
        # Create booster from string
        
        # Do we have a model_str as character?
        if (!is.character(model_str)) {
          stop("gpb.Booster: Can only use a string as model_str")
        }
        
        if (substr(model_str, 20, 20) == "1") {
          # has gp_model
          
          private$has_gp_model = TRUE
          save_data = RJSONIO::fromJSON(content = model_str)
          # Create booster from string
          handle <- .Call(
            LGBM_BoosterLoadModelFromString_R
            , save_data[["booster_str"]]
          )
          # create gp_model from list
          private$gp_model <- gpb.GPModel$new(model_list = save_data[["gp_model_str"]])
          if (!is.null(save_data[["raw_data"]])) {
            
            private$train_set <- gpb.Dataset(
              data = matrix(unlist(save_data[["raw_data"]]$data),
                            nrow = length(save_data[["raw_data"]]$data),
                            byrow = TRUE),
              label = save_data[["raw_data"]]$label)
            
          } else {
            
            if (private$gp_model$get_likelihood_name() == "gaussian") {
              private$residual_loaded_from_file <- save_data[["residual"]]
            } else {
              private$fixed_effect_train_loaded_from_file <- save_data[["fixed_effect_train"]]
              private$label_loaded_from_file <- save_data[["label"]]
            }
            private$gp_model_prediction_data_loaded_from_file <- TRUE
            
          }
          
        } else { # has no gp_model
          
          handle <- .Call(
            LGBM_BoosterLoadModelFromString_R
            , model_str
          )
        }
        
      } else {
        
        # Booster non existent
        stop(
          "gpb.Booster: Need at least either training dataset, "
          , "model file, or model_str to create booster instance"
        )
        
      }
      
      # Check whether the handle was created properly if it was not stopped earlier by a stop call
      if (isTRUE(gpb.is.null.handle(x = handle))) {
        
        stop("gpb.Booster: cannot create Booster handle")
        
      } else {
        
        # Create class
        class(handle) <- "gpb.Booster.handle"
        private$handle <- handle
        private$num_class <- 1L
        .Call(
          LGBM_BoosterGetNumClasses_R
          , private$handle
          , private$num_class
        )
        
      }
      
      self$params <- params
      
      return(invisible(NULL))
      
    },
    
    # Set training data name
    set_train_data_name = function(name) {
      
      # Set name
      private$name_train_set <- name
      return(invisible(self))
      
    },
    
    # Add validation data
    add_valid = function(data, name, valid_set_gp = NULL, use_gp_model_for_validation = TRUE) {
      
      # Check if data is gpb.Dataset
      if (!gpb.check.r6.class(object = data, name = "gpb.Dataset")) {
        stop("gpb.Booster.add_valid: Can only use gpb.Dataset as validation data")
      }
      
      # Check if predictors are identical
      if (!identical(data$.__enclos_env__$private$predictor, private$init_predictor)) {
        stop(
          "gpb.Booster.add_valid: Failed to add validation data; "
          , "you should use the same predictor for these data"
        )
      }
      
      # Check if names are character
      if (!is.character(name)) {
        stop("gpb.Booster.add_valid: Can only use characters as data name")
      }
      
      # Add validation data to booster
      .Call(
        LGBM_BoosterAddValidData_R
        , private$handle
        , data$.__enclos_env__$private$get_handle()
      )
      
      # Store private information
      private$valid_sets <- c(private$valid_sets, data)
      private$name_valid_sets <- c(private$name_valid_sets, name)
      private$num_dataset <- private$num_dataset + 1L
      private$is_predicted_cur_iter <- c(private$is_predicted_cur_iter, FALSE)
      
      # Note: Validation using the GP model is only done in R if there are custom evaluation functions in feval, 
      #        otherwise it is directly done in C++. See the function Eval() in regression_metric.hpp
      if (private$has_gp_model) {
        
        private$use_gp_model_for_validation <- use_gp_model_for_validation
        
        if (use_gp_model_for_validation) {
          
          if (!is.null(valid_set_gp)) {
            if (!is.list(valid_set_gp)) {
              stop("gpb.Booster.add_valid: Can only use lists as valid_set_gp")
            }
            private$valid_sets_gp <- append(private$valid_sets_gp, list(valid_set_gp))
          }
          
        }
        
      } else {
        private$use_gp_model_for_validation <- FALSE
      }
      
      return(invisible(self))
      
    },
    
    # Reset parameters of booster
    reset_parameter = function(params, ...) {
      
      if (methods::is(self$params, "list")) {
        params <- modifyList(self$params, params)
      }
      
      params <- modifyList(params, list(...))
      params_str <- gpb.params2str(params = params)
      .Call(
        LGBM_BoosterResetParameter_R
        , private$handle
        , params_str
      )
      self$params <- params
      
      return(invisible(self))
      
    },
    
    # Perform boosting update iteration
    update = function(train_set = NULL, fobj = NULL) {
      
      if (is.null(train_set)) {
        if (private$train_set$.__enclos_env__$private$version != private$train_set_version) {
          train_set <- private$train_set
        }
      }
      
      # Check if training set is not null
      if (!is.null(train_set)) {
        
        # Check if training set is gpb.Dataset
        if (!gpb.check.r6.class(object = train_set, name = "gpb.Dataset")) {
          stop("gpb.Booster.update: Only can use gpb.Dataset as training data")
        }
        
        # Check if predictors are identical
        if (!identical(train_set$predictor, private$init_predictor)) {
          stop("gpb.Booster.update: Change train_set failed, you should use the same predictor for these data")
        }
        
        # Reset training data on booster
        .Call(
          LGBM_BoosterResetTrainingData_R
          , private$handle
          , train_set$.__enclos_env__$private$get_handle()
        )
        
        # Store private train set
        private$train_set <- train_set
        private$train_set_version <- train_set$.__enclos_env__$private$version
        
      }
      
      # Check if objective is empty
      if (is.null(fobj)) {
        if (private$set_objective_to_none) {
          stop("gpb.Booster.update: cannot update due to null objective function")
        }
        # Boost iteration from known objective
        .Call(
          LGBM_BoosterUpdateOneIter_R
          , private$handle
        )
        
      } else {
        
        # Check if objective is function
        if (!is.function(fobj)) {
          stop("gpb.Booster.update: fobj should be a function")
        }
        if (!private$set_objective_to_none) {
          self$reset_parameter(params = list(objective = "none"))
          private$set_objective_to_none <- TRUE
        }
        # Perform objective calculation
        gpair <- fobj(private$inner_predict(1L), private$train_set)
        
        # Check for gradient and hessian as list
        if (is.null(gpair$grad) || is.null(gpair$hess)) {
          stop("gpb.Booster.update: custom objective should
            return a list with attributes (hess, grad)")
        }
        
        # Return custom boosting gradient/hessian
        .Call(
          LGBM_BoosterUpdateOneIterCustom_R
          , private$handle
          , gpair$grad
          , gpair$hess
          , length(gpair$grad)
        )
        
      }
      
      # Loop through each iteration
      for (i in seq_along(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }
      
      return(invisible(self))
      
    },
    
    # Return one iteration behind
    rollback_one_iter = function() {
      
      # Return one iteration behind
      .Call(
        LGBM_BoosterRollbackOneIter_R
        , private$handle
      )
      
      # Loop through each iteration
      for (i in seq_along(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }
      
      return(invisible(self))
      
    },
    
    # Get current iteration
    current_iter = function() {
      
      cur_iter <- 0L
      .Call(
        LGBM_BoosterGetCurrentIteration_R
        , private$handle
        , cur_iter
      )
      return(cur_iter)
      
    },
    
    # Get upper bound
    upper_bound = function() {
      
      upper_bound <- 0.0
      .Call(
        LGBM_BoosterGetUpperBoundValue_R
        , private$handle
        , upper_bound
      )
      return(upper_bound)
      
    },
    
    # Get lower bound
    lower_bound = function() {
      
      lower_bound <- 0.0
      .Call(
        LGBM_BoosterGetLowerBoundValue_R
        , private$handle
        , lower_bound
      )
      return(lower_bound)
      
    },
    
    # Evaluate data on metrics
    eval = function(data, name, feval = NULL) {
      
      # Check if dataset is gpb.Dataset
      if (!gpb.check.r6.class(object = data, name = "gpb.Dataset")) {
        stop("gpb.Booster.eval: Can only use gpb.Dataset to eval")
      }
      
      # Check for identical data
      data_idx <- 0L
      if (identical(data, private$train_set)) {
        data_idx <- 1L
      } else {
        
        # Check for validation data
        if (length(private$valid_sets) > 0L) {
          
          # Loop through each validation set
          for (i in seq_along(private$valid_sets)) {
            
            # Check for identical validation data with training data
            if (identical(data, private$valid_sets[[i]])) {
              
              # Found identical data, skip
              data_idx <- i + 1L
              break
              
            }
            
          }
          
        }
        
      }
      
      # Check if evaluation was not done
      if (data_idx == 0L) {
        
        # Add validation data by name
        self$add_valid(data, name)
        data_idx <- private$num_dataset
        
      }
      
      # Evaluate data
      return(
        private$inner_eval(
          data_name = name
          , data_idx = data_idx
          , feval = feval
        )
      )
      
    },
    
    # Evaluation training data
    eval_train = function(feval = NULL) {
      return(private$inner_eval(private$name_train_set, 1L, feval))
    },
    
    # Evaluation validation data
    eval_valid = function(feval = NULL) {
      
      # Create ret list
      ret <- list()
      
      # Check if validation is empty
      if (length(private$valid_sets) <= 0L) {
        return(ret)
      }
      
      # Loop through each validation set
      for (i in seq_along(private$valid_sets)) {
        ret <- append(
          x = ret
          , values = private$inner_eval(private$name_valid_sets[[i]], i + 1L, feval)
        )
      }
      
      return(ret)
      
    },
    
    # Save model
    save_model = function(filename, start_iteration = NULL, num_iteration = NULL,
                          feature_importance_type = 0L, save_raw_data = FALSE, ...) {
      
      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      # Check if start iteration is non existent
      if (is.null(start_iteration)) {
        start_iteration <- 0L
      }
      
      # Save gp_model
      if (private$has_gp_model) {
        
        model_str <- self$save_model_to_string(start_iteration = start_iteration,
                                               num_iteration = num_iteration, 
                                               feature_importance_type = feature_importance_type,
                                               save_raw_data = save_raw_data)
        write(model_str, file=filename)
        
        
      } else {# has no gp_model
        
        # Save booster model
        .Call(
          LGBM_BoosterSaveModel_R
          , private$handle
          , as.integer(num_iteration)
          , as.integer(feature_importance_type)
          , filename
        )
      }
      
      return(invisible(self))
    },
    
    # Save model to string
    save_model_to_string = function(start_iteration = NULL, num_iteration = NULL,
                                    feature_importance_type = 0L, save_raw_data = FALSE, ...) {
      
      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      # Check if start iteration is non existent
      if (is.null(start_iteration)) {
        start_iteration <- 0L
      }
      
      bst_model_str <- .Call(
        LGBM_BoosterSaveModelToString_R
        , private$handle
        , as.integer(start_iteration)
        , as.integer(num_iteration)
        , as.integer(feature_importance_type)
      )
      
      if (private$has_gp_model) {
        
        if (is.null(private$train_set$.__enclos_env__$private$raw_data)) {
          stop("Cannot save to file or string when " , sQuote("free_raw_data = TRUE"), ".",
               " Set ", sQuote("free_raw_data = FALSE"), " when you construct the gpb.Dataset")
        }
        save_data <- list()
        save_data[["has_gp_model"]] <- 1L
        save_data[["booster_str"]] <- bst_model_str
        save_data[["gp_model_str"]] <- private$gp_model$model_to_list(include_response_data = FALSE)
        
        if (save_raw_data) {
          
          save_data[["raw_data"]] <- list()
          save_data[["raw_data"]][["label"]] <- as.vector(private$train_set$.__enclos_env__$private$info$label)
          save_data[["raw_data"]][["data"]] <- private$train_set$.__enclos_env__$private$raw_data
          if (is.matrix(save_data[["raw_data"]][["data"]])) {
            if (dim(save_data[["raw_data"]][["data"]])[2] == 1) {
              save_data[["raw_data"]][["data"]] <- as.vector(save_data[["raw_data"]][["data"]])
            }
          }
          
        } else {# do not save raw_data
          
          predictor <- Predictor$new(private$handle, ...)
          fixed_effect_train = predictor$predict( data = private$train_set$.__enclos_env__$private$raw_data
                                                  , start_iteration = start_iteration
                                                  , num_iteration = num_iteration
                                                  , rawscore = TRUE
                                                  , predleaf = FALSE
                                                  , predcontrib = FALSE
                                                  , header = header
                                                  , reshape = FALSE )
          if (private$gp_model$get_likelihood_name() == "gaussian") {
            residual = private$train_set$.__enclos_env__$private$info$label - fixed_effect_train
            save_data[["residual"]] <- as.vector(residual)
          } else {
            save_data[["fixed_effect_train"]] <- as.vector(fixed_effect_train)
            save_data[["label"]] <- as.vector(private$train_set$.__enclos_env__$private$info$label)
          }
          
        }# end !save_raw_data
        
        model_str <- RJSONIO::toJSON(save_data, digits=17)
        
      } else {# has no gp_model
        
        model_str <- bst_model_str
        
      }
      
      return(model_str)
      
    },
    
    # Dump model in memory
    dump_model = function(num_iteration = NULL, feature_importance_type = 0L) {
      
      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      
      model_str <- .Call(
        LGBM_BoosterDumpModel_R
        , private$handle
        , as.integer(num_iteration)
        , as.integer(feature_importance_type)
      )
      
      return(model_str)
      
    },
    
    # Predict on new data
    predict = function(data,
                       start_iteration = NULL,
                       num_iteration = NULL,
                       pred_latent = FALSE,
                       predleaf = FALSE,
                       predcontrib = FALSE,
                       header = FALSE,
                       reshape = FALSE,
                       group_data_pred = NULL,
                       group_rand_coef_data_pred = NULL,
                       gp_coords_pred = NULL,
                       gp_rand_coef_data_pred = NULL,
                       cluster_ids_pred = NULL,
                       predict_cov_mat = FALSE,
                       predict_var = FALSE,
                       cov_pars = NULL,
                       ignore_gp_model = FALSE,
                       rawscore = NULL,
                       vecchia_pred_type = NULL,
                       num_neighbors_pred = NULL,
                       ...) {
      
      if (!is.null(rawscore)) {
        stop("predict: The argument 'raw_score' is discontinued. 
             Use the renamed equivalent argument 'pred_latent' instead")
      }
      if (!is.null(vecchia_pred_type)) {
        stop("predict: The argument 'vecchia_pred_type' is discontinued. 
             Use the function 'set_prediction_data' to specify this")
      }
      if (!is.null(num_neighbors_pred)) {
        stop("predict: The argument 'num_neighbors_pred' is discontinued. 
             Use the function 'set_prediction_data' to specify this")
      }
      
      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      # Check if start iteration is non existent
      if (is.null(start_iteration)) {
        start_iteration <- 0L
      }
      
      # Predict on new data
      params <- list(...)
      predictor <- Predictor$new(
        modelfile = private$handle
        , params = params
      )
      
      if (private$has_gp_model & !predcontrib & !ignore_gp_model) {
        
        if (is.null(private$train_set$.__enclos_env__$private$raw_data) & 
            !private$gp_model_prediction_data_loaded_from_file) {
          stop("predict: cannot make predictions for random effects.
                Set ", sQuote("free_raw_data = FALSE"), " when you construct the gpb.Dataset")
        } else if (is.null(private$train_set$.__enclos_env__$private$raw_data) & 
                   private$gp_model_prediction_data_loaded_from_file) {
          if (start_iteration != 0L) {
            stop("predict: cannot use the option ", sQuote("start_iteration")," after loading 
                 from file without raw data. Set ", sQuote("save_raw_data = TRUE"), " when you save the model")
          }
        }
        
        random_effect_mean <- NA
        pred_var_cov <- NA
        response_mean <- NA
        response_var <- NA
        
        if(private$gp_model$get_likelihood_name() == "gaussian"){
          
          # Either use raw_data or data loaded from file for determining residual
          if (private$gp_model_prediction_data_loaded_from_file & 
              is.null(private$train_set$.__enclos_env__$private$raw_data)) {
            # Do not use raw_data but saved residual
            residual = private$residual_loaded_from_file
          } else {
            # Use raw_data
            fixed_effect_train = predictor$predict( data = private$train_set$.__enclos_env__$private$raw_data
                                                    , start_iteration = start_iteration
                                                    , num_iteration = num_iteration
                                                    , rawscore = TRUE
                                                    , predleaf = FALSE
                                                    , predcontrib = FALSE
                                                    , header = header
                                                    , reshape = FALSE )
            residual = private$train_set$.__enclos_env__$private$info$label - fixed_effect_train
          }
          
          # Note: we need to provide the response variable y as this was not saved
          #   in the gp_model ("in C++") for Gaussian data but was overwritten during training
          random_effect_pred = private$gp_model$predict( y = residual
                                                         , group_data_pred = group_data_pred
                                                         , group_rand_coef_data_pred = group_rand_coef_data_pred
                                                         , gp_coords_pred = gp_coords_pred
                                                         , gp_rand_coef_data_pred = gp_rand_coef_data_pred
                                                         , cluster_ids_pred = cluster_ids_pred
                                                         , predict_cov_mat = predict_cov_mat
                                                         , predict_var = predict_var
                                                         , cov_pars = cov_pars
                                                         , X_pred = NULL
                                                         , predict_response = !pred_latent )
          fixed_effect = predictor$predict( data = data
                                            , start_iteration = start_iteration
                                            , num_iteration = num_iteration
                                            , rawscore = TRUE
                                            , predleaf = FALSE
                                            , predcontrib = FALSE
                                            , header = header
                                            , reshape = FALSE )
          
          if (length(fixed_effect) != length(random_effect_pred$mu)){
            warning("Number of data points in fixed effect (tree ensemble) and random effect are not equal")
          }
          
          if (pred_latent) {
            if(predict_cov_mat){
              pred_var_cov <- random_effect_pred$cov
            } else if(predict_var){
              pred_var_cov <- random_effect_pred$var
            }
            random_effect_mean <- random_effect_pred$mu
          } 
          else {
            if(predict_cov_mat){
              response_var <- random_effect_pred$cov
            } else if(predict_var){
              response_var <- random_effect_pred$var
            }
            response_mean <- random_effect_pred$mu + fixed_effect
            fixed_effect <- NULL
          }
          
        }# end Gaussian data
        else{# non-Gaussian data
          
          y <- NULL
          # Either use raw_data or data loaded from file for determining fixed_effect_train
          if (private$gp_model_prediction_data_loaded_from_file & 
              is.null(private$train_set$.__enclos_env__$private$raw_data)) {
            # Do not use raw_data but used saved fixed_effect_train and label
            fixed_effect_train = private$fixed_effect_train_loaded_from_file
            y <- private$label_loaded_from_file
          } else {
            # Use raw_data
            fixed_effect_train = predictor$predict( data = private$train_set$.__enclos_env__$private$raw_data
                                                    , start_iteration = start_iteration
                                                    , num_iteration = num_iteration
                                                    , rawscore = TRUE
                                                    , predleaf = FALSE
                                                    , predcontrib = FALSE
                                                    , header = header
                                                    , reshape = FALSE )
            if (private$gp_model$.__enclos_env__$private$model_has_been_loaded_from_saved_file){
              # The label is never saved in the gp_model, so we need to provide it to the predict function when the model as loaded from file
              y <- private$train_set$.__enclos_env__$private$info$label
            }
          }
          
          fixed_effect = predictor$predict( data = data
                                            , start_iteration = start_iteration
                                            , num_iteration = num_iteration
                                            , rawscore = TRUE
                                            , predleaf = FALSE
                                            , predcontrib = FALSE
                                            , header = header
                                            , reshape = FALSE )
          
          if (pred_latent) {
            
            # Note: we don't need to provide the response variable y as this is saved
            #   in the gp_model ("in C++") for non-Gaussian data. y is only not NULL when
            #   the model was loaded from a file
            random_effect_pred = private$gp_model$predict( group_data_pred = group_data_pred
                                                          , group_rand_coef_data_pred = group_rand_coef_data_pred
                                                          , gp_coords_pred = gp_coords_pred
                                                          , gp_rand_coef_data_pred = gp_rand_coef_data_pred
                                                          , cluster_ids_pred = cluster_ids_pred
                                                          , predict_cov_mat = predict_cov_mat
                                                          , predict_var = predict_var
                                                          , cov_pars = cov_pars
                                                          , X_pred = NULL
                                                          , predict_response = FALSE
                                                          , offset = fixed_effect_train
                                                          , y = y )
            
            if (length(fixed_effect) != length(random_effect_pred$mu)){
              warning("Number of data points in fixed effect (tree ensemble) and random effect are not equal")
            }
            
            if(predict_cov_mat){
              pred_var_cov <- random_effect_pred$cov
            } else if(predict_var){
              pred_var_cov <- random_effect_pred$var
            }
            random_effect_mean <- random_effect_pred$mu
            
          }# end pred_latent
          else {# predict response variable for non-Gaussian data
            
            pred_resp <- private$gp_model$predict( group_data_pred = group_data_pred
                                                  , group_rand_coef_data_pred = group_rand_coef_data_pred
                                                  , gp_coords_pred = gp_coords_pred
                                                  , gp_rand_coef_data_pred = gp_rand_coef_data_pred
                                                  , cluster_ids_pred = cluster_ids_pred
                                                  , predict_cov_mat = predict_cov_mat
                                                  , predict_var = predict_var
                                                  , cov_pars = cov_pars
                                                  , X_pred = NULL
                                                  , predict_response = TRUE
                                                  , offset = fixed_effect_train
                                                  , offset_pred = fixed_effect
                                                  , y = y )
            
            response_mean <-  pred_resp$mu
            response_var <-  pred_resp$var
            fixed_effect <- NA
            
          }# end not pred_latent
          
        }# end non-Gaussian data
        
        return(list(fixed_effect = fixed_effect,
                    random_effect_mean = random_effect_mean,
                    random_effect_cov = pred_var_cov,
                    response_mean = response_mean,
                    response_var = response_var))
        
      }# end GPBoost prediction
      else {# no gp_model or predcontrib or ignore_gp_model
        return(
          predictor$predict(
            data = data
            , start_iteration = start_iteration
            , num_iteration = num_iteration
            , rawscore = pred_latent
            , predleaf = predleaf
            , predcontrib = predcontrib
            , header = header
            , reshape = reshape
          )
        )
      }
    },
    
    # Transform into predictor
    to_predictor = function() {
      return(Predictor$new(modelfile = private$handle))
    },
    
    # Used for save
    raw = NA,
    
    # Save model to temporary file for in-memory saving
    save = function() {
      
      # Overwrite model in object
      self$raw <- self$save_model_to_string(NULL)
      
      return(invisible(NULL))
      
    }
    
  ),
  private = list(
    handle = NULL,
    train_set = NULL,
    name_train_set = "training",
    gp_model = NULL,
    has_gp_model = FALSE,
    valid_sets_gp = list(),
    use_gp_model_for_validation = TRUE,
    residual_loaded_from_file = NULL,
    label_loaded_from_file = NULL,
    fixed_effect_train_loaded_from_file = NULL,
    gp_model_prediction_data_loaded_from_file = FALSE,
    valid_sets = list(),
    name_valid_sets = list(),
    predict_buffer = list(),
    is_predicted_cur_iter = list(),
    num_class = 1L,
    num_dataset = 0L,
    init_predictor = NULL,
    eval_names = NULL,
    higher_better_inner_eval = NULL,
    set_objective_to_none = FALSE,
    train_set_version = 0L,
    # Predict data
    inner_predict = function(idx) {
      
      # Store data name
      data_name <- private$name_train_set
      
      # Check for id bigger than 1
      if (idx > 1L) {
        data_name <- private$name_valid_sets[[idx - 1L]]
      }
      
      # Check for unknown dataset (over the maximum provided range)
      if (idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }
      
      # Check for prediction buffer
      if (is.null(private$predict_buffer[[data_name]])) {
        
        # Store predictions
        npred <- 0L
        .Call(
          LGBM_BoosterGetNumPredict_R
          , private$handle
          , as.integer(idx - 1L)
          , npred
        )
        private$predict_buffer[[data_name]] <- numeric(npred)
        
      }
      
      # Check if current iteration was already predicted
      if (!private$is_predicted_cur_iter[[idx]]) {
        
        # Use buffer
        .Call(
          LGBM_BoosterGetPredict_R
          , private$handle
          , as.integer(idx - 1L)
          , private$predict_buffer[[data_name]]
        )
        private$is_predicted_cur_iter[[idx]] <- TRUE
      }
      
      return(private$predict_buffer[[data_name]])
    },
    
    # Get evaluation information
    get_eval_info = function() {
      
      # Check for evaluation names emptiness
      if (is.null(private$eval_names)) {
        
        # Get evaluation names
        eval_names <- .Call(
          LGBM_BoosterGetEvalNames_R
          , private$handle
        )
        
        # Check names' length
        if (length(eval_names) > 0L) {
          
          # Parse and store privately names
          private$eval_names <- eval_names
          
          # some metrics don't map cleanly to metric names, for example "ndcg@1" is just the
          # ndcg metric evaluated at the first "query result" in learning-to-rank
          metric_names <- gsub("@.*", "", eval_names)
          private$higher_better_inner_eval <- .METRICS_HIGHER_BETTER()[metric_names]
          
        }
        
      }
      
      return(private$eval_names)
      
    },
    
    # Perform inner evaluation
    inner_eval = function(data_name, data_idx, feval = NULL) {
      
      # Check for unknown dataset (over the maximum provided range)
      if (data_idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }
      
      # Get evaluation information
      private$get_eval_info()
      
      # Prepare return
      ret <- list()
      
      # Check evaluation names existence
      if (length(private$eval_names) > 0L) {
        
        # Create evaluation values
        tmp_vals <- numeric(length(private$eval_names))
        .Call(
          LGBM_BoosterGetEval_R
          , private$handle
          , as.integer(data_idx - 1L)
          , tmp_vals
        )
        
        # Loop through all evaluation names
        for (i in seq_along(private$eval_names)) {
          
          # Store evaluation and append to return
          res <- list()
          res$data_name <- data_name
          res$name <- private$eval_names[i]
          res$value <- tmp_vals[i]
          res$higher_better <- private$higher_better_inner_eval[i]
          ret <- append(ret, list(res))
          
        }
        
      }
      
      # Check if there are evaluation metrics
      if (!is.null(feval)) {
        
        # Check if evaluation metric is a function
        if (!is.function(feval)) {
          stop("gpb.Booster.eval: feval should be a function")
        }
        
        # Prepare data
        data <- private$train_set
        
        # Check if data to assess is existing differently
        if (data_idx > 1L) {
          data <- private$valid_sets[[data_idx - 1L]]
        }
        
        # Perform function evaluation
        predval <- private$inner_predict(data_idx)
        # Note: the following is only done in R if there are custom evaluation functions in feval, 
        #        otherwise it is directly done in C++. See the function Eval() in regression_metric.hpp
        if (private$has_gp_model & private$use_gp_model_for_validation & data_idx > 1) {
          
          if (length(private$valid_sets_gp) == 0) {
            stop("gpb.Booster.add_valid: Validation data for the GP model not provided")
          }
          valid_set_gp <- private$valid_sets_gp[[data_idx-1]]
          fixed_effect_train = private$inner_predict(1)
          
          residual = private$train_set$.__enclos_env__$private$info$label-fixed_effect_train
          random_effect_pred = private$gp_model$predict(y=residual,
                                                        group_data_pred = valid_set_gp[["group_data_pred"]],
                                                        group_rand_coef_data_pred = valid_set_gp[["group_rand_coef_data_pred"]],
                                                        gp_coords_pred = valid_set_gp[["gp_coords_pred"]],
                                                        gp_rand_coef_data_pred = valid_set_gp[["gp_rand_coef_data_pred"]],
                                                        cluster_ids_pred = valid_set_gp[["cluster_ids_pred"]],
                                                        predict_cov_mat = FALSE,
                                                        predict_var = FALSE,
                                                        predict_response = FALSE,
                                                        cov_pars = NULL,
                                                        X_pred = NULL,
                                                        use_saved_data = FALSE)$mu
          
          predval = predval + random_effect_pred
          
        }
        res <- feval(predval, data)
        
        # Check for name correctness
        if (is.null(res$name) || is.null(res$value) ||  is.null(res$higher_better)) {
          stop("gpb.Booster.eval: custom eval function should return a
            list with attribute (name, value, higher_better)");
        }
        
        # Append names and evaluation
        res$data_name <- data_name
        ret <- append(ret, list(res))
      }
      
      return(ret)
      
    }
    
  )
)

#' @name predict.gpb.Booster
#' @title Prediction function for \code{gpb.Booster} objects
#' @description Prediction function for \code{gpb.Booster} objects
#' @inheritParams GPModel_shared_params
#' @param num_neighbors_pred an \code{integer} specifying the number of neighbors for making predictions.
#' This is discontinued here. Use the function 'set_prediction_data' to specify this
#' @param vecchia_pred_type A \code{string} specifying the type of Vecchia approximation used for making predictions.
#' This is discontinued here. Use the function 'set_prediction_data' to specify this
#' @param object Object of class \code{gpb.Booster}
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param start_iteration int or NULL, optional (default=NULL)
#'                        Start index of the iteration to predict.
#'                        If NULL or <= 0, starts from the first iteration.
#' @param num_iteration int or NULL, optional (default=NULL)
#'                      Limit number of iterations in the prediction.
#'                      If NULL, if the best iteration exists and start_iteration is NULL or <= 0, the
#'                      best iteration is used; otherwise, all iterations from start_iteration are used.
#'                      If <= 0, all iterations from start_iteration are used (no limits).
#' @param pred_latent  If TRUE latent variables, both fixed effects (tree-ensemble) 
#' and random effects (\code{gp_model}) are predicted. Otherwise, the response variable 
#' (label) is predicted. Depending on how the argument 'pred_latent' is set,
#' different values are returned from this function; see the 'Value' section for more details. 
#' If there is no \code{gp_model}, this argument corresponds to 'raw_score' in LightGBM.
#' @param predleaf whether predict leaf index instead.
#' @param predcontrib return per-feature contributions for each record.
#' @param header only used for prediction for text file. True if text file has header
#' @param reshape whether to reshape the vector of predictions to a matrix form when there are several
#'                prediction outputs per case.
#' @param cov_pars A \code{vector} containing covariance parameters which are used if the 
#' \code{gp_model} has not been trained or if predictions should be made for other 
#' parameters than the trained ones
#' @param ignore_gp_model A \code{boolean}. If TRUE, predictions are only made for the tree ensemble part
#' and the \code{gp_model} is ignored
#' @param rawscore This is discontinued. Use the renamed equivalent argument 
#' \code{pred_latent} instead
#' @param ... Additional named arguments passed to the \code{predict()} method of
#'            the \code{gpb.Booster} object passed to \code{object}. 
#' @return either a list with vectors or a single vector / matrix depending on 
#' whether there is a \code{gp_model} or not
#'   If there is a \code{gp_model}, the result dict contains the following entries.
#'     1. If \code{pred_latent} is TRUE, the dict contains the following 3 entries:
#'       - result["fixed_effect"] are the predictions from the tree-ensemble.
#'       - result["random_effect_mean"] are the predicted means of the \code{gp_model}.
#'       - result["random_effect_cov"] are the predicted covariances or variances of the \code{gp_model}
#'   (only if 'predict_var' or 'predict_cov' is TRUE).
#'     2. If \code{pred_latent} is FALSE, the dict contains the following 2 entries:
#'       - result["response_mean"] are the predicted means of the response variable (Label) taking into account
#'     both the fixed effects (tree-ensemble) and the random effects (\code{gp_model})
#'       - result["response_var"] are the predicted  covariances or variances of the response variable
#'   (only if 'predict_var' or 'predict_cov' is TRUE)
#'   If there is no \code{gp_model} or \code{predcontrib} or \code{ignore_gp_model} 
#'     are TRUE, the result contains predictions from the tree-booster only.
#' 
#' @examples
#' 
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' \donttest{
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
#' 
#' # Train model
#' bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 16,
#'                learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
#'                verbose = 0)
#' # Estimated random effects model
#' summary(gp_model)
#' 
#' # Make predictions
#' # Predict latent variables
#' pred <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
#'                 predict_var = TRUE, pred_latent = TRUE)
#' pred$random_effect_mean # Predicted latent random effects mean
#' pred$random_effect_cov # Predicted random effects variances
#' pred$fixed_effect # Predicted fixed effects from tree ensemble
#' # Predict response variable
#' pred_resp <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
#'                      predict_var = TRUE, pred_latent = FALSE)
#' pred_resp$response_mean # Predicted response mean
#' # For Gaussian data: pred$random_effect_mean + pred$fixed_effect = pred_resp$response_mean
#' pred$random_effect_mean + pred$fixed_effect - pred_resp$response_mean
#' 
#' #--------------------Combine tree-boosting and Gaussian process model----------------
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
#'                     likelihood = "gaussian")
#' # Train model
#' bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 8,
#'                learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
#'                verbose = 0)
#' # Estimated random effects model
#' summary(gp_model)
#' # Make predictions
# Predict latent variables
#' pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
#'                 predict_var = TRUE, pred_latent = TRUE)
#' pred$random_effect_mean # Predicted latent random effects mean
#' pred$random_effect_cov # Predicted random effects variances
#' pred$fixed_effect # Predicted fixed effects from tree ensemble
#' # Predict response variable
#' pred_resp <- predict(bst, data = X_test, gp_coords_pred = coords_test,
#'                      predict_var = TRUE, pred_latent = FALSE)
#' pred_resp$response_mean # Predicted response mean
#' }
#' @author Fabio Sigrist, authors of the LightGBM R package
#' @export
predict.gpb.Booster <- function(object,
                                data,
                                start_iteration = NULL,
                                num_iteration = NULL,
                                pred_latent = FALSE,
                                predleaf = FALSE,
                                predcontrib = FALSE,
                                header = FALSE,
                                reshape = FALSE,
                                group_data_pred = NULL,
                                group_rand_coef_data_pred = NULL,
                                gp_coords_pred = NULL,
                                gp_rand_coef_data_pred = NULL,
                                cluster_ids_pred = NULL,
                                predict_cov_mat = FALSE,
                                predict_var = FALSE,
                                cov_pars = NULL,
                                ignore_gp_model = FALSE,
                                rawscore = NULL,
                                vecchia_pred_type = NULL,
                                num_neighbors_pred = NULL,
                                ...) {
  
  if (!gpb.is.Booster(x = object)) {
    stop("predict.gpb.Booster: object should be an ", sQuote("gpb.Booster"))
  }
  
  # Return booster predictions
  return(
    object$predict(
      data = data
      , start_iteration = start_iteration
      , num_iteration = num_iteration
      , pred_latent = pred_latent
      , predleaf =  predleaf
      , predcontrib =  predcontrib
      , header = header
      , reshape = reshape
      , group_data_pred = group_data_pred
      , group_rand_coef_data_pred = group_rand_coef_data_pred
      , gp_coords_pred = gp_coords_pred
      , gp_rand_coef_data_pred = gp_rand_coef_data_pred
      , cluster_ids_pred = cluster_ids_pred
      , predict_cov_mat = predict_cov_mat
      , predict_var = predict_var
      , cov_pars = cov_pars
      , ignore_gp_model = ignore_gp_model
      , rawscore = rawscore
      , vecchia_pred_type = vecchia_pred_type
      , num_neighbors_pred = num_neighbors_pred
      , ...
    )
  )
}

#' @name gpb.load
#' @title Load GPBoost model
#' @description  Load GPBoost takes in either a file path or model string.
#'               If both are provided, Load will default to loading from file
#'               Boosters with gp_models can only be loaded from file.
#' @param filename path of model file
#' @param model_str a str containing the model
#'
#' @return gpb.Booster
#'
#' @examples
#' \donttest{
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' # Train model and make prediction
#' gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
#' bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 16,
#'                learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
#'                verbose = 0)
#' pred <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
#'                 predict_var= TRUE, pred_latent = TRUE)
#' # Save model to file
#' filename <- tempfile(fileext = ".json")
#' gpb.save(bst,filename = filename)
#' # Load from file and make predictions again
#' bst_loaded <- gpb.load(filename = filename)
#' pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test[,1],
#'                        predict_var= TRUE, pred_latent = TRUE)
#' # Check equality
#' pred$fixed_effect - pred_loaded$fixed_effect
#' pred$random_effect_mean - pred_loaded$random_effect_mean
#' pred$random_effect_cov - pred_loaded$random_effect_cov
#' }
#' @author Fabio Sigrist, authors of the LightGBM R package
#' @export
gpb.load <- function(filename = NULL, model_str = NULL) {
  
  filename_provided <- !is.null(filename)
  model_str_provided <- !is.null(model_str)
  
  if (filename_provided) {
    if (!is.character(filename)) {
      stop("gpb.load: filename should be character")
    }
    if (!file.exists(filename)) {
      stop(sprintf("gpb.load: file '%s' passed to filename does not exist", filename))
    }
    return(invisible(Booster$new(modelfile = filename)))
  }
  
  if (model_str_provided) {
    if (!is.character(model_str)) {
      stop("gpb.load: model_str should be character")
    }
    return(invisible(Booster$new(model_str = model_str)))
  }
  
  stop("gpb.load: either filename or model_str must be given")
}

#' @name gpb.save
#' @title Save GPBoost model
#' @description Save GPBoost model
#' @param booster Object of class \code{gpb.Booster}
#' @param filename saved filename
#' @param start_iteration int or NULL, optional (default=NULL)
#'                        Start index of the iteration to predict.
#'                        If NULL or <= 0, starts from the first iteration.
#' @param num_iteration int or NULL, optional (default=NULL)
#'                      Limit number of iterations in the prediction.
#'                      If NULL, if the best iteration exists and start_iteration is NULL or <= 0, the
#'                      best iteration is used; otherwise, all iterations from start_iteration are used.
#'                      If <= 0, all iterations from start_iteration are used (no limits).
#' @param save_raw_data If TRUE, the raw data (predictor / covariate data) for the Booster is also saved.
#' Enable this option if you want to change \code{start_iteration} or \code{num_iteration} at prediction time after loading.
#' @param ... Additional named arguments passed to the \code{predict()} method of
#'            the \code{gpb.Booster} object passed to \code{object}. 
#'            This is only used when there is a gp_model and when save_raw_data=FALSE
#'
#' @return gpb.Booster
#'
#' @examples
#' \donttest{
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' # Train model and make prediction
#' gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
#' bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 16,
#'                learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
#'                verbose = 0)
#' pred <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
#'                 predict_var= TRUE, pred_latent = TRUE)
#' # Save model to file
#' filename <- tempfile(fileext = ".json")
#' gpb.save(bst,filename = filename)
#' # Load from file and make predictions again
#' bst_loaded <- gpb.load(filename = filename)
#' pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test[,1],
#'                        predict_var= TRUE, pred_latent = TRUE)
#' # Check equality
#' pred$fixed_effect - pred_loaded$fixed_effect
#' pred$random_effect_mean - pred_loaded$random_effect_mean
#' pred$random_effect_cov - pred_loaded$random_effect_cov
#' }
#' @author Fabio Sigrist, authors of the LightGBM R package
#' @export
gpb.save <- function(booster, filename, start_iteration = NULL, 
                     num_iteration = NULL, save_raw_data = FALSE, ...) {
  
  if (!gpb.is.Booster(x = booster)) {
    stop("gpb.save: booster should be an ", sQuote("gpb.Booster"))
  }
  
  if (!(is.character(filename) && length(filename) == 1L)) {
    stop("gpb.save: filename should be a string")
  }
  
  # Store booster
  return(
    invisible(booster$save_model(
      filename = filename
      , start_iteration = start_iteration
      , num_iteration = num_iteration
      , save_raw_data = save_raw_data
      , ...
    ))
  )
  
}

#' @name gpb.dump
#' @title Dump GPBoost model to json
#' @description Dump GPBoost model to json
#' @param booster Object of class \code{gpb.Booster}
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#'
#' @return json format of model
#'
#' @examples
#' \donttest{
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- gpb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 10L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , early_stopping_rounds = 5L
#' )
#' json_model <- gpb.dump(model)
#' }
#' @export
gpb.dump <- function(booster, num_iteration = NULL) {
  
  if (!gpb.is.Booster(x = booster)) {
    stop("gpb.save: booster should be an ", sQuote("gpb.Booster"))
  }
  
  # Return booster at requested iteration
  return(booster$dump_model(num_iteration =  num_iteration))
  
}

#' @name gpb.get.eval.result
#' @title Get record evaluation result from booster
#' @description Given a \code{gpb.Booster}, return evaluation results for a
#'              particular metric on a particular dataset.
#' @param booster Object of class \code{gpb.Booster}
#' @param data_name Name of the dataset to return evaluation results for.
#' @param eval_name Name of the evaluation metric to return results for.
#' @param iters An integer vector of iterations you want to get evaluation results for. If NULL
#'              (the default), evaluation results for all iterations will be returned.
#' @param is_err TRUE will return evaluation error instead
#'
#' @return numeric vector of evaluation result
#'
#' @examples
#' \donttest{
#' # train a regression model
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- gpb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#' )
#'
#' # Examine valid data_name values
#' print(setdiff(names(model$record_evals), "start_iter"))
#'
#' # Examine valid eval_name values for dataset "test"
#' print(names(model$record_evals[["test"]]))
#'
#' # Get L2 values for "test" dataset
#' gpb.get.eval.result(model, "test", "l2")
#' }
#' @export
gpb.get.eval.result <- function(booster, data_name, eval_name, iters = NULL, is_err = FALSE) {
  
  # Check if booster is booster
  if (!gpb.is.Booster(x = booster)) {
    stop("gpb.get.eval.result: Can only use ", sQuote("gpb.Booster"), " to get eval result")
  }
  
  # Check if data and evaluation name are characters or not
  if (!is.character(data_name) || !is.character(eval_name)) {
    stop("gpb.get.eval.result: data_name and eval_name should be characters")
  }
  
  # NOTE: "start_iter" exists in booster$record_evals but is not a valid data_name
  data_names <- setdiff(names(booster$record_evals), "start_iter")
  if (!(data_name %in% data_names)) {
    stop(paste0(
      "gpb.get.eval.result: data_name "
      , shQuote(data_name)
      , " not found. Only the following datasets exist in record evals: ["
      , paste(data_names, collapse = ", ")
      , "]"
    ))
  }
  
  # Check if evaluation result is existing
  eval_names <- names(booster$record_evals[[data_name]])
  if (!(eval_name %in% eval_names)) {
    stop(paste0(
      "gpb.get.eval.result: eval_name "
      , shQuote(eval_name)
      , " not found. Only the following eval_names exist for dataset "
      , shQuote(data_name)
      , ": ["
      , paste(eval_names, collapse = ", ")
      , "]"
    ))
    stop("gpb.get.eval.result: wrong eval name")
  }
  
  result <- booster$record_evals[[data_name]][[eval_name]][[.EVAL_KEY()]]
  
  # Check if error is requested
  if (is_err) {
    result <- booster$record_evals[[data_name]][[eval_name]][[.EVAL_ERR_KEY()]]
  }
  
  # Check if iteration is non existant
  if (is.null(iters)) {
    return(as.numeric(result))
  }
  
  # Parse iteration and booster delta
  iters <- as.integer(iters)
  delta <- booster$record_evals$start_iter - 1.0
  iters <- iters - delta
  
  # Return requested result
  return(as.numeric(result[iters]))
}
