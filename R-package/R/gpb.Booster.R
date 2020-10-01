#' @importFrom R6 R6Class
Booster <- R6::R6Class(
  classname = "gpb.Booster",
  cloneable = FALSE,
  public = list(

    best_iter = -1,
    best_score = NA,
    record_evals = list(),

    # Finalize will free up the handles
    finalize = function() {

      # Check the need for freeing handle
      if (!gpb.is.null.handle(private$handle)) {

        # Freeing up handle
        gpb.call("LGBM_BoosterFree_R", ret = NULL, private$handle)
        private$handle <- NULL

      }

    },

    # Initialize will create a starter booster
    initialize = function(params = list(),
                          train_set = NULL,
                          modelfile = NULL,
                          model_str = NULL,
                          gp_model = NULL,
                          ...) {

      R_ARCH <- Sys.getenv("R_ARCH")
      if(R_ARCH == "/i386"){
        warning("It is not recommended to run the tree-boosting functionality of GPBoost in its 32-bit version. Use the 64-bit version instead.")
      }
      
      # Create parameters and handle
      if (!is.null(gp_model)) {
        params["has_gp_model"] <- TRUE
      }
      params <- append(params, list(...))
      params_str <- gpb.params2str(params)
      handle <- 0.0

      # Attempts to create a handle for the dataset
      try({

        # Check if training dataset is not null
        if (!is.null(train_set)) {

          # Check if training dataset is gpb.Dataset or not
          if (!gpb.check.r6.class(train_set, "gpb.Dataset")) {
            stop("gpb.Booster: Can only use gpb.Dataset as training data")
          }
          
          # Check if gp_model is not null
          if (!is.null(gp_model)) {
            
            # Check if gp_model is gpb.Dataset or not
            if (!gpb.check.r6.class(gp_model, "GPModel")) {
              stop("gpb.Booster: Can only use GPModel as gp_model")
            }
            
            train_set$construct()
           
            if (gp_model$get_num_data() != train_set$dim()[1]) {
              stop("gpb.Booster: Number of data points in gp_model and train_set are not equal")
            }
            
            private$has_gp_model <- TRUE
            private$gp_model <- gp_model
            # Store booster handle
            handle <- gpb.call("LGBM_GPBoosterCreate_R", 
                               ret = handle, 
                               train_set$.__enclos_env__$private$get_handle(), 
                               params_str,
                               gp_model$.__enclos_env__$private$get_handle())
            
          } else {
            
            # Store booster handle
            handle <- gpb.call("LGBM_BoosterCreate_R", 
                               ret = handle, 
                               train_set$.__enclos_env__$private$get_handle(), 
                               params_str)
            
          }



          # Create private booster information
          private$train_set <- train_set
          private$num_dataset <- 1
          private$init_predictor <- train_set$.__enclos_env__$private$predictor

          # Check if predictor is existing
          if (!is.null(private$init_predictor)) {

            # Merge booster
            gpb.call("LGBM_BoosterMerge_R",
                     ret = NULL,
                     handle,
                     private$init_predictor$.__enclos_env__$private$handle)

          }

          # Check current iteration
          private$is_predicted_cur_iter <- c(private$is_predicted_cur_iter, FALSE)

        } else if (!is.null(modelfile)) {

          # Do we have a model file as character?
          if (!is.character(modelfile)) {
            stop("gpb.Booster: Can only use a string as model file path")
          }

          # Create booster from model
          handle <- gpb.call("LGBM_BoosterCreateFromModelfile_R",
                             ret = handle,
                             gpb.c_str(modelfile))

        } else if (!is.null(model_str)) {

          # Do we have a model_str as character?
          if (!is.character(model_str)) {
            stop("gpb.Booster: Can only use a string as model_str")
          }

          # Create booster from model
          handle <- gpb.call("LGBM_BoosterLoadModelFromString_R",
                             ret = handle,
                             gpb.c_str(model_str))

        } else {

          # Booster non existent
          stop("gpb.Booster: Need at least either training dataset, model file, or model_str to create booster instance")

        }

      })

      # Check whether the handle was created properly if it was not stopped earlier by a stop call
      if (gpb.is.null.handle(handle)) {

        stop("gpb.Booster: cannot create Booster handle")

      } else {

        # Create class
        class(handle) <- "gpb.Booster.handle"
        private$handle <- handle
        private$num_class <- 1L
        private$num_class <- gpb.call("LGBM_BoosterGetNumClasses_R",
                                      ret = private$num_class,
                                      private$handle)

      }

    },

    # Set training data name
    set_train_data_name = function(name) {

      # Set name
      private$name_train_set <- name
      return(invisible(self))

    },

    # Add validation data
    add_valid = function(data, name, valid_set_gp = NULL, use_gp_model_for_validation = FALSE) {

      # Check if data is gpb.Dataset
      if (!gpb.check.r6.class(data, "gpb.Dataset")) {
        stop("gpb.Booster.add_valid: Can only use gpb.Dataset as validation data")
      }

      # Check if predictors are identical
      if (!identical(data$.__enclos_env__$private$predictor, private$init_predictor)) {
        stop("gpb.Booster.add_valid: Failed to add validation data; you should use the same predictor for these data")
      }

      # Check if names are character
      if (!is.character(name)) {
        stop("gpb.Booster.add_valid: Can only use characters as data name")
      }

      # Add validation data to booster
      gpb.call("LGBM_BoosterAddValidData_R",
               ret = NULL,
               private$handle,
               data$.__enclos_env__$private$get_handle())

      # Store private information
      private$valid_sets <- c(private$valid_sets, data)
      private$name_valid_sets <- c(private$name_valid_sets, name)
      private$num_dataset <- private$num_dataset + 1
      private$is_predicted_cur_iter <- c(private$is_predicted_cur_iter, FALSE)
      
      # Note: Validation using the GP model is only done in R if there are custom evaluation functions in feval, 
      #        otherwise it is directly done in C++. See the function Eval() in regression_metric.hpp
      if (private$has_gp_model & use_gp_model_for_validation) {

        if (!is.null(valid_set_gp)) {
          if (!is.list(valid_set_gp)) {
            stop("gpb.Booster.add_valid: Can only use lists as valid_set_gp")
          }
          private$valid_sets_gp = append(private$valid_sets_gp, list(valid_set_gp))
        }

        private$use_gp_model_for_validation = use_gp_model_for_validation
        
      }
      
      # Return self
      return(invisible(self))

    },

    # Reset parameters of booster
    reset_parameter = function(params, ...) {

      # Append parameters
      params <- append(params, list(...))
      params_str <- gpb.params2str(params)

      # Reset parameters
      gpb.call("LGBM_BoosterResetParameter_R",
               ret = NULL,
               private$handle,
               params_str)

      # Return self
      return(invisible(self))

    },

    # Perform boosting update iteration
    update = function(train_set = NULL, fobj = NULL) {

      # Check if training set is not null
      if (!is.null(train_set)) {

        # Check if training set is gpb.Dataset
        if (!gpb.check.r6.class(train_set, "gpb.Dataset")) {
          stop("gpb.Booster.update: Only can use gpb.Dataset as training data")
        }

        # Check if predictors are identical
        if (!identical(train_set$predictor, private$init_predictor)) {
          stop("gpb.Booster.update: Change train_set failed, you should use the same predictor for these data")
        }

        # Reset training data on booster
        gpb.call("LGBM_BoosterResetTrainingData_R",
                 ret = NULL,
                 private$handle,
                 train_set$.__enclos_env__$private$get_handle())

        # Store private train set
        private$train_set = train_set

      }

      # Check if objective is empty
      if (is.null(fobj)) {
        if (private$set_objective_to_none) {
          stop("gpb.Booster.update: cannot update due to null objective function")
        }
        # Boost iteration from known objective
        ret <- gpb.call("LGBM_BoosterUpdateOneIter_R", ret = NULL, private$handle)

      } else {

        # Check if objective is function
        if (!is.function(fobj)) {
          stop("gpb.Booster.update: fobj should be a function")
        }
        if (!private$set_objective_to_none) {
          self$reset_parameter(params = list(objective = "none"))
          private$set_objective_to_none = TRUE
        }
        # Perform objective calculation
        gpair <- fobj(private$inner_predict(1), private$train_set)

        # Check for gradient and hessian as list
        if(is.null(gpair$grad) || is.null(gpair$hess)){
          stop("gpb.Booster.update: custom objective should
            return a list with attributes (hess, grad)")
        }

        # Return custom boosting gradient/hessian
        ret <- gpb.call("LGBM_BoosterUpdateOneIterCustom_R",
                        ret = NULL,
                        private$handle,
                        gpair$grad,
                        gpair$hess,
                        length(gpair$grad))

      }

      # Loop through each iteration
      for (i in seq_along(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }

      return(ret)

    },

    # Return one iteration behind
    rollback_one_iter = function() {

      # Return one iteration behind
      gpb.call("LGBM_BoosterRollbackOneIter_R",
               ret = NULL,
               private$handle)

      # Loop through each iteration
      for (i in seq_along(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }

      # Return self
      return(invisible(self))

    },

    # Get current iteration
    current_iter = function() {

      cur_iter <- 0L
      gpb.call("LGBM_BoosterGetCurrentIteration_R",
               ret = cur_iter,
               private$handle)

    },

    # Evaluate data on metrics
    eval = function(data, name, feval = NULL) {

      # Check if dataset is gpb.Dataset
      if (!gpb.check.r6.class(data, "gpb.Dataset")) {
        stop("gpb.Booster.eval: Can only use gpb.Dataset to eval")
      }

      # Check for identical data
      data_idx <- 0
      if (identical(data, private$train_set)) {
        data_idx <- 1
      } else {

        # Check for validation data
        if (length(private$valid_sets) > 0) {

          # Loop through each validation set
          for (i in seq_along(private$valid_sets)) {

            # Check for identical validation data with training data
            if (identical(data, private$valid_sets[[i]])) {

              # Found identical data, skip
              data_idx <- i + 1
              break

            }

          }

        }

      }

      # Check if evaluation was not done
      if (data_idx == 0) {

        # Add validation data by name
        self$add_valid(data, name)
        data_idx <- private$num_dataset

      }

      # Evaluate data
      private$inner_eval(name, data_idx, feval)

    },

    # Evaluation training data
    eval_train = function(feval = NULL) {
      private$inner_eval(private$name_train_set, 1, feval)
    },

    # Evaluation validation data
    eval_valid = function(feval = NULL) {

      # Create ret list
      ret = list()

      # Check if validation is empty
      if (length(private$valid_sets) <= 0) {
        return(ret)
      }

      # Loop through each validation set
      for (i in seq_along(private$valid_sets)) {
        ret <- append(ret, private$inner_eval(private$name_valid_sets[[i]], i + 1, feval))
      }

      # Return ret
      return(ret)

    },

    # Save model
    save_model = function(filename, num_iteration = NULL) {

      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      # Save booster model
      gpb.call("LGBM_BoosterSaveModel_R",
               ret = NULL,
               private$handle,
               as.integer(num_iteration),
               gpb.c_str(filename))

      # Return self
      return(invisible(self))
    },

    # Save model to string
    save_model_to_string = function(num_iteration = NULL) {

      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      # Return model string
      return(gpb.call.return.str("LGBM_BoosterSaveModelToString_R",
                                 private$handle,
                                 as.integer(num_iteration)))

    },

    # Dump model in memory
    dump_model = function(num_iteration = NULL) {

      # Check if number of iteration is non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      # Return dumped model
      gpb.call.return.str("LGBM_BoosterDumpModel_R",
                          private$handle,
                          as.integer(num_iteration))

    },

    # Predict on new data
    predict = function(data,
                       num_iteration = NULL,
                       rawscore = FALSE,
                       predleaf = FALSE,
                       predcontrib = FALSE,
                       header = FALSE,
                       reshape = FALSE,
                       group_data_pred = NULL,
                       group_rand_coef_data_pred = NULL,
                       gp_coords_pred = NULL,
                       gp_rand_coef_data_pred = NULL,
                       cluster_ids_pred = NULL,
                       vecchia_pred_type = NULL,
                       num_neighbors_pred = -1,
                       predict_cov_mat = FALSE,...) {

      # Check if number of iteration is  non existent
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      # Predict on new data
      predictor <- Predictor$new(private$handle, ...)
      
      if (private$has_gp_model) {
        # Check for empty data
        if (is.null(private$train_set$.__enclos_env__$private$raw_data)) {
          stop("predict: cannot make predictions for Gaussian process.
                Set ", sQuote("free_raw_data = FALSE"), " when you construct gpb.Dataset")
        }
        
        fixed_effect_train = predictor$predict(private$train_set$.__enclos_env__$private$raw_data,
                                              num_iteration, FALSE, FALSE, FALSE, FALSE, FALSE)
        residual = private$train_set$.__enclos_env__$private$info$label-fixed_effect_train
        random_effect_pred = private$gp_model$predict(y=residual,
                                           group_data_pred = group_data_pred,
                                           group_rand_coef_data_pred = group_rand_coef_data_pred,
                                           gp_coords_pred = gp_coords_pred,
                                           gp_rand_coef_data_pred = gp_rand_coef_data_pred,
                                           cluster_ids_pred = cluster_ids_pred,
                                           predict_cov_mat = predict_cov_mat,
                                           cov_pars = NULL,
                                           X_pred = NULL,
                                           vecchia_pred_type = vecchia_pred_type,
                                           num_neighbors_pred = num_neighbors_pred)
        fixed_effect = predictor$predict(data, num_iteration, rawscore, predleaf, predcontrib, header, reshape)
        
        if (length(fixed_effect) != length(random_effect_pred$mu)){
          warning("Number of data points in fixed effect (tree ensemble) and random effect are not equal")
        }
        
        return(list(fixed_effect = fixed_effect,
                    random_effect_mean = random_effect_pred$mu,
                    random_effect_cov = random_effect_pred$cov))
      }
      else {
        predictor$predict(data, num_iteration, rawscore, predleaf, predcontrib, header, reshape)
      }
      

    },

    # Transform into predictor
    to_predictor = function() {
      Predictor$new(private$handle)
    },

    # Used for save
    raw = NA,

    # Save model to temporary file for in-memory saving
    save = function() {

      # Overwrite model in object
      self$raw <- self$save_model_to_string(NULL)

    }

  ),
  private = list(
    handle = NULL,
    train_set = NULL,
    gp_model = NULL,
    has_gp_model = FALSE,
    name_train_set = "training",
    valid_sets = list(),
    valid_sets_gp = list(),
    use_gp_model_for_validation = FALSE,
    name_valid_sets = list(),
    predict_buffer = list(),
    is_predicted_cur_iter = list(),
    num_class = 1,
    num_dataset = 0,
    init_predictor = NULL,
    eval_names = NULL,
    higher_better_inner_eval = NULL,
    set_objective_to_none = FALSE,
    # Predict data
    inner_predict = function(idx) {

      # Store data name
      data_name <- private$name_train_set

      # Check for id bigger than 1
      if (idx > 1) {
        data_name <- private$name_valid_sets[[idx - 1]]
      }

      # Check for unknown dataset (over the maximum provided range)
      if (idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }

      # Check for prediction buffer
      if (is.null(private$predict_buffer[[data_name]])) {

        # Store predictions
        npred <- 0L
        npred <- gpb.call("LGBM_BoosterGetNumPredict_R",
                          ret = npred,
                          private$handle,
                          as.integer(idx - 1))
        private$predict_buffer[[data_name]] <- numeric(npred)

      }

      # Check if current iteration was already predicted
      if (!private$is_predicted_cur_iter[[idx]]) {

        # Use buffer
        private$predict_buffer[[data_name]] <- gpb.call("LGBM_BoosterGetPredict_R",
                                                        ret = private$predict_buffer[[data_name]],
                                                        private$handle,
                                                        as.integer(idx - 1))
        private$is_predicted_cur_iter[[idx]] <- TRUE
      }

      # Return prediction buffer
      return(private$predict_buffer[[data_name]])
    },

    # Get evaluation information
    get_eval_info = function() {

      # Check for evaluation names emptiness
      if (is.null(private$eval_names)) {

        # Get evaluation names
        names <- gpb.call.return.str("LGBM_BoosterGetEvalNames_R",
                                     private$handle)

        # Check names' length
        if (nchar(names) > 0) {

          # Parse and store privately names
          names <- strsplit(names, "\t")[[1]]
          private$eval_names <- names
          private$higher_better_inner_eval <- grepl("^ndcg|^map|^auc$", names)

        }

      }

      # Return evaluation names
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
      if (length(private$eval_names) > 0) {

        # Create evaluation values
        tmp_vals <- numeric(length(private$eval_names))
        tmp_vals <- gpb.call("LGBM_BoosterGetEval_R",
                             ret = tmp_vals,
                             private$handle,
                             as.integer(data_idx - 1))

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
        if (data_idx > 1) {
          data <- private$valid_sets[[data_idx - 1]]
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
                                                        cov_pars = NULL,
                                                        X_pred = NULL,
                                                        use_saved_data = FALSE)$mu
          
          predval = predval + random_effect_pred
          
        }
        res <- feval(predval, data)

        # Check for name correctness
        if(is.null(res$name) || is.null(res$value) ||  is.null(res$higher_better)) {
          stop("gpb.Booster.eval: custom eval function should return a
            list with attribute (name, value, higher_better)");
        }

        # Append names and evaluation
        res$data_name <- data_name
        ret <- append(ret, list(res))
      }

      # Return ret
      return(ret)

    }

  )
)


#' Predict method for GPBoost model
#'
#' Predicted values based on class \code{gpb.Booster}
#'
#' @param object Object of class \code{gpb.Booster}
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#' @param rawscore whether the prediction should be returned in the for of original untransformed
#'        sum of predictions from boosting iterations' results. E.g., setting \code{rawscore=TRUE} for
#'        logistic regression would result in predictions for log-odds instead of probabilities.
#' @param predleaf whether predict leaf index instead.
#' @param predcontrib return per-feature contributions for each record.
#' @param header only used for prediction for text file. True if text file has header
#' @param reshape whether to reshape the vector of predictions to a matrix form when there are several
#'        prediction outputs per case.
#' @param ... Additional named arguments passed to the \code{predict()} method of
#'            the \code{gpb.Booster}. In particular, this includes prediction data for the GPModel (if there is one)
#' @return
#' For regression or binary classification, it returns a vector of length \code{nrows(data)}.
#' For multiclass classification, either a \code{num_class * nrows(data)} vector or
#' a \code{(nrows(data), num_class)} dimension matrix is returned, depending on
#' the \code{reshape} value.
#'
#' When \code{predleaf = TRUE}, the output is a matrix object with the
#' number of columns corresponding to the number of trees.
#'
#' @examples
#' ## SEE ALSO THE HELP OF 'gpb.train' FOR MORE EXAMPLES
#' \dontrun{
#' library(gpboost)
#' 
#' #--------------------Example without a Gaussian process or random effects model--------------
#' # Non-linear function for simulation
#' f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
#' x <- seq(from=0,to=1,length.out=200)
#' plot(x,f1d(x),type="l",lwd=2,col="red",main="Mean function")
#' # Function that simulates data. Two covariates of which only one has an effect
#' sim_data <- function(n){
#'   X=matrix(runif(2*n),ncol=2)
#'   # mean function plus noise
#'   y=f1d(X[,1])+rnorm(n,sd=0.1)
#'   return(list(X=X,y=y))
#' }
#' # Simulate data
#' n <- 1000
#' set.seed(1)
#' data <- sim_data(2 * n)
#' dtrain <- gpb.Dataset(data = data$X[1:n,], label = data$y[1:n])
#' Xtest <- data$X[1:n + n,]
#' ytest <- data$y[1:n + n]
#' 
#' # Train model
#' print("Train boosting model")
#' bst <- gpb.train(data = dtrain,
#'                  nrounds = 40,
#'                  learning_rate = 0.1,
#'                  max_depth = 6,
#'                  min_data_in_leaf = 5,
#'                  objective = "regression_l2",
#'                  verbose = 0)
#'
#' # Make predictions and compare fit to truth
#' x <- seq(from=0,to=1,length.out=200)
#' Xtest_plot <- cbind(x,rep(0,length(x)))
#' pred_plot <- predict(bst, data = Xtest_plot)
#' plot(x,f1d(x),type="l",ylim = c(-0.25,3.25), col = "red", lwd = 2,
#'      main = "Comparison of true and fitted value")
#' lines(x,pred_plot, col = "blue", lwd = 2)
#' legend("bottomright", legend = c("truth", "fitted"),
#'        lwd=2, col = c("red", "blue"), bty = "n")
#'        
#' # Prediction accuracy       
#' pred <- predict(bst, data = Xtest)
#' err <- mean((ytest-pred)^2)
#' print(paste("test-RMSE =", err))
#' 
#' 
#' #--------------------Combine tree-boosting and Gaussian process model----------------
#' # Simulate data
#' # Function for non-linear mean. Two covariates of which only one has an effect
#' f1d=function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
#' set.seed(2)
#' n <- 200 # number of samples
#' X=matrix(runif(2*n),ncol=2)
#' y <- f1d(X[,1]) # mean
#' # Add Gaussian process
#' sigma2_1 <- 1^2 # marginal variance of GP
#' rho <- 0.1 # range parameter
#' sigma2 <- 0.1^2 # error variance
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' D <- as.matrix(dist(coords))
#' Sigma = sigma2_1*exp(-D/rho)+diag(1E-20,n)
#' C = t(chol(Sigma))
#' b_1=rnorm(n) # simulate random effect
#' eps <- C %*% b_1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- y + eps + xi # add random effects and error to data
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
#' # Default optimizer for covariance parameters is Fisher scoring.
#' # This can be changed as follows:
#' # re_params <- list(optimizer_cov = "gradient_descent", lr_cov = 0.05,
#' #                   use_nesterov_acc = TRUE, acc_rate_cov = 0.5)
#' # gp_model$set_optim_params(params=re_params)
#'
#' # Train model
#' print("Train boosting with Gaussian process model")
#' bst <- gpboost(data = X,
#'                label = y,
#'                gp_model = gp_model,
#'                nrounds = 8,
#'                learning_rate = 0.1,
#'                max_depth = 6,
#'                min_data_in_leaf = 5,
#'                objective = "regression_l2",
#'                verbose = 0)
#' print("Estimated random effects model")
#' summary(gp_model)
#'
#' # Make predictions
#' set.seed(1)
#' ntest <- 5
#' Xtest=matrix(runif(2*ntest),ncol=2)
#' # prediction locations (=features) for Gaussian process
#' coords_test <- cbind(runif(ntest),runif(ntest))/10
#' pred <- predict(bst, data = Xtest, gp_coords_pred = coords_test,
#'                 predict_cov_mat =TRUE)
#' print("Predicted (posterior) mean of GP")
#' pred$random_effect_mean
#' print("Predicted (posterior) covariance matrix of GP")
#' pred$random_effect_cov
#' print("Predicted fixed effect from tree ensemble")
#' pred$fixed_effect
#' }
#'
#' @rdname predict.gpb.Booster
#' @export
predict.gpb.Booster <- function(object,
                                data,
                                num_iteration = NULL,
                                rawscore = FALSE,
                                predleaf = FALSE,
                                predcontrib = FALSE,
                                header = FALSE,
                                reshape = FALSE,
                                ...) {

  # Check booster existence
  if (!gpb.is.Booster(object)) {
    stop("predict.gpb.Booster: object should be an ", sQuote("gpb.Booster"))
  }

  # Return booster predictions
  object$predict(data,
                 num_iteration,
                 rawscore,
                 predleaf,
                 predcontrib,
                 header,
                 reshape, ...)
}

#' Load model
#'
#' Load model from saved model file or string
#' Load takes in either a file path or model string
#' If both are provided, Load will default to loading from file
#'
#' @param filename path of model file
#' @param model_str a str containing the model
#'
#' @return gpb.Booster
#'
#' @examples
#' \dontrun{
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- gpb.train(params = params,
#'                    data = dtrain,
#'                    nrounds = 10,
#'                    valids = valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#' gpb.save(model, "model.txt")
#' load_booster <- gpb.load(filename = "model.txt")
#' model_string <- model$save_model_to_string(NULL) # saves best iteration
#' load_booster_from_str <- gpb.load(model_str = model_string)
#' }
#'
#' @rdname gpb.load
#' @export
gpb.load <- function(filename = NULL, model_str = NULL){

  if (is.null(filename) && is.null(model_str)) {
    stop("gpb.load: either filename or model_str must be given")
  }

  # Load from filename
  if (!is.null(filename) && !is.character(filename)) {
    stop("gpb.load: filename should be character")
  }

  # Return new booster
  if (!is.null(filename) && !file.exists(filename)) stop("gpb.load: file does not exist for supplied filename")
  if (!is.null(filename)) return(invisible(Booster$new(modelfile = filename)))

  # Load from model_str
  if (!is.null(model_str) && !is.character(model_str)) {
    stop("gpb.load: model_str should be character")
  }
  # Return new booster
  if (!is.null(model_str)) return(invisible(Booster$new(model_str = model_str)))

}

#' Save model
#'
#' Save model
#'
#' @param booster Object of class \code{gpb.Booster}
#' @param filename saved filename
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#'
#' @return gpb.Booster
#'
#' @examples
#' \dontrun{
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- gpb.train(params = params,
#'                    data = dtrain,
#'                    nrounds = 10,
#'                    valids = valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#' gpb.save(model, "model.txt")
#' }
#'
#' @rdname gpb.save
#' @export
gpb.save <- function(booster, filename, num_iteration = NULL){

  # Check if booster is booster
  if (!gpb.is.Booster(booster)) {
    stop("gpb.save: booster should be an ", sQuote("gpb.Booster"))
  }

  # Check if file name is character
  if (!is.character(filename)) {
    stop("gpb.save: filename should be a character")
  }

  # Store booster
  invisible(booster$save_model(filename, num_iteration))

}

#' Dump model to json
#'
#' Dump model to json
#'
#' @param booster Object of class \code{gpb.Booster}
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#'
#' @return json format of model
#'
#' @examples
#' \dontrun{
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- gpb.train(params = params,
#'                    data = dtrain,
#'                    nrounds = 10,
#'                    valids = valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#' json_model <- gpb.dump(model)
#' }
#' 
#' @rdname gpb.dump
#' @export
gpb.dump <- function(booster, num_iteration = NULL){

  # Check if booster is booster
  if (!gpb.is.Booster(booster)) {
    stop("gpb.dump: booster should be an ", sQuote("gpb.Booster"))
  }

  # Return booster at requested iteration
  booster$dump_model(num_iteration)

}

#' Get record evaluation result from booster
#'
#' Get record evaluation result from booster
#' @param booster Object of class \code{gpb.Booster}
#' @param data_name name of dataset
#' @param eval_name name of evaluation
#' @param iters iterations, NULL will return all
#' @param is_err TRUE will return evaluation error instead
#'
#' @return vector of evaluation result
#'
#' @examples
#' \dontrun{
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- gpb.train(params = params,
#'                    data = dtrain,
#'                    nrounds = 10,
#'                    valids = valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#' gpb.get.eval.result(model, "test", "l2")
#' }
#' @rdname gpb.get.eval.result
#' @export
gpb.get.eval.result <- function(booster, data_name, eval_name, iters = NULL, is_err = FALSE) {

  # Check if booster is booster
  if (!gpb.is.Booster(booster)) {
    stop("gpb.get.eval.result: Can only use ", sQuote("gpb.Booster"), " to get eval result")
  }

  # Check if data and evaluation name are characters or not
  if (!is.character(data_name) || !is.character(eval_name)) {
    stop("gpb.get.eval.result: data_name and eval_name should be characters")
  }

  # Check if recorded evaluation is existing
  if (is.null(booster$record_evals[[data_name]])) {
    stop("gpb.get.eval.result: wrong data name")
  }

  # Check if evaluation result is existing
  if (is.null(booster$record_evals[[data_name]][[eval_name]])) {
    stop("gpb.get.eval.result: wrong eval name")
  }

  # Create result
  result <- booster$record_evals[[data_name]][[eval_name]]$eval

  # Check if error is requested
  if (is_err) {
    result <- booster$record_evals[[data_name]][[eval_name]]$eval_err
  }

  # Check if iteration is non existant
  if (is.null(iters)) {
    return(as.numeric(result))
  }

  # Parse iteration and booster delta
  iters <- as.integer(iters)
  delta <- booster$record_evals$start_iter - 1
  iters <- iters - delta

  # Return requested result
  as.numeric(result[iters])
}
