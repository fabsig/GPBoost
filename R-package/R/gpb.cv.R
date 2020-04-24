#' @importFrom R6 R6Class
CVBooster <- R6::R6Class(
  classname = "gpb.CVBooster",
  cloneable = FALSE,
  public = list(
    best_iter = -1,
    best_score = NA,
    record_evals = list(),
    boosters = list(),
    initialize = function(x) {
      self$boosters <- x
    },
    reset_parameter = function(new_params) {
      for (x in boosters) { x$reset_parameter(new_params) }
      self
    }
  )
)

#' @title Main CV logic for GPBoost
#' @description Cross validation logic used by GPBoost
#' @name gpb.cv
#' @inheritParams gpb_shared_params
#' @param data A matrix with covariate (feature) data or a \code{gpb.Dataset} object containing covariate data for training
#' @param nfold The original dataset is randomly partitioned into \code{nfold} equal size subsamples.
#' @param showsd \code{boolean}, whether to show standard deviation of cross validation
#' @param stratified a \code{boolean} indicating whether sampling of folds should be stratified
#'        by the values of outcome labels.
#' @param folds \code{list} provides a possibility to use a list of pre-defined CV folds
#'        (each element must be a vector of test fold's indices). When folds are supplied,
#'        the \code{nfold} and \code{stratified} parameters are ignored.
#' @param fit_GP_cov_pars_OOS Boolean (default = FALSE). If TRUE, the covariance parameters of the 
#'            GPModel model are estimated using the out-of-sample (OOS) predictions 
#'            on the validation data using the optimal number of iterations (after performing the CV)
#' @param ... Other parameters, see Parameters.rst for more information.
#'
#' @return a trained model \code{gpb.CVBooster}.
#'
#' @examples
#' \dontrun{
#' require(gpboost)
#'
#' #--------------------Cross validation for tree-boosting without GP or random effects----------------
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
#' dtrain <- gpb.Dataset(data$X[1:n,], label = data$y[1:n])
#' nrounds <- 100
#' params <- list(learning_rate = 0.1,
#'                max_depth = 6,
#'                min_data_in_leaf = 5,
#'                objective = "regression_l2")
#'
#' print("Running cross validation with mean squared error")
#' bst <- gpb.cv(params = params,
#'               data = dtrain,
#'               nrounds = nrounds,
#'               nfold = 10,
#'               eval = "l2",
#'               early_stopping_rounds = 5)
#' print(paste0("Optimal number of iterations: ", bst$best_iter))
#'
#' print("Running cross validation with mean absolute error")
#' bst <- gpb.cv(params = params,
#'               data = dtrain,
#'               nrounds = nrounds,
#'               nfold = 10,
#'               eval = "l1",
#'               early_stopping_rounds = 5)
#' print(paste0("Optimal number of iterations: ", bst$best_iter))
#'
#'
#' #--------------------Custom loss function----------------
#' # Cross validation can also be done with a cutomized loss function
#' # Define custom loss (quantile loss)
#' quantile_loss <- function(preds, dtrain) {
#'   alpha <- 0.95
#'   labels <- getinfo(dtrain, "label")
#'   y_diff <- as.numeric(labels-preds)
#'   dummy <- ifelse(y_diff<0,1,0)
#'   quantloss <- mean((alpha-dummy)*y_diff)
#'   return(list(name = "quant_loss", value = quantloss, higher_better = FALSE))
#' }
#'
#' print("Running cross validation, with cutomsized loss function (quantile loss)")
#' bst <- gpb.cv(params = params,
#'               data = dtrain,
#'               nrounds = nrounds,
#'               nfold = 10,
#'               eval = quantile_loss,
#'               early_stopping_rounds = 5)
#' print(paste0("Optimal number of iterations: ", bst$best_iter))
#'
#'
#' #--------------------Combine tree-boosting and grouped random effects model----------------
#' # Simulate data
#' f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
#' set.seed(1)
#' n <- 1000 # number of samples
#' X <- matrix(runif(2*n),ncol=2)
#' y <- f1d(X[,1]) # mean
#' # Add grouped random effects
#' m <- 25 # number of categories / levels for grouping variable
#' group <- rep(1,n) # grouping variable
#' for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
#' sigma2_1 <- 1^2 # random effect variance
#' sigma2 <- 0.1^2 # error variance
#' # incidence matrix relating grouped random effects to samples
#' Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
#' b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
#' eps <- Z1 %*% b1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- y + eps + xi # add random effects and error to data
#'
#' # Create random effects model and dataset
#' gp_model <- GPModel(group_data = group)
#' dtrain <- gpb.Dataset(X, label = y)
#' params <- list(learning_rate = 0.05,
#'                max_depth = 6,
#'                min_data_in_leaf = 5,
#'                objective = "regression_l2")
#'
#' print("Running cross validation for GPBoost model")
#' bst <- gpb.cv(params = params,
#'               data = dtrain,
#'               gp_model = gp_model,
#'               nrounds = 100,
#'               nfold = 10,
#'               eval = "l2",
#'               early_stopping_rounds = 5)
#' print(paste0("Optimal number of iterations: ", bst$best_iter))
#'
#' # Include random effect predictions for validation (observe the lower test error)
#' gp_model <- GPModel(group_data = group)
#' print("Running cross validation for GPBoost model and use_gp_model_for_validation = TRUE")
#' bst <- gpb.cv(params = params,
#'               data = dtrain,
#'               gp_model = gp_model,
#'               use_gp_model_for_validation = TRUE,
#'               nrounds = 100,
#'               nfold = 10,
#'               eval = "l2",
#'               early_stopping_rounds = 5)
#' print(paste0("Optimal number of iterations: ", bst$best_iter))
#' }
#' @export
gpb.cv <- function(params = list(),
                   data,
                   nrounds = 100,
                   nfold = 4,
                   label = NULL,
                   obj = NULL,
                   gp_model = NULL,
                   use_gp_model_for_validation = FALSE,
                   fit_GP_cov_pars_OOS = FALSE,
                   train_gp_model_cov_pars = TRUE,
                   weight = NULL,
                   eval = NULL,
                   verbose = 1,
                   record = TRUE,
                   eval_freq = 1L,
                   showsd = FALSE,
                   stratified = TRUE,
                   folds = NULL,
                   init_model = NULL,
                   colnames = NULL,
                   categorical_feature = NULL,
                   early_stopping_rounds = NULL,
                   callbacks = list(),
                   reset_data = FALSE,
                   ...) {
  
  # Setup temporary variables
  addiction_params <- list(...)
  params <- append(params, addiction_params)
  params$verbose <- verbose
  params <- gpb.check.obj(params, obj)
  params <- gpb.check.eval(params, eval)
  fobj <- NULL
  feval <- NULL
  
  params$use_gp_model_for_validation <- use_gp_model_for_validation
  params$train_gp_model_cov_pars <- train_gp_model_cov_pars
  if (!is.null(gp_model)) {
    params["has_gp_model"] <- TRUE
  }
  
  if (nrounds <= 0) {
    stop("nrounds should be greater than zero")
  }
  
  # Check for objective (function or not)
  if (is.function(params$objective)) {
    fobj <- params$objective
    params$objective <- "NONE"
  }
  
  # Check for loss (function or not)
  if (is.function(eval)) {
    feval <- eval
  }
  
  # Check for parameters
  gpb.check.params(params)
  
  # Init predictor to empty
  predictor <- NULL
  
  # Check for boosting from a trained model
  if (is.character(init_model)) {
    predictor <- Predictor$new(init_model)
  } else if (gpb.is.Booster(init_model)) {
    predictor <- init_model$to_predictor()
  }
  
  # Set the iteration to start from / end to (and check for boosting from a trained model, again)
  begin_iteration <- 1
  if (!is.null(predictor)) {
    begin_iteration <- predictor$current_iter() + 1
  }
  # Check for number of rounds passed as parameter - in case there are multiple ones, take only the first one
  n_trees <- c("num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees", "num_round", "num_rounds", "num_boost_round", "n_estimators")
  if (any(names(params) %in% n_trees)) {
    end_iteration <- begin_iteration + params[[which(names(params) %in% n_trees)[1]]] - 1
  } else {
    end_iteration <- begin_iteration + nrounds - 1
  }
  
  # Check for training dataset type correctness
  if (!gpb.is.Dataset(data)) {
    if (is.null(label)) {
      stop("Labels must be provided for gpb.cv")
    }
    data <- gpb.Dataset(data, label = label)
  }
  
  if (!is.null(gp_model)) {
    if (gp_model$get_num_data() != data$dim()[1]) {
      stop("Different number of samples in data and gp_model")
    }
  }
  
  # Check for weights
  if (!is.null(weight)) {
    data$setinfo("weight", weight)
  }
  
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
  
  # Construct datasets, if needed
  data$construct()
  
  # Check for folds
  if (!is.null(folds)) {
    
    # Check for list of folds or for single value
    if (!is.list(folds) || length(folds) < 2) {
      stop(sQuote("folds"), " must be a list with 2 or more elements that are vectors of indices for each CV-fold")
    }
    
    # Set number of folds
    nfold <- length(folds)
    
  } else {
    
    # Check fold value
    if (nfold <= 1) {
      stop(sQuote("nfold"), " must be > 1")
    }
    
    # Create folds
    folds <- generate.cv.folds(nfold,
                               nrow(data),
                               stratified,
                               getinfo(data, "label"),
                               getinfo(data, "group"),
                               params)
    
  }
  
  # Add printing log callback
  if (verbose > 0 && eval_freq > 0) {
    callbacks <- add.cb(callbacks, cb.print.evaluation(eval_freq))
  }
  
  # Add evaluation log callback
  if (record) {
    callbacks <- add.cb(callbacks, cb.record.evaluation())
  }
  
  # Check for early stopping passed as parameter when adding early stopping callback
  early_stop <- c("early_stopping_round", "early_stopping_rounds", "early_stopping", "n_iter_no_change")
  if (any(names(params) %in% early_stop)) {
    if (params[[which(names(params) %in% early_stop)[1]]] > 0) {
      callbacks <- add.cb(callbacks, cb.early.stop(params[[which(names(params) %in% early_stop)[1]]], verbose = verbose))
    }
  } else {
    if (!is.null(early_stopping_rounds)) {
      if (early_stopping_rounds > 0) {
        callbacks <- add.cb(callbacks, cb.early.stop(early_stopping_rounds, verbose = verbose))
      }
    }
  }
  
  # Categorize callbacks
  cb <- categorize.callbacks(callbacks)
  
  # Construct booster using a list apply, check if requires group or not
  if (!is.list(folds[[1]])) {
    bst_folds <- lapply(seq_along(folds), function(k) {
      dtest <- slice(data, folds[[k]])
      dtrain <- slice(data, seq_len(nrow(data))[-folds[[k]]])
      setinfo(dtrain, "weight", getinfo(data, "weight")[-folds[[k]]])
      setinfo(dtrain, "init_score", getinfo(data, "init_score")[-folds[[k]]])
      setinfo(dtest, "weight", getinfo(data, "weight")[folds[[k]]])
      setinfo(dtest, "init_score", getinfo(data, "init_score")[folds[[k]]])
      
      if (!is.null(gp_model)) {
        
        folds[[k]] = sort(folds[[k]], decreasing = FALSE)##needs to be sorted since sorting is done in slice
        
        group_data_pred <- NULL
        group_data <- gp_model$get_group_data()
        if (!is.null(group_data)) {
          group_data_pred <- group_data[folds[[k]],]
          group_data <- group_data[-folds[[k]],]
        }
        
        group_rand_coef_data_pred <- NULL
        group_rand_coef_data <- gp_model$get_group_rand_coef_data()
        if (!is.null(group_rand_coef_data)) {
          group_rand_coef_data_pred <- group_rand_coef_data[folds[[k]],]
          group_rand_coef_data <- group_rand_coef_data[-folds[[k]],]
        }
        
        gp_coords_pred <- NULL
        gp_coords <- gp_model$get_gp_coords()
        if (!is.null(gp_coords)) {
          gp_coords_pred <- gp_coords[folds[[k]],]
          gp_coords <- gp_coords[-folds[[k]],]
        }
        
        gp_rand_coef_data_pred <- NULL
        gp_rand_coef_data <- gp_model$get_gp_rand_coef_data()
        if (!is.null(gp_rand_coef_data)) {
          gp_rand_coef_data_pred <- gp_rand_coef_data[folds[[k]],]
          gp_rand_coef_data <- gp_rand_coef_data[-folds[[k]],]
        }
        
        cluster_ids_pred <- NULL
        cluster_ids <- gp_model$get_cluster_ids()
        if (!is.null(cluster_ids)) {
          cluster_ids_pred <- cluster_ids[folds[[k]]]
          cluster_ids <- cluster_ids[-folds[[k]]]
        }

        vecchia_approx <- gp_model$.__enclos_env__$private$vecchia_approx
        num_neighbors <- gp_model$.__enclos_env__$private$num_neighbors
        vecchia_ordering <- gp_model$.__enclos_env__$private$vecchia_ordering
        vecchia_pred_type <- gp_model$.__enclos_env__$private$vecchia_pred_type
        num_neighbors_pred <- gp_model$.__enclos_env__$private$num_neighbors_pred
        cov_function <- gp_model$get_cov_function()
        cov_fct_shape <- gp_model$get_cov_fct_shape()
        ind_effect_group_rand_coef <- gp_model$get_ind_effect_group_rand_coef()

        gp_model_train <- gpb.GPModel$new(group_data=group_data,
                                      group_rand_coef_data=group_rand_coef_data,
                                      ind_effect_group_rand_coef=ind_effect_group_rand_coef,
                                      gp_coords=gp_coords,
                                      gp_rand_coef_data=gp_rand_coef_data,
                                      cov_function=cov_function,
                                      cov_fct_shape=cov_fct_shape,
                                      vecchia_approx=vecchia_approx,
                                      num_neighbors=num_neighbors,
                                      vecchia_ordering=vecchia_ordering,
                                      vecchia_pred_type=vecchia_pred_type,
                                      num_neighbors_pred=num_neighbors_pred,
                                      cluster_ids=cluster_ids,
                                      free_raw_data=TRUE)
        gp_model_train$set_optim_params(params = gp_model$get_optim_params())
        
        valid_set_gp <- NULL
        if (use_gp_model_for_validation) {
          gp_model_train$set_prediction_data(group_data_pred = group_data_pred, group_rand_coef_data_pred = group_rand_coef_data_pred,
                                             gp_coords_pred = gp_coords_pred, gp_rand_coef_data_pred = gp_rand_coef_data_pred,
                                             cluster_ids_pred = cluster_ids_pred)
          if (!is.null(feval)) {
            # Note: Validation using the GP model is only done in R if there are custom evaluation functions in feval, 
            #        otherwise it is directly done in C++. See the function Eval() in regression_metric.hpp
            valid_set_gp <- list(group_data_pred = group_data_pred, group_rand_coef_data_pred = group_rand_coef_data_pred,
                                 gp_coords_pred = gp_coords_pred, gp_rand_coef_data_pred = gp_rand_coef_data_pred,
                                 cluster_ids_pred = cluster_ids_pred)
          }
          
        }
        
        booster <- Booster$new(params, dtrain, gp_model = gp_model_train)
      } else {
        booster <- Booster$new(params, dtrain)
      }
      
      booster$add_valid(dtest, "valid", valid_set_gp = valid_set_gp, use_gp_model_for_validation = use_gp_model_for_validation)
      list(booster = booster)
    })
  } else {
    bst_folds <- lapply(seq_along(folds), function(k) {
      dtest <- slice(data, folds[[k]]$fold)
      dtrain <- slice(data, (seq_len(nrow(data)))[-folds[[k]]$fold])
      setinfo(dtrain, "weight", getinfo(data, "weight")[-folds[[k]]$fold])
      setinfo(dtrain, "init_score", getinfo(data, "init_score")[-folds[[k]]$fold])
      setinfo(dtrain, "group", getinfo(data, "group")[-folds[[k]]$group])
      setinfo(dtest, "weight", getinfo(data, "weight")[folds[[k]]$fold])
      setinfo(dtest, "init_score", getinfo(data, "init_score")[folds[[k]]$fold])
      setinfo(dtest, "group", getinfo(data, "group")[folds[[k]]$group])
      booster <- Booster$new(params, dtrain)
      booster$add_valid(dtest, "valid")
      list(booster = booster)
    })
  }
  
  # Create new booster
  cv_booster <- CVBooster$new(bst_folds)
  
  # Callback env
  env <- CB_ENV$new()
  env$model <- cv_booster
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
    msg <- lapply(cv_booster$boosters, function(fd) {
      fd$booster$update(fobj = fobj)
      fd$booster$eval_valid(feval = feval)
    })
    
    # Prepare collection of evaluation results
    merged_msg <- gpb.merge.cv.result(msg)
    
    # Write evaluation result in environment
    env$eval_list <- merged_msg$eval_list
    
    # Check for standard deviation requirement
    if(showsd) {
      env$eval_err_list <- merged_msg$eval_err_list
    }
    
    # Loop through env
    for (f in cb$post_iter) {
      f(env)
    }
    
    # Check for early stopping and break if needed
    if (env$met_early_stop) break
    
  }
  
  if (record && is.na(env$best_score)) {
    if (env$eval_list[[1]]$higher_better[1] == TRUE) {
      cv_booster$best_iter <- unname(which.max(unlist(cv_booster$record_evals[[2]][[1]][[1]])))
      cv_booster$best_score <- cv_booster$record_evals[[2]][[1]][[1]][[cv_booster$best_iter]]
    } else {
      cv_booster$best_iter <- unname(which.min(unlist(cv_booster$record_evals[[2]][[1]][[1]])))
      cv_booster$best_score <- cv_booster$record_evals[[2]][[1]][[1]][[cv_booster$best_iter]]
    }
  }
  
  if (!is.null(gp_model) & fit_GP_cov_pars_OOS) {
    pred_fixed_effect_OOS <- rep(NA,nrow(data))
    for (k in 1:nfold) {
      fd <- cv_booster$boosters[[k]]
      ##Predict on OOS data
      predictor <- Predictor$new(fd$booster$.__enclos_env__$private$handle)
      # print(fd$booster$.__enclos_env__$private$valid_sets)
      pred_fixed_effect_OOS[folds[[k]]] = predictor$predict(data$.__enclos_env__$private$raw_data[folds[[k]],],
                                                  cv_booster$best_iter, FALSE, FALSE, FALSE, FALSE, FALSE)
    }
    message("Fitting GPModel on out-of-sample data...")
    gp_model$fit(y = data$.__enclos_env__$private$info$label - pred_fixed_effect_OOS)
    summary(gp_model)
    
  }
  
  if (reset_data) {
    lapply(cv_booster$boosters, function(fd) {
      # Store temporarily model data elsewhere
      booster_old <- list(best_iter = fd$booster$best_iter,
                          best_score = fd$booster$best_score,
                          record_evals = fd$booster$record_evals)
      # Reload model
      fd$booster <- gpb.load(model_str = fd$booster$save_model_to_string())
      fd$booster$best_iter <- booster_old$best_iter
      fd$booster$best_score <- booster_old$best_score
      fd$booster$record_evals <- booster_old$record_evals
    })

  }
  
  # Return booster
  return(cv_booster)
  
}

# Generates random (stratified if needed) CV folds
generate.cv.folds <- function(nfold, nrows, stratified, label, group, params) {
  
  # Check for group existence
  if (is.null(group)) {
    
    # Shuffle
    rnd_idx <- sample.int(nrows)
    
    # Request stratified folds
    if (isTRUE(stratified) && params$objective %in% c("binary", "multiclass") && length(label) == length(rnd_idx)) {
      
      y <- label[rnd_idx]
      y <- factor(y)
      folds <- gpb.stratified.folds(y, nfold)
      
    } else {
      
      # Make simple non-stratified folds
      folds <- list()
      
      # Loop through each fold
      for (i in seq_len(nfold)) {
        kstep <- length(rnd_idx) %/% (nfold - i + 1)
        folds[[i]] <- rnd_idx[seq_len(kstep)]
        rnd_idx <- rnd_idx[-seq_len(kstep)]
      }
      
    }
    
  } else {
    
    # When doing group, stratified is not possible (only random selection)
    if (nfold > length(group)) {
      stop("\n\tYou requested too many folds for the number of available groups.\n")
    }
    
    # Degroup the groups
    ungrouped <- inverse.rle(list(lengths = group, values = seq_along(group)))
    
    # Can't stratify, shuffle
    rnd_idx <- sample.int(length(group))
    
    # Make simple non-stratified folds
    folds <- list()
    
    # Loop through each fold
    for (i in seq_len(nfold)) {
      kstep <- length(rnd_idx) %/% (nfold - i + 1)
      folds[[i]] <- list(fold = which(ungrouped %in% rnd_idx[seq_len(kstep)]),
                         group = rnd_idx[seq_len(kstep)])
      rnd_idx <- rnd_idx[-seq_len(kstep)]
    }
    
  }
  
  # Return folds
  return(folds)
  
}

# Creates CV folds stratified by the values of y.
# It was borrowed from caret::gpb.stratified.folds and simplified
# by always returning an unnamed list of fold indices.
#' @importFrom stats quantile
gpb.stratified.folds <- function(y, k = 10) {
  
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
    if (cuts < 2) { cuts <- 2 }
    if (cuts > 5) { cuts <- 5 }
    y <- cut(y,
             unique(stats::quantile(y, probs = seq.int(0, 1, length.out = cuts))),
             include.lowest = TRUE)
    
  }
  
  if (k < length(y)) {
    
    ## Reset levels so that the possible levels and
    ## the levels in the vector are the same
    y <- factor(as.character(y))
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
      if (numInClass[i] %% k > 0) {
        seqVector <- c(seqVector, sample.int(k, numInClass[i] %% k))
      }
      
      ## Shuffle the integers for fold assignment and assign to this classes's data
      foldVector[y == dimnames(numInClass)$y[i]] <- sample(seqVector)
      
    }
    
  } else {
    
    foldVector <- seq(along = y)
    
  }
  
  # Return data
  out <- split(seq(along = y), foldVector)
  names(out) <- NULL
  out
}

gpb.merge.cv.result <- function(msg, showsd = TRUE) {
  
  # Get CV message length
  if (length(msg) == 0) {
    stop("gpb.cv: size of cv result error")
  }
  
  # Get evaluation message length
  eval_len <- length(msg[[1]])
  
  # Is evaluation message empty?
  if (eval_len == 0) {
    stop("gpb.cv: should provide at least one metric for CV")
  }
  
  # Get evaluation results using a list apply
  eval_result <- lapply(seq_len(eval_len), function(j) {
    as.numeric(lapply(seq_along(msg), function(i) {
      msg[[i]][[j]]$value }))
  })
  
  # Get evaluation
  ret_eval <- msg[[1]]
  
  # Go through evaluation length items
  for (j in seq_len(eval_len)) {
    ret_eval[[j]]$value <- mean(eval_result[[j]])
  }
  
  # Preinit evaluation error
  ret_eval_err <- NULL
  
  # Check for standard deviation
  if (showsd) {
    
    # Parse standard deviation
    for (j in seq_len(eval_len)) {
      ret_eval_err <- c(ret_eval_err,
                        sqrt(mean(eval_result[[j]] ^ 2) - mean(eval_result[[j]]) ^ 2))
    }
    
    # Convert to list
    ret_eval_err <- as.list(ret_eval_err)
    
  }
  
  # Return errors
  list(eval_list = ret_eval,
       eval_err_list = ret_eval_err)
  
}
