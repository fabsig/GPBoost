#' @title Main function for training with GPBoost
#' @name gpb.train
#' @description Main function for training with GPBoost
#' @inheritParams gpb_shared_params
#' @param ... Other parameters, see Parameters.rst for more information.
#' @return a trained booster model \code{gpb.Booster}.
#'
#' @examples
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
#' Xtrain <- data$X[1:n,]
#' ytrain <- data$y[1:n]
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
#' #--------------------Using validation set-------------------------
#' \dontrun{
#' # valids is a list of gpb.Dataset, each of them is tagged with a name
#' dtest <- gpb.Dataset.create.valid(dtrain, data = Xtest, label = ytest)
#' valids <- list(test = dtest)
#'
#' # To train with valids, use gpb.train, which contains more advanced features
#' # valids allows us to monitor the evaluation result on all data in the list
#' print("Training using gpb.train with validation data ")
#' bst <- gpb.train(data = dtrain,
#'                  nrounds = 100,
#'                  learning_rate = 0.1,
#'                  max_depth = 6,
#'                  min_data_in_leaf = 5,
#'                  objective = "regression_l2",
#'                  verbose = 1,
#'                  valids = valids,
#'                  early_stopping_rounds = 5)
#' print(paste0("Optimal number of iterations: ", bst$best_iter))
#'
#' # We can change evaluation metrics, or use multiple evaluation metrics
#' print("Train using gpb.train with multiple validation metrics")
#' bst <- gpb.train(data = dtrain,
#'                  nrounds = 100,
#'                  learning_rate = 0.1,
#'                  max_depth = 6,
#'                  min_data_in_leaf = 5,
#'                  objective = "regression_l2",
#'                  verbose = 1,
#'                  valids = valids,
#'                  eval = c("l2", "l1"),
#'                  early_stopping_rounds = 5)
#'print(paste0("Optimal number of iterations: ", bst$best_iter))
#'
#'
#'#--------------------Nesterov accelerated boosting-------------------------
#' dtrain <- gpb.Dataset(data = Xtrain, label = ytrain)
#' dtest <- gpb.Dataset.create.valid(dtrain, data = Xtest, label = ytest)
#' valids <- list(test = dtest)
#' print("Training using gpb.train with Nesterov acceleration")
#' bst <- gpb.train(data = dtrain,
#'                  nrounds = 100,
#'                  learning_rate = 0.01,
#'                  max_depth = 6,
#'                  min_data_in_leaf = 5,
#'                  objective = "regression_l2",
#'                  verbose = 1,
#'                  valids = valids,
#'                  early_stopping_rounds = 5,
#'                  use_nesterov_acc = TRUE)
#' # Compare fit to truth
#' x <- seq(from=0,to=1,length.out=200)
#' Xtest_plot <- cbind(x,rep(0,length(x)))
#' pred_plot <- predict(bst, data = Xtest_plot)
#' plot(x,f1d(x),type="l",ylim = c(-0.25,3.25), col = "red", lwd = 2,
#'      main = "Comparison of true and fitted value")
#' lines(x,pred_plot, col = "blue", lwd = 2)
#' legend("bottomright", legend = c("truth", "fitted"),
#'        lwd=2, col = c("red", "blue"), bty = "n")
#'
#'               
#' #--------------------Combine tree-boosting and grouped random effects model----------------
#' ## SEE ALSO THE HELP OF 'gpboost' FOR MORE EXAMPLES (in particular a Gaussian process example)
#' # Simulate data
#' # Function for non-linear mean. Two covariates of which only one has an effect
#' f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
#' x=seq(from=0,to=1,length.out=200)
#' plot(x,f1d(x),type="l",lwd=2,col="red",main="Mean function")
#' set.seed(1)
#' n <- 1000 # number of samples
#' X=matrix(runif(2*n),ncol=2)
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
#'
#' #--------------------Training----------------
#' # Create random effects model
#' gp_model <- GPModel(group_data = group)
#' # The properties of the optimizer for the Gaussian process or 
#' #   random effects model can be set as follows
#' # re_params <- list(trace=TRUE,optimizer_cov="gradient_descent",
#' #                   lr_cov = 0.05, use_nesterov_acc = TRUE)
#' re_params <- list(optimizer_cov="fisher_scoring")
#' gp_model$set_optim_params(params=re_params)
#' print("Train boosting with random effects model")
#' bst <- gpboost(data = X,
#'                label = y,
#'                gp_model = gp_model,
#'                nrounds = 16,
#'                learning_rate = 0.05,
#'                max_depth = 6,
#'                min_data_in_leaf = 5,
#'                objective = "regression_l2",
#'                verbose = 0)
#'
#' # Same thing using the gpb.train function
#' print("Training with gpb.train")
#' dtrain <- gpb.Dataset(data = X, label = y)
#' bst <- gpb.train(data = dtrain,
#'                  gp_model = gp_model,
#'                  nrounds = 16,
#'                  learning_rate = 0.05,
#'                  max_depth = 6,
#'                  min_data_in_leaf = 5,
#'                  objective = "regression_l2",
#'                  verbose = 0)
#'
#' print("Estimated random effects model")
#' summary(gp_model)
#' 
#'
#'
#' #--------------------Prediction--------------
#' group_test <- 1:m
#' x <- seq(from=0,to=1,length.out=m)
#' Xtest <- cbind(x,rep(0,length(x)))
#' pred <- predict(bst, data = Xtest, group_data_pred = group_test)

#' # Compare fit to truth: random effects
#' pred_random_effect <- pred$random_effect_mean
#' plot(b1, pred_random_effect, xlab="truth", ylab="predicted",
#'      main="Comparison of true and predicted random effects")
#' abline(a=0,b=1)
#' # Compare fit to truth: fixed effect (mean function)
#' pred_mean <- pred$fixed_effect
#' plot(x,f1d(x),type="l",ylim = c(-0.25,3.25), col = "red", lwd = 2,
#'      main = "Comparison of true and fitted value")
#' points(x,pred_mean, col = "blue", lwd = 2)
#' legend("bottomright", legend = c("truth", "fitted"),
#'        lwd=2, col = c("red", "blue"), bty = "n")
#'
#'
#' #--------------------Using validation set-------------------------
#' set.seed(1)
#' train_ind <- sample.int(n,size=900)
#' dtrain <- gpb.Dataset(data = X[train_ind,], label = y[train_ind])
#' dtest <- gpb.Dataset.create.valid(dtrain, data = X[-train_ind,], label = y[-train_ind])
#' valids <- list(test = dtest)
#' gp_model <- GPModel(group_data = group[train_ind])
#'
#' print("Training with validation data and use_gp_model_for_validation = FALSE ")
#' bst <- gpb.train(data = dtrain,
#'                  gp_model = gp_model,
#'                  nrounds = 100,
#'                  learning_rate = 0.05,
#'                  max_depth = 6,
#'                  min_data_in_leaf = 5,
#'                  objective = "regression_l2",
#'                  verbose = 1,
#'                  valids = valids,
#'                  early_stopping_rounds = 5,
#'                  use_gp_model_for_validation = FALSE)
#' print(paste0("Optimal number of iterations: ", bst$best_iter))
#'
#' # Include random effect predictions for validation (observe the lower test error)
#' gp_model <- GPModel(group_data = group[train_ind])
#' gp_model$set_prediction_data(group_data_pred = group[-train_ind])
#' print("Training with validation data and use_gp_model_for_validation = TRUE ")
#' bst <- gpb.train(data = dtrain,
#'                  gp_model = gp_model,
#'                  nrounds = 5,
#'                  learning_rate = 0.05,
#'                  max_depth = 6,
#'                  min_data_in_leaf = 5,
#'                  objective = "regression_l2",
#'                  verbose = 1,
#'                  valids = valids,
#'                  early_stopping_rounds = 5,
#'                  use_gp_model_for_validation = TRUE)
#' print(paste0("Optimal number of iterations: ", bst$best_iter))
#' }
#' 
#' #--------------------GPBoostOOS algorithm: GP parameters estimated out-of-sample----------------
#' \dontrun{
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
#' # Create random effects model and dataset
#' gp_model <- GPModel(group_data = group)
#' dtrain <- gpb.Dataset(X, label = y)
#' params <- list(learning_rate = 0.05,
#'                max_depth = 6,
#'                min_data_in_leaf = 5,
#'                objective = "regression_l2",
#'                has_gp_model =TRUE)
#' # Stage 1: run cross-validation to (i) determine to optimal number of iterations
#' #           and (ii) to estimate the GPModel on the out-of-sample data
#' cvbst <- gpb.cv(params = params,
#'                 data = dtrain,
#'                 gp_model = gp_model,
#'                 nrounds = 100,
#'                 nfold = 4,
#'                 eval = "l2",
#'                 early_stopping_rounds = 5,
#'                 use_gp_model_for_validation = FALSE,
#'                 fit_GP_cov_pars_OOS = TRUE)
#' print(paste0("Optimal number of iterations: ", cvbst$best_iter))
#' # Fitted model (note: ideally, one would have to find the optimal combination of 
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
#' @export
gpb.train <- function(data,
                      nrounds = 100,
                      obj = NULL,
                      gp_model = NULL,
                      train_gp_model_cov_pars = TRUE,
                      params = list(),
                      valids = list(),
                      early_stopping_rounds = NULL,
                      use_gp_model_for_validation = FALSE,
                      eval = NULL,
                      verbose = 1,
                      record = TRUE,
                      eval_freq = 1,
                      init_model = NULL,
                      colnames = NULL,
                      categorical_feature = NULL,
                      callbacks = list(),
                      reset_data = FALSE,
                      ...) {

  # Setup temporary variables
  additional_params <- list(...)
  params <- append(params, additional_params)
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
  
  if (use_gp_model_for_validation) {
    
    if (is.null(gp_model)) {
      stop("gp_model missing but is should be used for validation")
    }
    
    if (is.null(gp_model$.__enclos_env__$private$num_data_pred)) {
      stop("Prediction data for gp_model has not been set. Call gp_model$set_prediction_data() first")
    }
    
    if (length(valids)>1) {
      stop("Can use only one validation set when use_gp_model_for_validation = TRUE")
    }
    
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
    if (use_gp_model_for_validation) {
      # Note: if this option should be added, it can be done similarly as in gpb.cv using booster$add_valid(..., valid_set_gp = valid_set_gp, ...)
      warning("Using the Gaussian process for making predictions for the validation data is currently not supported for custom validation functions. If you need this feature, contact the developer of this package.")
    }
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
  n_rounds <- c("num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees", "num_round", "num_rounds", "num_boost_round", "n_estimators")
  if (any(names(params) %in% n_rounds)) {
    end_iteration <- begin_iteration + params[[which(names(params) %in% n_rounds)[1]]] - 1
  } else {
    end_iteration <- begin_iteration + nrounds - 1
  }


  # Check for training dataset type correctness
  if (!gpb.is.Dataset(data)) {
    stop("gpb.train: data only accepts gpb.Dataset object")
  }

  # Check for validation dataset type correctness
  if (length(valids) > 0) {

    # One or more validation dataset

    # Check for list as input and type correctness by object
    if (!is.list(valids) || !all(vapply(valids, gpb.is.Dataset, logical(1)))) {
      stop("gpb.train: valids must be a list of gpb.Dataset elements")
    }

    # Attempt to get names
    evnames <- names(valids)

    # Check for names existance
    if (is.null(evnames) || !all(nzchar(evnames))) {
      stop("gpb.train: each element of the valids must have a name tag")
    }
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
  vaild_contain_train <- FALSE
  train_data_name <- "train"
  reduced_valid_sets <- list()

  # Parse validation datasets
  if (length(valids) > 0) {

    # Loop through all validation datasets using name
    for (key in names(valids)) {

      # Use names to get validation datasets
      valid_data <- valids[[key]]

      # Check for duplicate train/validation dataset
      if (identical(data, valid_data)) {
        vaild_contain_train <- TRUE
        train_data_name <- key
        next
      }

      # Update parameters, data
      valid_data$update_params(params)
      valid_data$set_reference(data)
      reduced_valid_sets[[key]] <- valid_data

    }

  }

  # Add printing log callback
  if (verbose > 0 && eval_freq > 0) {
    callbacks <- add.cb(callbacks, cb.print.evaluation(eval_freq))
  }

  # Add evaluation log callback
  if (record && length(valids) > 0) {
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

  # "Categorize" callbacks
  cb <- categorize.callbacks(callbacks)

  # Construct booster with datasets
  booster <- Booster$new(params = params, train_set = data, gp_model = gp_model)
  if (vaild_contain_train) { booster$set_train_data_name(train_data_name) }
  for (key in names(reduced_valid_sets)) {
    booster$add_valid(reduced_valid_sets[[key]], key)
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
    if (length(valids) > 0) {

      # Validation has training dataset?
      if (vaild_contain_train) {
        eval_list <- append(eval_list, booster$eval_train(feval = feval))
      }

      # Has no validation dataset
      eval_list <- append(eval_list, booster$eval_valid(feval = feval))
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

  # When early stopping is not activated, we compute the best iteration / score ourselves by selecting the first metric and the first dataset
  if (record && length(valids) > 0 && is.na(env$best_score)) {
    if (env$eval_list[[1]]$higher_better[1] == TRUE) {
      booster$best_iter <- unname(which.max(unlist(booster$record_evals[[2]][[1]][[1]])))
      booster$best_score <- booster$record_evals[[2]][[1]][[1]][[booster$best_iter]]
    } else {
      booster$best_iter <- unname(which.min(unlist(booster$record_evals[[2]][[1]][[1]])))
      booster$best_score <- booster$record_evals[[2]][[1]][[1]][[booster$best_iter]]
    }
  }

  # Check for booster model conversion to predictor model
  if (reset_data) {

    # Store temporarily model data elsewhere
    booster_old <- list(best_iter = booster$best_iter,
                        best_score = booster$best_score,
                        record_evals = booster$record_evals)

    # Reload model
    booster <- gpb.load(model_str = booster$save_model_to_string())
    booster$best_iter <- booster_old$best_iter
    booster$best_score <- booster_old$best_score
    booster$record_evals <- booster_old$record_evals

  }

  # Return booster
  return(booster)

}
