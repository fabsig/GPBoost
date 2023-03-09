# Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
# Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
# Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.

#' @name gpb_shared_params
#' @title Shared parameter docs
#' @description Parameter docs shared by \code{gpb.train}, \code{gpb.cv}, and \code{gpboost}
#' @param callbacks List of callback functions that are applied at each iteration.
#' @param data a \code{gpb.Dataset} object, used for training. Some functions, such as \code{\link{gpb.cv}},
#'             may allow you to pass other types of data like \code{matrix} and then separately supply
#'             \code{label} as a keyword argument.
#' @param early_stopping_rounds int. Activates early stopping. Requires at least one validation data
#'                              and one metric. When this parameter is non-null,
#'                              training will stop if the evaluation of any metric on any validation set
#'                              fails to improve for \code{early_stopping_rounds} consecutive boosting rounds.
#'                              If training stops early, the returned model will have attribute \code{best_iter}
#'                              set to the iteration number of the best iteration.
#' @param eval evaluation function(s). This can be a character vector, function, or list with a mixture of
#'             strings and functions.
#'
#'             \itemize{
#'                 \item{\bold{a. character vector}:
#'                     If you provide a character vector to this argument, it should contain strings with valid
#'                     evaluation metrics.
#'                     See \href{https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters}{
#'                     the "metric" section of the parameter documentation}
#'                     for a list of valid metrics.
#'                 }
#'                 \item{\bold{b. function}:
#'                      You can provide a custom evaluation function. This
#'                      should accept the keyword arguments \code{preds} and \code{dtrain} and should return a named
#'                      list with three elements:
#'                      \itemize{
#'                          \item{\code{name}: A string with the name of the metric, used for printing
#'                              and storing results.
#'                          }
#'                          \item{\code{value}: A single number indicating the value of the metric for the
#'                              given predictions and true values
#'                          }
#'                          \item{
#'                              \code{higher_better}: A boolean indicating whether higher values indicate a better fit.
#'                              For example, this would be \code{FALSE} for metrics like MAE or RMSE.
#'                          }
#'                      }
#'                 }
#'                 \item{\bold{c. list}:
#'                     If a list is given, it should only contain character vectors and functions.
#'                     These should follow the requirements from the descriptions above.
#'                 }
#'             }
#' @param eval_freq evaluation output frequency, only effect when verbose > 0
#' @param valids a list of \code{gpb.Dataset} objects, used for validation
#' @param record Boolean, TRUE will record iteration message to \code{booster$record_evals}
#' @param colnames feature names, if not null, will use this to overwrite the names in dataset
#' @param categorical_feature categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g.
#'                            \code{c(1L, 10L)} to say "the first and tenth columns").
#' @param init_model path of model file of \code{gpb.Booster} object, will continue training from this model
#' @param nrounds number of boosting iterations (= number of trees). This is the most important tuning parameter for boosting
#' @param obj objective function, can be character or custom objective function. Examples include
#'            \code{regression}, \code{regression_l1}, \code{huber},
#'            \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}
#' @param params list of ("tuning") parameters. 
#' See \href{https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst}{the parameter documentation} for more information. 
#' A few key parameters:
#'            \itemize{
#'                \item{\code{learning_rate}}{ The learning rate, also called shrinkage or damping parameter 
#'                (default = 0.1). An important tuning parameter for boosting. Lower values usually 
#'                lead to higher predictive accuracy but more boosting iterations are needed }
#'                \item{\code{num_leaves}}{ Number of leaves in a tree. Tuning parameter for 
#'                tree-boosting (default = 31)}
#'                \item{\code{min_data_in_leaf}}{ Minimal number of samples per leaf. Tuning parameter for 
#'                tree-boosting (default = 20)}
#'                \item{\code{max_depth}}{ Maximal depth of a tree. Tuning parameter for tree-boosting (default = no limit)}
#'                \item{\code{leaves_newton_update}}{ Set this to TRUE to do a Newton update step for the tree leaves 
#'                after the gradient step. Applies only to Gaussian process boosting (GPBoost algorithm)}
#'                \item{\code{train_gp_model_cov_pars}}{ If TRUE, the covariance parameters of the Gaussian process 
#'                are stimated in every boosting iterations, 
#'                otherwise the gp_model parameters are not estimated. In the latter case, you need to 
#'                either esimate them beforehand or provide the values via 
#'                the 'init_cov_pars' parameter when creating the gp_model (default = TRUE).}
#'                \item{\code{use_gp_model_for_validation}}{ If TRUE, the Gaussian process is also used 
#'                (in addition to the tree model) for calculating predictions on the validation data 
#'                (default = TRUE)}
#'                \item{\code{use_nesterov_acc}}{ Set this to TRUE to do boosting with Nesterov acceleration (default = FALSE). 
#'                Can currently only be used for tree_learner = "serial" (default option)}
#'                \item{\code{nesterov_acc_rate}}{ Acceleration rate for momentum step in case Nesterov accelerated 
#'                boosting is used (default = 0.5)}
#'                \item{\code{oosting}}{ Boosting type. \code{"gbdt"}, \code{"rf"}, \code{"dart"} or \code{"goss"}.
#'                Only \code{"gbdt"} allows for doing Gaussian process boosting.}
#'                \item{num_threads}{ Number of threads. For the best speed, set this to
#'                             the number of real CPU cores(\code{parallel::detectCores(logical = FALSE)}),
#'                             not the number of threads (most CPU using hyper-threading to generate 2 threads
#'                             per CPU core).}
#'            }
#' @param verbose verbosity for output, if <= 0, also will disable the print of evaluation during training
#' @param gp_model A \code{GPModel} object that contains the random effects (Gaussian process and / or grouped random effects) model
#' @param use_gp_model_for_validation Boolean. If TRUE, the \code{gp_model} 
#' (Gaussian process and/or random effects) is also used (in addition to the tree model) for calculating 
#' predictions on the validation data. If FALSE, the \code{gp_model} (random effects part) is ignored 
#' for making predictions and only the tree ensemble is used for making predictions for calculating the validation / test error.
#' @param train_gp_model_cov_pars Boolean. If TRUE, the covariance parameters 
#' of the \code{gp_model} (Gaussian process and/or random effects) are estimated in every 
#' boosting iterations, otherwise the \code{gp_model} parameters are not estimated. 
#' In the latter case, you need to either estimate them beforehand or provide the values via 
#' the \code{init_cov_pars} parameter when creating the \code{gp_model}
#' @section Early Stopping:
#'
#'          "early stopping" refers to stopping the training process if the model's performance on a given
#'          validation set does not improve for several consecutive iterations.
#'
#'          If multiple arguments are given to \code{eval}, their order will be preserved. If you enable
#'          early stopping by setting \code{early_stopping_rounds} in \code{params}, by default all
#'          metrics will be considered for early stopping.
#'
#'          If you want to only consider the first metric for early stopping, pass
#'          \code{first_metric_only = TRUE} in \code{params}. Note that if you also specify \code{metric}
#'          in \code{params}, that metric will be considered the "first" one. If you omit \code{metric},
#'          a default metric will be used based on your choice for the parameter \code{obj} (keyword argument)
#'          or \code{objective} (passed into \code{params}).
#' @keywords internal
NULL

#' @name gpboost
#' @title Train a GPBoost model
#' @description Simple interface for training a GPBoost model.
#' @inheritParams gpb_shared_params
#' @param label Vector of labels, used if \code{data} is not an \code{\link{gpb.Dataset}}
#' @param weight vector of response values. If not NULL, will set to dataset
#' @param ... Additional arguments passed to \code{\link{gpb.train}}. For example
#'     \itemize{
#'        \item{\code{valids}: a list of \code{gpb.Dataset} objects, used for validation}
#'        \item{\code{obj}: objective function, can be character or custom objective function. Examples include
#'                   \code{regression}, \code{regression_l1}, \code{huber},
#'                    \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}}
#'        \item{\code{eval}: evaluation function, can be (a list of) character or custom eval function}
#'        \item{\code{record}: Boolean, TRUE will record iteration message to \code{booster$record_evals}}
#'        \item{\code{colnames}: feature names, if not null, will use this to overwrite the names in dataset}
#'        \item{\code{categorical_feature}: categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g. \code{c(1L, 10L)} to
#'                            say "the first and tenth columns").}
#'        \item{\code{reset_data}: Boolean, setting it to TRUE (not the default value) will transform the booster model
#'                          into a predictor model which frees up memory and the original datasets}
#'     }
#' @inheritSection gpb_shared_params Early Stopping
#' @return a trained \code{gpb.Booster}
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
#' 
#' # Train model
#' bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 16,
#'                learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
#'                objective = "regression_l2", verbose = 0)
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
#' \donttest{
#' #--------------------Combine tree-boosting and Gaussian process model----------------
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
#'                     likelihood = "gaussian")
#' # Train model
#' bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 8,
#'                learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
#'                objective = "regression_l2", verbose = 0)
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
gpboost <- function(data,
                    label = NULL,
                    weight = NULL,
                    params = list(),
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
                    early_stopping_rounds = NULL,
                    init_model = NULL,
                    colnames = NULL,
                    categorical_feature = NULL,
                    callbacks = list(),
                    ...) {
  
  # validate inputs early to avoid unnecessary computation
  if (nrounds <= 0L) {
    stop("nrounds should be greater than zero")
  }
  
  # Set data to a temporary variable
  dtrain <- data
  
  # Check whether data is gpb.Dataset, if not then create gpb.Dataset manually
  if (!gpb.is.Dataset(x = dtrain)) {
    dtrain <- gpb.Dataset(data = data, label = label, weight = weight)
  }
  
  train_args <- list(
    "params" = params
    , "data" = dtrain
    , "nrounds" = nrounds
    , "gp_model" = gp_model
    , "use_gp_model_for_validation" = use_gp_model_for_validation
    , "train_gp_model_cov_pars" = train_gp_model_cov_pars
    , "valids" = valids
    , "obj" = obj
    , "eval" = eval
    , "verbose" = verbose
    , "record" = record
    , "eval_freq" = eval_freq
    , "early_stopping_rounds" = early_stopping_rounds
    , "init_model" = init_model
    , "colnames" = colnames
    , "categorical_feature" = categorical_feature
    , "callbacks" = callbacks
  )

  train_args <- append(train_args, list(...))
  
  if (! "valids" %in% names(train_args)) {
    train_args[["valids"]] <- list()
  }
  
  # Set validation as oneself
  if (verbose > 0L) {
    train_args[["valids"]][["train"]] <- dtrain
  }
  
  # Train a model using the regular way
  bst <- do.call(
    what = gpb.train
    , args = train_args
  )
  
  # # Store model under a specific name
  # bst$save_model(filename = save_name)
  
  return(bst)
}

#' @name agaricus.train
#' @title Training part from Mushroom Data Set
#' @description This data set is originally from the Mushroom data set,
#'              UCI Machine Learning Repository.
#'              This data set includes the following fields:
#'
#'               \itemize{
#'                   \item{\code{label}: the label for each record}
#'                   \item{\code{data}: a sparse Matrix of \code{dgCMatrix} class, with 126 columns.}
#'                }
#'
#' @references
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
#' School of Information and Computer Science.
#'
#' @docType data
#' @keywords datasets
#' @usage data(agaricus.train)
#' @format A list containing a label vector, and a dgCMatrix object with 6513
#' rows and 127 variables
NULL

#' @name agaricus.test
#' @title Test part from Mushroom Data Set
#' @description This data set is originally from the Mushroom data set,
#'              UCI Machine Learning Repository.
#'              This data set includes the following fields:
#'
#'              \itemize{
#'                  \item{\code{label}: the label for each record}
#'                  \item{\code{data}: a sparse Matrix of \code{dgCMatrix} class, with 126 columns.}
#'              }
#' @references
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
#' School of Information and Computer Science.
#'
#' @docType data
#' @keywords datasets
#' @usage data(agaricus.test)
#' @format A list containing a label vector, and a dgCMatrix object with 1611
#' rows and 126 variables
NULL

#' @name bank
#' @title Bank Marketing Data Set
#' @description This data set is originally from the Bank Marketing data set,
#'              UCI Machine Learning Repository.
#'
#'              It contains only the following: bank.csv with 10% of the examples and 17 inputs,
#'              randomly selected from 3 (older version of this dataset with less inputs).
#'
#' @references
#' http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#'
#' S. Moro, P. Cortez and P. Rita. (2014)
#' A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems
#'
#' @docType data
#' @keywords datasets
#' @usage data(bank)
#' @format A data.table with 4521 rows and 17 variables
NULL

#' @name GPBoost_data
#' @title Example data for the GPBoost package
#' @description Simulated example data for the GPBoost package
#'              This data set includes the following fields:
#'               \itemize{
#'                   \item{\code{y}: response variable}
#'                   \item{\code{X}: a matrix with covariate information}
#'                   \item{\code{group_data}: a matrix with categorical grouping variables}
#'                   \item{\code{coords}: a matrix with spatial coordinates}
#'                   \item{\code{X_test}: a matrix with covariate information for predictions}
#'                   \item{\code{group_data_test}: a matrix with categorical grouping variables for predictions}
#'                   \item{\code{coords_test}: a matrix with spatial coordinates for predictions}
#'                }
#'
#' @docType data
#' @keywords datasets
#' @usage data(GPBoost_data)
NULL

#' @name y
#' @title Example data for the GPBoost package
#' @description Response variable for the example data of the GPBoost package
#'
#' @docType data
#' @keywords datasets
#' @usage data(GPBoost_data)
NULL

#' @name X
#' @title Example data for the GPBoost package
#' @description A matrix with covariate data for the example data of the GPBoost package
#'
#' @docType data
#' @keywords datasets
#' @usage data(GPBoost_data)
NULL

#' @name group_data
#' @title Example data for the GPBoost package
#' @description A matrix with categorical grouping variables for the example data of the GPBoost package
#'
#' @docType data
#' @keywords datasets
#' @usage data(GPBoost_data)
NULL

#' @name coords
#' @title Example data for the GPBoost package
#' @description A matrix with spatial coordinates for the example data of the GPBoost package
#'
#' @docType data
#' @keywords datasets
#' @usage data(GPBoost_data)
NULL

#' @name X_test
#' @title Example data for the GPBoost package
#' @description A matrix with covariate information for the predictions for the example data of the GPBoost package
#'
#' @docType data
#' @keywords datasets
#' @usage data(GPBoost_data)
NULL

#' @name group_data_test
#' @title Example data for the GPBoost package
#' @description A matrix with categorical grouping variables for predictions for the example data of the GPBoost package
#'
#' @docType data
#' @keywords datasets
#' @usage data(GPBoost_data)
NULL

#' @name coords_test
#' @title Example data for the GPBoost package
#' @description A matrix with spatial coordinates for predictions for the example data of the GPBoost package
#'
#' @docType data
#' @keywords datasets
#' @usage data(GPBoost_data)
NULL

# Various imports
#' @import methods
#' @importFrom Matrix Matrix
#' @importFrom R6 R6Class
#' @useDynLib lib_gpboost , .registration = TRUE
NULL

# Suppress false positive warnings from R CMD CHECK about
# "unrecognized global variable"
globalVariables(c(
  "."
  , ".N"
  , ".SD"
  , "abs_contribution"
  , "bar_color"
  , "Contribution"
  , "Cover"
  , "Feature"
  , "Frequency"
  , "Gain"
  , "internal_count"
  , "internal_value"
  , "leaf_index"
  , "leaf_parent"
  , "leaf_value"
  , "node_parent"
  , "split_feature"
  , "split_gain"
  , "split_index"
  , "tree_index"
))
