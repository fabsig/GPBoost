#' @name GPModel_shared_params
#' @title Shared parameter docs
#' @description Parameter docs shared by \code{GPModel}, \code{gpb.cv}, and \code{gpboost}
#' @param likelihood A \code{string} specifying the likelihood function (distribution) of the response variable
#' Default = "gaussian"
#' @param group_data A \code{vector} or \code{matrix} with labels of group levels for grouped random effects
#' @param group_rand_coef_data A \code{vector} or \code{matrix} with covariate data for grouped random coefficients
#' @param ind_effect_group_rand_coef A \code{vector} with indices that relate every random coefficients 
#' to a "base" intercept grouped random effect. Counting starts at 1.
#' @param gp_coords A \code{matrix} with coordinates (features) for Gaussian process
#' @param gp_rand_coef_data A \code{vector} or \code{matrix} with covariate data for Gaussian process random coefficients
#' @param cov_function A \code{string} specifying the covariance function for the Gaussian process. 
#' The following covariance functions are available: "exponential", "gaussian", "matern", and "powered_exponential". 
#' We follow the notation and parametrization of Diggle and Ribeiro (2007) except for the Matern covariance 
#' where we follow Rassmusen and Williams (2006)
#' @param cov_fct_shape A \code{numeric} specifying the shape parameter of a covariance function 
#' (=smoothness parameter for Matern covariance, irrelevant for some covariance functions 
#' such as the exponential or Gaussian)
#' @param vecchia_approx A \code{boolean}. If true, the Vecchia approximation is used 
#' @param num_neighbors An \code{integer} specifying the number of neighbors for the Vecchia approximation
#' @param vecchia_ordering A \code{string} specifying the ordering used in the Vecchia approximation. 
#' "none" means the default ordering is used, "random" uses a random ordering
#' @param vecchia_pred_type A \code{string} specifying the type of Vecchia approximation used for making predictions. 
#' "order_obs_first_cond_obs_only" = observed data is ordered first and the neighbors are only observed points, 
#' "order_obs_first_cond_all" = observed data is ordered first and the neighbors are selected among all points 
#' (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, 
#' "latent_order_obs_first_cond_obs_only" = Vecchia approximation for the latent process and observed data is 
#' ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all" = Vecchia approximation 
#' for the latent process and observed data is ordered first and neighbors are selected among all points
#' @param num_neighbors_pred an \code{integer} specifying the number of neighbors for the Vecchia approximation 
#' for making predictions
#' @param cluster_ids A \code{vector} with IDs / labels indicating independent realizations of 
#' random effects / Gaussian processes (same values = same process realization)
#' @param free_raw_data If TRUE, the data (groups, coordinates, covariate data for random coefficients) 
#' is freed in R after initialization
#' @param y A \code{vector} with response variable data
#' @param X A \code{matrix} with covariate data for fixed effects ( = linear regression term)
#' @param params A \code{list} with parameters for the model fitting / optimization
#'             \itemize{
#'                \item{optimizer_cov}{ Optimizer used for estimating covariance parameters. 
#'                Options: "gradient_descent" or "fisher_scoring". Default="fisher_scoring" for Gaussian data
#'                and "gradient_descent" for other likelihoods.}
#'                \item{optimizer_coef}{ Optimizer used for estimating linear regression coefficients, if there are any 
#'                (for the GPBoost algorithm there are usually none). 
#'                Options: "gradient_descent" or "wls". Gradient descent steps are done simultaneously 
#'                with gradient descent steps for the covariance paramters. 
#'                "wls" refers to doing coordinate descent for the regression coefficients using weighted least squares.
#'                Default="wls" for Gaussian data and "gradient_descent" for other likelihoods.}
#'                \item{maxit}{ Maximal number of iterations for optimization algorithm. Default=1000.}
#'                \item{delta_rel_conv}{ Convergence criterion: stop optimization if relative change 
#'                in parameters is below this value. Default=1E-6.}
#'                \item{init_coef}{ Initial values for the regression coefficients (if there are any, can be NULL).
#'                Default=NULL.}
#'                \item{init_cov_pars}{ Initial values for covariance parameters of Gaussian process and 
#'                random effects (can be NULL). Default=NULL.}
#'                \item{lr_coef}{ Learning rate for fixed effect regression coefficients if gradient descent is used.
#'                Default=0.1.}
#'                \item{lr_cov}{ Learning rate for covariance parameters. If <= 0, internal default values are used.
#'                Default value = 0.1 for "gradient_descent" and 1. for "fisher_scoring"}
#'                \item{use_nesterov_acc}{ If TRUE Nesterov acceleration is used.
#'                This is used only for gradient descent. Default=TRUE}
#'                \item{acc_rate_coef}{ Acceleration rate for regression coefficients (if there are any) 
#'                for Nesterov acceleration. Default=0.5.}
#'                \item{acc_rate_cov}{ Acceleration rate for covariance parameters for Nesterov acceleration.
#'                Default=0.5.}
#'                \item{momentum_offset}{ Number of iterations for which no mometum is applied in the beginning.
#'                Default=2.}
#'                \item{trace}{ If TRUE, the value of the gradient is printed for some iterations.
#'                Useful for finding good learning rates. Default=FALSE.}
#'                \item{convergence_criterion}{ The convergence criterion used for terminating the optimization algorithm.
#'                Options: "relative_change_in_log_likelihood" (default) or "relative_change_in_parameters".}
#'                \item{std_dev}{ If TRUE, (asymptotic) standard deviations are calculated for the covariance parameters}
#'            }
NULL


#' @name gpb_shared_params
#' @title Shared parameter docs
#' @description Parameter docs shared by \code{gpb.train}, \code{gpb.cv}, and \code{gpboost}
#' @param callbacks list of callback functions
#'        List of callback functions that are applied at each iteration.
#' @param data A \code{gpb.Dataset} object. Some functions, such as \code{gpboost} and \code{gpb.cv}, 
#'             allow you to pass other types of data such as \code{matrix} and then separately 
#'             supply \code{label} as argument.
#' @param early_stopping_rounds int
#'        Activates early stopping.
#'        Requires at least one validation data and one metric
#'        If there's more than one, will check all of them except the training data
#'        Returns the model with (best_iter + early_stopping_rounds)
#'        If early stopping occurs, the model will have 'best_iter' field
#' @param eval_freq Evaluation output frequency, only effect when verbose > 0
#' @param init_model Path of model file of \code{gpb.Booster} object, will continue training from this model
#' @param nrounds Number of boosting iterations (= number of trees). This is the most important tuning parameter for boosting 
#' @param params List of parameters (many of them tuning paramters), see Parameters.rst for more information. A few key parameters:
#'            \itemize{
#'                \item{learning_rate}{ The learning rate, also called shrinkage or damping parameter 
#'                (default = 0.1). An important tuning parameter for boosting. Lower values usually 
#'                lead to higher predictive accuracy but more boosting iterations are needed }
#'                \item{num_leaves}{ Number of leaves in a tree. Tuning parameter for 
#'                tree-boosting (default = 127)}
#'                \item{min_data_in_leaf}{ Minimal number of samples per leaf. Tuning parameter for 
#'                tree-boosting (default = 20)}
#'                \item{max_depth}{ Maximal depth of a tree. Tuning parameter for tree-boosting (default = no limit)}
#'                \item{leaves_newton_update}{ Set this to TRUE to do a Newton update step for the tree leaves 
#'                after the gradient step. Applies only to Gaussian process boosting (GPBoost algorithm)}
#'                \item{train_gp_model_cov_pars}{ If TRUE, the covariance parameters of the Gaussian process 
#'                are stimated in every boosting iterations, 
#'                otherwise the gp_model parameters are not estimated. In the latter case, you need to 
#'                either esimate them beforehand or provide the values via 
#'                the 'init_cov_pars' parameter when creating the gp_model (default = TRUE).}
#'                \item{use_gp_model_for_validation}{ If TRUE, the Gaussian process is also used 
#'                (in addition to the tree model) for calculating predictions on the validation data 
#'                (default = TRUE)}
#'                \item{use_nesterov_acc}{ Set this to TRUE to do boosting with Nesterov acceleration. 
#'                Can currently only be used for tree_learner = "serial" (default option)}
#'                \item{nesterov_acc_rate}{ Acceleration rate for momentum step in case Nesterov accelerated 
#'                boosting is used (default = 0.5)}
#'                \item{boosting}{ Boosting type. \code{"gbdt"} or \code{"dart"}. Only "gpdt" allows for 
#'                doing Gaussian process boosting}
#'                \item{num_threads}{ Number of threads. For the best speed, set this to
#'                                   the number of real CPU cores, not the number of threads (most
#'                                   CPU using hyper-threading to generate 2 threads per CPU core).}
#'            }
#' @param verbose Verbosity for output, if <= 0, also will disable the print of evaluation during training
#' @param label Vector of response variables / labels, used if \code{data} is not an \code{\link{gpb.Dataset}}
#' @param weight Vector of weights for samples (default = NULL). This is currently not supported for Gaussian process boosting (i.e. it only affects the trees and the Gaussian process or random effects model ignores the weights)
#' @param reset_data Boolean, setting it to TRUE (not the default value) will transform the booster model into a predictor model which frees up memory and the original datasets
#' @param valids A list of \code{gpb.Dataset} objects, used as validation data
#' @param obj Objective function, can be character or custom objective function (default = "regression_l2"). Examples include
#'        \code{regression_l2}, \code{regression_l1}, \code{huber},
#'        \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}. Currently only "regression_l2" supports Gaussian process boosting.
#' @param eval Evaluation function, can be (a list of) \code{strings} (character) or a custom evaluation function.
#'        If this is defined by strings, this is equivalent to specifying the parameter \code{metric}.
#'        For a list of all possible metrics, see https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
#' @param metric A list of \code{strings} for specifying the validation metric.
#'        This can equivalently also be specified in the parameter \code{eval}.
#'        For a list of all possible metrics, see https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
#' @param record Boolean, TRUE will record iteration message to \code{booster$record_evals}
#' @param colnames Feature (covariate) names, if not null, will use this to overwrite the names in dataset
#' @param categorical_feature List of str or int
#'        type int represents index,
#'        type str represents feature names
#' @param callbacks list of callback functions
#'        List of callback functions that are applied at each iteration.
#' @param gp_model A \code{GPModel} object that contains the random effects (Gaussian process and / or grouped random effects) model
#' @param use_gp_model_for_validation Boolean (default = TRUE). If TRUE, the Gaussian process is also used (in addition to the tree model) for calculating predictions on the validation data
#' @param train_gp_model_cov_pars Boolean (default = TRUE). If TRUE, the covariance parameters of the Gaussian process are estimated in every boosting iterations, 
#'                otherwise the gp_model parameters are not estimated. In the latter case, you need to either esimate them beforehand or provide the values via 
#'                the 'init_cov_pars' parameter when creating the gp_model
NULL

#' Training part from Mushroom Data Set
#'
#' This data set is originally from the Mushroom data set,
#' UCI Machine Learning Repository.
#'
#' This data set includes the following fields:
#'
#' \itemize{
#'  \item \code{label} the label for each record
#'  \item \code{data} a sparse Matrix of \code{dgCMatrix} class, with 126 columns.
#' }
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
#' @name agaricus.train
#' @usage data(agaricus.train)
#' @format A list containing a label vector, and a dgCMatrix object with 6513
#' rows and 127 variables
NULL

#' Test part from Mushroom Data Set
#'
#' This data set is originally from the Mushroom data set,
#' UCI Machine Learning Repository.
#'
#' This data set includes the following fields:
#'
#' \itemize{
#'  \item \code{label} the label for each record
#'  \item \code{data} a sparse Matrix of \code{dgCMatrix} class, with 126 columns.
#' }
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
#' @name agaricus.test
#' @usage data(agaricus.test)
#' @format A list containing a label vector, and a dgCMatrix object with 1611
#' rows and 126 variables
NULL

#' Bank Marketing Data Set
#'
#' This data set is originally from the Bank Marketing data set,
#' UCI Machine Learning Repository.
#'
#' It contains only the following: bank.csv with 10% of the examples and 17 inputs,
#' randomly selected from 3 (older version of this dataset with less inputs).
#'
#' @references
#' http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#'
#' S. Moro, P. Cortez and P. Rita. (2014)
#' A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems
#'
#' @docType data
#' @keywords datasets
#' @name bank
#' @usage data(bank)
#' @format A data.table with 4521 rows and 17 variables
NULL

# Various imports
#' @import methods
#' @importFrom R6 R6Class
#' @useDynLib lib_gpboost , .registration = TRUE
NULL

# Suppress false positive warnings from R CMD CHECK about
# "unrecognized global variable"
globalVariables(c(
  "."
  , ".N"
  , ".SD"
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
  , "abs_contribution"
))