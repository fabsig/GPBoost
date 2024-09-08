# Copyright (c) 2020 - 2024 Fabio Sigrist. All rights reserved.
# 
# Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.

#' @name GPModel_shared_params
#' @title Documentation for parameters shared by \code{GPModel}, \code{gpb.cv}, and \code{gpboost}
#' @description Documentation for parameters shared by \code{GPModel}, \code{gpb.cv}, and \code{gpboost}
#' @param likelihood A \code{string} specifying the likelihood function (distribution) of the response variable. 
#' Available options:
#' \itemize{
#' \item{ "gaussian" }
#' \item{ "bernoulli_probit": binary data with Bernoulli likelihood and a probit link function }
#' \item{ "bernoulli_logit": binary data with Bernoulli likelihood and a logit link function }
#' \item{ "gamma": gamma distribution with a with log link function }
#' \item{ "poisson": Poisson distribution with a with log link function }
#' \item{ "negative_binomial": negative binomial distribution with a with log link function }
#' \item{ Note: other likelihoods could be implemented upon request }
#' }
#' @param group_data A \code{vector} or \code{matrix} whose columns are categorical grouping variables. 
#' The elements being group levels defining grouped random effects.
#' The elements of 'group_data' can be integer, double, or character.
#' The number of columns corresponds to the number of grouped (intercept) random effects
#' @param group_rand_coef_data A \code{vector} or \code{matrix} with numeric covariate data 
#' for grouped random coefficients
#' @param ind_effect_group_rand_coef A \code{vector} with integer indices that 
#' indicate the corresponding categorical grouping variable (=columns) in 'group_data' for 
#' every covariate in 'group_rand_coef_data'. Counting starts at 1.
#' The length of this index vector must equal the number of covariates in 'group_rand_coef_data'.
#' For instance, c(1,1,2) means that the first two covariates (=first two columns) in 'group_rand_coef_data'
#' have random coefficients corresponding to the first categorical grouping variable (=first column) in 'group_data',
#' and the third covariate (=third column) in 'group_rand_coef_data' has a random coefficient
#' corresponding to the second grouping variable (=second column) in 'group_data'
#' @param drop_intercept_group_rand_effect A \code{vector} of type \code{logical} (boolean). 
#' Indicates whether intercept random effects are dropped (only for random coefficients). 
#' If drop_intercept_group_rand_effect[k] is TRUE, the intercept random effect number k is dropped / not included. 
#' Only random effects with random slopes can be dropped.
#' @param gp_coords A \code{matrix} with numeric coordinates (= inputs / features) for defining Gaussian processes
#' @param gp_rand_coef_data A \code{vector} or \code{matrix} with numeric covariate data for  
#' Gaussian process random coefficients
#' @param cov_function A \code{string} specifying the covariance function for the Gaussian process. 
#' Available options:
#' \itemize{
#' \item{"exponential": Exponential covariance function (using the parametrization of Diggle and Ribeiro, 2007) }
#' \item{"gaussian": Gaussian, aka squared exponential, covariance function (using the parametrization of Diggle and Ribeiro, 2007) }
#' \item{ "matern": Matern covariance function with the smoothness specified by 
#' the \code{cov_fct_shape} parameter (using the parametrization of Rasmussen and Williams, 2006) }
#' \item{"powered_exponential": powered exponential covariance function with the exponent specified by 
#' the \code{cov_fct_shape} parameter (using the parametrization of Diggle and Ribeiro, 2007) }
#' \item{ "wendland": Compactly supported Wendland covariance function (using the parametrization of Bevilacqua et al., 2019, AOS) }
#' \item{ "matern_space_time": Spatio-temporal Matern covariance function with different range parameters for space and time. 
#' Note that the first column in \code{gp_coords} must correspond to the time dimension }
#' \item{ "matern_ard": anisotropic Matern covariance function with Automatic Relevance Determination (ARD), 
#' i.e., with a different range parameter for every coordinate dimension / column of \code{gp_coords} }
#' \item{ "gaussian_ard": anisotropic Gaussian, aka squared exponential, covariance function with Automatic Relevance Determination (ARD), 
#' i.e., with a different range parameter for every coordinate dimension / column of \code{gp_coords} }
#' }
#' @param cov_fct_shape A \code{numeric} specifying the shape parameter of the covariance function 
#' (=smoothness parameter for Matern covariance)  
#' This parameter is irrelevant for some covariance functions such as the exponential or Gaussian
#' @param gp_approx A \code{string} specifying the large data approximation
#' for Gaussian processes. Available options: 
#' \itemize{
#' \item{"none": No approximation }
#' \item{"vecchia": A Vecchia approximation; see Sigrist (2022, JMLR) for more details }
#' \item{"tapering": The covariance function is multiplied by 
#' a compactly supported Wendland correlation function }
#' \item{"fitc": Fully Independent Training Conditional approximation aka 
#' modified predictive process approximation; see Gyger, Furrer, and Sigrist (2024) for more details }
#' \item{"full_scale_tapering": A full scale approximation combining an 
#' inducing point / predictive process approximation with tapering on the residual process; 
#' see Gyger, Furrer, and Sigrist (2024) for more details }
#' }
#' @param cov_fct_taper_range A \code{numeric} specifying the range parameter 
#' of the Wendland covariance function and Wendland correlation taper function. 
#' We follow the notation of Bevilacqua et al. (2019, AOS)
#' @param cov_fct_taper_shape A \code{numeric} specifying the shape (=smoothness) parameter 
#' of the Wendland covariance function and Wendland correlation taper function. 
#' We follow the notation of Bevilacqua et al. (2019, AOS)
#' @param num_neighbors An \code{integer} specifying the number of neighbors for 
#' the Vecchia approximation. Note: for prediction, the number of neighbors can 
#' be set through the 'num_neighbors_pred' parameter in the 'set_prediction_data'
#' function. By default, num_neighbors_pred = 2 * num_neighbors. Further, 
#' the type of Vecchia approximation used for making predictions is set through  
#' the 'vecchia_pred_type' parameter in the 'set_prediction_data' function
#' @param vecchia_ordering A \code{string} specifying the ordering used in 
#' the Vecchia approximation. Available options:
#' \itemize{
#' \item{"none": the default ordering in the data is used }
#' \item{"random": a random ordering }
#' \item{"time": ordering accorrding to time (only for space-time models) }
#' \item{"time_random_space": ordering according to time and randomly for all 
#' spatial points with the same time points (only for space-time models) }
#' }
#' @param ind_points_selection A \code{string} specifying the method for choosing inducing points
#' Available options:
#' \itemize{
#' \item{"kmeans++: the k-means++ algorithm }
#' \item{"cover_tree": the cover tree algorithm }
#' \item{"random": random selection from data points }
#' }
#' @param num_ind_points An \code{integer} specifying the number of inducing 
#' points / knots for, e.g., a predictive process approximation
#' @param cover_tree_radius A \code{numeric} specifying the radius (= "spatial resolution") 
#' for the cover tree algorithm
#' @param matrix_inversion_method A \code{string} specifying the method used for inverting covariance matrices. 
#' Available options:
#' \itemize{
#' \item{"cholesky": Cholesky factorization }
#' \item{"iterative": iterative methods. A combination of conjugate gradient, Lanczos algorithm, and other methods. 
#' 
#' This is currently only supported for the following cases: 
#' \itemize{
#' \item{likelihood != "gaussian" and gp_approx == "vecchia" (non-Gaussian likelihoods with a Vecchia-Laplace approximation) }
#' \item{likelihood == "gaussian" and gp_approx == "full_scale_tapering" (Gaussian likelihood with a full-scale tapering approximation) }
#' }
#' }
#' }
#' @param seed An \code{integer} specifying the seed used for model creation 
#' (e.g., random ordering in Vecchia approximation)
#' @param vecchia_pred_type A \code{string} specifying the type of Vecchia approximation used for making predictions.
#' Default value if vecchia_pred_type = NULL: "order_obs_first_cond_obs_only". 
#' Available options:
#' \itemize{
#' \item{"order_obs_first_cond_obs_only": Vecchia approximation for the observable process and observed training data is 
#' ordered first and the neighbors are only observed training data points }
#' \item{"order_obs_first_cond_all": Vecchia approximation for the observable process and observed training data is 
#' ordered first and the neighbors are selected among all points (training + prediction) }
#' \item{"latent_order_obs_first_cond_obs_only": Vecchia approximation for the latent process and observed data is 
#' ordered first and neighbors are only observed points}
#' \item{"latent_order_obs_first_cond_all": Vecchia approximation 
#' for the latent process and observed data is ordered first and neighbors are selected among all points }
#' \item{"order_pred_first": Vecchia approximation for the observable process and prediction data is 
#' ordered first for making predictions. This option is only available for Gaussian likelihoods }
#' }
#' @param num_neighbors_pred an \code{integer} specifying the number of neighbors for the Vecchia approximation 
#' for making predictions. Default value if NULL: num_neighbors_pred = 2 * num_neighbors
#' @param cg_delta_conv_pred a \code{numeric} specifying the tolerance level for L2 norm of residuals for 
#' checking convergence in conjugate gradient algorithms when being used for prediction
#' Default value if NULL: 1e-3
#' @param nsim_var_pred an \code{integer} specifying the number of samples when simulation 
#' is used for calculating predictive variances
#' Default value if NULL: 1000
#' @param rank_pred_approx_matrix_lanczos an \code{integer} specifying the rank 
#' of the matrix for approximating predictive covariances obtained using the Lanczos algorithm
#' Default value if NULL: 1000
#' @param cluster_ids A \code{vector} with elements indicating independent realizations of 
#' random effects / Gaussian processes (same values = same process realization).
#' The elements of 'cluster_ids' can be integer, double, or character.
#' @param free_raw_data A \code{boolean}. If TRUE, the data (groups, coordinates, covariate data for random coefficients) 
#' is freed in R after initialization
#' @param y A \code{vector} with response variable data
#' @param X A \code{matrix} with numeric covariate data for the 
#' fixed effects linear regression term (if there is one)
#' @param params A \code{list} with parameters for the estimation / optimization
#'             \itemize{
#'                \item{optimizer_cov: \code{string} (default = "lbfgs"). 
#'                Optimizer used for estimating covariance parameters. 
#'                Options: "gradient_descent", "lbfgs", "fisher_scoring", "newton", "nelder_mead", "adam".
#'                If there are additional auxiliary parameters for non-Gaussian likelihoods, 
#'                'optimizer_cov' is also used for those }
#'                \item{optimizer_coef: \code{string} (default = "wls" for Gaussian likelihoods and "lbfgs" for other likelihoods). 
#'                Optimizer used for estimating linear regression coefficients, if there are any 
#'                (for the GPBoost algorithm there are usually none). 
#'                Options: "gradient_descent", "lbfgs", "wls", "nelder_mead", "adam". Gradient descent steps are done simultaneously 
#'                with gradient descent steps for the covariance parameters. 
#'                "wls" refers to doing coordinate descent for the regression coefficients using weighted least squares.
#'                If 'optimizer_cov' is set to "nelder_mead", "lbfgs", or "adam", 
#'                'optimizer_coef' is automatically also set to the same value.}
#'                \item{maxit: \code{integer} (default = 1000). 
#'                Maximal number of iterations for optimization algorithm }
#'                \item{delta_rel_conv: \code{numeric} (default = 1E-6 except for "nelder_mead" for which the default is 1E-8). 
#'                Convergence tolerance. The algorithm stops if the relative change 
#'                in either the (approximate) log-likelihood or the parameters is below this value. 
#'                For "adam", the L2 norm of the gradient is used instead of the relative change in the log-likelihood. 
#'                If < 0, internal default values are used }
#'                \item{convergence_criterion: \code{string} (default = "relative_change_in_log_likelihood"). 
#'                The convergence criterion used for terminating the optimization algorithm.
#'                Options: "relative_change_in_log_likelihood" or "relative_change_in_parameters" }
#'                \item{init_coef: \code{vector} with \code{numeric} elements (default = NULL). 
#'                Initial values for the regression coefficients (if there are any, can be NULL) }
#'                \item{init_cov_pars: \code{vector} with \code{numeric} elements (default = NULL). 
#'                Initial values for covariance parameters of Gaussian process and 
#'                random effects (can be NULL). The order it the same as the order 
#'                of the parameters in the summary function: first is the error variance 
#'                (only for "gaussian" likelihood), next follow the variances of the 
#'                grouped random effects (if there are any, in the order provided in 'group_data'), 
#'                and then follow the marginal variance and the range of the Gaussian process. 
#'                If there are multiple Gaussian processes, then the variances and ranges follow alternatingly.
#'                If 'init_cov_pars = NULL', an internal choice is used that depends on the 
#'                likelihood and the random effects type and covariance function. 
#'                If you select the option 'trace = TRUE' in the 'params' argument, 
#'                you will see the first initial covariance parameters in iteration 0. }
#'                \item{lr_coef: \code{numeric} (default = 0.1). 
#'                Learning rate for fixed effect regression coefficients if gradient descent is used }
#'                \item{lr_cov: \code{numeric} (default = 0.1 for "gradient_descent" and 1. otherwise). 
#'                Initial learning rate for covariance parameters if a gradient-based optimization method is used 
#'                \itemize{
#'                \item{If lr_cov < 0, internal default values are used (0.1 for "gradient_descent" and 1. otherwise) }
#'                \item{If there are additional auxiliary parameters for non-Gaussian likelihoods, 
#'                'lr_cov' is also used for those }
#'                \item{For "lbfgs", this is divided by the norm of the gradient in the first iteration }}}
#'                \item{use_nesterov_acc: \code{boolean} (default = TRUE). 
#'                If TRUE Nesterov acceleration is used.
#'                This is used only for gradient descent }
#'                \item{acc_rate_coef: \code{numeric} (default = 0.5). 
#'                Acceleration rate for regression coefficients (if there are any) 
#'                for Nesterov acceleration }
#'                \item{acc_rate_cov: \code{numeric} (default = 0.5). 
#'                Acceleration rate for covariance parameters for Nesterov acceleration }
#'                \item{momentum_offset: \code{integer} (Default = 2)}. 
#'                Number of iterations for which no momentum is applied in the beginning.
#'                \item{trace: \code{boolean} (default = FALSE). 
#'                If TRUE, information on the progress of the parameter
#'                optimization is printed}
#'                \item{std_dev: \code{boolean} (default = TRUE). 
#'                If TRUE, approximate standard deviations are calculated for the covariance and linear regression parameters 
#'                (= square root of diagonal of the inverse Fisher information for Gaussian likelihoods and 
#'                square root of diagonal of a numerically approximated inverse Hessian for non-Gaussian likelihoods) }
#'                \item{init_aux_pars: \code{vector} with \code{numeric} elements (default = NULL). 
#'                Initial values for additional parameters for non-Gaussian likelihoods 
#'                (e.g., shape parameter of a gamma or negative_binomial likelihood) }
#'                \item{estimate_aux_pars: \code{boolean} (default = TRUE). 
#'                If TRUE, additional parameters for non-Gaussian likelihoods 
#'                are also estimated (e.g., shape parameter of a gamma or negative_binomial likelihood) }
#'                \item{cg_max_num_it: \code{integer} (default = 1000). 
#'                Maximal number of iterations for conjugate gradient algorithms }
#'                \item{cg_max_num_it_tridiag: \code{integer} (default = 1000). 
#'                Maximal number of iterations for conjugate gradient algorithm 
#'                when being run as Lanczos algorithm for tridiagonalization }
#'                \item{cg_delta_conv: \code{numeric} (default = 1E-2).
#'                Tolerance level for L2 norm of residuals for checking convergence 
#'                in conjugate gradient algorithm when being used for parameter estimation }
#'                \item{num_rand_vec_trace: \code{integer} (default = 50). 
#'                Number of random vectors (e.g., Rademacher) for stochastic approximation of the trace of a matrix }
#'                \item{reuse_rand_vec_trace: \code{boolean} (default = TRUE). 
#'                If true, random vectors (e.g., Rademacher) for stochastic approximations 
#'                of the trace of a matrix are sampled only once at the beginning of 
#'                the parameter estimation and reused in later trace approximations.
#'                Otherwise they are sampled every time a trace is calculated }
#'                \item{seed_rand_vec_trace: \code{integer} (default = 1). 
#'                Seed number to generate random vectors (e.g., Rademacher) }
#'                \item{piv_chol_rank: \code{integer} (default = 50). 
#'                Rank of the pivoted Cholesky decomposition used as 
#'                preconditioner in conjugate gradient algorithms }
#'                \item{cg_preconditioner_type: \code{string}.
#'                Type of preconditioner used for conjugate gradient algorithms.
#'                \itemize{
#'                  \item Options for non-Gaussian likelihoods and gp_approx = "vecchia": 
#'                    \itemize{
#'                      \item{"Sigma_inv_plus_BtWB" (= default): (B^T * (D^-1 + W) * B) as preconditioner for inverting (B^T * D^-1 * B + W), 
#'                  where B^T * D^-1 * B approx= Sigma^-1 }
#'                  }
#'                      \item{"piv_chol_on_Sigma": (Lk * Lk^T + W^-1) as preconditioner for inverting (B^-1 * D * B^-T + W^-1), 
#'                  where Lk is a low-rank pivoted Cholesky approximation for Sigma and B^-1 * D * B^-T approx= Sigma }
#'                  \item Options for likelihood = "gaussian" and gp_approx = "full_scale_tapering": 
#'                    \itemize{
#'                      \item{"predictive_process_plus_diagonal" (= default): predictive process preconditiioner }
#'                      \item{"none": no preconditioner }
#'                  }
#'                }
#'               }
#'            }
#' @param offset A \code{numeric} \code{vector} with 
#' additional fixed effects contributions that are added to the linear predictor (= offset). 
#' The length of this vector needs to equal the number of training data points.
#' @param fixed_effects This is discontinued. Use the renamed equivalent argument \code{offset} instead
#' @param group_data_pred A \code{vector} or \code{matrix} with elements being group levels 
#' for which predictions are made (if there are grouped random effects in the \code{GPModel})
#' @param group_rand_coef_data_pred A \code{vector} or \code{matrix} with covariate data 
#' for grouped random coefficients (if there are some in the \code{GPModel})
#' @param gp_coords_pred A \code{matrix} with prediction coordinates (=features) for 
#' Gaussian process (if there is a GP in the \code{GPModel})
#' @param gp_rand_coef_data_pred A \code{vector} or \code{matrix} with covariate data for 
#' Gaussian process random coefficients (if there are some in the \code{GPModel})
#' @param cluster_ids_pred A \code{vector} with elements indicating the realizations of 
#' random effects / Gaussian processes for which predictions are made 
#' (set to NULL if you have not specified this when creating the \code{GPModel})
#' @param X_pred A \code{matrix} with prediction covariate data for the 
#' fixed effects linear regression term (if there is one in the \code{GPModel})
#' @param predict_cov_mat A \code{boolean}. If TRUE, the (posterior) 
#' predictive covariance is calculated in addition to the (posterior) predictive mean
#' @param predict_var A \code{boolean}. If TRUE, the (posterior) 
#' predictive variances are calculated
#' @param vecchia_approx Discontinued. Use the argument \code{gp_approx} instead


NULL


#' @importFrom R6 R6Class
gpb.GPModel <- R6::R6Class(
  # Class for random effects model (Gaussian process, grouped random effects, mixed effects models, etc.)
  # Author: Fabio Sigrist
  classname = "GPModel",
  cloneable = FALSE,
  public = list(
    
    # Finalize will free up the handles
    finalize = function() {
      .Call(
        GPB_REModelFree_R
        , private$handle
      )
      private$handle <- NULL
      return(invisible(NULL))
    },
    
    # Initialize will create a GPModel
    initialize = function(likelihood = "gaussian",
                          group_data = NULL,
                          group_rand_coef_data = NULL,
                          ind_effect_group_rand_coef = NULL,
                          drop_intercept_group_rand_effect = NULL,
                          gp_coords = NULL,
                          gp_rand_coef_data = NULL,
                          cov_function = "exponential",
                          cov_fct_shape = 0.5,
                          gp_approx = "none",
                          cov_fct_taper_range = 1.,
                          cov_fct_taper_shape = 0.,
                          num_neighbors = 20L,
                          vecchia_ordering = "random",
                          ind_points_selection = "kmeans++",
                          num_ind_points = 500L,
                          cover_tree_radius = 1.,
                          matrix_inversion_method = "cholesky",
                          seed = 0L,
                          cluster_ids = NULL,
                          free_raw_data = FALSE,
                          modelfile = NULL,
                          model_list = NULL,
                          vecchia_approx = NULL,
                          vecchia_pred_type = NULL,
                          num_neighbors_pred = NULL) {
      
      if (!is.null(vecchia_approx)) {
        stop("GPModel: The argument 'vecchia_approx' is discontinued. Use the argument 'gp_approx' instead")
      }
      if (!is.null(vecchia_pred_type)) {
        stop("GPModel: The argument 'vecchia_pred_type' is discontinued. Use the function 'set_prediction_data' to specify this")
      }
      if (!is.null(num_neighbors_pred)) {
        stop("GPModel: The argument 'num_neighbors_pred' is discontinued. Use the function 'set_prediction_data' to specify this")
      }
      
      if (!is.null(modelfile) | !is.null(model_list)){
        # Load model from file or list
        if (!is.null(modelfile)) {
          if (!(is.character(modelfile) && length(modelfile) == 1L)) {
            stop("gpb.GPModel: modelfile should be a string")
          }
          if (!file.exists(modelfile)) {
            stop(sprintf("gpb.GPModel: file '%s' passed to modelfile does not exist", modelfile))
          }
          # Load data
          model_list = RJSONIO::fromJSON(content=modelfile)
        } else {
          if (!is.list(model_list)) {
            stop("gpb.GPModel: Can only use a list as model_list")
          }
        }
        # Make sure that data in correct format
        MAYBE_CONVERT_TO_MATRIX <- c("cov_pars","group_data", "group_rand_coef_data",
                                     "gp_coords", "gp_rand_coef_data",
                                     "ind_effect_group_rand_coef",
                                     "cluster_ids","coefs","X")
        for (feature in MAYBE_CONVERT_TO_MATRIX) {
          if (!is.null(model_list[[feature]])) {
            if (is.list(model_list[[feature]])) {
              model_list[[feature]] <- matrix(unlist(model_list[[feature]]),
                                              nrow = length(model_list[[feature]]),
                                              byrow = TRUE)
              if (dim(model_list[[feature]])[2]==1) {
                model_list[[feature]] <- as.vector(model_list[[feature]])
              }
            }
          }
        }
        # Set feature data overwriting arguments for constructor
        group_data = model_list[["group_data"]]
        private$nb_groups = model_list[["nb_groups"]]
        group_rand_coef_data = model_list[["group_rand_coef_data"]]
        ind_effect_group_rand_coef = model_list[["ind_effect_group_rand_coef"]]
        drop_intercept_group_rand_effect = model_list[["drop_intercept_group_rand_effect"]]
        gp_coords = model_list[["gp_coords"]]
        gp_rand_coef_data = model_list[["gp_rand_coef_data"]]
        cov_function = model_list[["cov_function"]]
        cov_fct_shape = model_list[["cov_fct_shape"]]
        gp_approx = model_list[["gp_approx"]]
        cov_fct_taper_range = model_list[["cov_fct_taper_range"]]
        cov_fct_taper_shape = model_list[["cov_fct_taper_shape"]]
        num_neighbors = model_list[["num_neighbors"]]
        vecchia_ordering = model_list[["vecchia_ordering"]]
        num_ind_points = model_list[["num_ind_points"]]
        ind_points_selection = model_list[["ind_points_selection"]]
        cover_tree_radius = model_list[["cover_tree_radius"]]
        seed = model_list[["seed"]]
        cluster_ids = model_list[["cluster_ids"]]
        likelihood = model_list[["likelihood"]]
        matrix_inversion_method = model_list[["matrix_inversion_method"]]
        # Set additionally required data
        private$model_has_been_loaded_from_saved_file = TRUE
        private$cov_pars_loaded_from_file = model_list[["cov_pars"]]
        if (!is.null(model_list[["y"]])) {
          private$y_loaded_from_file = model_list[["y"]]
        }
        private$has_covariates = model_list[["has_covariates"]]
        if (model_list[["has_covariates"]]) {
          private$coefs_loaded_from_file = model_list[["coefs"]]
          private$num_coef = model_list[["num_coef"]]
          private$X_loaded_from_file = model_list[["X"]]
          if (is.null(colnames(private$X_loaded_from_file))) {
            private$coef_names <- c(private$coef_names,paste0("Covariate_",1:private$num_coef))
          } else {
            private$coef_names <- c(private$coef_names,colnames(private$X_loaded_from_file))
          }
        }
        private$model_fitted = model_list[["model_fitted"]]
      }# end !is.null(modelfile) | !is.null(model_list)
      
      if(likelihood == "gaussian"){
        private$cov_par_names <- c("Error_term")
      }else{
        private$cov_par_names <- c()
      }
      if (is.null(group_data) & is.null(gp_coords)) {
        stop("GPModel: Both ", sQuote("group_data"), " and " , sQuote("gp_coords"),
             " are NULL. Provide at least one of them.")
      }
      private$matrix_inversion_method <- as.character(matrix_inversion_method)
      private$seed <- as.integer(seed)
      # Set data for grouped random effects
      group_data_c_str <- NULL
      if (!is.null(group_data)) {
        # Check for correct format
        if (!(is.data.frame(group_data) | is.matrix(group_data) | 
              is.numeric(group_data) | is.character(group_data) | is.factor(group_data))) {
          stop("GPModel: Can only use the following types for as ", sQuote("group_data"),": ",
               sQuote("data.frame"), ", ", sQuote("matrix"), ", ", sQuote("character"),
               ", ", sQuote("numeric"), ", ", sQuote("factor"))
        }
        if (is.data.frame(group_data) | is.numeric(group_data) |
            is.character(group_data) | is.factor(group_data)) {
          group_data <- as.matrix(group_data)
        }
        private$num_group_re <- as.integer(dim(group_data)[2])
        private$num_data <- as.integer(dim(group_data)[1])
        private$group_data <- group_data
        if (is.null(colnames(private$group_data))) {
          new_name <- paste0("Group_", 1:private$num_group_re)
        } else {
          new_name <- colnames(private$group_data)
        }
        private$cov_par_names <- c(private$cov_par_names, new_name)
        private$re_comp_names <- c(private$re_comp_names, new_name)
        # Convert to correct format for passing to C
        group_data <- as.vector(group_data)
        group_data_unique <- unique(group_data)
        group_data_unique_c_str <- lapply(group_data_unique, gpb.c_str)
        group_data_c_str <- unlist(group_data_unique_c_str[match(group_data, group_data_unique)])
        # Version 2: slower than above (not used)
        # group_data_c_str <- unlist(lapply(group_data,gpb.c_str))
        # group_data_c_str <- c()# Version 3: much slower
        # for (i in 1:length(group_data)) {
        #   group_data_c_str <- c(group_data_c_str,gpb.c_str(group_data[i]))
        # }
        faux <- function(x) length(unique(x))
        private$nb_groups <- apply(private$group_data,2,faux)
        # Set data for grouped random coefficients
        if (!is.null(group_rand_coef_data)) {
          if (is.numeric(group_rand_coef_data)) {
            group_rand_coef_data <- as.matrix(group_rand_coef_data)
          }
          if (is.matrix(group_rand_coef_data)) {
            # Check whether matrix is the correct type first ("double")
            if (storage.mode(group_rand_coef_data) != "double") {
              storage.mode(group_rand_coef_data) <- "double"
            }
          } else {
            stop("GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("group_rand_coef_data"))
          }
          if (dim(group_rand_coef_data)[1] != private$num_data) {
            stop("GPModel: Number of data points in ", sQuote("group_rand_coef_data"), 
                 " does not match number of data points in ", sQuote("group_data"))
          }
          if (is.null(ind_effect_group_rand_coef)) {
            stop("GPModel: Indices of grouped random effects (", 
                 sQuote("ind_effect_group_rand_coef"), ") for random slopes in ", 
                 sQuote("group_rand_coef_data"), " not provided")
          }
          if (dim(group_rand_coef_data)[2] != length(ind_effect_group_rand_coef)) {
            stop("GPModel: Number of random coefficients in ", 
                 sQuote("group_rand_coef_data"), " does not match number in ", 
                 sQuote("ind_effect_group_rand_coef"))
          }
          if (storage.mode(ind_effect_group_rand_coef) != "integer") {
            storage.mode(ind_effect_group_rand_coef) <- "integer"
          }
          if (!is.null(drop_intercept_group_rand_effect)) {
            if (length(drop_intercept_group_rand_effect) != private$num_group_re) {
              stop("GPModel: Length of ", sQuote("drop_intercept_group_rand_effect"), 
                   " does not match number of random effects")
            }
            if (storage.mode(drop_intercept_group_rand_effect) != "logical") {
              stop("GPModel: Can only use ", sQuote("logical"), " as ",
                   sQuote("drop_intercept_group_rand_effect"))
            }
          }
          private$num_group_rand_coef <- as.integer(dim(group_rand_coef_data)[2])
          private$group_rand_coef_data <- group_rand_coef_data
          group_rand_coef_data <- as.vector(matrix((private$group_rand_coef_data))) #convert to correct format for sending to C
          ind_effect_group_rand_coef <- as.vector(ind_effect_group_rand_coef)
          private$ind_effect_group_rand_coef <- ind_effect_group_rand_coef
          private$drop_intercept_group_rand_effect <- drop_intercept_group_rand_effect
          offset = 1
          if(likelihood != "gaussian"){
            offset = 0
          }
          counter_re <- rep(1,private$num_group_re)
          for (ii in 1:private$num_group_rand_coef) {
            if (is.null(colnames(private$group_rand_coef_data))) {
              new_name <- paste0(private$cov_par_names[ind_effect_group_rand_coef[ii]+offset],
                                 "_rand_coef_nb_",counter_re[ind_effect_group_rand_coef[ii]])
              counter_re[ind_effect_group_rand_coef[ii]] <- counter_re[ind_effect_group_rand_coef[ii]] + 1
            } else {
              new_name <- paste0(private$cov_par_names[ind_effect_group_rand_coef[ii]+offset],
                                 "_rand_coef_",colnames(private$group_rand_coef_data)[ii])
            }
            private$cov_par_names <- c(private$cov_par_names,new_name)
            private$re_comp_names <- c(private$re_comp_names,new_name)
          }
          if (!is.null(private$drop_intercept_group_rand_effect)) {
            if (sum(private$drop_intercept_group_rand_effect) > 0) {
              ind_drop <- ((1:private$num_group_re) + (likelihood == "gaussian"))[private$drop_intercept_group_rand_effect]
              private$cov_par_names <- private$cov_par_names[-ind_drop]
              private$re_comp_names <- private$re_comp_names[-which(private$drop_intercept_group_rand_effect)]
            }
          }
        } # End set data for grouped random coefficients
      } # End set data for grouped random effects
      # Set data for Gaussian process part
      if (!is.null(gp_coords)) {
        # Check for correct format
        if (!(is.data.frame(gp_coords) | is.matrix(gp_coords) | 
              is.numeric(gp_coords))) {
          stop("GPModel: Can only use the following types for as ", sQuote("gp_coords"),": ",
               sQuote("data.frame"), ", ", sQuote("matrix"), ", ", sQuote("numeric"))
        }
        if (is.data.frame(gp_coords) | is.numeric(gp_coords)) {
          gp_coords <- as.matrix(gp_coords)
        }
        if (!is.null(private$num_data)) {
          if (dim(gp_coords)[1] != private$num_data) {
            stop("GPModel: Number of data points in ", sQuote("gp_coords"), 
                 " does not match number of data points in ", sQuote("group_data"))
          }
        } else {
          private$num_data <- as.integer(dim(gp_coords)[1])
        }
        private$num_gp <- 1L
        private$dim_coords <- as.integer(dim(gp_coords)[2])
        private$gp_coords <- gp_coords
        gp_coords <- as.vector(matrix(private$gp_coords)) #convert to correct format for sending to C
        private$cov_function <- as.character(cov_function)
        private$cov_fct_shape <- as.numeric(cov_fct_shape)
        private$gp_approx <- as.character(gp_approx)
        private$cov_fct_taper_range <- as.numeric(cov_fct_taper_range)
        private$cov_fct_taper_shape <- as.numeric(cov_fct_taper_shape)
        private$num_neighbors <- as.integer(num_neighbors)
        private$vecchia_ordering <- as.character(vecchia_ordering)
        private$num_ind_points <- as.integer(num_ind_points)
        private$cover_tree_radius <- as.numeric(cover_tree_radius)
        private$ind_points_selection <- as.character(ind_points_selection)
        if (private$cov_function == "matern_space_time" | private$cov_function == "exponential_space_time") {
          private$cov_par_names <- c(private$cov_par_names,"GP_var", "GP_range_time", "GP_range_space")
        } else if (private$cov_function == "matern_ard" | private$cov_function == "gaussian_ard" | 
                   private$cov_function == "exponential_ard") {
          if (is.null(colnames(gp_coords))) {
            private$cov_par_names <- c(private$cov_par_names,"GP_var", paste0("GP_range_",1:private$dim_coords))
          } else {
            private$cov_par_names <- c(private$cov_par_names,"GP_var", paste0("GP_range_",colnames(gp_coords)))
          }
        } else if (private$cov_function == "wendland") {
          private$cov_par_names <- c(private$cov_par_names,"GP_var")
        } else if (private$cov_function == "matern_estimate_shape") {
          private$cov_par_names <- c(private$cov_par_names,"GP_var", "GP_range", "GP_smoothness")
        } else {
          private$cov_par_names <- c(private$cov_par_names,"GP_var", "GP_range")
        }
        private$re_comp_names <- c(private$re_comp_names,"GP")
        # Set data for GP random coefficients
        if (!is.null(gp_rand_coef_data)) {
          if (is.numeric(gp_rand_coef_data)) {
            gp_rand_coef_data <- as.matrix(gp_rand_coef_data)
          }
          if (is.matrix(gp_rand_coef_data)) {
            # Check whether matrix is the correct type first ("double")
            if (storage.mode(gp_rand_coef_data) != "double") {
              storage.mode(gp_rand_coef_data) <- "double"
            }
          } else {
            stop("GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("gp_rand_coef_data"))
          }
          if (dim(gp_rand_coef_data)[1] != private$num_data) {
            stop("GPModel: Number of data points in ", sQuote("gp_rand_coef_data"), " does not match number of data points")
          }
          private$num_gp_rand_coef <- as.integer(dim(gp_rand_coef_data)[2])
          private$gp_rand_coef_data <- gp_rand_coef_data
          gp_rand_coef_data <- as.vector(matrix(private$gp_rand_coef_data)) #convert to correct format for sending to C
          for (ii in 1:private$num_gp_rand_coef) {
            if (is.null(colnames(private$gp_rand_coef_data))) {
              if (private$cov_function == "matern_space_time" | private$cov_function == "exponential_space_time") {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_nb_", ii,"_var"),
                                           paste0("GP_rand_coef_nb_", ii,"_range_time"),
                                           paste0("GP_rand_coef_nb_", ii,"_range_space"))
              } else if (private$cov_function == "matern_ard" | private$cov_function == "gaussian_ard" | 
                         private$cov_function == "exponential_ard") {
                if (is.null(colnames(gp_coords))) {
                  private$cov_par_names <- c(private$cov_par_names,paste0("GP_rand_coef_nb_", ii,"_var"), 
                                             paste0(paste0("GP_rand_coef_nb_", ii,"_var"),1:private$dim_coords))
                } else {
                  private$cov_par_names <- c(private$cov_par_names,"GP_var", 
                                             paste0(paste0("GP_rand_coef_nb_", ii,"_range"),colnames(gp_coords)))
                }
              } else if (private$cov_function == "wendland") {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_nb_", ii,"_var"))
              } else if (private$cov_function == "matern_estimate_shape") {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_nb_", ii,"_var"),
                                           paste0("GP_rand_coef_nb_", ii,"_range"),
                                           paste0("GP_rand_coef_nb_", ii,"_smoothness"))
              } else {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_nb_", ii,"_var"),
                                           paste0("GP_rand_coef_nb_", ii,"_range"))
              }
            } else {
              if (private$cov_function == "matern_space_time" | private$cov_function == "exponential_space_time") {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_var"),
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_range_time"),
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_range_space"))
              } else if (private$cov_function == "matern_ard" | private$cov_function == "gaussian_ard" | 
                         private$cov_function == "exponential_ard") {
                if (is.null(colnames(gp_coords))) {
                  private$cov_par_names <- c(private$cov_par_names,paste0("GP_rand_coef_nb_", ii,"_var"), 
                                             paste0(paste0("GP_rand_coef_nb_", colnames(private$gp_rand_coef_data)[ii],"_var"),1:private$dim_coords))
                } else {
                  private$cov_par_names <- c(private$cov_par_names,"GP_var", 
                                             paste0(paste0("GP_rand_coef_nb_", colnames(private$gp_rand_coef_data)[ii],"_range"),colnames(gp_coords)))
                }
              } else if (private$cov_function == "wendland") {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_var"))
              }  else if (private$cov_function == "matern_estimate_shape") {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_var"),
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_range"),
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_smoothness"))
              } else {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_var"),
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_range"))
              }
            }
            private$re_comp_names <- c(private$re_comp_names,paste0("GP_rand_coef_nb_", ii))
          }
        } # End set data for GP random coefficients
      } # End set data for Gaussian process part
      # Set IDs for independent processes (cluster_ids)
      if (!is.null(cluster_ids)) {
        if (is.vector(cluster_ids)) {
          if (length(cluster_ids) != private$num_data) {
            stop("GPModel: Length of ", sQuote("cluster_ids"), " does not match number of data points")
          }
          private$cluster_ids = cluster_ids
          # Convert cluster_ids to int and save conversion map
          if (storage.mode(cluster_ids) != "integer") {
            create_map <- TRUE
            if (storage.mode(cluster_ids) == "double") {
              if (all(cluster_ids == floor(cluster_ids))) {
                create_map <- FALSE
                cluster_ids <- as.integer(cluster_ids)
              }
            }
            if (create_map) {
              private$cluster_ids_map_to_int <- structure(1:length(unique(cluster_ids)),names=c(unique(cluster_ids)))
              cluster_ids = private$cluster_ids_map_to_int[cluster_ids] 
            }
          }
        } else {
          stop("GPModel: Can only use ", sQuote("vector"), " as ", sQuote("cluster_ids"))
        }
        cluster_ids <- as.vector(cluster_ids)
      } # End set IDs for independent processes (cluster_ids)
      private$determine_num_cov_pars(likelihood)
      # Create handle for the GPModel
      handle <- NULL
      # Create handle for the GPModel
      handle <- .Call(
        GPB_CreateREModel_R
        , private$num_data
        , cluster_ids
        , group_data_c_str
        , private$num_group_re
        , group_rand_coef_data
        , private$ind_effect_group_rand_coef
        , private$num_group_rand_coef
        , private$drop_intercept_group_rand_effect
        , private$num_gp
        , gp_coords
        , private$dim_coords
        , gp_rand_coef_data
        , private$num_gp_rand_coef
        , private$cov_function
        , private$cov_fct_shape
        , private$gp_approx
        , private$cov_fct_taper_range
        , private$cov_fct_taper_shape
        , private$num_neighbors
        , private$vecchia_ordering
        , private$num_ind_points
        , private$cover_tree_radius
        , private$ind_points_selection
        , likelihood
        , private$matrix_inversion_method
        , private$seed
      )
      # Check whether the handle was created properly if it was not stopped earlier by a stop call
      if (gpb.is.null.handle(handle)) {
        stop("GPModel: Cannot create handle")
      } else {
        # Add class label
        class(handle) <- "GPModel.handle"
        private$handle <- handle
      }
      private$free_raw_data <- free_raw_data
      # Should we free raw data?
      if (isTRUE(private$free_raw_data)) {
        private$group_data <- NULL
        private$group_rand_coef_data <- NULL
        private$gp_coords <- NULL
        private$gp_rand_coef_data <- NULL
        private$cluster_ids <- NULL
      }
      if (private$model_has_been_loaded_from_saved_file) {
        self$set_optim_params(params = model_list[["params"]])
      }
    }, # End initialize
    
    # Find parameters that minimize the negative log-likelihood (=MLE)
    fit = function(y,
                   X = NULL,
                   params = list(),
                   offset = NULL,
                   fixed_effects = NULL) {
      
      if (!is.null(fixed_effects)) {
        stop("The argument 'fixed_effects' is discontinued. Use the renamed equivalent argument 'offset' instead")
      }
      if (gpb.is.null.handle(private$handle)) {
        stop("fit.GPModel: Gaussian process model has not been initialized")
      }
      if ((private$num_cov_pars == 1L && self$get_likelihood_name() == "gaussian") ||
          (private$num_cov_pars == 0L && self$get_likelihood_name() != "gaussian")) {
        stop("fit.GPModel: No random effects (grouped, spatial, etc.) have been defined")
      }
      if (!is.vector(y)) {
        if (is.matrix(y)) {
          if (dim(y)[2] != 1) {
            stop("fit.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("y"))
          }
        } else{
          stop("fit.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("y"))
        }
      }
      if (storage.mode(y) != "double") {
        storage.mode(y) <- "double"
      }
      y <- as.vector(y)
      if (length(y) != private$num_data) {
        stop("fit.GPModel: Number of data points in ", sQuote("y"), " does not match number of data points of initialized model")
      }# end handling of y
      if (!is.null(offset)) {
        private$has_offset <- TRUE
        if (!is.vector(offset)) {
          if (is.matrix(offset)) {
            if (dim(offset)[2] != 1) {
              stop("fit.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("offset"))
            }
          } else{
            stop("fit.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("offset"))
          }
        }
        if (storage.mode(offset) != "double") {
          storage.mode(offset) <- "double"
        }
        offset <- as.vector(offset)
        if (length(offset) != private$num_data) {
          stop("fit.GPModel: Number of data points in ", sQuote("offset"), " does not match number of data points of initialized model")
        }
      } # end handling of offset
      # Set data linear fixed-effects
      if (!is.null(X)) {
        if (is.numeric(X)) {
          X <- as.matrix(X)
        }
        if (is.matrix(X)) {
          if (storage.mode(X) != "double") {
            storage.mode(X) <- "double"
          }
        } else {
          stop("fit.GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("X"))
        }
        if (dim(X)[1] != private$num_data) {
          stop("fit.GPModel: Number of data points in ", sQuote("X"), " does not match number of data points of initialized model")
        }
        private$has_covariates <- TRUE
        private$num_coef <- as.integer(dim(X)[2])
        if (is.null(colnames(X))) {
          private$coef_names <- c(private$coef_names,paste0("Covariate_",1:private$num_coef))
        } else {
          private$coef_names <- c(private$coef_names,colnames(X))
        }
        X <- as.vector(matrix(X))#matrix() is needed in order that all values are contiguous in memory (when colnames is not NULL)
      } else {
        private$has_covariates <- FALSE
      }
      # end handling of X
      self$set_optim_params(params)
      if (is.null(X)) {
        .Call(
          GPB_OptimCovPar_R
          , private$handle
          , y
          , offset
        )
      } else {
        .Call(
          GPB_OptimLinRegrCoefCovPar_R
          , private$handle
          , y
          , X
          , private$num_coef
          , offset
        )
      }
      if (private$params$trace) {
        message(paste0("GPModel: Number of iterations until convergence: ", 
                       self$get_num_optim_iter()))
      }
      private$model_fitted <- TRUE
      return(invisible(self))
    },
    
    # Evaluate the negative log-likelihood
    neg_log_likelihood = function(cov_pars, 
                                  y,
                                  fixed_effects = NULL,
                                  aux_pars = NULL) {
      if (gpb.is.null.handle(private$handle)) {
        stop("GPModel: Gaussian process model has not been initialized")
      }
      if ((private$num_cov_pars == 1L && self$get_likelihood_name() =="gaussian") ||
          (private$num_cov_pars == 0L && self$get_likelihood_name() !="gaussian")) {
        stop("GPModel.neg_log_likelihood: No random effects (grouped, spatial, etc.) have been defined")
      }
      if (is.vector(cov_pars)) {
        if (storage.mode(cov_pars) != "double") {
          storage.mode(cov_pars) <- "double"
        }
        cov_pars <- as.vector(cov_pars)
      } else {
        stop("GPModel.neg_log_likelihood: Can only use ", sQuote("vector"), " as ", 
             sQuote("cov_pars"))
      }
      if (length(cov_pars) != private$num_cov_pars) {
        stop("GPModel.neg_log_likelihood: Number of parameters in ", sQuote("cov_pars"), 
             " does not correspond to numbers of parameters")
      }
      if (!is.vector(y)) {
        if (is.matrix(y)) {
          if (dim(y)[2] != 1) {
            stop("GPModel.neg_log_likelihood: Can only use ", sQuote("vector"), " as ", sQuote("y"))
          }
        } else{
          stop("GPModel.neg_log_likelihood: Can only use ", sQuote("vector"), " as ", sQuote("y"))
        }
      }
      if (storage.mode(y) != "double") {
        storage.mode(y) <- "double"
      }
      y <- as.vector(y)
      if (length(y) != private$num_data) {
        stop("GPModel.neg_log_likelihood: Number of data points in ", sQuote("y"), " does not match number of data points of initialized model")
      }
      if (!is.null(fixed_effects)) {
        if (is.vector(fixed_effects)) {
          if (storage.mode(fixed_effects) != "double") {
            storage.mode(fixed_effects) <- "double"
          }
          fixed_effects <- as.vector(fixed_effects)
        } else {
          stop("GPModel.neg_log_likelihood: Can only use ", sQuote("vector"), " as ", sQuote("fixed_effects"))
        }
        if (length(fixed_effects) != private$num_data) {
          stop("GPModel.neg_log_likelihood: Length of ", sQuote("fixed_effects"), " does not match number of observed data points")
        }
      }# end fixed_effects
      if (!is.null(aux_pars)) {
        self$set_optim_params(params = list(init_aux_pars = aux_pars))
      }
      negll <- 0.
      .Call(
        GPB_EvalNegLogLikelihood_R
        , private$handle
        , y
        , cov_pars
        , fixed_effects
        , negll
      )
      return(negll)
    },
    
    # Set configuration parameters for the optimizer
    set_optim_params = function(params = list()) {
      if (gpb.is.null.handle(private$handle)) {
        stop("GPModel: Gaussian process model has not been initialized")
      }
      private$update_params(params)
      # prepare for calling C++
      optimizer_cov_c_str <- NULL
      if (!is.null(params[["optimizer_cov"]])) {
        optimizer_cov_c_str <- params[["optimizer_cov"]]
      }
      init_cov_pars <- NULL
      if (!is.null(params[["init_cov_pars"]])) {
        init_cov_pars <- params[["init_cov_pars"]]
      }
      optimizer_coef_c_str <- NULL
      if (!is.null(params[["optimizer_coef"]])) {
        optimizer_coef_c_str <- params[["optimizer_coef"]]
      }
      cg_preconditioner_type_c_str <- NULL
      if (!is.null(params[["cg_preconditioner_type"]])) {
        cg_preconditioner_type_c_str <- params[["cg_preconditioner_type"]]
      }
      init_aux_pars <- NULL
      if (!is.null(params[["init_aux_pars"]])) {
        init_aux_pars <- params[["init_aux_pars"]]
      }
      .Call(
        GPB_SetOptimConfig_R
        , private$handle
        , init_cov_pars
        , private$params[["lr_cov"]]
        , private$params[["acc_rate_cov"]]
        , private$params[["maxit"]]
        , private$params[["delta_rel_conv"]]
        , private$params[["use_nesterov_acc"]]
        , private$params[["nesterov_schedule_version"]]
        , private$params[["trace"]]
        , optimizer_cov_c_str
        , private$params[["momentum_offset"]]
        , private$params[["convergence_criterion"]]
        , private$params[["std_dev"]]
        , private$num_coef
        , private$params[["init_coef"]]
        , private$params[["lr_coef"]]
        , private$params[["acc_rate_coef"]]
        , optimizer_coef_c_str
        , private$params[["cg_max_num_it"]]
        , private$params[["cg_max_num_it_tridiag"]]
        , private$params[["cg_delta_conv"]]
        , private$params[["num_rand_vec_trace"]]
        , private$params[["reuse_rand_vec_trace"]]
        , cg_preconditioner_type_c_str
        , private$params[["seed_rand_vec_trace"]]
        , private$params[["piv_chol_rank"]]
        , init_aux_pars
        , private$params[["estimate_aux_pars"]]
      )
      return(invisible(self))
    },
    
    get_optim_params = function() {
      params <- private$params
      params$optimizer_cov <- .Call(
        GPB_GetOptimizerCovPars_R
        , private$handle
      )
      params$optimizer_coef <- .Call(
        GPB_GetOptimizerCoef_R
        , private$handle
      )
      params$cg_preconditioner_type <- .Call(
        GPB_GetCGPreconditionerType_R
        , private$handle
      )
      init_cov_pars <- numeric(private$num_cov_pars)
      .Call(
        GPB_GetInitCovPar_R
        , private$handle
        , init_cov_pars
      )
      if (sum(abs(init_cov_pars - rep(-1,private$num_cov_pars))) < 1E-6) {
        params["init_cov_pars"] <- list(NULL)
      } else {
        params$init_cov_pars <- init_cov_pars
      }
      if (self$get_num_aux_pars() > 0) {
        init_aux_pars <- numeric(self$get_num_aux_pars())
        .Call(
          GPB_GetInitAuxPars_R
          , private$handle
          , init_aux_pars
        )
        if (sum(abs(init_aux_pars - rep(-1,self$get_num_aux_pars()))) < 1E-6) {
          params["init_aux_pars"] <- list(NULL)
        } else {
          params$init_aux_pars <- init_aux_pars
        }
      } else {
        params["init_aux_pars"] <- list(NULL)
      }
      return(params)
    },
    
    get_cov_pars = function() {
      if (private$model_has_been_loaded_from_saved_file) {
        cov_pars <- private$cov_pars_loaded_from_file
      } else {
        private$update_cov_par_names(self$get_likelihood_name())
        if (private$params[["std_dev"]]) {
          optim_pars <- numeric(2 * private$num_cov_pars)
        } else {
          optim_pars <- numeric(private$num_cov_pars)
        }
        .Call(
          GPB_GetCovPar_R
          , private$handle
          , private$params[["std_dev"]]
          , optim_pars
        )
        cov_pars <- optim_pars[1:private$num_cov_pars]
      }
      names(cov_pars) <- private$cov_par_names
      if (private$params[["std_dev"]] & self$get_likelihood_name() == "gaussian") {
        cov_pars_std_dev <- optim_pars[1:private$num_cov_pars+private$num_cov_pars]
        cov_pars <- rbind(cov_pars,cov_pars_std_dev)
        rownames(cov_pars) <- c("Param.", "Std. dev.")
      }
      return(cov_pars)
    },
    
    get_coef = function() {
      if (private$model_has_been_loaded_from_saved_file) {
        coef <- private$coefs_loaded_from_file
      } else {
        if (is.null(private$num_coef)) {
          stop("GPModel: ", sQuote("fit"), " has not been called")
        }
        if (private$params[["std_dev"]]) {
          optim_pars <- numeric(2 * private$num_coef)
        } else {
          optim_pars <- numeric(private$num_coef)
        }
        .Call(
          GPB_GetCoef_R
          , private$handle
          , private$params[["std_dev"]]
          , optim_pars
        )
        coef <- optim_pars[1:private$num_coef]
      }
      names(coef) <- private$coef_names
      if (private$params[["std_dev"]]) {
        coef_std_dev <- optim_pars[1:private$num_coef+private$num_coef]
        coef <- rbind(coef,coef_std_dev)
        rownames(coef) <- c("Param.", "Std. dev.")
      }
      return(coef)
    },
    
    get_aux_pars = function() {
      num_aux_pars <- self$get_num_aux_pars()
      if (num_aux_pars > 0) {
        aux_pars <- numeric(num_aux_pars)
        aux_pars_name <- .Call(
          GPB_GetAuxPars_R
          , private$handle
          , aux_pars
        )
        names(aux_pars) <- rep(aux_pars_name, num_aux_pars)
      } else {
        aux_pars <- NULL
      }
      return(aux_pars)
    },
    
    set_prediction_data = function(vecchia_pred_type = NULL,
                                   num_neighbors_pred = NULL,
                                   cg_delta_conv_pred = NULL,
                                   nsim_var_pred = NULL,
                                   rank_pred_approx_matrix_lanczos = NULL,
                                   group_data_pred = NULL,
                                   group_rand_coef_data_pred = NULL,
                                   gp_coords_pred = NULL,
                                   gp_rand_coef_data_pred = NULL,
                                   cluster_ids_pred = NULL,
                                   X_pred = NULL) {
      num_data_pred <- 0
      group_data_pred_c_str <- NULL
      # Set data for grouped random effects
      if (!is.null(group_data_pred)) {
        # Check for correct format and set meta data
        if (!(is.data.frame(group_data_pred) | is.matrix(group_data_pred) |
              is.numeric(group_data_pred) | is.character(group_data_pred) |
              is.factor(group_data_pred))) {
          stop("set_prediction_data: Can only use the following types for as ", sQuote("group_data_pred"), ": ",
               sQuote("data.frame"), ", ", sQuote("matrix"), ", ", sQuote("character"),
               ", ", sQuote("numeric"), ", ", sQuote("factor"))
        }
        if (is.data.frame(group_data_pred) | is.numeric(group_data_pred) |
            is.character(group_data_pred) | is.factor(group_data_pred)) {
          group_data_pred <- as.matrix(group_data_pred)
        }
        if (dim(group_data_pred)[2] != private$num_group_re) {
          stop("set_prediction_data: Number of grouped random effects in ", sQuote("group_data_pred"), " is not correct")
        }
        num_data_pred <- as.integer(dim(group_data_pred)[1])
        group_data_pred <- as.vector(group_data_pred)
        group_data_pred_unique <- unique(group_data_pred)
        group_data_pred_unique_c_str <- lapply(group_data_pred_unique,gpb.c_str)
        group_data_pred_c_str <- unlist(group_data_pred_unique_c_str[match(group_data_pred,group_data_pred_unique)])
        # Set data for grouped random coefficients
        if (!is.null(group_rand_coef_data_pred)) {
          if (is.numeric(group_rand_coef_data_pred)) {
            group_rand_coef_data_pred <- as.matrix(group_rand_coef_data_pred)
          }
          if (is.matrix(group_rand_coef_data_pred)) {
            if (storage.mode(group_rand_coef_data_pred) != "double") {
              storage.mode(group_rand_coef_data_pred) <- "double"
            }
          } else {
            stop("set_prediction_data: Can only use ", sQuote("matrix"), " as group_rand_coef_data_pred")
          }
          if (dim(group_rand_coef_data_pred)[1] != num_data_pred) {
            stop("set_prediction_data: Number of data points in ", sQuote("group_rand_coef_data_pred"), " does not match number of data points in ", sQuote("group_data_pred"))
          }
          if (dim(group_rand_coef_data_pred)[2] != private$num_group_rand_coef) {
            stop("set_prediction_data: Number of random coef in ", sQuote("group_rand_coef_data_pred"), " is not correct")
          }
          group_rand_coef_data_pred <- as.vector(matrix(group_rand_coef_data_pred))
        } # End set data for grouped random coefficients
      } # End set data for grouped random effects
      # Set data for Gaussian process
      if (!is.null(gp_coords_pred)) {
        if (is.numeric(gp_coords_pred)) {
          gp_coords_pred <- as.matrix(gp_coords_pred)
        }
        if (is.matrix(gp_coords_pred)) {
          if (storage.mode(gp_coords_pred) != "double") {
            storage.mode(gp_coords_pred) <- "double"
          }
        } else {
          stop("set_prediction_data: Can only use ", sQuote("matrix"), " as ", sQuote("gp_coords_pred"))
        }
        if (num_data_pred != 0) {
          if (dim(gp_coords_pred)[1] != num_data_pred) {
            stop("set_prediction_data: Number of data points in ", sQuote("gp_coords_pred"), " does not match number of data points in ", sQuote("group_data_pred"))
          }
        } else {
          num_data_pred <- as.integer(dim(gp_coords_pred)[1])
        }
        if (dim(gp_coords_pred)[2] != private$dim_coords) {
          stop("set_prediction_data: Dimension / number of coordinates in ", sQuote("gp_coords_pred"), " is not correct")
        }
        gp_coords_pred <- as.vector(matrix(gp_coords_pred))
        # Set data for GP random coefficients
        if (!is.null(gp_rand_coef_data_pred)) {
          if (is.numeric(gp_rand_coef_data_pred)) {
            gp_rand_coef_data_pred <- as.matrix(gp_rand_coef_data_pred)
          }
          if (is.matrix(gp_rand_coef_data_pred)) {
            if (storage.mode(gp_rand_coef_data_pred) != "double") {
              storage.mode(gp_rand_coef_data_pred) <- "double"
            }
          } else {
            stop("set_prediction_data: Can only use ", sQuote("matrix"), " as ", sQuote("gp_rand_coef_data_pred"))
          }
          if (dim(gp_rand_coef_data_pred)[1] != num_data_pred) {
            stop("set_prediction_data: Number of data points in ", sQuote("gp_rand_coef_data_pred"), " does not match number of data points")
          }
          if (dim(gp_rand_coef_data_pred)[2] != num_gp_rand_coef) {
            stop("set_prediction_data: Number of covariates in ", sQuote("gp_rand_coef_data_pred"), " is not correct")
          }
          gp_rand_coef_data_pred <- as.vector(matrix(gp_rand_coef_data_pred))
        } # End set data for GP random coefficients
      } # End set data for Gaussian process
      # Set data linear fixed-effects
      if (!is.null(X_pred)) {
        if(!private$has_covariates){
          stop("set_prediction_data: Covariate data provided in ", sQuote("X_pred"), " but model has no linear predictor")
        }
        if (is.numeric(X_pred)) {
          X_pred <- as.matrix(X_pred)
        }
        if (is.matrix(X_pred)) {
          if (storage.mode(X_pred) != "double") {
            storage.mode(X_pred) <- "double"
          }
        } else {
          stop("set_prediction_data: Can only use ", sQuote("matrix"), " as ", sQuote("X_pred"))
        }
        if (dim(X_pred)[1] != num_data_pred) {
          stop("set_prediction_data: Number of data points in ", sQuote("X_pred"), " is not correct")
        }
        if (dim(X_pred)[2] != private$num_coef) {
          stop("set_prediction_data: Number of covariates in ", sQuote("X_pred"), " is not correct")
        }
        X_pred <- as.vector(matrix(X_pred))
      } # End set data linear fixed-effects
      # Set cluster_ids for independent processes
      if (!is.null(cluster_ids_pred)) {
        if (is.vector(cluster_ids_pred)) {
          if (is.null(private$cluster_ids_map_to_int) & storage.mode(cluster_ids_pred) != "integer") {
            error_message <- TRUE
            if (storage.mode(cluster_ids_pred) == "double") {
              if (all(cluster_ids_pred == floor(cluster_ids_pred))) {
                error_message <- FALSE
                cluster_ids_pred <- as.integer(cluster_ids_pred)
              }
            }
            if (error_message) {
              stop("set_prediction_data: cluster_ids_pred needs to be of type int as the data provided in cluster_ids when initializing the model was also int (or cluster_ids was not provided)")
            }
          }
          if (!is.null(private$cluster_ids_map_to_int)) {
            cluster_ids_pred_map_to_int <- structure(1:length(unique(cluster_ids_pred)),names=c(unique(cluster_ids_pred)))
            for (key in names(cluster_ids_pred_map_to_int)) {
              if (key %in% names(private$cluster_ids_map_to_int)) {
                cluster_ids_pred_map_to_int[key] = private$cluster_ids_map_to_int[key]
              } else {
                cluster_ids_pred_map_to_int[key] = cluster_ids_pred_map_to_int[key] + length(private$cluster_ids_map_to_int)
              }
            }
            cluster_ids_pred <- cluster_ids_pred_map_to_int[cluster_ids_pred]
          }
        } else {
          stop("set_prediction_data: Can only use ", sQuote("vector"), " as ", sQuote("cluster_ids_pred"))
        }
        if (length(cluster_ids_pred) != num_data_pred) {
          stop("set_prediction_data: Length of ", sQuote("cluster_ids_pred"), " does not match number of predicted data points")
        }
        cluster_ids_pred <- as.vector(cluster_ids_pred)
      } # End set cluster_ids for independent processes
      private$num_data_pred <- num_data_pred
      if (!is.null(vecchia_pred_type)) {
        private$vecchia_pred_type <- vecchia_pred_type
      }
      if (!is.null(num_neighbors_pred)) {
        private$num_neighbors_pred <- as.integer(num_neighbors_pred)
      }
      if (!is.null(cg_delta_conv_pred)) {
        private$cg_delta_conv_pred <- as.numeric(cg_delta_conv_pred)
      }
      if (!is.null(nsim_var_pred)) {
        private$nsim_var_pred <- as.integer(nsim_var_pred)
      }
      if (!is.null(rank_pred_approx_matrix_lanczos)) {
        private$rank_pred_approx_matrix_lanczos <- as.integer(rank_pred_approx_matrix_lanczos)
      }
      .Call(
        GPB_SetPredictionData_R
        , private$handle
        , num_data_pred
        , cluster_ids_pred
        , group_data_pred_c_str
        , group_rand_coef_data_pred
        , gp_coords_pred
        , gp_rand_coef_data_pred
        , X_pred
        , private$vecchia_pred_type
        , private$num_neighbors_pred
        , private$cg_delta_conv_pred
        , private$nsim_var_pred
        , private$rank_pred_approx_matrix_lanczos
      )
      return(invisible(self))
    },
    
    predict = function(y = NULL,
                       group_data_pred = NULL,
                       group_rand_coef_data_pred = NULL,
                       gp_coords_pred = NULL,
                       gp_rand_coef_data_pred = NULL,
                       cluster_ids_pred = NULL,
                       predict_cov_mat = FALSE,
                       predict_var = FALSE,
                       cov_pars = NULL,
                       X_pred = NULL,
                       use_saved_data = FALSE,
                       predict_response = TRUE,
                       offset = NULL,
                       offset_pred = NULL,
                       fixed_effects = NULL,
                       fixed_effects_pred = NULL,
                       vecchia_pred_type = NULL,
                       num_neighbors_pred = NULL) {
      
      if (!is.null(fixed_effects)) {
        stop("The argument 'fixed_effects' is discontinued. Use the renamed equivalent argument 'offset' instead ")
      }
      if (!is.null(fixed_effects_pred)) {
        stop("The argument 'fixed_effects_pred' is discontinued. Use the renamed equivalent argument 'offset_pred' instead ")
      }
      if (!is.null(vecchia_pred_type)) {
        stop("predict.GPModel: The argument 'vecchia_pred_type' is discontinued. Use the function 'set_prediction_data' to specify this")
      }
      if (!is.null(vecchia_pred_type)) {
        stop("predict.GPModel: The argument 'vecchia_pred_type' is discontinued. Use the function 'set_prediction_data' to specify this")
      }
      if (!is.null(num_neighbors_pred)) {
        stop("predict.GPModel: The argument 'num_neighbors_pred' is discontinued. Use the function 'set_prediction_data' to specify this")
      }
      if (private$model_has_been_loaded_from_saved_file) {
        if (is.null(y)) {
          y <- private$y_loaded_from_file
        }
        if (is.null(cov_pars)) {
          if (is.matrix(private$cov_pars_loaded_from_file)) {
            cov_pars <- private$cov_pars_loaded_from_file[1,]
          } else {
            cov_pars <- private$cov_pars_loaded_from_file
          }
        }
      }
      if(predict_cov_mat && predict_var){
        predict_cov_mat <- TRUE
        predict_var <- FALSE
      }
      if (gpb.is.null.handle(private$handle)) {
        stop("predict.GPModel: Gaussian process model has not been initialized")
      }
      group_data_pred_c_str <- NULL
      if (!is.null(y)) {
        if (!is.vector(y)) {
          if (is.matrix(y)) {
            if (dim(y)[2] != 1) {
              stop("predict.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("y"))
            }
          } else {
            stop("predict.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("y"))
          }
        }
        if (storage.mode(y) != "double") {
          storage.mode(y) <- "double"
        }
        y <- as.vector(y)
        if (length(y) != private$num_data) {
          stop("predict.GPModel: Number of data points in ", sQuote("y"), " does not match number of data points of initialized model")
        }
      }
      if (!is.null(cov_pars)) {
        if (is.vector(cov_pars)) {
          if (storage.mode(cov_pars) != "double") {
            storage.mode(cov_pars) <- "double"
          }
          cov_pars <- as.vector(cov_pars)
        } else {
          stop("predict.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("cov_pars"))
        }
        if (length(cov_pars) != private$num_cov_pars) {
          stop("predict.GPModel: Number of parameters in ", sQuote("cov_pars"), " does not correspond to numbers of parameters of model")
        }
      }
      if (!use_saved_data) {
        num_data_pred <- 0
        # Set data for grouped random effects
        if (private$num_group_re > 0) {
          if (is.null(group_data_pred)) {
            stop("predict.GPModel: the argument ", sQuote("group_data_pred"), " is missing")
          } else {
            # Check for correct format and set meta data
            if (!(is.data.frame(group_data_pred) | is.matrix(group_data_pred) |
                  is.numeric(group_data_pred) | is.character(group_data_pred) |
                  is.factor(group_data_pred))) {
              stop("predict.GPModel: Can only use the following types for as ", sQuote("group_data_pred"), ": ",
                   sQuote("data.frame"), ", ", sQuote("matrix"), ", ", sQuote("character"),
                   ", ", sQuote("numeric"), ", ", sQuote("factor"))
            }
            if (is.data.frame(group_data_pred) | is.numeric(group_data_pred) |
                is.character(group_data_pred) | is.factor(group_data_pred)) {
              group_data_pred <- as.matrix(group_data_pred)
            }
            if (dim(group_data_pred)[2] != private$num_group_re) {
              stop("predict.GPModel: Number of grouped random effects in ", sQuote("group_data_pred"), " does not correspond to the number of random effects in the training data")
            }
            num_data_pred <- as.integer(dim(group_data_pred)[1])
            group_data_pred <- as.vector(group_data_pred)
            group_data_pred_unique <- unique(group_data_pred)
            group_data_pred_unique_c_str <- lapply(group_data_pred_unique,gpb.c_str)
            group_data_pred_c_str <- unlist(group_data_pred_unique_c_str[match(group_data_pred,group_data_pred_unique)])
          }
        } # End set data for grouped random coefficients
        # Set data for grouped random coefficients
        if (private$num_group_rand_coef > 0) {
          if (is.null(group_rand_coef_data_pred)) {
            stop("predict.GPModel: the argument ", sQuote("group_rand_coef_data_pred"), " is missing")
          } else {
            if (is.numeric(group_rand_coef_data_pred)) {
              group_rand_coef_data_pred <- as.matrix(group_rand_coef_data_pred)
            }
            if (is.matrix(group_rand_coef_data_pred)) {
              if (storage.mode(group_rand_coef_data_pred) != "double") {
                storage.mode(group_rand_coef_data_pred) <- "double"
              }
            } else {
              stop("predict.GPModel: Can only use ", sQuote("matrix"), " as group_rand_coef_data_pred")
            }
            if (dim(group_rand_coef_data_pred)[1] != num_data_pred) {
              stop("predict.GPModel: Number of data points in ", sQuote("group_rand_coef_data_pred"), " does not match number of data points in ", sQuote("group_data_pred"))
            }
            if (dim(group_rand_coef_data_pred)[2] != private$num_group_rand_coef) {
              stop("predict.GPModel: Number of covariates in ", sQuote("group_rand_coef_data_pred"), " is not correct")
            }
            group_rand_coef_data_pred <- as.vector(matrix(group_rand_coef_data_pred))
          }
        } # End set data for grouped random coefficients
        # Set data for Gaussian process
        if (private$num_gp > 0) {
          if (is.null(gp_coords_pred)) {
            stop("predict.GPModel: the argument ", sQuote("gp_coords_pred"), " is missing")
          } else {
            if (is.numeric(gp_coords_pred)) {
              gp_coords_pred <- as.matrix(gp_coords_pred)
            }
            if (is.matrix(gp_coords_pred)) {
              if (storage.mode(gp_coords_pred) != "double") {
                storage.mode(gp_coords_pred) <- "double"
              }
            } else {
              stop("predict.GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("gp_coords_pred"))
            }
            if (num_data_pred != 0) {
              if (dim(gp_coords_pred)[1] != num_data_pred) {
                stop("predict.GPModel: Number of data points in ", sQuote("gp_coords_pred"), " does not match number of data points in ", sQuote("group_data_pred"))
              }
            } else {
              num_data_pred <- as.integer(dim(gp_coords_pred)[1])
            }
            if (dim(gp_coords_pred)[2] != private$dim_coords) {
              stop("predict.GPModel: Dimension / number of coordinates in ", sQuote("gp_coords_pred"), " is not correct")
            }
            gp_coords_pred <- as.vector(matrix(gp_coords_pred))
          }
        } # End set data for Gaussian process
        # Set data for GP random coefficients
        if (private$num_gp_rand_coef > 0) {
          if (is.null(gp_rand_coef_data_pred)) {
            stop("predict.GPModel: the argument ", sQuote("gp_rand_coef_data_pred"), " is missing")
          } else {
            if (is.numeric(gp_rand_coef_data_pred)) {
              gp_rand_coef_data_pred <- as.matrix(gp_rand_coef_data_pred)
            }
            if (is.matrix(gp_rand_coef_data_pred)) {
              if (storage.mode(gp_rand_coef_data_pred) != "double") {
                storage.mode(gp_rand_coef_data_pred) <- "double"
              }
            } else {
              stop("predict.GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("gp_rand_coef_data_pred"))
            }
            if (dim(gp_rand_coef_data_pred)[1] != num_data_pred) {
              stop("predict.GPModel: Number of data points in ", sQuote("gp_rand_coef_data_pred"), " does not match number of data points")
            }
            if (dim(gp_rand_coef_data_pred)[2] != private$num_gp_rand_coef) {
              stop("predict.GPModel: Number of covariates in ", sQuote("gp_rand_coef_data_pred"), " is not correct")
            }
            gp_rand_coef_data_pred <- as.vector(matrix(gp_rand_coef_data_pred))
          }
        } # End set data for GP random coefficients
        # Set data for linear fixed-effects
        if(private$has_covariates){
          if(is.null(X_pred)){
            stop("predict.GPModel: No covariate data is provided in ", sQuote("X_pred"), " but model has linear predictor")
          }
          if (is.numeric(X_pred)) {
            X_pred <- as.matrix(X_pred)
          }
          if (is.matrix(X_pred)) {
            if (storage.mode(X_pred) != "double") {
              storage.mode(X_pred) <- "double"
            }
          } else {
            stop("predict.GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("X_pred"))
          }
          if (dim(X_pred)[1] != num_data_pred) {
            stop("predict.GPModel: Number of data points in ", sQuote("X_pred"), " is not correct")
          }
          if (dim(X_pred)[2] != private$num_coef) {
            stop("predict.GPModel: Number of covariates in ", sQuote("X_pred"), " is not correct")
          }
          if (private$model_has_been_loaded_from_saved_file) {
            if (is.matrix(private$coefs_loaded_from_file)) {
              coefs <- private$coefs_loaded_from_file[1,]
            } else {
              coefs <- private$coefs_loaded_from_file
            }
            if (is.null(offset)) {
              offset <- as.vector(private$X_loaded_from_file %*% coefs)
            } else {
              offset <- offset + as.vector(private$X_loaded_from_file %*% coefs)
            }
            if (is.null(offset_pred)) {
              offset_pred <- as.vector(X_pred %*% coefs)
            } else {
              offset_pred <- offset_pred + as.vector(X_pred %*% coefs)
            }
            X_pred <- NULL
          } else {
            X_pred <- as.vector(matrix(X_pred))
          }
        } # End set data for linear fixed-effects
        # Set cluster_ids for independent processes
        if (!is.null(cluster_ids_pred)) {
          if (is.vector(cluster_ids_pred)) {
            if (is.null(private$cluster_ids_map_to_int) & storage.mode(cluster_ids_pred) != "integer") {
              error_message <- TRUE
              if (storage.mode(cluster_ids_pred) == "double") {
                if (all(cluster_ids_pred == floor(cluster_ids_pred))) {
                  error_message <- FALSE
                  cluster_ids_pred <- as.integer(cluster_ids_pred)
                }
              }
              if (error_message) {
                stop("predict.GPModel: cluster_ids_pred needs to be of type int as the data provided in cluster_ids when initializing the model was also int (or cluster_ids was not provided)")
              }
            }
            if (!is.null(private$cluster_ids_map_to_int)) {
              cluster_ids_pred_map_to_int <- structure(1:length(unique(cluster_ids_pred)),names=c(unique(cluster_ids_pred)))
              for (key in names(cluster_ids_pred_map_to_int)) {
                if (key %in% names(private$cluster_ids_map_to_int)) {
                  cluster_ids_pred_map_to_int[key] = private$cluster_ids_map_to_int[key]
                } else {
                  cluster_ids_pred_map_to_int[key] = cluster_ids_pred_map_to_int[key] + length(private$cluster_ids_map_to_int)
                }
              }
              cluster_ids_pred <- cluster_ids_pred_map_to_int[cluster_ids_pred]
            }
          } else {
            stop("predict.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("cluster_ids_pred"))
          }
          if (length(cluster_ids_pred) != num_data_pred) {
            stop("predict.GPModel: Length of ", sQuote("cluster_ids_pred"), 
                 " does not match number of predicted data points")
          }
          cluster_ids_pred <- as.vector(cluster_ids_pred)
        } # End set cluster_ids for independent processes
      } else { # use_saved_data
        cluster_ids_pred <- NULL
        group_data_pred_c_str <- NULL
        group_rand_coef_data_pred <- NULL
        gp_coords_pred <- NULL
        gp_rand_coef_data_pred <- NULL
        X_pred <- NULL
        num_data_pred <- private$num_data_pred
        if (is.null(private$num_data_pred)) {
          stop("predict.GPModel: No data has been set for making predictions. 
               Call set_prediction_data first")
        }
      }
      if (storage.mode(predict_cov_mat) != "logical") {
        stop("predict.GPModel: Can only use ", sQuote("logical"), " as ", sQuote("predict_cov_mat"))
      }
      if (storage.mode(predict_var) != "logical") {
        stop("predict.GPModel: Can only use ", sQuote("logical"), " as ", sQuote("predict_var"))
      }
      if (storage.mode(predict_response) != "logical") {
        stop("predict.GPModel: Can only use ", sQuote("logical"), " as ", sQuote("predict_response"))
      }
      if (!is.null(offset)) {
        if (is.vector(offset)) {
          if (storage.mode(offset) != "double") {
            storage.mode(offset) <- "double"
          }
          offset <- as.vector(offset)
        } else {
          stop("predict.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("offset"))
        }
        if (length(offset) != private$num_data) {
          stop("predict.GPModel: Length of ", sQuote("offset"), " does not match number of observed data points")
        }
      }# end offset
      if (!is.null(offset_pred)) {
        if (is.vector(offset_pred)) {
          if (storage.mode(offset_pred) != "double") {
            storage.mode(offset_pred) <- "double"
          }
          offset_pred <- as.vector(offset_pred)
        } else {
          stop("predict.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("offset_pred"))
        }
        if (length(offset_pred) != num_data_pred) {
          stop("predict.GPModel: Length of ", sQuote("offset"), " does not match number of predicted data points")
        }
      }# end offset_pred
      # Pre-allocate empty vector
      if (predict_var) {
        preds <- numeric(num_data_pred * 2)
      } else if (predict_cov_mat) {
        preds <- numeric(num_data_pred * (1 + num_data_pred))
      } else {
        preds <- numeric(num_data_pred)
      }
      .Call(
        GPB_PredictREModel_R
        , private$handle
        , y
        , num_data_pred
        , predict_cov_mat
        , predict_var
        , predict_response
        , cluster_ids_pred
        , group_data_pred_c_str
        , group_rand_coef_data_pred
        , gp_coords_pred
        , gp_rand_coef_data_pred
        , cov_pars
        , X_pred
        , use_saved_data
        , offset
        , offset_pred
        , preds
      )
      # Process C++ output
      pred_mean <- preds[1:num_data_pred]
      pred_cov_mat <- NA
      pred_var <- NA
      if (predict_var) {
        pred_var <- preds[1:num_data_pred + num_data_pred]
      } else if (predict_cov_mat) {
        pred_cov_mat <- matrix(preds[1:(num_data_pred^2) + num_data_pred],ncol=num_data_pred)
      }
      return(list(mu=pred_mean,cov=pred_cov_mat,var=pred_var))
    },
    
    predict_training_data_random_effects = function(predict_var = FALSE) {
      if(isTRUE(private$model_has_been_loaded_from_saved_file)){
        stop("GPModel: 'predict_training_data_random_effects' is currently not 
        implemented for models that have been loaded from a saved file")
      }
      num_re_comps = private$num_group_re + private$num_group_rand_coef + 
        private$num_gp + private$num_gp_rand_coef
      if (!is.null(private$drop_intercept_group_rand_effect)) {
        num_re_comps <- num_re_comps - sum(private$drop_intercept_group_rand_effect)
      }
      if (storage.mode(predict_var) != "logical") {
        stop("predict_training_data_random_effects.GPModel: Can only use ", 
             sQuote("logical"), " as ", sQuote("predict_var"))
      }
      if (predict_var) {
        re_preds <- numeric(private$num_data * num_re_comps * 2)
      } else {
        re_preds <- numeric(private$num_data * num_re_comps)
      }
      
      .Call(
        GPB_PredictREModelTrainingDataRandomEffects_R
        , private$handle
        , NULL
        , NULL
        , NULL
        , predict_var
        , re_preds
      )
      if (predict_var) {
        re_preds <- matrix(re_preds, ncol = 2 * num_re_comps, 
                           dimnames = list(NULL, c(private$re_comp_names,
                                                   paste0(private$re_comp_names, "_var"))))
      } else {
        re_preds <- matrix(re_preds, ncol = num_re_comps, 
                           dimnames = list(NULL, private$re_comp_names))
      }
      return(re_preds)
    },
    
    get_group_data = function() {##TODO: get all this data from C++ and don't double save this here in R
      if(isTRUE(private$free_raw_data)){
        stop("GPModel: cannot return ", sQuote("group_data"), ",
          please set ", sQuote("free_raw_data = FALSE"), " when you create the ", sQuote("GPModel"))
      }
      return(private$group_data)
    },
    
    get_group_rand_coef_data = function() {
      if(isTRUE(private$free_raw_data)){
        stop("GPModel: cannot return ", sQuote("group_rand_coef_data"), ",
          please set ", sQuote("free_raw_data = FALSE"), " when you create the ", sQuote("GPModel"))
      }
      return(private$group_rand_coef_data)
    },
    
    get_gp_coords = function() {
      if(isTRUE(private$free_raw_data)){
        stop("GPModel: cannot return ", sQuote("gp_coords"), ",
          please set ", sQuote("free_raw_data = FALSE"), " when you create the ", sQuote("GPModel"))
      }
      return(private$gp_coords)
    },
    
    get_gp_rand_coef_data = function() {
      if(isTRUE(private$free_raw_data)){
        stop("GPModel: cannot return ", sQuote("gp_rand_coef_data"), ",
          please set ", sQuote("free_raw_data = FALSE"), " when you create the ", sQuote("GPModel"))
      }
      return(private$gp_rand_coef_data)
    },
    
    get_cluster_ids = function() {
      if(isTRUE(private$free_raw_data)){
        stop("GPModel: cannot return ", sQuote("cluster_ids"), ",
             please set ", sQuote("free_raw_data = FALSE"), " when you create the ", sQuote("GPModel"))
      }
      return(private$cluster_ids)
    },
    
    get_response_data = function() {
      response_data <- numeric(private$num_data)
      .Call(
        GPB_GetResponseData_R
        , private$handle
        , response_data
      )
      return(response_data)
    },
    
    get_covariate_data = function() {
      if (!private$has_covariates) {
        stop("GPModel: Model has no covariate data for linear predictor")
      }
      covariate_data <- numeric(private$num_data * private$num_coef)
      .Call(
        GPB_GetCovariateData_R
        , private$handle
        , covariate_data
      )
      covariate_data <- matrix(covariate_data,ncol=private$num_coef)
      return(covariate_data)
    },
    
    get_num_data = function() {
      return(private$num_data)
    },
    
    get_num_optim_iter = function() {
      num_it <- integer(1)
      .Call(
        GPB_GetNumIt_R
        , private$handle
        , num_it
      )
      return(num_it)
    },
    
    get_num_aux_pars = function() {
      num_aux_pars <- integer(1)
      .Call(
        GPB_GetNumAuxPars_R
        , private$handle
        , num_aux_pars
      )
      return(num_aux_pars)
    },
    
    get_current_neg_log_likelihood = function() {
      negll <- 0.
      .Call(
        GPB_GetCurrentNegLogLikelihood_R
        , private$handle
        , negll
      )
      return(negll + 0.) # add 0. to avoid undesired copy override issues
    },
    
    get_likelihood_name = function() {
      ll_name <- .Call(
        GPB_GetLikelihoodName_R
        , private$handle
      )
      return(ll_name)
    },
    
    set_likelihood = function(likelihood) {
      if (!is.character(likelihood)) {
        stop("set_likelihood: Can only use ", sQuote("character"), " as ", sQuote("likelihood"))
      }
      private$update_cov_par_names(likelihood)
      .Call(
        GPB_SetLikelihood_R
        , private$handle
        , likelihood
      )
      return(invisible(self))
    },
    
    model_to_list = function(include_response_data=TRUE) {
      if (isTRUE(private$free_raw_data)) {
        stop("model_to_list: cannot convert to json when free_raw_data=TRUE has been set")
      }
      if (private$has_offset) {
        warning("An 'offset' was provided for estimation / fitting. Saving this 'offset' is currently not 
                implemented. Please pass the 'offset' again when you call the 'predict()' function 
                (in addition to a potential 'offset_pred' parameter for prediction points")
      }
      model_list <- list()
      # Parameters
      model_list[["params"]] <- self$get_optim_params()
      model_list[["likelihood"]] <- self$get_likelihood_name()
      model_list[["cov_pars"]] <- self$get_cov_pars()
      # Response data
      if (include_response_data) {
        model_list[["y"]] <- self$get_response_data()
      }
      # Random effects / GP data
      model_list[["group_data"]] <- self$get_group_data()
      model_list[["nb_groups"]] <- private$nb_groups
      model_list[["group_rand_coef_data"]] <- self$get_group_rand_coef_data()
      model_list[["gp_coords"]] <- self$get_gp_coords()
      model_list[["gp_rand_coef_data"]] <- self$get_gp_rand_coef_data()
      model_list[["ind_effect_group_rand_coef"]] <- private$ind_effect_group_rand_coef
      model_list[["drop_intercept_group_rand_effect"]] <- private$drop_intercept_group_rand_effect
      model_list[["cluster_ids"]] <- self$get_cluster_ids()
      model_list[["num_neighbors"]] <- private$num_neighbors
      model_list[["vecchia_ordering"]] <- private$vecchia_ordering
      model_list[["cov_function"]] <- private$cov_function
      model_list[["cov_fct_shape"]] <- private$cov_fct_shape
      model_list[["gp_approx"]] <- private$gp_approx
      model_list[["cov_fct_taper_range"]] <- private$cov_fct_taper_range
      model_list[["cov_fct_taper_shape"]] <- private$cov_fct_taper_shape
      model_list[["num_ind_points"]] <- private$num_ind_points
      model_list[["cover_tree_radius"]] <- private$cover_tree_radius
      model_list[["ind_points_selection"]] <- private$ind_points_selection
      model_list[["matrix_inversion_method"]] <- private$matrix_inversion_method
      model_list[["seed"]] <- private$seed
      # Covariate data
      model_list[["has_covariates"]] <- private$has_covariates
      if (private$has_covariates) {
        model_list[["coefs"]] <- self$get_coef()
        model_list[["num_coef"]] <- private$num_coef
        model_list[["X"]] <- self$get_covariate_data()
      }
      # Additional likelihood parameters (e.g., shape parameter for a gamma or negative_binomial likelihood)
      model_list[["params"]]["init_aux_pars"] <- self$get_aux_pars()
      # Note: for simplicity, this is put into 'init_aux_pars'. When loading the model, 'init_aux_pars' are correctly set
      model_list[["model_fitted"]] <- private$model_fitted
      # Make sure that data is saved in correct format by RJSONIO::toJSON
      MAYBE_CONVERT_TO_VECTOR <- c("cov_pars","group_data", "group_rand_coef_data",
                                   "gp_coords", "gp_rand_coef_data",
                                   "ind_effect_group_rand_coef",
                                   "drop_intercept_group_rand_effect",
                                   "cluster_ids","coefs","X","nb_groups", "aux_pars")
      for (feature in MAYBE_CONVERT_TO_VECTOR) {
        if (!is.null(model_list[[feature]])) {
          if (is.vector(model_list[[feature]])) {
            model_list[[feature]] <- as.vector(model_list[[feature]])
          }
          if (is.matrix(model_list[[feature]])) {
            if (dim(model_list[[feature]])[2] == 1) {
              model_list[[feature]] <- as.vector(model_list[[feature]])
            }
          }
        }
      }
      return(model_list)
    },
    
    save = function(filename) {
      if (!(is.character(filename) && length(filename) == 1L)) {
        stop("save.GPModel: filename should be a string")
      }
      if (isTRUE(private$free_raw_data)) {
        stop("save.GPModel: cannot save when free_raw_data=TRUE has been set")
      }
      # Use RJSONIO R package since jsonlite and rjson omit the last digit of a double
      save_data_json <- RJSONIO::toJSON(self$model_to_list(include_response_data=TRUE), digits=17)
      write(save_data_json, file=filename)
      return(invisible(self))
    },
    
    summary = function() {
      cov_pars <- self$get_cov_pars()
      cat("=====================================================\n")
      if (private$model_fitted && !private$model_has_been_loaded_from_saved_file) {
        cat("Model summary:\n")
        ll <- -self$get_current_neg_log_likelihood()
        npar <- private$num_cov_pars
        if (private$has_covariates) {
          npar <- npar + private$num_coef
        }
        aic <- 2*npar - 2*ll
        bic <- npar*log(self$get_num_data()) - 2*ll
        print(round(c("Log-lik"=ll, "AIC"=aic, "BIC"=bic),digits=2))
        cat(paste0("Nb. observations: ", self$get_num_data(),"\n"))
        if ((private$num_group_re + private$num_group_rand_coef) > 0) {
          outstr <- "Nb. groups: "
          for (i in 1:private$num_group_re) {
            if (i > 1) {
              outstr <- paste0(outstr,", ")
            }
            outstr <- paste0(outstr,private$nb_groups[i]," (",
                             private$re_comp_names[i],")")
          }
          outstr <- paste0(outstr,"\n")
          cat(outstr)
        }
        cat("-----------------------------------------------------\n")
      }
      cat("Covariance parameters (random effects):\n")
      if (is.matrix(cov_pars)) {
        print(round(t(cov_pars),4))
      } else {
        cov_pars <- t(t(cov_pars))
        colnames(cov_pars) <- "Param."
        print(round(cov_pars,4))
      }
      if (private$has_covariates) {
        coefs <- self$get_coef()
        cat("-----------------------------------------------------\n")
        cat("Linear regression coefficients (fixed effects):\n")
        if (private$params$std_dev) {
          z_values <- coefs[1,] / coefs[2,]
          p_values <- 2 * exp(pnorm(-abs(z_values), log.p = TRUE))
          coefs_summary <- cbind(t(coefs),"z value"=z_values,"P(>|z|)"=p_values)
          print(round(coefs_summary,4))
        } else {
          coefs <- t(t(coefs))
          colnames(coefs) <- "Param."
          print(round(coefs,4))
        }
      }
      if (self$get_num_aux_pars() > 0) {
        aux_pars <- self$get_aux_pars()
        aux_pars <- t(t(aux_pars))
        colnames(aux_pars) <- "Param."
        cat("-----------------------------------------------------\n")
        cat("Additional parameters:\n")
        print(round(aux_pars,4))
      }
      if (private$params$maxit == self$get_num_optim_iter()) {
        cat("-----------------------------------------------------\n")
        cat("Note: no convergence after the maximal number of iterations\n")
      }
      cat("=====================================================\n")
    }
    
  ), # end public
  
  private = list(
    handle = NULL,
    num_data = NULL,
    num_group_re = 0L,
    num_group_rand_coef = 0L,
    num_cov_pars = 0L,
    num_gp = 0L,
    dim_coords = 2L,
    num_gp_rand_coef = 0L,
    has_covariates = FALSE,
    has_offset = FALSE,
    num_coef = 0,
    group_data = NULL,
    nb_groups = NULL,
    group_rand_coef_data = NULL,
    ind_effect_group_rand_coef = NULL,
    drop_intercept_group_rand_effect = NULL,
    gp_coords = NULL,
    gp_rand_coef_data = NULL,
    cov_function = "exponential",
    cov_fct_shape = 0.5,
    gp_approx = "none",
    cov_fct_taper_range = 1.,
    cov_fct_taper_shape = 0.,
    num_neighbors = 20L,
    vecchia_ordering = "random",
    vecchia_pred_type = NULL,
    num_neighbors_pred = -1,
    cg_delta_conv_pred = -1,
    nsim_var_pred = -1,
    rank_pred_approx_matrix_lanczos = -1,
    num_ind_points = 500L,
    cover_tree_radius = 1.,
    ind_points_selection = "kmeans++",
    matrix_inversion_method = "cholesky",
    seed = 0L,
    cluster_ids = NULL,
    cluster_ids_map_to_int = NULL,
    free_raw_data = FALSE,
    cov_par_names = NULL,
    re_comp_names = NULL,
    coef_names = NULL,
    num_data_pred = NULL,
    model_has_been_loaded_from_saved_file = FALSE,
    y_loaded_from_file = NULL,
    cov_pars_loaded_from_file = NULL,
    coefs_loaded_from_file = NULL,
    X_loaded_from_file = NULL,
    model_fitted = FALSE,
    params = list(maxit = 1000L,
                  delta_rel_conv = -1., # default value is set in C++
                  init_coef = NULL,
                  lr_coef = 0.1,
                  lr_cov = -1., # default value is set in C++
                  use_nesterov_acc = TRUE,
                  acc_rate_coef = 0.5,
                  acc_rate_cov = 0.5,
                  nesterov_schedule_version = 0L,
                  momentum_offset = 2L,
                  trace = FALSE,
                  convergence_criterion = "relative_change_in_log_likelihood",
                  std_dev = FALSE,
                  cg_max_num_it = 1000L,
                  cg_max_num_it_tridiag = 1000L,
                  cg_delta_conv = 1e-2,
                  num_rand_vec_trace = 50L,
                  reuse_rand_vec_trace = TRUE,
                  seed_rand_vec_trace = 1L,
                  piv_chol_rank = 50L,
                  estimate_aux_pars = TRUE),
    
    determine_num_cov_pars = function(likelihood) {
      if (private$cov_function == "matern_space_time" | private$cov_function == "exponential_space_time" | private$cov_function == "matern_estimate_shape") {
        num_par_per_GP <- 3L
      } else if (private$cov_function == "matern_ard" | private$cov_function == "gaussian_ard" | private$cov_function == "exponential_ard") {
        num_par_per_GP <- 1L + private$dim_coords
      } else if (private$cov_function == "wendland") {
        num_par_per_GP <- 1L
      } else {
        num_par_per_GP <- 2L
      }
      private$num_cov_pars <- private$num_group_re + private$num_group_rand_coef + 
        num_par_per_GP * (private$num_gp + private$num_gp_rand_coef)
      if (!is.null(private$drop_intercept_group_rand_effect)) {
        private$num_cov_pars <- private$num_cov_pars - sum(private$drop_intercept_group_rand_effect)
      }
      if (likelihood == "gaussian"){
        private$num_cov_pars <- private$num_cov_pars + 1L
      }
      storage.mode(private$num_cov_pars) <- "integer"
    },
    
    update_params = function(params) {
      ## Check format of parameters
      numeric_params <- c("lr_cov", "acc_rate_cov", "delta_rel_conv",
                          "lr_coef", "acc_rate_coef", "cg_delta_conv")
      integer_params <- c("maxit", "nesterov_schedule_version",
                          "momentum_offset", "cg_max_num_it", "cg_max_num_it_tridiag",
                          "num_rand_vec_trace", "seed_rand_vec_trace",
                          "piv_chol_rank")
      character_params <- c("optimizer_cov", "convergence_criterion",
                            "optimizer_coef", "cg_preconditioner_type")
      logical_params <- c("use_nesterov_acc", "trace", "std_dev", 
                          "reuse_rand_vec_trace", "estimate_aux_pars")
      if (!is.null(params[["init_cov_pars"]])) {
        if (is.vector(params[["init_cov_pars"]])) {
          if (storage.mode(params[["init_cov_pars"]]) != "double") {
            storage.mode(params[["init_cov_pars"]]) <- "double"
          }
          params[["init_cov_pars"]] <- as.vector(params[["init_cov_pars"]])
        } else {
          stop("GPModel: Can only use ", sQuote("vector"), " as ", sQuote("params$init_cov_pars"))
        }
        if (length(params[["init_cov_pars"]]) != private$num_cov_pars) {
          stop("GPModel: Number of parameters in ", sQuote("params$init_cov_pars"), 
               " is not correct")
        }
      }
      if (!is.null(params[["init_coef"]])) {
        if (is.vector(params[["init_coef"]])) {
          if (storage.mode(params[["init_coef"]]) != "double") {
            storage.mode(params[["init_coef"]]) <- "double"
          }
          params[["init_coef"]] <- as.vector(params[["init_coef"]])
          num_coef <- as.integer(length(params[["init_coef"]]))
          if (is.null(private$num_coef) | private$num_coef==0) {
            private$num_coef <- num_coef
          }
        } else {
          stop("GPModel: Can only use ", sQuote("vector"), " as ", sQuote("init_coef"))
        }
        if (length(params[["init_coef"]]) != private$num_coef) {
          stop("GPModel: Number of parameters in ", sQuote("init_coef"), 
               " does not correspond to numbers of covariates in ", sQuote("X"))
        }
      }
      if (!is.null(params[["init_aux_pars"]])) {
        if (is.vector(params[["init_aux_pars"]])) {
          if (storage.mode(params[["init_aux_pars"]]) != "double") {
            storage.mode(params[["init_aux_pars"]]) <- "double"
          }
          params[["init_aux_pars"]] <- as.vector(params[["init_aux_pars"]])
        } else {
          stop("GPModel: Can only use ", sQuote("vector"), " as ", sQuote("params$init_aux_pars"))
        }
        if (length(params[["init_aux_pars"]]) != self$get_num_aux_pars()) {
          stop("GPModel: Number of parameters in ", sQuote("params$init_aux_pars"), 
               " is not correct")
        }
      }
      ## Update private$params
      for (param in names(params)) {
        if (param %in% numeric_params & !is.null(params[[param]])) {
          params[[param]] <- as.numeric(params[[param]])
        }
        if (param %in% integer_params & !is.null(params[[param]])) {
          params[[param]] <- as.integer(params[[param]])
        }
        if (param %in% character_params & !is.null(params[[param]])) {
          if (!is.character(params[[param]])) {
            stop("GPModel: Can only use ", sQuote("character"), " as ", param)
          }
        }
        if (param %in% logical_params & !is.null(params[[param]])) {
          if (!is.logical(params[[param]])) {
            stop("GPModel: Can only use ", sQuote("logical"), " as ", param)
          }
        }
        if (param %in% names(private$params)) {
          if (is.null(params[[param]])) {
            private$params[param] <- list(NULL)
          }else{
            private$params[[param]] <- params[[param]]
          }
        }
        else if (!(param %in% c("optimizer_cov", "optimizer_coef", "cg_preconditioner_type", 
                                "init_cov_pars", "init_aux_pars"))){
          stop(paste0("GPModel: Unknown parameter: ", param))
        }
      }
    },
    
    update_cov_par_names = function(likelihood) {
      if (!is.character(likelihood)) {
        stop("set_likelihood: Can only use ", sQuote("character"), " as ", sQuote("likelihood"))
      }
      private$determine_num_cov_pars(likelihood)
      if (likelihood != "gaussian" && "Error_term" %in% private$cov_par_names){
        private$cov_par_names <- private$cov_par_names["Error_term" != private$cov_par_names]
      }
      if (likelihood == "gaussian" && !("Error_term" %in% private$cov_par_names)){
        private$cov_par_names <- c("Error_term",private$cov_par_names)
      }
    },
    
    # Get handle
    get_handle = function() {
      if (gpb.is.null.handle(private$handle)) {
        stop("GPModel: model has not been initialized")
      }
      private$handle
    }
  )# end private
)

#' Create a \code{GPModel} object
#'
#' Create a \code{GPModel} which contains a Gaussian process and / or mixed effects model with grouped random effects
#'
#' @inheritParams GPModel_shared_params 
#' @param num_neighbors_pred an \code{integer} specifying the number of neighbors for making predictions.
#' This is discontinued here. Use the function 'set_prediction_data' to specify this
#' @param vecchia_pred_type A \code{string} specifying the type of Vecchia approximation used for making predictions.
#' This is discontinued here. Use the function 'set_prediction_data' to specify this
#'
#' @return A \code{GPModel} containing ontains a Gaussian process and / or mixed effects model with grouped random effects
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' data(GPBoost_data, package = "gpboost")
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
#' 
#' #--------------------Gaussian process model----------------
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
#'                     likelihood="gaussian")
#'
#' #--------------------Combine Gaussian process with grouped random effects----------------
#' gp_model <- GPModel(group_data = group_data,
#'                     gp_coords = coords, cov_function = "exponential",
#'                     likelihood="gaussian")
#' @author Fabio Sigrist
#' @export
GPModel <- function(likelihood = "gaussian",
                    group_data = NULL,
                    group_rand_coef_data = NULL,
                    ind_effect_group_rand_coef = NULL,
                    drop_intercept_group_rand_effect = NULL,
                    gp_coords = NULL,
                    gp_rand_coef_data = NULL,
                    cov_function = "exponential",
                    cov_fct_shape = 0.5,
                    gp_approx = "none",
                    cov_fct_taper_range = 1.,
                    cov_fct_taper_shape = 0.,
                    num_neighbors = 20L,
                    vecchia_ordering = "random",
                    ind_points_selection = "kmeans++",
                    num_ind_points = 500L,
                    cover_tree_radius = 1.,
                    matrix_inversion_method = "cholesky",
                    seed = 0L,
                    cluster_ids = NULL,
                    free_raw_data = FALSE,
                    vecchia_approx = NULL,
                    vecchia_pred_type = NULL,
                    num_neighbors_pred = NULL) {
  
  # Create new GPModel
  invisible(gpb.GPModel$new(likelihood = likelihood
                            , group_data = group_data
                            , group_rand_coef_data = group_rand_coef_data
                            , ind_effect_group_rand_coef = ind_effect_group_rand_coef
                            , drop_intercept_group_rand_effect = drop_intercept_group_rand_effect
                            , gp_coords = gp_coords
                            , gp_rand_coef_data = gp_rand_coef_data
                            , cov_function = cov_function
                            , cov_fct_shape = cov_fct_shape
                            , gp_approx = gp_approx
                            , cov_fct_taper_range = cov_fct_taper_range
                            , cov_fct_taper_shape = cov_fct_taper_shape
                            , num_neighbors = num_neighbors
                            , vecchia_ordering = vecchia_ordering
                            , ind_points_selection = ind_points_selection
                            , num_ind_points = num_ind_points
                            , cover_tree_radius = cover_tree_radius
                            , matrix_inversion_method = matrix_inversion_method
                            , seed = seed
                            , cluster_ids = cluster_ids
                            , free_raw_data = free_raw_data
                            , vecchia_approx = vecchia_approx
                            , vecchia_pred_type = vecchia_pred_type
                            , num_neighbors_pred = num_neighbors_pred))
  
}

#' Generic 'fit' method for a \code{GPModel}
#'
#' Generic 'fit' method for a \code{GPModel}
#' 
#' @param gp_model a \code{GPModel}
#' @inheritParams GPModel_shared_params
#' 
#' @author Fabio Sigrist
#' @export 
fit <- function(gp_model, y, X, params, offset = NULL, fixed_effects = NULL) UseMethod("fit")

#' Fits a \code{GPModel}
#'
#' Estimates the parameters of a \code{GPModel} by maximizing the marginal likelihood
#'
#' @param gp_model a \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @return A fitted \code{GPModel}
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' # Add intercept column
#' X1 <- cbind(rep(1,dim(X)[1]),X)
#' X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
#' fit(gp_model, y = y, X = X1, params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
#'                 X_pred = X_test1, predict_var = TRUE)
#' pred$mu # Predicted mean
#' pred$var # Predicted variances
#' # Also predict covariance matrix
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
#'                 X_pred = X_test1, predict_cov_mat = TRUE)
#' pred$mu # Predicted mean
#' pred$cov # Predicted covariance
#'  
#' #--------------------Gaussian process model----------------
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
#'                     likelihood="gaussian")
#' fit(gp_model, y = y, X = X1, params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, gp_coords_pred = coords_test, 
#'                 X_pred = X_test1, predict_cov_mat = TRUE)
#' pred$mu # Predicted (posterior) mean of GP
#' pred$cov # Predicted (posterior) covariance matrix of GP
#' }
#' 
#' @method fit GPModel 
#' @rdname fit.GPModel
#' @author Fabio Sigrist
#' @export
fit.GPModel <- function(gp_model,
                        y,
                        X = NULL,
                        params = list(),
                        offset = NULL,
                        fixed_effects = NULL) {
  
  # Fit model
  invisible(gp_model$fit(y = y,
                         X = X,
                         params = params,
                         offset = offset,
                         fixed_effects = fixed_effects))
  
}

#' Fits a \code{GPModel}
#'
#' Estimates the parameters of a \code{GPModel} by maximizing the marginal likelihood
#'
#' @inheritParams GPModel_shared_params 
#' @param num_neighbors_pred an \code{integer} specifying the number of neighbors for making predictions.
#' This is discontinued here. Use the function 'set_prediction_data' to specify this
#' @param vecchia_pred_type A \code{string} specifying the type of Vecchia approximation used for making predictions.
#' This is discontinued here. Use the function 'set_prediction_data' to specify this
#'
#' @return A fitted \code{GPModel}
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' # Add intercept column
#' X1 <- cbind(rep(1,dim(X)[1]),X)
#' X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1,
#'                        likelihood="gaussian", params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
#'                 X_pred = X_test1, predict_var = TRUE)
#' pred$mu # Predicted mean
#' pred$var # Predicted variances
#' # Also predict covariance matrix
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
#'                 X_pred = X_test1, predict_cov_mat = TRUE)
#' pred$mu # Predicted mean
#' pred$cov # Predicted covariance
#'
#' #--------------------Two crossed random effects and a random slope----------------
#' gp_model <- fitGPModel(group_data = group_data, likelihood="gaussian",
#'                        group_rand_coef_data = X[,2],
#'                        ind_effect_group_rand_coef = 1,
#'                        y = y, X = X1, params = list(std_dev = TRUE))
#' summary(gp_model)
#'
#' #--------------------Gaussian process model----------------
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        likelihood="gaussian", y = y, X = X1, params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, gp_coords_pred = coords_test, 
#'                 X_pred = X_test1, predict_cov_mat = TRUE)
#' pred$mu # Predicted (posterior) mean of GP
#' pred$cov # Predicted (posterior) covariance matrix of GP
#'
#' #--------------------Gaussian process model with Vecchia approximation----------------
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        gp_approx = "vecchia", num_neighbors = 20,
#'                        likelihood="gaussian", y = y)
#' summary(gp_model)
#'
#' #--------------------Gaussian process model with random coefficients----------------
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        gp_rand_coef_data = X[,2], y=y,
#'                        likelihood = "gaussian", params = list(std_dev = TRUE))
#' summary(gp_model)
#'
#' #--------------------Combine Gaussian process with grouped random effects----------------
#' gp_model <- fitGPModel(group_data = group_data,
#'                        gp_coords = coords, cov_function = "exponential",
#'                        likelihood = "gaussian", y = y, X = X1, params = list(std_dev = TRUE))
#' summary(gp_model)
#' }
#' 
#' @rdname fitGPModel
#' @author Fabio Sigrist
#' @export fitGPModel
fitGPModel <- function(likelihood = "gaussian",
                       group_data = NULL,
                       group_rand_coef_data = NULL,
                       ind_effect_group_rand_coef = NULL,
                       drop_intercept_group_rand_effect = NULL,
                       gp_coords = NULL,
                       gp_rand_coef_data = NULL,
                       cov_function = "exponential",
                       cov_fct_shape = 0.5,
                       gp_approx = "none",
                       cov_fct_taper_range = 1.,
                       cov_fct_taper_shape = 0.,
                       num_neighbors = 20L,
                       vecchia_ordering = "random",
                       ind_points_selection = "kmeans++",
                       num_ind_points = 500L,
                       cover_tree_radius = 1.,
                       matrix_inversion_method = "cholesky",
                       seed = 0L,
                       cluster_ids = NULL,
                       free_raw_data = FALSE,
                       y,
                       X = NULL,
                       params = list(),
                       vecchia_approx = NULL,
                       vecchia_pred_type = NULL,
                       num_neighbors_pred = NULL,
                       offset = NULL,
                       fixed_effects = NULL) {
  #Create model
  gpmodel <- gpb.GPModel$new(likelihood = likelihood
                             , group_data = group_data
                             , group_rand_coef_data = group_rand_coef_data
                             , ind_effect_group_rand_coef = ind_effect_group_rand_coef
                             , drop_intercept_group_rand_effect = drop_intercept_group_rand_effect
                             , gp_coords = gp_coords
                             , gp_rand_coef_data = gp_rand_coef_data
                             , cov_function = cov_function
                             , cov_fct_shape = cov_fct_shape
                             , gp_approx = gp_approx
                             , cov_fct_taper_range = cov_fct_taper_range
                             , cov_fct_taper_shape = cov_fct_taper_shape
                             , num_neighbors = num_neighbors
                             , vecchia_ordering = vecchia_ordering
                             , ind_points_selection = ind_points_selection
                             , num_ind_points = num_ind_points
                             , cover_tree_radius = cover_tree_radius
                             , matrix_inversion_method = matrix_inversion_method
                             , seed = seed
                             , cluster_ids = cluster_ids
                             , free_raw_data = free_raw_data
                             , vecchia_approx = vecchia_approx
                             , vecchia_pred_type = vecchia_pred_type
                             , num_neighbors_pred = num_neighbors_pred)
  # Fit model
  gpmodel$fit(y = y,
              X = X,
              params = params,
              offset = offset,
              fixed_effects = fixed_effects)
  return(gpmodel)
  
}

#' Summary for a \code{GPModel}
#'
#' Summary for a \code{GPModel}
#'
#' @param object a \code{GPModel}
#' @param ... (not used, ignore this, simply here that there is no CRAN warning)
#'
#' @return Summary of a (fitted) \code{GPModel}
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' data(GPBoost_data, package = "gpboost")
#' # Add intercept column
#' X1 <- cbind(rep(1,dim(X)[1]),X)
#' X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1,
#'                        likelihood="gaussian", params = list(std_dev = TRUE))
#' summary(gp_model)
#'
#'
#' \donttest{
#' #--------------------Gaussian process model----------------
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        likelihood="gaussian", y = y, X = X1, params = list(std_dev = TRUE))
#' summary(gp_model)
#' }
#' 
#' @method summary GPModel 
#' @rdname summary.GPModel
#' @author Fabio Sigrist
#' @export
summary.GPModel <- function(object, ...){
  object$summary()
  return(invisible(object))
}

#' Make predictions for a \code{GPModel}
#'
#' Make predictions for a \code{GPModel}
#'
#' @param object a \code{GPModel}
#' @param y Observed data (can be NULL, e.g. when the model has been estimated 
#' already and the same data is used for making predictions)
#' @param cov_pars A \code{vector} containing covariance parameters which are used if the 
#' \code{GPModel} has not been trained or if predictions should be made for other 
#' parameters than the trained ones
#' @param use_saved_data A \code{boolean}. If TRUE, predictions are done using 
#' a priory set data via the function '$set_prediction_data' (this option is not used by users directly)
#' @param predict_response A \code{boolean}. If TRUE, the response variable (label) 
#' is predicted, otherwise the latent random effects
#' @param offset_pred A \code{numeric} \code{vector} with 
#' additional fixed effects contributions that are added to the linear predictor for the prediction points (= offset). 
#' The length of this vector needs to equal the number of prediction points.
#' @param fixed_effects_pred This is discontinued. Use the renamed equivalent argument \code{offset_pred} instead
#' @param ... (not used, ignore this, simply here that there is no CRAN warning)
#' @inheritParams GPModel_shared_params 
#' @param num_neighbors_pred an \code{integer} specifying the number of neighbors for making predictions.
#' This is discontinued here. Use the function 'set_prediction_data' to specify this
#' @param vecchia_pred_type A \code{string} specifying the type of Vecchia approximation used for making predictions.
#' This is discontinued here. Use the function 'set_prediction_data' to specify this
#'
#' @return Predictions from a \code{GPModel}. A list with three entries is returned:
#' \itemize{
#' \item{ "mu" (first entry): predictive (=posterior) mean. For (generalized) linear mixed
#' effects models, i.e., models with a linear regression term, this consists of the sum of 
#' fixed effects and random effects predictions }
#' \item{ "cov" (second entry): predictive (=posterior) covariance matrix. 
#' This is NULL if 'predict_cov_mat=FALSE'  }
#' \item{ "var" (third entry) : predictive (=posterior) variances. 
#' This is NULL if 'predict_var=FALSE'  }
#' }
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' # Add intercept column
#' X1 <- cbind(rep(1,dim(X)[1]),X)
#' X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1,
#'                        likelihood="gaussian", params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
#'                 X_pred = X_test1, predict_var = TRUE)
#' pred$mu # Predicted mean
#' pred$var # Predicted variances
#' # Also predict covariance matrix
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
#'                 X_pred = X_test1, predict_cov_mat = TRUE)
#' pred$mu # Predicted mean
#' pred$cov # Predicted covariance
#'
#'
#' #--------------------Gaussian process model----------------
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        likelihood="gaussian", y = y, X = X1, params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, gp_coords_pred = coords_test, 
#'                 X_pred = X_test1, predict_cov_mat = TRUE)
#' pred$mu # Predicted (posterior) mean of GP
#' pred$cov # Predicted (posterior) covariance matrix of GP
#' }
#' 
#' @rdname predict.GPModel
#' @author Fabio Sigrist
#' @export
predict.GPModel <- function(object,
                            y = NULL,
                            group_data_pred = NULL,
                            group_rand_coef_data_pred = NULL,
                            gp_coords_pred = NULL,
                            gp_rand_coef_data_pred = NULL,
                            cluster_ids_pred = NULL,
                            predict_cov_mat = FALSE,
                            predict_var = FALSE,
                            cov_pars = NULL,
                            X_pred = NULL,
                            use_saved_data = FALSE,
                            predict_response = TRUE,
                            offset = NULL,
                            offset_pred = NULL, 
                            fixed_effects = NULL,
                            fixed_effects_pred = NULL, 
                            vecchia_pred_type = NULL,
                            num_neighbors_pred = NULL,...){
  return(object$predict( y = y
                         , group_data_pred = group_data_pred
                         , group_rand_coef_data_pred = group_rand_coef_data_pred
                         , gp_coords_pred = gp_coords_pred
                         , gp_rand_coef_data_pred = gp_rand_coef_data_pred
                         , cluster_ids_pred = cluster_ids_pred
                         , predict_cov_mat = predict_cov_mat
                         , predict_var = predict_var
                         , cov_pars = cov_pars
                         , X_pred = X_pred
                         , use_saved_data = use_saved_data
                         , predict_response = predict_response
                         , offset = offset
                         , offset_pred = offset_pred
                         , fixed_effects = fixed_effects
                         , fixed_effects_pred = fixed_effects_pred
                         , vecchia_pred_type = vecchia_pred_type
                         , num_neighbors_pred = num_neighbors_pred
                         , ... ))
}

#' @name saveGPModel
#' @title Save a \code{GPModel}
#' @description Save a \code{GPModel}
#' @param gp_model a \code{GPModel}
#' @param filename filename for saving
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' # Add intercept column
#' X1 <- cbind(rep(1,dim(X)[1]),X)
#' X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)
#' 
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
#'                 X_pred = X_test1, predict_var = TRUE)
#' # Save model to file
#' filename <- tempfile(fileext = ".json")
#' saveGPModel(gp_model,filename = filename)
#' # Load from file and make predictions again
#' gp_model_loaded <- loadGPModel(filename = filename)
#' pred_loaded <- predict(gp_model_loaded, group_data_pred = group_data_test[,1], 
#'                        X_pred = X_test1, predict_var = TRUE)
#' # Check equality
#' pred$mu - pred_loaded$mu
#' pred$var - pred_loaded$var
#' }
#' @rdname saveGPModel
#' @importFrom RJSONIO toJSON
#' @author Fabio Sigrist
#' @export
#' 
saveGPModel <- function(gp_model, filename) {
  
  if (!gpb.check.r6.class(gp_model, "GPModel")) {
    stop("saveGPModel: gp_model needs to be a ", sQuote("GPModel"))
  }
  
  if (!(is.character(filename) && length(filename) == 1L)) {
    stop("saveGPModel: filename should be a string")
  }
  
  # Save GPModel
  return(invisible(gp_model$save(filename = filename)))
  
}

#' @name loadGPModel
#' @title Load a \code{GPModel} from a file
#' @description Load a \code{GPModel} from a file
#' @param filename filename for loading
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' # Add intercept column
#' X1 <- cbind(rep(1,dim(X)[1]),X)
#' X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)
#' 
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
#'                 X_pred = X_test1, predict_var = TRUE)
#' # Save model to file
#' filename <- tempfile(fileext = ".json")
#' saveGPModel(gp_model,filename = filename)
#' # Load from file and make predictions again
#' gp_model_loaded <- loadGPModel(filename = filename)
#' pred_loaded <- predict(gp_model_loaded, group_data_pred = group_data_test[,1], 
#'                        X_pred = X_test1, predict_var = TRUE)
#' # Check equality
#' pred$mu - pred_loaded$mu
#' pred$var - pred_loaded$var
#' }
#' @rdname loadGPModel
#' @importFrom RJSONIO fromJSON
#' @author Fabio Sigrist
#' @export
loadGPModel <- function(filename){
  
  if (!(is.character(filename) && length(filename) == 1L)) {
    stop("loadGPModel: filename should be a string")
  }
  if (!file.exists(filename)) {
    stop(sprintf("loadGPModel: file '%s' passed to filename does not exist", filename))
  }
  
  return(invisible(gpb.GPModel$new(modelfile = filename)))
  
}

#' Set parameters for estimation of the covariance parameters
#' 
#' Set parameters for optimization of the covariance parameters of a \code{GPModel}
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' gp_model <- GPModel(group_data = group_data, likelihood="gaussian")
#' set_optim_params(gp_model, params=list(optimizer_cov="nelder_mead"))
#' }
#' 
#' @author Fabio Sigrist
#' @export 
#' 
set_optim_params <- function(gp_model,
                             params = list()) UseMethod("set_optim_params")

#' Set parameters for estimation of the covariance parameters
#' 
#' Set parameters for optimization of the covariance parameters of a \code{GPModel}
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' gp_model <- GPModel(group_data = group_data, likelihood="gaussian")
#' set_optim_params(gp_model, params=list(optimizer_cov="nelder_mead"))
#' }
#' @method set_optim_params GPModel 
#' @rdname set_optim_params.GPModel
#' @author Fabio Sigrist
#' @export 
#' 
set_optim_params.GPModel <- function(gp_model
                                     , params = list()) {
  
  if (!gpb.check.r6.class(gp_model, "GPModel")) {
    stop("set_optim_params.GPModel: gp_model needs to be a ", sQuote("GPModel"))
  }
  
  invisible(gp_model$set_optim_params(params = params))
}

#' Set prediction data for a \code{GPModel}
#' 
#' Set the data required for making predictions with a \code{GPModel} 
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' set.seed(1)
#' train_ind <- sample.int(length(y),size=250)
#' gp_model <- GPModel(group_data = group_data[train_ind,1], likelihood="gaussian")
#' set_prediction_data(gp_model, group_data_pred = group_data[-train_ind,1])
#' }
#' 
#' @author Fabio Sigrist
#' @export 
#' 
set_prediction_data <- function(gp_model,
                                vecchia_pred_type = NULL,
                                num_neighbors_pred = NULL,
                                cg_delta_conv_pred = NULL,
                                nsim_var_pred = NULL,
                                rank_pred_approx_matrix_lanczos = NULL,
                                group_data_pred = NULL,
                                group_rand_coef_data_pred = NULL,
                                gp_coords_pred = NULL,
                                gp_rand_coef_data_pred = NULL,
                                cluster_ids_pred = NULL,
                                X_pred = NULL) UseMethod("set_prediction_data")

#' Set prediction data for a \code{GPModel}
#' 
#' Set the data required for making predictions with a \code{GPModel} 
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' set.seed(1)
#' train_ind <- sample.int(length(y),size=250)
#' gp_model <- GPModel(group_data = group_data[train_ind,1], likelihood="gaussian")
#' set_prediction_data(gp_model, group_data_pred = group_data[-train_ind,1])
#' }
#' @method set_prediction_data GPModel 
#' @rdname set_prediction_data.GPModel
#' @author Fabio Sigrist
#' @export 
#' 
set_prediction_data.GPModel <- function(gp_model
                                        , vecchia_pred_type = NULL
                                        , num_neighbors_pred = NULL
                                        , cg_delta_conv_pred = NULL
                                        , nsim_var_pred = NULL
                                        , rank_pred_approx_matrix_lanczos = NULL
                                        , group_data_pred = NULL
                                        , group_rand_coef_data_pred = NULL
                                        , gp_coords_pred = NULL
                                        , gp_rand_coef_data_pred = NULL
                                        , cluster_ids_pred = NULL
                                        , X_pred = NULL) {
  
  if (!gpb.check.r6.class(gp_model, "GPModel")) {
    stop("set_prediction_data.GPModel: gp_model needs to be a ", sQuote("GPModel"))
  }
  
  invisible(gp_model$set_prediction_data(vecchia_pred_type = vecchia_pred_type
                                         , num_neighbors_pred = num_neighbors_pred
                                         , cg_delta_conv_pred = cg_delta_conv_pred
                                         , nsim_var_pred = nsim_var_pred
                                         , rank_pred_approx_matrix_lanczos = rank_pred_approx_matrix_lanczos
                                         , group_data_pred = group_data_pred
                                         , group_rand_coef_data_pred = group_rand_coef_data_pred
                                         , gp_coords_pred = gp_coords_pred
                                         , gp_rand_coef_data_pred = gp_rand_coef_data_pred
                                         , cluster_ids_pred = cluster_ids_pred
                                         , X_pred = X_pred))
}

#' Evaluate the negative log-likelihood
#' 
#' Evaluate the negative log-likelihood. If there is a linear fixed effects
#' predictor term, this needs to be calculated "manually" prior to calling this 
#' function (see example below)
#' 
#' 
#' @param gp_model A \code{GPModel}
#' @param cov_pars A \code{vector} with \code{numeric} elements. 
#' Covariance parameters of Gaussian process and  random effects
#' @param aux_pars A \code{vector} with \code{numeric} elements. 
#' Additional parameters for non-Gaussian likelihoods (e.g., shape parameter of a gamma or negative_binomial likelihood)
#' @inheritParams GPModel_shared_params
#' @param fixed_effects A \code{numeric} \code{vector} with fixed effects, e.g., containing a linear predictor. 
#' The length of this vector needs to equal the number of training data points.
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' gp_model <- GPModel(group_data = group_data, likelihood="gaussian")
#' X1 <- cbind(rep(1,dim(X)[1]), X)
#' coef <- c(0.1, 0.1, 0.1)
#' fixed_effects <- as.numeric(X1 %*% coef)
#' neg_log_likelihood(gp_model, y = y, cov_pars = c(0.1,1,1), 
#'                    fixed_effects = fixed_effects)
#' }
#' @author Fabio Sigrist
#' @export 
#' 
neg_log_likelihood <- function(gp_model
                               , cov_pars
                               , y
                               , fixed_effects = NULL
                               , aux_pars = NULL) UseMethod("neg_log_likelihood")

#' Evaluate the negative log-likelihood
#' 
#' Evaluate the negative log-likelihood. If there is a linear fixed effects
#' predictor term, this needs to be calculated "manually" prior to calling this 
#' function (see example below)
#' 
#' @param gp_model A \code{GPModel}
#' @param cov_pars A \code{vector} with \code{numeric} elements. 
#' Covariance parameters of Gaussian process and  random effects
#' @param aux_pars A \code{vector} with \code{numeric} elements. 
#' Additional parameters for non-Gaussian likelihoods (e.g., shape parameter of a gamma or negative_binomial likelihood)
#' @inheritParams GPModel_shared_params
#' @param fixed_effects A \code{numeric} \code{vector} with fixed effects, e.g., containing a linear predictor. 
#' The length of this vector needs to equal the number of training data points.
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' gp_model <- GPModel(group_data = group_data, likelihood="gaussian")
#' X1 <- cbind(rep(1,dim(X)[1]), X)
#' coef <- c(0.1, 0.1, 0.1)
#' fixed_effects <- as.numeric(X1 %*% coef)
#' neg_log_likelihood(gp_model, y = y, cov_pars = c(0.1,1,1), 
#'                    fixed_effects = fixed_effects)
#' }
#' @method neg_log_likelihood GPModel 
#' @rdname neg_log_likelihood.GPModel
#' @author Fabio Sigrist
#' @export 
#' 
neg_log_likelihood.GPModel <- function(gp_model
                                       , cov_pars
                                       , y
                                       , fixed_effects = NULL
                                       , aux_pars = NULL) {
  
  if (!gpb.check.r6.class(gp_model, "GPModel")) {
    stop("neg_log_likelihood.GPModel: gp_model needs to be a ", sQuote("GPModel"))
  }
  
  gp_model$neg_log_likelihood(cov_pars = cov_pars, 
                              y = y, 
                              fixed_effects = fixed_effects,
                              aux_pars = aux_pars)
}

#' Predict ("estimate") training data random effects for a \code{GPModel}
#' 
#' Predict ("estimate") training data random effects for a \code{GPModel}
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' # Add intercept column
#' X1 <- cbind(rep(1,dim(X)[1]),X)
#' X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)
#' 
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
#' all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
#' first_occurences <- match(unique(group_data[,1]), group_data[,1])
#' unique_training_data_random_effects <- all_training_data_random_effects[first_occurences]
#' head(unique_training_data_random_effects)
#' }
#' @author Fabio Sigrist
#' @export 
#' 
predict_training_data_random_effects <- function(gp_model,
                                                 predict_var = FALSE) UseMethod("predict_training_data_random_effects")

#' Predict ("estimate") training data random effects for a \code{GPModel}
#' 
#' Predict ("estimate") training data random effects for a \code{GPModel} 
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' # Add intercept column
#' X1 <- cbind(rep(1,dim(X)[1]),X)
#' X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)
#' 
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
#' all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
#' first_occurences <- match(unique(group_data[,1]), group_data[,1])
#' unique_training_data_random_effects <- all_training_data_random_effects[first_occurences]
#' head(unique_training_data_random_effects)
#' }
#' @method predict_training_data_random_effects GPModel 
#' @rdname predict_training_data_random_effects.GPModel
#' @author Fabio Sigrist
#' @export 
#' 
predict_training_data_random_effects.GPModel <- function(gp_model,
                                                         predict_var = FALSE) {
  
  if (!gpb.check.r6.class(gp_model, "GPModel")) {
    stop("predict_training_data_random_effects.GPModel: gp_model needs to be a ", sQuote("GPModel"))
  }
  
  return(gp_model$predict_training_data_random_effects(predict_var = predict_var))
}

#' Get (estimated) covariance parameters
#' 
#' Get (estimated) covariance parameters and standard deviations (if std_dev=TRUE was set in \code{fit})
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
#' get_cov_pars(gp_model)
#' }
#' 
#' @author Fabio Sigrist
#' @export 
#' 
get_cov_pars <- function(gp_model) UseMethod("get_cov_pars")

#' Get (estimated) covariance parameters
#' 
#' Get (estimated) covariance parameters and standard deviations (if std_dev=TRUE was set in \code{fit})
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
#' get_cov_pars(gp_model)
#' }
#' @method get_cov_pars GPModel 
#' @rdname get_cov_pars.GPModel
#' @author Fabio Sigrist
#' @export 
#' 
get_cov_pars.GPModel <- function(gp_model) {
  
  if (!gpb.check.r6.class(gp_model, "GPModel")) {
    stop("get_cov_pars.GPModel: gp_model needs to be a ", sQuote("GPModel"))
  }
  
  gp_model$get_cov_pars()
}

#' Get (estimated) linear regression coefficients
#' 
#' Get (estimated) linear regression coefficients and standard deviations (if std_dev=TRUE was set in \code{fit})
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
#' get_coef(gp_model)
#' }
#' 
#' @author Fabio Sigrist
#' @export 
#' 
get_coef <- function(gp_model) UseMethod("get_coef")

#' Get (estimated) linear regression coefficients
#' 
#' Get (estimated) linear regression coefficients and standard deviations (if std_dev=TRUE was set in \code{fit})
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
#' get_coef(gp_model)
#' }
#' @method get_coef GPModel 
#' @rdname get_coef.GPModel
#' @author Fabio Sigrist
#' @export 
#' 
get_coef.GPModel <- function(gp_model) {
  
  if (!gpb.check.r6.class(gp_model, "GPModel")) {
    stop("get_coef.GPModel: gp_model needs to be a ", sQuote("GPModel"))
  }
  
  gp_model$get_coef()
}

#' Get (estimated) auxiliary (additional) parameters of the likelihood
#' 
#' Get (estimated) auxiliary (additional) parameters of the likelihood such as the shape parameter of a gamma or
#' a negative binomial distribution. Some likelihoods (e.g., bernoulli_logit or poisson) have no auxiliary parameters
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
#' y_pos <- exp(y)
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y_pos, X = X1, likelihood="gamma")
#' get_aux_pars(gp_model)
#' }
#' 
#' @author Fabio Sigrist
#' @export 
#' 
get_aux_pars <- function(gp_model) UseMethod("get_aux_pars")

#' Get (estimated) auxiliary (additional) parameters of the likelihood
#' 
#' Get (estimated) auxiliary (additional) parameters of the likelihood such as the shape parameter of a gamma or
#' a negative binomial distribution. Some likelihoods (e.g., bernoulli_logit or poisson) have no auxiliary parameters
#' 
#' @param gp_model A \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' data(GPBoost_data, package = "gpboost")
#' X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
#' y_pos <- exp(y)
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y_pos, X = X1, likelihood="gamma")
#' get_aux_pars(gp_model)
#' }
#' @method get_aux_pars GPModel 
#' @rdname get_aux_pars.GPModel
#' @author Fabio Sigrist
#' @export 
#' 
get_aux_pars.GPModel <- function(gp_model) {
  
  if (!gpb.check.r6.class(gp_model, "GPModel")) {
    stop("get_aux_pars.GPModel: gp_model needs to be a ", sQuote("GPModel"))
  }
  
  gp_model$get_aux_pars()
}

#' Auxiliary function to create categorical variables for nested grouped random effects
#' 
#' Auxiliary function to create categorical variables for nested grouped random effects 
#' 
#' @param outer_var A \code{vector} containing the outer categorical grouping variable
#' within which the \code{inner_var is} nested in. Can be of type integer, double, or character.
#' @param inner_var A \code{vector} containing the inner nested categorical grouping variable
#'
#' @return A \code{vector} containing a categorical variable such that inner_var is nested in outer_var
#'
#' @examples
#' \donttest{
#' # Fit a model with Time as categorical fixed effects variables and Diet and Chick
#' #   as random effects, where Chick is nested in Diet using lme4
#' chick_nested_diet <- get_nested_categories(ChickWeight$Diet, ChickWeight$Chick)
#' fixed_effects_matrix <- model.matrix(weight ~ as.factor(Time), data = ChickWeight)
#' mod_gpb <- fitGPModel(X = fixed_effects_matrix, 
#'                       group_data = cbind(diet=ChickWeight$Diet, chick_nested_diet), 
#'                       y = ChickWeight$weight, params = list(std_dev = TRUE))
#' summary(mod_gpb)
#' # This does (almost) the same thing as the following code using lme4:
#' # mod_lme4 <-  lmer(weight ~ as.factor(Time) + (1 | Diet/Chick), data = ChickWeight, REML = FALSE)
#' # summary(mod_lme4)
#' }
#' @rdname get_nested_categories
#' @author Fabio Sigrist
#' @export 
#' 
get_nested_categories <- function(outer_var, inner_var) {
  nested_var <- rep(NA, length(outer_var))
  nb_groups <- 0
  for(i in unique(outer_var)) {# loop over outer variable
    aux_var <- as.numeric(inner_var[outer_var == i])
    nested_var[outer_var == i] <- match(aux_var, unique(aux_var)) + nb_groups
    nb_groups <- nb_groups + length(unique(aux_var))
  }
  return(nested_var)
}

