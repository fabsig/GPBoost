# Copyright (c) 2020 Fabio Sigrist. All rights reserved.
# 
# Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.

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
#' The following covariance functions are available:
#' "exponential", "gaussian", "matern", "powered_exponential", "wendland", and "exponential_tapered".
#' For "exponential", "gaussian", and "powered_exponential", we follow the notation and parametrization of Diggle and Ribeiro (2007).
#' For "matern", we follow the notation of Rassmusen and Williams (2006).
#' For "wendland", we follow the notation of Bevilacqua et al. (2019).
#' A covariance function with the suffix "_tapered" refers to a covariance function that is multiplied by 
#' a compactly supported Wendland covariance function (= tapering)
#' @param cov_fct_shape A \code{numeric} specifying the shape parameter of the covariance function 
#' (=smoothness parameter for Matern and Wendland covariance). For the Wendland covariance function, 
#' we follow the notation of Bevilacqua et al. (2019)). 
#' This parameter is irrelevant for some covariance functions such as the exponential or Gaussian.
#' @param cov_fct_taper_range A \code{numeric} specifying the range parameter of the Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2019)
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
#'                Options: "gradient_descent", "fisher_scoring", and "nelder_mead". Default = "gradient_descent"}
#'                \item{optimizer_coef}{ Optimizer used for estimating linear regression coefficients, if there are any 
#'                (for the GPBoost algorithm there are usually none). 
#'                Options: "gradient_descent", "wls", and "nelder_mead". Gradient descent steps are done simultaneously 
#'                with gradient descent steps for the covariance parameters. 
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
#'                \item{trace}{ If TRUE, information on the progress of the parameter optimization is printed. Default=FALSE.}
#'                \item{convergence_criterion}{ The convergence criterion used for terminating the optimization algorithm.
#'                Options: "relative_change_in_log_likelihood" (default) or "relative_change_in_parameters".}
#'                \item{std_dev}{ If TRUE, (asymptotic) standard deviations are calculated for the covariance parameters}
#'            }
#' @param fixed_effects A \code{vector} of optional external fixed effects which are held fixed during training. 

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
      
      # Check the need for freeing handle
      if (!gpb.is.null.handle(private$handle)) {
        
        # Freeing up handle
        .Call(
          GPB_REModelFree_R
          , private$handle
        )
        private$handle <- NULL
        
      }
      
    },
    
    # Initialize will create a GPModel
    initialize = function(likelihood = "gaussian",
                          group_data = NULL,
                          group_rand_coef_data = NULL,
                          ind_effect_group_rand_coef = NULL,
                          gp_coords = NULL,
                          gp_rand_coef_data = NULL,
                          cov_function = "exponential",
                          cov_fct_shape = 0,
                          cov_fct_taper_range = 1,
                          vecchia_approx = FALSE,
                          num_neighbors = 30L,
                          vecchia_ordering = "none",
                          vecchia_pred_type = "order_obs_first_cond_obs_only",
                          num_neighbors_pred = num_neighbors,
                          cluster_ids = NULL,
                          free_raw_data = FALSE,
                          modelfile = NULL,
                          model_list = NULL) {
      
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
        group_rand_coef_data = model_list[["group_rand_coef_data"]]
        ind_effect_group_rand_coef = model_list[["ind_effect_group_rand_coef"]]
        gp_coords = model_list[["gp_coords"]]
        gp_rand_coef_data = model_list[["gp_rand_coef_data"]]
        cov_function = model_list[["cov_function"]]
        cov_fct_shape = model_list[["cov_fct_shape"]]
        cov_fct_taper_range = model_list[["cov_fct_taper_range"]]
        vecchia_approx = model_list[["vecchia_approx"]]
        num_neighbors = model_list[["num_neighbors"]]
        vecchia_ordering = model_list[["vecchia_ordering"]]
        vecchia_pred_type = model_list[["vecchia_pred_type"]]
        num_neighbors_pred = model_list[["num_neighbors_pred"]]
        cluster_ids = model_list[["cluster_ids"]]
        likelihood = model_list[["likelihood"]]
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
        }
      }# end !is.null(modelfile)
      
      if(likelihood == "gaussian"){
        private$cov_par_names <- c("Error_term")
      }else{
        private$cov_par_names <- c()
      }
      if (is.null(group_data) & is.null(gp_coords)) {
        stop("GPModel: Both ", sQuote("group_data"), " and " , sQuote("gp_coords"),
             " are NULL. Provide at least one of them.")
      }
      private$vecchia_approx <- as.logical(vecchia_approx)
      # Set data for grouped random effects
      group_data_c_str <- NULL
      if (!is.null(group_data)) {
        
        # Check for correct format and set meta data
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
          private$cov_par_names <- c(private$cov_par_names,paste0("Group_",1:private$num_group_re))
        } else {
          private$cov_par_names <- c(private$cov_par_names,colnames(private$group_data))
        }
        # Convert to correct format for passing to C
        group_data <- as.vector(group_data)
        group_data_unique <- unique(group_data)
        group_data_unique_c_str <- lapply(group_data_unique,gpb.c_str)
        group_data_c_str <- unlist(group_data_unique_c_str[match(group_data,group_data_unique)])
        # Version 2: slower than above
        # group_data_c_str <- unlist(lapply(group_data,gpb.c_str))
        # group_data_c_str <- c()# Version 3: much slower
        # for (i in 1:length(group_data)) {
        #   group_data_c_str <- c(group_data_c_str,gpb.c_str(group_data[i]))
        # }
        
        # Set data for random coefficients for grouped REs
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
            
            stop("GPModel: Can only use ", sQuote("matrix"), " as ",sQuote("group_rand_coef_data"))
            
          }
          
          if (dim(group_rand_coef_data)[1] != private$num_data) {
            stop("GPModel: Number of data points in ", sQuote("group_rand_coef_data"), " does not match number of data points in ", sQuote("group_data"))
          }
          
          if (is.null(ind_effect_group_rand_coef)) {
            stop("GPModel: Indices of grouped random effects (", sQuote("ind_effect_group_rand_coef"), ") for random slopes in ", sQuote("group_rand_coef_data"), " not provided")
          }
          
          if (dim(group_rand_coef_data)[2] != length(ind_effect_group_rand_coef)) {
            stop("GPModel: Number of random coefficients in ", sQuote("group_rand_coef_data"), "does not match number in ", sQuote("ind_effect_group_rand_coef"))
          }
          
          if (storage.mode(ind_effect_group_rand_coef) != "integer") {
            storage.mode(ind_effect_group_rand_coef) <- "integer"
          }
          
          private$num_group_rand_coef <- as.integer(dim(group_rand_coef_data)[2])
          private$group_rand_coef_data <- group_rand_coef_data
          group_rand_coef_data <- as.vector(matrix((private$group_rand_coef_data))) #convert to correct format for sending to C
          ind_effect_group_rand_coef <- as.vector(ind_effect_group_rand_coef)
          private$ind_effect_group_rand_coef <- ind_effect_group_rand_coef
          counter_re <- rep(1,private$num_group_re)
          for (ii in 1:private$num_group_rand_coef) {
            if (is.null(colnames(private$group_rand_coef_data))) {
              private$cov_par_names <- c(private$cov_par_names,
                                         paste0(private$cov_par_names[ind_effect_group_rand_coef[ii]+1],
                                                "_rand_coef_nb_",counter_re[ind_effect_group_rand_coef[ii]]))
              counter_re[ind_effect_group_rand_coef[ii]] <- counter_re[ind_effect_group_rand_coef[ii]] + 1
            }else{
              private$cov_par_names <- c(private$cov_par_names,
                                         paste0(private$cov_par_names[ind_effect_group_rand_coef[ii]+1],
                                                "_rand_coef_",colnames(private$group_rand_coef_data)[ii]))
            }
          }
          
        }
        
      }
      
      # Set data for (spatial, spatio-temporal, or temporal) Gaussian process
      if (!is.null(gp_coords)) {
        
        if (is.numeric(gp_coords)) {
          gp_coords <- as.matrix(gp_coords)
        }
        
        if (is.matrix(gp_coords)) {
          
          # Check whether matrix is the correct type first ("double")
          if (storage.mode(gp_coords) != "double") {
            storage.mode(gp_coords) <- "double"
          }
          
        } else {
          
          stop("GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("gp_coords"))
          
        }
        
        if (!is.null(private$num_data)) {
          
          if (dim(gp_coords)[1] != private$num_data) {
            stop("GPModel: Number of data points in ", sQuote("gp_coords"), " does not match number of data points in ", sQuote("group_data"))
          }
          
        } else {
          
          private$num_data <- as.integer(dim(gp_coords)[1])
          
        }
        
        private$num_gp <- 1L
        private$dim_coords <- as.integer(dim(gp_coords)[2])
        private$gp_coords <- gp_coords
        gp_coords <- as.vector(matrix(private$gp_coords)) #convert to correct format for sending to C
        private$cov_function <- cov_function
        private$cov_fct_shape <- as.numeric(cov_fct_shape)
        private$cov_fct_taper_range <- as.numeric(cov_fct_taper_range)
        private$num_neighbors <- as.integer(num_neighbors)
        private$vecchia_ordering <- vecchia_ordering
        private$vecchia_pred_type <- vecchia_pred_type
        private$num_neighbors_pred <- as.integer(num_neighbors_pred)
        if (private$cov_function == "wendland") {
          private$cov_par_names <- c(private$cov_par_names,"GP_var")
        } else {
          private$cov_par_names <- c(private$cov_par_names,"GP_var","GP_range")
        }
        
        # Set data for GP random coef
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
              if (private$cov_function == "wendland") {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_nb_", ii,"_var"))
              } else {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_nb_", ii,"_var"),
                                           paste0("GP_rand_coef_nb_", ii,"_range"))
              }
              
            } else {
              if (private$cov_function == "wendland") {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_var"))
              } else {
                private$cov_par_names <- c(private$cov_par_names,
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_var"),
                                           paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_range"))
              }
            }
          }
        }
        
      }
      
      # Set IDs for independent processes (cluster_ids)
      if (!is.null(cluster_ids)) {
        
        if (is.vector(cluster_ids)) {
          if (length(cluster_ids) != private$num_data) {
            stop("GPModel: Length of ", sQuote("cluster_ids"), "does not match number of data points")
          }
          private$cluster_ids = cluster_ids
          
          # Convert cluster_ids to int and save conversion map
          if (storage.mode(cluster_ids) != "integer") {
            
            private$cluster_ids_map_to_int <- structure(1:length(unique(cluster_ids)),names=c(unique(cluster_ids)))
            cluster_ids = private$cluster_ids_map_to_int[cluster_ids]
            
          }
          
        } else {
          
          stop("GPModel: Can only use ", sQuote("vector"), " as ", sQuote("cluster_ids"))
          
        }
        
        cluster_ids <- as.vector(cluster_ids)
        
      }
      
      private$determine_num_cov_pars(likelihood)
      
      # Create handle
      handle <- NULL
      
      # Attempts to create a handle for the GPModel
      # try({
      
      # Store handle
      handle <- .Call(
        GPB_CreateREModel_R
        , private$num_data
        , cluster_ids
        , group_data_c_str
        , private$num_group_re
        , group_rand_coef_data
        , ind_effect_group_rand_coef
        , private$num_group_rand_coef
        , private$num_gp
        , gp_coords
        , private$dim_coords
        , gp_rand_coef_data
        , private$num_gp_rand_coef
        , private$cov_function
        , private$cov_fct_shape
        , private$cov_fct_taper_range
        , private$vecchia_approx
        , private$num_neighbors
        , vecchia_ordering
        , private$vecchia_pred_type
        , private$num_neighbors_pred
        , likelihood
      )
      # })
      
      # Check whether the handle was created properly if it was not stopped earlier by a stop call
      if (gpb.is.null.handle(handle)) {
        
        stop("GPModel: Cannot create handle")
        
      } else {
        
        # Create class
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
        private$cluster_ids_map_to_int <- NULL
      }
      
      if (!is.null(modelfile)){
        self$set_optim_params(params = model_list[["params"]])
        self$set_optim_coef_params(params = model_list[["params"]])
      }
      
    },
    
    # Find parameters that minimize the negative log-ligelihood (=MLE)
    fit = function(y,
                   X = NULL,
                   params = list(),
                   fixed_effects = NULL) {
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
      }
      
      if (!is.null(fixed_effects)) {
        
        if (!is.null(X)) {
          stop("fit.GPModel: cannot provide both X and fixed_effects")
        }
        
        if (!is.vector(fixed_effects)) {
          
          if (is.matrix(fixed_effects)) {
            
            if (dim(fixed_effects)[2] != 1) {
              
              stop("fit.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("fixed_effects"))
              
            }
            
          } else{
            
            stop("fit.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("fixed_effects"))
            
          }
          
        }
        
        if (storage.mode(fixed_effects) != "double") {
          storage.mode(fixed_effects) <- "double"
        }
        
        fixed_effects <- as.vector(fixed_effects)
        
        if (length(fixed_effects) != private$num_data) {
          stop("fit.GPModel: Number of data points in ", sQuote("fixed_effects"), " does not match number of data points of initialized model")
        }
        
      }
      
      # Set data linear fixed-effects
      if (!is.null(X)) {
        
        if (is.numeric(X)) {
          X <- as.matrix(X)
        }
        
        if (is.matrix(X)) {
          
          # Check whether matrix is the correct type first ("double")
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
        
        self$set_optim_coef_params(params)
        
      } else {
        private$has_covariates <- FALSE
      }
      
      self$set_optim_params(params)
      
      if (is.null(X)) {
        
        .Call(
          GPB_OptimCovPar_R
          , private$handle
          , y
          , fixed_effects
        )
        
      } else {
        
        .Call(
          GPB_OptimLinRegrCoefCovPar_R
          , private$handle
          , y
          , X
          , private$num_coef
        )
        
      }
      
      if (private$params$trace) {
        message(paste0("GPModel: Number of iterations until convergence: ", self$get_num_optim_iter()))
      }
      
      return(invisible(self))
      
    },
    
    # Evaluate the negative log-ligelihood
    neg_log_likelihood = function(cov_pars, y) {
      
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
        
        stop("GPModel.neg_log_likelihood: Can only use ", sQuote("vector"), " as ", sQuote("cov_pars"))
        
      }
      
      if (length(cov_pars) != private$num_cov_pars) {
        stop("GPModel.neg_log_likelihood: Number of parameters in ", sQuote("cov_pars"), " does not correspond to numbers of parameters")
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
      
      negll = 0.
      .Call(
        GPB_EvalNegLogLikelihood_R
        , private$handle
        , y
        , cov_pars
        , negll
      )
      
      return(negll)
      
    },
    
    # Set configuration parameters for the optimizer
    set_optim_params = function(params = list()) {
      if (gpb.is.null.handle(private$handle)) {
        stop("GPModel: Gaussian process model has not been initialized")
      }
      
      ##Check format of configuration parameters
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
          stop("GPModel: Number of parameters in ", sQuote("params$init_cov_pars"), " does not correspond to numbers of parameters")
        }
        
      }
      
      if (!is.null(params[["use_nesterov_acc"]])) {
        if (storage.mode(params[["use_nesterov_acc"]]) != "logical") {
          stop("GPModel: Can only use ", sQuote("logical"), " as ", sQuote("params$use_nesterov_acc"))
        }
      }
      if (!is.null(params[["trace"]])) {
        if (storage.mode(params[["trace"]]) != "logical") {
          stop("GPModel: Can only use ", sQuote("logical"), " as ", sQuote("params$trace"))
        }
      }
      if (!is.null(params[["acc_rate_cov"]])) {
        params[["acc_rate_cov"]] <- as.numeric(params[["acc_rate_cov"]])
      }
      if (!is.null(params[["nesterov_schedule_version"]])) {
        params[["nesterov_schedule_version"]] = as.integer(params[["nesterov_schedule_version"]])
      }
      if (!is.null(params[["lr_cov"]])) {
        params[["lr_cov"]] <- as.numeric(params[["lr_cov"]])
      }
      if (!is.null(params[["maxit"]])) {
        params[["maxit"]] <- as.integer(params[["maxit"]])
      }
      if (!is.null(params[["optimizer_cov"]])) {
        if (!is.character(params[["optimizer_cov"]])) {
          stop("GPModel: Can only use ", sQuote("character"), " as ", sQuote("optimizer_cov"))
        }
      }
      if (!is.null(params[["convergence_criterion"]])) {
        if (!is.character(params[["convergence_criterion"]])) {
          stop("GPModel: Can only use ", sQuote("character"), " as ", sQuote("convergence_criterion"))
        }
      }
      if (!is.null(params[["delta_rel_conv"]])) {
        params[["delta_rel_conv"]] <- as.numeric(params[["delta_rel_conv"]])
      }
      if (!is.null(params[["momentum_offset"]])) {
        params[["momentum_offset"]] <- as.integer(params[["momentum_offset"]])
      }
      if (!is.null(params[["std_dev"]])) {
        if (storage.mode(params[["std_dev"]]) != "logical") {
          stop("fit.GPModel: Can only use ", sQuote("logical"), " as ", sQuote("std_dev"))
        }
      }
      
      private$update_params(params)
      
      lr_cov <- private$params[["lr_cov"]]
      acc_rate_cov <- private$params[["acc_rate_cov"]]
      maxit <- private$params[["maxit"]]
      delta_rel_conv <- private$params[["delta_rel_conv"]]
      use_nesterov_acc <- private$params[["use_nesterov_acc"]]
      nesterov_schedule_version <- private$params[["nesterov_schedule_version"]]
      trace <- private$params[["trace"]]
      momentum_offset <- private$params[["momentum_offset"]]
      convergence_criterion <- private$params[["convergence_criterion"]]
      optimizer_cov_c_str <- NULL
      init_cov_pars <- NULL
      if (!is.null(params[["optimizer_cov"]])) {
        optimizer_cov_c_str <- params[["optimizer_cov"]]
      }
      if (!is.null(params[["init_cov_pars"]])) {
        init_cov_pars <- params[["init_cov_pars"]]
      }
      std_dev <- private$params[["std_dev"]]
      
      .Call(
        GPB_SetOptimConfig_R
        , private$handle
        , init_cov_pars
        , lr_cov
        , acc_rate_cov
        , maxit
        , delta_rel_conv
        , use_nesterov_acc
        , nesterov_schedule_version
        , trace
        , optimizer_cov_c_str
        , momentum_offset
        , convergence_criterion
        , std_dev
      )
      
      return(invisible(self))
     
      
    },
    
    get_optim_params = function() {
      
      params <- private$params
      # Get covariance parameters optimizer
      params$optimizer_cov <- .Call(
        GPB_GetOptimizerCovPars_R
        , private$handle
      )
      # Get linear regression coefficients optimizer
      params$optimizer_coef <- .Call(
        GPB_GetOptimizerCoef_R
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
      }
      else{
        params$init_cov_pars <- init_cov_pars
      }
      
      return(params)
    },
    
    # Set configuration parameters for the optimizer
    set_optim_coef_params = function(params = list()) {
      num_coef <- NULL
      
      if (gpb.is.null.handle(private$handle)) {
        stop("GPModel: Gaussian process model has not been initialized")
      }
      
      if (!is.null(params[["init_coef"]])) {
        
        if (is.vector(params[["init_coef"]])) {
          
          if (storage.mode(params[["init_coef"]]) != "double") {
            storage.mode(params[["init_coef"]]) <- "double"
          }
          
          params[["init_coef"]] <- as.vector(params[["init_coef"]])
          num_coef <- as.integer(length(params[["init_coef"]]))
          if (is.null(private$num_coef)) {
            private$num_coef <- num_coef
          }
          
        } else {
          
          stop("GPModel: Can only use ", sQuote("vector"), " as ", sQuote("init_coef"))
          
        }
        
        if (length(params[["init_coef"]]) != private$num_coef) {
          stop("GPModel: Number of parameters in ", sQuote("init_coef"), " does not correspond to numbers of covariates in ", sQuote("X"))
        }
        
      }
      
      if (!is.null(params[["lr_coef"]])) {
        params[["lr_coef"]] <- as.numeric(params[["lr_coef"]])
      }
      if (!is.null(params[["acc_rate_coef"]])) {
        params[["acc_rate_coef"]] <- as.numeric(params[["acc_rate_coef"]])
      }
      if (!is.null(params[["optimizer_coef"]])) {
        if (!is.character(params[["optimizer_coef"]])) {
          stop("GPModel: Can only use ", sQuote("character"), " as ", sQuote("optimizer_coef"))
        }
      }
      
      private$update_params(params)
      
      init_coef <- private$params[["init_coef"]]
      lr_coef <- private$params[["lr_coef"]]
      acc_rate_coef <- private$params[["acc_rate_coef"]]
      optimizer_coef_c_str <- NULL
      if (!is.null(params[["optimizer_coef"]])) {
        optimizer_coef_c_str <- params[["optimizer_coef"]]
      }
      
      .Call(
        GPB_SetOptimCoefConfig_R
        , private$handle
        , num_coef
        , init_coef
        , lr_coef
        , acc_rate_coef
        , optimizer_coef_c_str
      )
      
      return(invisible(self))
      
    },
    
    get_cov_pars = function() {
      
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
      names(cov_pars) <- private$cov_par_names
      if (private$params[["std_dev"]]) {
        cov_pars_std_dev <- optim_pars[1:private$num_cov_pars+private$num_cov_pars]
        cov_pars <- rbind(cov_pars,cov_pars_std_dev)
        rownames(cov_pars) <- c("Param.", "Std. dev.")
      }
      
      return(cov_pars)
      
    },
    
    get_coef = function() {
      
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
      names(coef) <- private$coef_names
      if (private$params[["std_dev"]]) {
        coef_std_dev <- optim_pars[1:private$num_coef+private$num_coef]
        coef <- rbind(coef,coef_std_dev)
        rownames(coef) <- c("Param.", "Std. dev.")
      }
      
      return(coef)
      
    },
    
    set_prediction_data = function(group_data_pred = NULL,
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
          stop("predict.GPModel: Can only use the following types for as ", sQuote("group_data_pred"), ": ",
               sQuote("data.frame"), ", ", sQuote("matrix"), ", ", sQuote("character"),
               ", ", sQuote("numeric"), ", ", sQuote("factor"))
        }
        
        if (is.data.frame(group_data_pred) | is.numeric(group_data_pred) |
            is.character(group_data_pred) | is.factor(group_data_pred)) {
          group_data_pred <- as.matrix(group_data_pred)
        }
        
        if (dim(group_data_pred)[2] != private$num_group_re) {
          stop("predict.GPModel: Number of grouped random effects in ", sQuote("group_data_pred"), " is not correct")
        }
        
        num_data_pred <- as.integer(dim(group_data_pred)[1])
        group_data_pred <- as.vector(group_data_pred)
        group_data_pred_unique <- unique(group_data_pred)
        group_data_pred_unique_c_str <- lapply(group_data_pred_unique,gpb.c_str)
        group_data_pred_c_str <- unlist(group_data_pred_unique_c_str[match(group_data_pred,group_data_pred_unique)])
        
        # Set data for random coef for grouped REs
        if (!is.null(group_rand_coef_data_pred)) {
          
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
            stop("predict.GPModel: Number of random coef in ", sQuote("group_rand_coef_data_pred"), " is not correct")
          }
          
          group_rand_coef_data_pred <- as.vector(matrix(group_rand_coef_data_pred))
          
        }
        
      }
      
      # Set data for Gaussian process
      if (!is.null(gp_coords_pred)) {
        
        if (is.numeric(gp_coords_pred)) {
          gp_coords_pred <- as.matrix(gp_coords_pred)
        }
        
        if (is.matrix(gp_coords_pred)) {
          
          # Check whether matrix is the correct type first ("double")
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
        
        # Set data for GP random coef
        if (!is.null(gp_rand_coef_data_pred)) {
          
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
          if (dim(gp_rand_coef_data_pred)[2] != num_gp_rand_coef) {
            stop("predict.GPModel: Number of covariates in ", sQuote("gp_rand_coef_data_pred"), " is not correct")
          }
          
          gp_rand_coef_data_pred <- as.vector(matrix(gp_rand_coef_data_pred))
          
        }
        
      }
      
      # Set data linear fixed-effects
      if (!is.null(X_pred)) {
        
        if(!private$has_covariates){
          stop("predict.GPModel: Covariate data provided in ", sQuote("X_pred"), " but model has no linear predictor")
        }
        
        if (is.numeric(X_pred)) {
          X_pred <- as.matrix(X_pred)
        }
        
        if (is.matrix(X_pred)) {
          
          # Check whether matrix is the correct type first ("double")
          if (storage.mode(X_pred) != "double") {
            storage.mode(X_pred) <- "double"
          }
          
        } else {
          
          stop("GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("X_pred"))
          
        }
        
        if (dim(X_pred)[1] != num_data_pred) {
          stop("GPModel: Number of data points in ", sQuote("X_pred"), " is not correct")
        }
        if (dim(X_pred)[2] != private$num_coef) {
          stop("GPModel: Number of covariates in ", sQuote("X_pred"), " is not correct")
        }
        
        X_pred <- as.vector(matrix(X_pred))
        
      }
      
      # Set cluster_ids for independent processes
      if (!is.null(cluster_ids_pred)) {
        
        if (is.vector(cluster_ids_pred)) {
          
          if (is.null(private$cluster_ids_map_to_int) & storage.mode(cluster_ids_pred) != "integer") {
            stop("predict.GPModel: cluster_ids_pred needs to be of type int as the data provided in cluster_ids when initializing the model was also int (or cluster_ids was not provided)")
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
          stop("predict.GPModel: Length of ", sQuote("cluster_ids_pred"), " does not match number of predicted data points")
        }
        
        cluster_ids_pred <- as.vector(cluster_ids_pred)
        
      }
      
      private$num_data_pred <- num_data_pred
      
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
      )
      
      return(invisible(self))
      
    },
    
    predict = function(y = NULL,
                       group_data_pred = NULL,
                       group_rand_coef_data_pred = NULL,
                       gp_coords_pred = NULL,
                       gp_rand_coef_data_pred = NULL,
                       vecchia_pred_type = NULL,
                       num_neighbors_pred = NULL,
                       cluster_ids_pred = NULL,
                       predict_cov_mat = FALSE,
                       predict_var = FALSE,
                       cov_pars = NULL,
                       X_pred = NULL,
                       use_saved_data = FALSE,
                       predict_response = FALSE,
                       fixed_effects = NULL,
                       fixed_effects_pred = NULL) {
      
      if (private$model_has_been_loaded_from_saved_file) {
        
        if (is.null(y)) {
          y <- private$y_loaded_from_file
        }
        if (is.null(cov_pars)) {
          cov_pars <- private$cov_pars_loaded_from_file
        }
        
      }
      
      if(predict_cov_mat && predict_var){
        predict_cov_mat <- TRUE
        predict_var <- FALSE
      }
      
      group_data_pred_c_str <- NULL
      if (!is.null(vecchia_pred_type)) {
        private$vecchia_pred_type <- vecchia_pred_type
      }
      vecchia_pred_type_c_str <- private$vecchia_pred_type
      
      if (!is.null(num_neighbors_pred)) {
        private$num_neighbors_pred <- as.integer(num_neighbors_pred)
      }
      
      if (gpb.is.null.handle(private$handle)) {
        stop("predict.GPModel: Gaussian process model has not been initialized")
      }
      
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
        if (!is.null(group_data_pred)) {
          
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
          
          # Set data for random coef for grouped REs
          if (!is.null(group_rand_coef_data_pred)) {
            
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
          
        }
        
        # Set data for Gaussian process
        if (!is.null(gp_coords_pred)) {
          
          if (is.numeric(gp_coords_pred)) {
            gp_coords_pred <- as.matrix(gp_coords_pred)
          }
          
          if (is.matrix(gp_coords_pred)) {
            
            # Check whether matrix is the correct type first ("double")
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
          
          # Set data for GP random coef
          if (!is.null(gp_rand_coef_data_pred)) {
            
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
          
        }
        
        # Set data for linear fixed-effects
        
        if(private$has_covariates){
          
          if(is.null(X_pred)){
            stop("predict.GPModel: No covariate data is provided in ", sQuote("X_pred"), " but model has linear predictor")
          }
          
          if (is.numeric(X_pred)) {
            X_pred <- as.matrix(X_pred)
          }
          
          if (is.matrix(X_pred)) {
            
            # Check whether matrix is the correct type first ("double")
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
            
            if (is.null(fixed_effects)) {
              fixed_effects <- as.vector(private$X_loaded_from_file %*% private$coefs_loaded_from_file)
            } else {
              fixed_effects <- fixed_effects + as.vector(private$X_loaded_from_file %*% private$coefs_loaded_from_file)
            }
            
            if (is.null(fixed_effects_pred)) {
              fixed_effects_pred <- as.vector(X_pred %*% private$coefs_loaded_from_file)
            } else {
              fixed_effects_pred <- fixed_effects_pred + as.vector(X_pred %*% private$coefs_loaded_from_file)
            }
            
            
          } else {
            
            X_pred <- as.vector(matrix(X_pred))
            
          }
          
        } 
        
        # Set cluster_ids for independent processes
        if (!is.null(cluster_ids_pred)) {
          
          if (is.vector(cluster_ids_pred)) {
            
            if (is.null(private$cluster_ids_map_to_int) & storage.mode(cluster_ids_pred) != "integer") {
              stop("predict.GPModel: cluster_ids_pred needs to be of type int as the data provided in cluster_ids when initializing the model was also int (or cluster_ids was not provided)")
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
            stop("predict.GPModel: Length of ", sQuote("cluster_ids_pred"), " does not match number of predicted data points")
          }
          
          cluster_ids_pred <- as.vector(cluster_ids_pred)
          
        }
        
      } else {
        
        cluster_ids_pred <- NULL
        group_data_pred_c_str <- NULL
        group_rand_coef_data_pred <- NULL
        gp_coords_pred <- NULL
        gp_rand_coef_data_pred <- NULL
        X_pred <- NULL
        num_data_pred <- private$num_data_pred
        
        if (is.null(private$num_data_pred)) {
          stop("predict.GPModel: No data has been set for making predictions. Call set_prediction_data first")
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
      
      if (!is.null(fixed_effects)) {
        
        if (is.vector(fixed_effects)) {
          
          if (storage.mode(fixed_effects) != "double") {
            storage.mode(fixed_effects) <- "double"
          }
          
          fixed_effects <- as.vector(fixed_effects)
          
        } else {
          
          stop("predict.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("fixed_effects"))
          
        }
        
        if (length(fixed_effects) != private$num_data) {
          stop("predict.GPModel: Length of ", sQuote("fixed_effects"), " does not match number of observed data points")
        }
        
      }##end fixed_effects
      
      if (!is.null(fixed_effects_pred)) {
        
        if (is.vector(fixed_effects_pred)) {
          
          if (storage.mode(fixed_effects_pred) != "double") {
            storage.mode(fixed_effects_pred) <- "double"
          }
          
          fixed_effects_pred <- as.vector(fixed_effects_pred)
          
        } else {
          
          stop("predict.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("fixed_effects_pred"))
          
        }
        
        if (length(fixed_effects_pred) != num_data_pred) {
          stop("predict.GPModel: Length of ", sQuote("fixed_effects"), " does not match number of predicted data points")
        }
        
      }##end fixed_effects_pred
      
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
        , vecchia_pred_type_c_str
        , private$num_neighbors_pred
        , fixed_effects
        , fixed_effects_pred
        , preds
      )
      
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
    
    get_cov_function = function() {
      return(private$cov_function)
    },
    
    get_cov_fct_shape = function() {
      return(private$cov_fct_shape)
    },
    
    get_cov_fct_taper_range = function() {
      return(private$cov_fct_taper_range)
    },
    
    get_ind_effect_group_rand_coef = function() {
      return(private$ind_effect_group_rand_coef)
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
      private$determine_num_cov_pars(likelihood)
      if (likelihood != "gaussian" && "Error_term" %in% private$cov_par_names){
        private$cov_par_names <- private$cov_par_names["Error_term" != private$cov_par_names]
      }
      if (likelihood == "gaussian" && !("Error_term" %in% private$cov_par_names)){
        private$cov_par_names <- c("Error_term",private$cov_par_names)
      }
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
      model_list <- list()
      # Parameters
      model_list[["params"]] <- self$get_optim_params()
      model_list[["likelihood"]] <- self$get_likelihood_name()
      model_list[["cov_pars"]] <- self$get_cov_pars()
      # Response data
      if (include_response_data) {
        model_list[["y"]] <- self$get_response_data()
      }
      # Feature data
      model_list[["group_data"]] <- self$get_group_data()
      model_list[["group_rand_coef_data"]] <- self$get_group_rand_coef_data()
      model_list[["gp_coords"]] <- self$get_gp_coords()
      model_list[["gp_rand_coef_data"]] <- self$get_gp_rand_coef_data()
      model_list[["ind_effect_group_rand_coef"]] <- self$get_ind_effect_group_rand_coef()
      model_list[["cluster_ids"]] <- self$get_cluster_ids()
      model_list[["vecchia_approx"]] <- private$vecchia_approx
      model_list[["num_neighbors"]] <- private$num_neighbors
      model_list[["vecchia_ordering"]] <- private$vecchia_ordering
      model_list[["vecchia_pred_type"]] <- private$vecchia_pred_type
      model_list[["num_neighbors_pred"]] <- private$num_neighbors_pred
      model_list[["cov_function"]] <- private$cov_function
      model_list[["cov_fct_shape"]] <- private$cov_fct_shape
      model_list[["cov_fct_taper_range"]] <- private$cov_fct_taper_range
      # Covariate data
      model_list[["has_covariates"]] <- private$has_covariates
      if (private$has_covariates) {
        model_list[["coefs"]] <- self$get_coef()
        model_list[["num_coef"]] <- private$num_coef
        model_list[["X"]] <- self$get_covariate_data()
      }
      # Make sure that data is saved in correct format by RJSONIO::toJSON
      MAYBE_CONVERT_TO_VECTOR <- c("cov_pars","group_data", "group_rand_coef_data",
                                   "gp_coords", "gp_rand_coef_data",
                                   "ind_effect_group_rand_coef",
                                   "cluster_ids","coefs","X")
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
      
    }

  ),
  
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
    num_coef = 0,
    group_data = NULL,
    group_rand_coef_data = NULL,
    ind_effect_group_rand_coef = NULL,
    gp_coords = NULL,
    gp_rand_coef_data = NULL,
    cov_function = "exponential",
    cov_fct_shape = 0.,
    cov_fct_taper_range = 1.,
    vecchia_approx = FALSE,
    num_neighbors = 30L,
    vecchia_ordering = "none",
    vecchia_pred_type = "order_obs_first_cond_obs_only",
    num_neighbors_pred = 30L,
    cov_par_names = NULL,
    coef_names = NULL,
    cluster_ids = NULL,
    cluster_ids_map_to_int = NULL,
    free_raw_data = FALSE,
    num_data_pred = NULL,
    model_has_been_loaded_from_saved_file = FALSE,
    y_loaded_from_file = NULL,
    cov_pars_loaded_from_file = NULL,
    coefs_loaded_from_file = NULL,
    X_loaded_from_file = NULL,
    params = list(maxit = 1000L,
                  delta_rel_conv = 1E-6,
                  init_coef = NULL,
                  lr_coef = 0.1,
                  lr_cov = -1.,
                  use_nesterov_acc = TRUE,
                  acc_rate_coef = 0.5,
                  acc_rate_cov = 0.5,
                  nesterov_schedule_version = 0L,
                  momentum_offset = 2L,
                  trace = FALSE,
                  convergence_criterion = "relative_change_in_log_likelihood",
                  std_dev = FALSE),
    
    determine_num_cov_pars = function(likelihood) {
      if (private$cov_function == "wendland") {
        num_par_per_GP <- 1L
      } else {
        num_par_per_GP <- 2L
      }
      private$num_cov_pars <- private$num_group_re + private$num_group_rand_coef + 
        num_par_per_GP * (private$num_gp + private$num_gp_rand_coef)
      if (likelihood == "gaussian"){
        private$num_cov_pars <- private$num_cov_pars + 1L
      }
      storage.mode(private$num_cov_pars) <- "integer"
    },
    
    update_params = function(params) {
      for (param in names(params)) {
        if (param %in% names(private$params)) {
          if (is.null(params[[param]])) {
            private$params[param] <- list(NULL)
          }else{
            private$params[[param]] <- params[[param]]
          }
        }
        else if (!(param %in% c("optimizer_cov", "init_cov_pars", "optimizer_coef"))){
          stop(paste0("GPModel: Unknown parameter: ", param))
        }
      }
    },
    
    # Get handle
    get_handle = function() {
      
      if (gpb.is.null.handle(private$handle)) {
        stop("GPModel: model has not been initialized")
      }
      
      private$handle
      
    }
  )##end private
)

#' Create a \code{GPModel} object
#'
#' Create a \code{GPModel} which contains a Gaussian process and / or mixed effects model with grouped random effects
#'
#' @inheritParams GPModel_shared_params 
#'
#' @return A \code{GPModel} containing ontains a Gaussian process and / or mixed effects model with grouped random effects
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' library(gpboost)
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
GPModel <- function(group_data = NULL,
                    group_rand_coef_data = NULL,
                    ind_effect_group_rand_coef = NULL,
                    gp_coords = NULL,
                    gp_rand_coef_data = NULL,
                    cov_function = "exponential",
                    cov_fct_shape = 0,
                    cov_fct_taper_range = 1,
                    vecchia_approx = FALSE,
                    num_neighbors = 30L,
                    vecchia_ordering = "none",
                    vecchia_pred_type = "order_obs_first_cond_obs_only",
                    num_neighbors_pred = num_neighbors,
                    cluster_ids = NULL,
                    free_raw_data = FALSE,
                    likelihood = "gaussian") {
  
  # Create new GPModel
  invisible(gpb.GPModel$new(group_data = group_data,
                            group_rand_coef_data = group_rand_coef_data,
                            ind_effect_group_rand_coef = ind_effect_group_rand_coef,
                            gp_coords = gp_coords,
                            gp_rand_coef_data = gp_rand_coef_data,
                            cov_function = cov_function,
                            cov_fct_shape = cov_fct_shape,
                            cov_fct_taper_range = cov_fct_taper_range,
                            vecchia_approx = vecchia_approx,
                            num_neighbors = num_neighbors,
                            vecchia_ordering = vecchia_ordering,
                            vecchia_pred_type = vecchia_pred_type,
                            num_neighbors_pred = num_neighbors_pred,
                            cluster_ids = cluster_ids,
                            free_raw_data = free_raw_data,
                            likelihood = likelihood))
  
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
fit <- function(gp_model, y, X, params, fixed_effects = NULL) UseMethod("fit")

#' Fits a \code{GPModel}
#'
#' Estimates the parameters of a \code{GPModel} using maximum likelihood estimation
#'
#' @param gp_model a \code{GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @return A fitted \code{GPModel}
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
#' fit(gp_model, y = y, params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_var = TRUE)
#' pred$mu # Predicted mean
#' pred$var # Predicted variances
#' # Also predict covariance matrix
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_cov_mat = TRUE)
#' pred$mu # Predicted mean
#' pred$cov # Predicted covariance
#'  
#' \donttest{
#' #--------------------Gaussian process model----------------
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
#'                     likelihood="gaussian")
#' fit(gp_model, y = y, params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, gp_coords_pred = coords_test, predict_cov_mat = TRUE)
#' # Predicted (posterior/conditional) mean of GP
#' pred$mu
#' # Predicted (posterior/conditional) covariance matrix of GP
#' pred$cov
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
                        fixed_effects = NULL) {
  
  # Fit model
  invisible(gp_model$fit(y = y,
                         X = X,
                         params = params,
                         fixed_effects = fixed_effects))
  
}

#' Fits a \code{GPModel}
#'
#' Estimates the parameters of a \code{GPModel} using maximum likelihood estimation
#'
#' @inheritParams GPModel_shared_params 
#'
#' @return A fitted \code{GPModel}
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, likelihood="gaussian",
#'                        params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_var = TRUE)
#' pred$mu # Predicted mean
#' pred$var # Predicted variances
#' # Also predict covariance matrix
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_cov_mat = TRUE)
#' pred$mu # Predicted mean
#' pred$cov # Predicted covariance
#'
#'
#' \donttest{
#' #--------------------Mixed effects model: random effects and linear fixed effects----------------
#' X1 <- cbind(rep(1,length(y)),X) # Add intercept column
#' gp_model <- fitGPModel(group_data = group_data[,1], likelihood="gaussian",
#'                        y = y, X = X1, params = list(std_dev = TRUE))
#' summary(gp_model)
#'
#'
#' #--------------------Two crossed random effects and a random slope----------------
#' gp_model <- fitGPModel(group_data = group_data, likelihood="gaussian",
#'                        group_rand_coef_data = X[,2],
#'                        ind_effect_group_rand_coef = 1,
#'                        y = y, params = list(std_dev = TRUE))
#' summary(gp_model)
#'
#'
#' #--------------------Gaussian process model----------------
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        likelihood="gaussian", y = y, params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, gp_coords_pred = coords_test, predict_cov_mat = TRUE)
#' # Predicted (posterior/conditional) mean of GP
#' pred$mu
#' # Predicted (posterior/conditional) covariance matrix of GP
#' pred$cov
#'
#'
#' #--------------------Gaussian process model with linear mean function----------------
#' X1 <- cbind(rep(1,length(y)),X) # Add intercept column
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        likelihood="gaussian", y = y, X=X1, params = list(std_dev = TRUE))
#' summary(gp_model)
#'
#'
#' #--------------------Gaussian process model with Vecchia approximation----------------
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        vecchia_approx = TRUE, num_neighbors = 30,
#'                        likelihood="gaussian", y = y)
#' summary(gp_model)
#'
#'
#' #--------------------Gaussian process model with random coefficents----------------
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
#'                     gp_rand_coef_data = X[,2], likelihood = "gaussian")
#' fit(gp_model, y = y, params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        gp_rand_coef_data = X[,2], y=y,
#'                        likelihood = "gaussian", params = list(std_dev = TRUE))
#' summary(gp_model)
#'
#'
#' #--------------------Combine Gaussian process with grouped random effects----------------
#' gp_model <- fitGPModel(group_data = group_data,
#'                        gp_coords = coords, cov_function = "exponential",
#'                        likelihood = "gaussian", y = y, params = list(std_dev = TRUE))
#' summary(gp_model)
#' }
#' 
#' @rdname fitGPModel
#' @author Fabio Sigrist
#' @export fitGPModel
fitGPModel <- function(group_data = NULL,
                       group_rand_coef_data = NULL,
                       ind_effect_group_rand_coef = NULL,
                       gp_coords = NULL,
                       gp_rand_coef_data = NULL,
                       cov_function = "exponential",
                       cov_fct_shape = 0,
                       cov_fct_taper_range = 1,
                       vecchia_approx = FALSE,
                       num_neighbors = 30L,
                       vecchia_ordering = "none",
                       vecchia_pred_type = "order_obs_first_cond_obs_only",
                       num_neighbors_pred = num_neighbors,
                       cluster_ids = NULL,
                       free_raw_data = FALSE,
                       likelihood = "gaussian",
                       y,
                       X = NULL,
                       params = list()) {
  #Create model
  gpmodel <- gpb.GPModel$new(group_data = group_data,
                             group_rand_coef_data = group_rand_coef_data,
                             ind_effect_group_rand_coef = ind_effect_group_rand_coef,
                             gp_coords = gp_coords,
                             gp_rand_coef_data = gp_rand_coef_data,
                             cov_function = cov_function,
                             cov_fct_shape = cov_fct_shape,
                             cov_fct_taper_range = cov_fct_taper_range,
                             vecchia_approx = vecchia_approx,
                             num_neighbors = num_neighbors,
                             vecchia_ordering = vecchia_ordering,
                             vecchia_pred_type = vecchia_pred_type,
                             num_neighbors_pred = num_neighbors_pred,
                             cluster_ids = cluster_ids,
                             free_raw_data = free_raw_data,
                             likelihood = likelihood)
  # Fit model
  gpmodel$fit(y = y,
              X = X,
              params = params)
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
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, likelihood="gaussian",
#'                        params = list(std_dev = TRUE))
#' summary(gp_model)
#' 
#' \donttest{
#' #--------------------Gaussian process model----------------
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        likelihood="gaussian", y = y, params = list(std_dev = TRUE))
#' summary(gp_model)
#' }
#' 
#' @method summary GPModel 
#' @rdname summary.GPModel
#' @author Fabio Sigrist
#' @export
summary.GPModel <- function(object, ...){
  cov_pars <- object$get_cov_pars()
  message("Covariance parameters:")
  print(signif(cov_pars,6))
  if (object$.__enclos_env__$private$has_covariates) {
    coef <- object$get_coef()
    message("Linear regression coefficients:")
    print(signif(coef,6))
  }
  if (object$.__enclos_env__$private$params$maxit == object$get_num_optim_iter()) {
    cat("\n")
    message("Note: no convergence after the maximal number of iterations")
  }
  return(invisible(object))
}

#' Make predictions for a \code{GPModel}
#'
#' Make predictions for a \code{GPModel}
#'
#' @param object a \code{GPModel}
#' @param y Observed data (can be NULL, e.g. when the model has been estimated already and the same data is used for making predictions)
#' @param group_data_pred A \code{vector} or \code{matrix} with labels of group levels for which predictions are made (if there are grouped random effects in the \code{GPModel})
#' @param group_rand_coef_data_pred A \code{vector} or \code{matrix} with covariate data for grouped random coefficients (if there are some in the \code{GPModel})
#' @param gp_coords_pred A \code{matrix} with prediction coordinates (features) for Gaussian process (if there is a GP in the \code{GPModel})
#' @param gp_rand_coef_data_pred A \code{vector} or \code{matrix} with covariate data for Gaussian process random coefficients (if there are some in the \code{GPModel})
#' @param cluster_ids_pred A \code{vector} with IDs / labels indicating the realizations of random effects / Gaussian processes for which predictions are made (set to NULL if you have not specified this when creating the \code{GPModel})
#' @param predict_cov_mat A \code{boolean}. If TRUE, the (posterior / conditional) predictive covariance is calculated in addition to the (posterior / conditional) predictive mean
#' @param predict_var A \code{boolean}. If TRUE, the (posterior / conditional) predictive variances are calculated
#' @param cov_pars A \code{vector} containing covariance parameters (used if the \code{GPModel} has not been trained or if predictions should be made for other parameters than the estimated ones)
#' @param X_pred A \code{matrix} with covariate data for the linear regression term (if there is one in the \code{GPModel})
#' @param use_saved_data A \code{boolean}. If TRUE, predictions are done using a priory set data via the function '$set_prediction_data'  (this option is not used by users directly)
#' @param predict_response A \code{boolean}. If TRUE, the response variable (label) is predicted, otherwise the latent random effects (this is only relevant for non-Gaussian data)
#' @param ... (not used, ignore this, simply here that there is no CRAN warning)
#' @inheritParams GPModel_shared_params 
#'
#' @return Predictions made using a \code{GPModel}. It returns a list of length three. 
#' The first entry ('mu') is the predicted mean, the second entry ('cov') is the predicted covariance matrix 
#' (=NULL if 'predict_cov_mat=FALSE'), and the third entry ('var') are predicted variances 
#' (=NULL if 'predict_var=FALSE')
#'
#' @examples
#' # See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
#' 
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, likelihood="gaussian",
#'                        params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_var = TRUE)
#' pred$mu # Predicted mean
#' pred$var # Predicted variances
#' # Also predict covariance matrix
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_cov_mat = TRUE)
#' pred$mu # Predicted mean
#' pred$cov # Predicted covariance
#' 
#' \donttest{
#' #--------------------Gaussian process model----------------
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                        likelihood="gaussian", y = y, params = list(std_dev = TRUE))
#' summary(gp_model)
#' # Make predictions
#' pred <- predict(gp_model, gp_coords_pred = coords_test, predict_cov_mat = TRUE)
#' # Predicted (posterior/conditional) mean of GP
#' pred$mu
#' # Predicted (posterior/conditional) covariance matrix of GP
#' pred$cov
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
                            vecchia_pred_type = NULL,
                            num_neighbors_pred = -1,
                            predict_response = FALSE,...){
  invisible(object$predict(y = y,
                           group_data_pred = group_data_pred,
                           group_rand_coef_data_pred = group_rand_coef_data_pred,
                           gp_coords_pred = gp_coords_pred,
                           gp_rand_coef_data_pred = gp_rand_coef_data_pred,
                           cluster_ids_pred = cluster_ids_pred,
                           predict_cov_mat = predict_cov_mat,
                           predict_var = predict_var,
                           cov_pars = cov_pars,
                           X_pred = X_pred,
                           use_saved_data = use_saved_data,
                           vecchia_pred_type = vecchia_pred_type,
                           num_neighbors_pred = num_neighbors_pred,
                           predict_response = predict_response,
                           ...))
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
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, likelihood="gaussian")
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_var = TRUE)
#' # Save model to file
#' filename <- tempfile(fileext = ".json")
#' saveGPModel(gp_model,filename = filename)
#' # Load from file and make predictions again
#' gp_model_loaded <- loadGPModel(filename = filename)
#' pred_loaded <- predict(gp_model_loaded, group_data_pred = group_data_test[,1], predict_var = TRUE)
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
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' 
#' gp_model <- fitGPModel(group_data = group_data[,1], y = y, likelihood="gaussian")
#' pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_var = TRUE)
#' # Save model to file
#' filename <- tempfile(fileext = ".json")
#' saveGPModel(gp_model,filename = filename)
#' # Load from file and make predictions again
#' gp_model_loaded <- loadGPModel(filename = filename)
#' pred_loaded <- predict(gp_model_loaded, group_data_pred = group_data_test[,1], predict_var = TRUE)
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

#' Generic 'set_prediction_data' method for a \code{GPModel}
#'
#' Generic 'set_prediction_data' method for a \code{GPModel}
#' 
#' @param gp_model A \code{GPModel}
#' @param group_data_pred A \code{vector} or \code{matrix} with labels of group levels for which predictions are made (if there are grouped random effects in the \code{GPModel})
#' @param group_rand_coef_data_pred A \code{vector} or \code{matrix} with covariate data for grouped random coefficients (if there are some in the \code{GPModel})
#' @param gp_coords_pred A \code{matrix} with prediction coordinates (features) for Gaussian process (if there is a GP in the \code{GPModel})
#' @param gp_rand_coef_data_pred A \code{vector} or \code{matrix} with covariate data for Gaussian process random coefficients (if there are some in the \code{GPModel})
#' @param cluster_ids_pred A \code{vector} with IDs / labels indicating the realizations of random effects / Gaussian processes for which predictions are made (set to NULL if you have not specified this when creating the \code{GPModel})
#' @param X_pred A \code{matrix} with covariate data for the linear regression term (if there is one in the \code{GPModel})
#'
#' @examples
#' \donttest{
#' library(gpboost)
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
#' @param group_data_pred A \code{vector} or \code{matrix} with labels of group levels for which predictions are made (if there are grouped random effects in the \code{GPModel})
#' @param group_rand_coef_data_pred A \code{vector} or \code{matrix} with covariate data for grouped random coefficients (if there are some in the \code{GPModel})
#' @param gp_coords_pred A \code{matrix} with prediction coordinates (features) for Gaussian process (if there is a GP in the \code{GPModel})
#' @param gp_rand_coef_data_pred A \code{vector} or \code{matrix} with covariate data for Gaussian process random coefficients (if there are some in the \code{GPModel})
#' @param cluster_ids_pred A \code{vector} with IDs / labels indicating the realizations of random effects / Gaussian processes for which predictions are made (set to NULL if you have not specified this when creating the \code{GPModel})
#' @param X_pred A \code{matrix} with covariate data for the linear regression term (if there is one in the \code{GPModel})
#'
#' @return A \code{GPModel}
#'
#' @examples
#' \donttest{
#' library(gpboost)
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
set_prediction_data.GPModel <- function(gp_model,
                                        group_data_pred = NULL,
                                        group_rand_coef_data_pred = NULL,
                                        gp_coords_pred = NULL,
                                        gp_rand_coef_data_pred = NULL,
                                        cluster_ids_pred = NULL,
                                        X_pred = NULL) {
  
  if (!gpb.check.r6.class(gp_model, "GPModel")) {
    stop("set_prediction_data.GPModel: gp_model needs to be a ", sQuote("GPModel"))
  }
  
  invisible(gp_model$set_prediction_data(group_data_pred = group_data_pred,
                                         group_rand_coef_data_pred = group_rand_coef_data_pred,
                                         gp_coords_pred = gp_coords_pred,
                                         gp_rand_coef_data_pred = gp_rand_coef_data_pred,
                                         cluster_ids_pred = cluster_ids_pred,
                                         X_pred = X_pred))
}


