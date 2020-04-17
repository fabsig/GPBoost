#' @importFrom R6 R6Class
gpb.GPModel <- R6::R6Class(
  classname = "gpb.GPModel",
  cloneable = FALSE,
  public = list(
    
    # Finalize will free up the handles
    finalize = function() {
      
      # Check the need for freeing handle
      if (!gpb.is.null.handle(private$handle)) {
        
        # Freeing up handle
        gpb.call("GPB_REModelFree_R", ret = NULL, private$handle)
        private$handle <- NULL
        
      }
      
    },
    
    # Initialize will create a gpb.GPModel
    initialize = function(group_data = NULL,
                          group_rand_coef_data = NULL,
                          ind_effect_group_rand_coef = NULL,
                          gp_coords = NULL,
                          gp_rand_coef_data = NULL,
                          cov_function = "exponential",
                          cov_fct_shape = 0,
                          vecchia_approx = FALSE,
                          num_neighbors = 30L,
                          vecchia_ordering = "none",
                          vecchia_pred_type = "order_obs_first_cond_obs_only",
                          num_neighbors_pred = num_neighbors,
                          cluster_ids = NULL,
                          free_raw_data = FALSE) {
      if (is.null(group_data) & is.null(gp_coords)) {
        stop("gpb.GPModel: Both ", sQuote("group_data"), " and " , sQuote("gp_coords"),
             " are NULL. Provide at least one of them.")
      }
      private$cov_par_names <- c("Error_term")
      # Set data for grouped random effects
      group_data_c_str <- NULL
      if (!is.null(group_data)) {
        
        # Check for correct format and set meta data
        if (!(is.data.frame(group_data) | is.matrix(group_data) | 
              is.numeric(group_data) | is.character(group_data) | is.factor(group_data))) {
          stop("gpb.GPModel: Can only use the following types for as ", sQuote("group_data"),": ",
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
        private$num_cov_pars <- private$num_cov_pars + private$num_group_re
        if (is.null(colnames(private$group_data))) {
          private$cov_par_names <- c(private$cov_par_names,paste0("Group_",1:private$num_group_re))
        } else {
          private$cov_par_names <- c(private$cov_par_names,colnames(private$group_data))
        }
        # Convert to correct format for passing to C
        group_data <- as.vector(group_data)
        if (storage.mode(group_data) != "character") {
          storage.mode(group_data) <- "character"
        }
        group_data_c_str <- c()
        for (i in 1:length(group_data)) {
          group_data_c_str <- c(group_data_c_str,gpb.c_str(group_data[i]))
        }
        
        # Set data for random coef for grouped REs
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
            
            stop("gpb.GPModel: Can only use ", sQuote("matrix"), " as ",sQuote("group_rand_coef_data"))
            
          }
          
          if (dim(group_rand_coef_data)[1] != private$num_data) {
            stop("gpb.GPModel: Number of data points in ", sQuote("group_rand_coef_data"), " does not match number of data points in ", sQuote("group_data"))
          }
          
          if (is.null(ind_effect_group_rand_coef)) {
            stop("gpb.GPModel: Indices of grouped random effects (", sQuote("ind_effect_group_rand_coef"), ") for random slopes in ", sQuote("group_rand_coef_data"), " not provided")
          }
          
          if (dim(group_rand_coef_data)[2] != length(ind_effect_group_rand_coef)) {
            stop("gpb.GPModel: Number of random coefficients in ", sQuote("group_rand_coef_data"), "does not match number in ", sQuote("ind_effect_group_rand_coef"))
          }
          
          if (storage.mode(ind_effect_group_rand_coef) != "integer") {
            storage.mode(ind_effect_group_rand_coef) <- "integer"
          }
          
          private$num_group_rand_coef <- as.integer(dim(group_rand_coef_data)[2])
          private$num_cov_pars <- private$num_cov_pars + private$num_group_rand_coef
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
          
          stop("gpb.GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("gp_coords"))
          
        }
        
        if (!is.null(private$num_data)) {
          
          if (dim(gp_coords)[1] != private$num_data) {
            stop("gpb.GPModel: Number of data points in ", sQuote("gp_coords"), " does not match number of data points in ", sQuote("group_data"))
          }
          
        } else {
          
          private$num_data <- as.integer(dim(gp_coords)[1])
          
        }
        
        if (!(vecchia_ordering %in% private$SUPPORTED_VECCHIA_ORDERING)) {
          stop("gpb.GPModel: ", sQuote("vecchia_ordering"), " needs to be: ",
               paste(sQuote(private$SUPPORTED_VECCHIA_ORDERING),collapse=", "))
        }
        
        if (!(vecchia_pred_type %in% private$VECCHIA_PRED_TYPES)) {
          stop("gpb.GPModel: ", sQuote("vecchia_pred_type"), " needs to be: ",
               paste(sQuote(private$VECCHIA_PRED_TYPES),collapse=", "))
        }
        
        if (!(cov_function %in% private$SUPPORTED_COV_FUNCTIONS)) {
          stop("gpb.GPModel: ", sQuote("cov_function"), " needs to be: ",
               paste(sQuote(private$SUPPORTED_COV_FUNCTIONS),collapse=", "))
        }
        
        if (cov_function == "powered_exponential") {
          if (cov_fct_shape <= 0 || cov_fct_shape > 2) {
            stop(sQuote("cov_fct_shape"), " needs to be larger than 0 and smaller or equal than 2 for ", sQuote("cov_function=powered_exponential"))
          }
        }
        
        if (cov_function == "matern") {
          if (!(cov_fct_shape == 0.5 || cov_fct_shape == 1.5 || cov_fct_shape == 2.5)) {
            stop(sQuote("cov_fct_shape"), " needs to be 0.5, 1.5, or 2.5 for ", sQuote("cov_function=matern"))
          }
        }
        
        private$num_gp <- 1L
        private$dim_coords <- as.integer(dim(gp_coords)[2])
        private$gp_coords <- gp_coords
        gp_coords <- as.vector(matrix(private$gp_coords)) #convert to correct format for sending to C
        private$num_cov_pars <- private$num_cov_pars + 2L
        private$cov_function <- cov_function
        cov_function <- gpb.c_str(cov_function)
        private$cov_fct_shape <- as.numeric(cov_fct_shape)
        private$vecchia_approx <- as.logical(vecchia_approx)
        private$num_neighbors <- as.integer(num_neighbors)
        private$vecchia_ordering <- vecchia_ordering
        vecchia_ordering <- gpb.c_str(vecchia_ordering)
        private$vecchia_pred_type <- vecchia_pred_type
        private$num_neighbors_pred <- as.integer(num_neighbors_pred)
        private$cov_par_names <- c(private$cov_par_names,"GP_var","GP_range")
        
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
            
            stop("gpb.GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("gp_rand_coef_data"))
            
            
          }
          
          if (dim(gp_rand_coef_data)[1] != private$num_data) {
            stop("gpb.GPModel: Number of data points in ", sQuote("gp_rand_coef_data"), " does not match number of data points")
          }
          
          private$num_gp_rand_coef <- as.integer(dim(gp_rand_coef_data)[2])
          private$gp_rand_coef_data <- gp_rand_coef_data
          gp_rand_coef_data <- as.vector(matrix(private$gp_rand_coef_data)) #convert to correct format for sending to C
          private$num_cov_pars <- private$num_cov_pars + 2L*private$num_gp_rand_coef
          for (ii in 1:private$num_gp_rand_coef) {
            if (is.null(colnames(private$gp_rand_coef_data))) {
              private$cov_par_names <- c(private$cov_par_names,paste0("GP_rand_coef_nb_", ii,"_var"),paste0("GP_rand_coef_nb_", ii,"_range"))
            } else {
              private$cov_par_names <- c(private$cov_par_names,
                                         paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_var"),
                                         paste0("GP_rand_coef_", colnames(private$gp_rand_coef_data)[ii],"_range"))
            }
          }
        }
        
      }
      
      storage.mode(private$num_cov_pars) <- "integer"
      
      # Set IDs for independent processes (cluster_ids)
      if (!is.null(cluster_ids)) {
        
        if (is.vector(cluster_ids)) {
          
          # Check whether matrix is the correct type first ("integer")
          if (storage.mode(cluster_ids) != "integer") {
            storage.mode(cluster_ids) <- "integer"
          }
          
        } else {
          
          stop("gpb.GPModel: Can only use ", sQuote("vector"), " as ", sQuote("cluster_ids"))
          
        }
        
        if (length(cluster_ids) != private$num_data) {
          stop("gpb.GPModel: Length of ", sQuote("cluster_ids"), "does not match number of data points")
        }
        
        private$cluster_ids = cluster_ids
        cluster_ids <- as.vector(cluster_ids)
        
      }
      
      vecchia_pred_type_c_str <- gpb.c_str(private$vecchia_pred_type)
      
      # Create handle
      handle <- 0.0
      
      # Attempts to create a handle for the gpb.GPModel
      try({
        # Store handle
        handle <- gpb.call("GPB_CreateREModel_R",
                           ret = handle,
                           private$num_data,
                           cluster_ids,
                           group_data_c_str,
                           private$num_group_re,
                           group_rand_coef_data,
                           ind_effect_group_rand_coef,
                           private$num_group_rand_coef,
                           private$num_gp,
                           gp_coords,
                           private$dim_coords,
                           gp_rand_coef_data,
                           private$num_gp_rand_coef,
                           cov_function,
                           private$cov_fct_shape,
                           private$vecchia_approx,
                           private$num_neighbors,
                           vecchia_ordering,
                           vecchia_pred_type_c_str,
                           private$num_neighbors_pred)
      })
      
      # Check whether the handle was created properly if it was not stopped earlier by a stop call
      if (gpb.is.null.handle(handle)) {
        
        stop("gpb.GPModel: Cannot create handle")
        
      } else {
        
        # Create class
        class(handle) <- "gpb.GPModel.handle"
        private$handle <- handle
        
      }
      
      private$free_raw_data <- free_raw_data
      # Should we free raw data?
      if (isTRUE(free_raw_data)) {
        private$group_data <- NULL
        private$group_rand_coef_data <- NULL
        private$gp_coords <- NULL
        private$gp_rand_coef_data <- NULL
        private$cluster_ids <- NULL
      }
      
    },
    
    # Find parameters that minimize the negative log-ligelihood (=MLE) using (Nesterov accelerated) gradient descent
    fit = function(y,
                   X = NULL,
                   std_dev = FALSE,
                   params = list(optimizer_cov = "fisher_scoring",
                                 optimizer_coef = "wls",
                                 maxit = 1000L,
                                 delta_rel_conv = 1E-6,
                                 init_coef = NULL,
                                 init_cov_pars = NULL,
                                 lr_coef = 0.01,
                                 lr_cov = 0.01,
                                 use_nesterov_acc = FALSE,
                                 acc_rate_coef = 0.1,
                                 acc_rate_cov = 0.5,
                                 nesterov_schedule_version = 0L,
                                 momentum_offset = 2L,
                                 trace = FALSE)) {
      
      if (gpb.is.null.handle(private$handle)) {
        stop("gpb.GPModel: Gaussian process model has not been initialized")
      }
      
      if (storage.mode(std_dev) != "logical") {
        stop("gpb.GPModel: Can only use ", sQuote("logical"), " as ", sQuote("std_dev"))
      }
      else {
        private$std_dev <- std_dev
      }
      
      if (private$num_cov_pars == 1L) {
        stop("gpb.GPModel.fit: No random effects (grouped, spatial, etc.) have been defined")
      }
      
      if (!is.vector(y)) {
        
        if (is.matrix(y)) {
          
          if (dim(y)[2] != 1) {
            
            stop("gpb.GPModel.fit: Can only use ", sQuote("vector"), " as ", sQuote("y"))
            
          }
          
        } else{
          
          stop("gpb.GPModel.fit: Can only use ", sQuote("vector"), " as ", sQuote("y"))
          
        }
        
      }
      
      if (storage.mode(y) != "double") {
        storage.mode(y) <- "double"
      }
      
      y <- as.vector(y)
      
      if (length(y) != private$num_data) {
        stop("gpb.GPModel.fit: Number of data points in ", sQuote("y"), " does not match number of data points of initialized model")
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
          
          stop("gpb.GPModel.fit: Can only use ", sQuote("matrix"), " as ", sQuote("X"))
          
        }
        
        
        if (dim(X)[1] != private$num_data) {
          stop("gpb.GPModel.fit: Number of data points in ", sQuote("X"), " does not match number of data points of initialized model")
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
        
        gpb.call("GPB_OptimCovPar_R",
                 ret = NULL,
                 private$handle,
                 y,
                 private$std_dev)
        
      } else {
        
        gpb.call("GPB_OptimLinRegrCoefCovPar_R",
                 ret = NULL,
                 private$handle,
                 y,
                 X,
                 private$num_coef,
                 private$std_dev)
        
      }
      
      message(paste0("gpb.GPModel: Number of iterations until convergence: ", self$get_num_optim_iter()))
      
    },
    
    # Evaluate the negative log-ligelihood
    neg_log_likelihood = function(cov_pars, y) {
      
      if (gpb.is.null.handle(private$handle)) {
        stop("gpb.GPModel: Gaussian process model has not been initialized")
      }

      if (private$num_cov_pars == 1L) {
        stop("gpb.GPModel.neg_log_likelihood: No random effects (grouped, spatial, etc.) have been defined")
      }
      
      if (is.vector(cov_pars)) {
        
        if (storage.mode(cov_pars) != "double") {
          storage.mode(cov_pars) <- "double"
        }
        
        cov_pars <- as.vector(cov_pars)
        
      } else {
        
        stop("gpb.GPModel.neg_log_likelihood: Can only use ", sQuote("vector"), " as ", sQuote("cov_pars"))
        
      }
      
      if (length(cov_pars) != private$num_cov_pars) {
        stop("gpb.GPModel.neg_log_likelihood: Number of parameters in ", sQuote("cov_pars"), " does not correspond to numbers of parameters")
      }
      
      if (!is.vector(y)) {
        
        if (is.matrix(y)) {
          
          if (dim(y)[2] != 1) {
            
            stop("gpb.GPModel.neg_log_likelihood: Can only use ", sQuote("vector"), " as ", sQuote("y"))
            
          }
          
        } else{
          
          stop("gpb.GPModel.neg_log_likelihood: Can only use ", sQuote("vector"), " as ", sQuote("y"))
          
        }
        
      }
      
      if (storage.mode(y) != "double") {
        storage.mode(y) <- "double"
      }
      
      y <- as.vector(y)
      
      if (length(y) != private$num_data) {
        stop("gpb.GPModel.neg_log_likelihood: Number of data points in ", sQuote("y"), " does not match number of data points of initialized model")
      }
      
      negll = 0.
      gpb.call("GPB_EvalNegLogLikelihood_R",
               ret = negll,
               private$handle,
               y,
               cov_pars)
      
      return(negll)
      
    },
    
    # Set configuration parameters for the optimizer
    set_optim_params = function(params = list(optimizer_cov = "fisher_scoring",
                                              init_cov_pars = NULL,
                                              lr_cov = 0.01,
                                              maxit = 1000L,
                                              delta_rel_conv = 1E-6,
                                              use_nesterov_acc = FALSE,
                                              acc_rate_cov = 0.5,
                                              nesterov_schedule_version = 0L,
                                              momentum_offset = 2L,
                                              trace = FALSE)) {
      if (gpb.is.null.handle(private$handle)) {
        stop("gpb.GPModel: Gaussian process model has not been initialized")
      }
      
      ##Check format of configuration parameters
      if (!is.null(params[["init_cov_pars"]])) {
        if (is.vector(params[["init_cov_pars"]])) {
          
          if (storage.mode(params[["init_cov_pars"]]) != "double") {
            storage.mode(params[["init_cov_pars"]]) <- "double"
          }
          
          params[["init_cov_pars"]] <- as.vector(params[["init_cov_pars"]])
          
        } else {
          
          stop("gpb.GPModel.fit: Can only use ", sQuote("vector"), " as ", sQuote("params$init_cov_pars"))
          
        }
        
        if (length(params[["init_cov_pars"]]) != private$num_cov_pars) {
          stop("gpb.GPModel.fit: Number of parameters in ", sQuote("params$init_cov_pars"), " does not correspond to numbers of parameters")
        }
        
      }
      
      if (!is.null(params[["use_nesterov_acc"]])) {
        if (storage.mode(params[["use_nesterov_acc"]]) != "logical") {
          stop("gpb.GPModel: Can only use ", sQuote("logical"), " as ", sQuote("params$use_nesterov_acc"))
        }
      }
      
      if (!is.null(params[["trace"]])) {
        if (storage.mode(params[["trace"]]) != "logical") {
          stop("gpb.GPModel: Can only use ", sQuote("logical"), " as ", sQuote("params$trace"))
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
          stop("gpb.GPModel: Can only use ", sQuote("character"), " as ", sQuote("optimizer_cov"))
        }
      }
      if (!is.null(params[["delta_rel_conv"]])) {
        params[["delta_rel_conv"]] <- as.numeric(params[["delta_rel_conv"]])
      }
      if (!is.null(params[["momentum_offset"]])) {
        params[["momentum_offset"]] <- as.integer(params[["momentum_offset"]])
      }
      
      for (param in names(params)) {
        if (param %in% names(private$params)) {
          if (is.null(params[[param]])) {
            private$params[param] <- list(NULL)
          }else{
            private$params[[param]] <- params[[param]]
          }
        }
        else {
          stop(paste0("gpb.GPModel: Unknown parameter: ", param))
        }
      }
      
      init_cov_pars <- private$params[["init_cov_pars"]]
      lr_cov <- private$params[["lr_cov"]]
      acc_rate_cov <- private$params[["acc_rate_cov"]]
      maxit <- private$params[["maxit"]]
      delta_rel_conv <- private$params[["delta_rel_conv"]]
      use_nesterov_acc <- private$params[["use_nesterov_acc"]]
      nesterov_schedule_version <- private$params[["nesterov_schedule_version"]]
      trace <- private$params[["trace"]]
      optimizer_cov <- gpb.c_str(private$params[["optimizer_cov"]])
      momentum_offset <- private$params[["momentum_offset"]]
      
      gpb.call("GPB_SetOptimConfig_R",
               ret = NULL,
               private$handle,
               init_cov_pars,
               lr_cov,
               acc_rate_cov,
               maxit,
               delta_rel_conv,
               use_nesterov_acc,
               nesterov_schedule_version,
               trace,
               optimizer_cov,
               momentum_offset)
      
      return(invisible(NULL))
      
    },
    
    get_optim_params = function() {
      return(private$params)
    },
    
    # Set configuration parameters for the optimizer
    set_optim_coef_params = function(params = list(optimizer_coef = "wls",
                                                   init_coef = NULL,
                                                   lr_coef = 0.001,
                                                   acc_rate_coef = 0.5)) {
      num_coef=NULL
      
      if (gpb.is.null.handle(private$handle)) {
        stop("gpb.GPModel: Gaussian process model has not been initialized")
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
          
          stop("gpb.GPModel.fit: Can only use ", sQuote("vector"), " as ", sQuote("init_coef"))
          
        }
        
        if (length(params[["init_coef"]]) != private$num_coef) {
          stop("gpb.GPModel.fit: Number of parameters in ", sQuote("init_coef"), " does not correspond to numbers of covariates in ", sQuote("X"))
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
          stop("gpb.GPModel: Can only use ", sQuote("character"), " as ", sQuote("optimizer_coef"))
        }
      }
      
      for (param in names(params)) {
        if (param %in% names(private$params)) {
          if (is.null(params[[param]])) {
            private$params[param] <- list(NULL)
          }else{
            private$params[[param]] <- params[[param]]
          }
        }
        else {
          stop(paste0("gpb.GPModel: Unknown parameter: ", param))
        }
      }
      
      init_coef <- private$params[["init_coef"]]
      lr_coef <- private$params[["lr_coef"]]
      acc_rate_coef <- private$params[["acc_rate_coef"]]
      optimizer_coef <- gpb.c_str(private$params[["optimizer_coef"]])
      
      gpb.call("GPB_SetOptimCoefConfig_R",
               ret = NULL,
               private$handle,
               num_coef,
               init_coef,
               lr_coef,
               acc_rate_coef,
               optimizer_coef)
      
      return(invisible(NULL))
      
    },
    
    get_cov_pars = function() {
      
      if (private$std_dev) {
        optim_pars <- numeric(2 * private$num_cov_pars)
      } else {
        optim_pars <- numeric(private$num_cov_pars)
      }
      
      optim_pars <- gpb.call("GPB_GetCovPar_R",
                             ret = optim_pars,
                             private$handle,
                             private$std_dev)
      
      cov_pars <- optim_pars[1:private$num_cov_pars]
      names(cov_pars) <- private$cov_par_names
      if (private$std_dev) {
        cov_pars_std_dev <- optim_pars[1:private$num_cov_pars+private$num_cov_pars]
        cov_pars <- rbind(cov_pars,cov_pars_std_dev)
        rownames(cov_pars) <- c("Param.", "Std. dev.")
      }
      
      return(cov_pars)
      
    },
    
    get_coef = function() {
      
      if (is.null(private$num_coef)) {
        stop("gpb.GPModel: ", sQuote("fit"), " has not been called")
      }
      
      if (private$std_dev) {
        optim_pars <- numeric(2 * private$num_coef)
      } else {
        optim_pars <- numeric(private$num_coef)
      }
      
      optim_pars <- gpb.call("GPB_GetCoef_R",
                             ret = optim_pars,
                             private$handle,
                             private$std_dev)
      
      coef <- optim_pars[1:private$num_coef]
      names(coef) <- private$coef_names
      if (private$std_dev) {
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
      
      num_data_pred <- NULL
      
      # Set data for grouped random effects
      if (private$num_group_re > 0) {
        
        if (is.null(group_data_pred)) {
          stop("gpb.GPModel.predict: ", sQuote("group_data_pred"), " not provided ")
        }
        
        # Check for correct format and set meta data
        if (!(is.data.frame(group_data_pred) | is.matrix(group_data_pred) |
              is.numeric(group_data_pred) | is.character(group_data_pred) |
              is.factor(group_data_pred))) {
          stop("gpb.GPModel.predict: Can only use the following types for as ", sQuote("group_data_pred"), ": ",
               sQuote("data.frame"), ", ", sQuote("matrix"), ", ", sQuote("character"),
               ", ", sQuote("numeric"), ", ", sQuote("factor"))
        }
        
        if (is.data.frame(group_data_pred) | is.numeric(group_data_pred) |
            is.character(group_data_pred) | is.factor(group_data_pred)) {
          group_data_pred <- as.matrix(group_data_pred)
        }
        
        if (dim(group_data_pred)[2] != private$num_group_re) {
          stop("gpb.GPModel.predict: Number of grouped random effects in ", sQuote("group_data_pred"), " is not correct")
        }
        
        num_data_pred <- as.integer(dim(group_data_pred)[1])
        group_data_pred <- as.vector(group_data_pred)
        
        if (storage.mode(group_data_pred) != "character") {
          storage.mode(group_data_pred) <- "character"
        }
        
        group_data_pred_c_str <- c()
        for (i in 1:length(group_data_pred)) {
          group_data_pred_c_str <- c(group_data_pred_c_str,gpb.c_str(group_data_pred[i]))
        }
        group_data_pred <- group_data_pred_c_str
        
        # Set data for random coef for grouped REs
        if (private$num_group_rand_coef > 0) {
          
          if (is.null(group_rand_coef_data_pred)) {
            stop("gpb.GPModel.predict: ", sQuote("group_rand_coef_data_pred"), " not provided ")
          }
          
          if (is.numeric(group_rand_coef_data_pred)) {
            group_rand_coef_data_pred <- as.matrix(group_rand_coef_data_pred)
          }
          
          if (is.matrix(group_rand_coef_data_pred)) {
            
            if (storage.mode(group_rand_coef_data_pred) != "double") {
              storage.mode(group_rand_coef_data_pred) <- "double"
            }
            
          } else {
            
            stop("gpb.GPModel.predict: Can only use ", sQuote("matrix"), " as group_rand_coef_data_pred")
            
          }
          
          if (dim(group_rand_coef_data_pred)[1] != num_data_pred) {
            stop("gpb.GPModel.predict: Number of data points in ", sQuote("group_rand_coef_data_pred"), " does not match number of data points in ", sQuote("group_data_pred"))
          }
          
          if (dim(group_rand_coef_data_pred)[2] != private$num_group_rand_coef) {
            stop("gpb.GPModel.predict: Number of random coef in ", sQuote("group_rand_coef_data_pred"), " is not correct")
          }
          
          group_rand_coef_data_pred <- as.vector(matrix(group_rand_coef_data_pred))
          
        }
        
      }
      
      # Set data for (spatial, spatio-temporal, or temporal) Gaussian process
      if (private$num_gp > 0) {
        
        if (is.null(gp_coords_pred)) {
          stop("gpb.GPModel.predict: ", sQuote("gp_coords_pred"), " not provided ")
        }
        
        if (is.numeric(gp_coords_pred)) {
          gp_coords_pred <- as.matrix(gp_coords_pred)
        }
        
        if (is.matrix(gp_coords_pred)) {
          
          # Check whether matrix is the correct type first ("double")
          if (storage.mode(gp_coords_pred) != "double") {
            storage.mode(gp_coords_pred) <- "double"
          }
          
        } else {
          
          stop("gpb.GPModel.predict: Can only use ", sQuote("matrix"), " as ", sQuote("gp_coords_pred"))
          
        }
        
        if (!is.null(num_data_pred)) {
          
          if (dim(gp_coords_pred)[1] != num_data_pred) {
            stop("gpb.GPModel.predict: Number of data points in ", sQuote("gp_coords_pred"), " does not match number of data points in ", sQuote("group_data_pred"))
          }
          
        } else {
          
          num_data_pred <- as.integer(dim(gp_coords_pred)[1])
          
        }
        
        if (dim(gp_coords_pred)[2] != private$dim_coords) {
          stop("gpb.GPModel.predict: Dimension / number of coordinates in ", sQuote("gp_coords_pred"), " is not correct")
        }
        
        gp_coords_pred <- as.vector(matrix(gp_coords_pred))
        
        # Set data for GP random coef
        if (private$num_gp_rand_coef > 0) {
          
          if (is.null(gp_rand_coef_data_pred)) {
            stop("gpb.GPModel.predict: ", sQuote("gp_rand_coef_data_pred"), " not provided ")
          }
          
          if (is.numeric(gp_rand_coef_data_pred)) {
            gp_rand_coef_data_pred <- as.matrix(gp_rand_coef_data_pred)
          }
          
          if (is.matrix(gp_rand_coef_data_pred)) {
            
            if (storage.mode(gp_rand_coef_data_pred) != "double") {
              storage.mode(gp_rand_coef_data_pred) <- "double"
            }
            
          } else {
            
            stop("gpb.GPModel.predict: Can only use ", sQuote("matrix"), " as ", sQuote("gp_rand_coef_data_pred"))
            
          }
          
          if (dim(gp_rand_coef_data_pred)[1] != num_data_pred) {
            stop("gpb.GPModel.predict: Number of data points in ", sQuote("gp_rand_coef_data_pred"), " does not match number of data points")
          }
          
          gp_rand_coef_data_pred <- as.vector(matrix(gp_rand_coef_data_pred))
          
        }
        
      }
      
      # Set data linear fixed-effects
      if (!is.null(X_pred)) {
        
        if (is.numeric(X_pred)) {
          X_pred <- as.matrix(X_pred)
        }
        
        if (is.matrix(X_pred)) {
          
          # Check whether matrix is the correct type first ("double")
          if (storage.mode(X_pred) != "double") {
            storage.mode(X_pred) <- "double"
          }
          
        } else {
          
          stop("gpb.GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("X_pred"))
          
        }
        
        if (dim(X_pred)[1] != num_data_pred) {
          stop("gpb.GPModel: Number of data points in ", sQuote("X_pred"), " is not correct")
        }
        
        if (dim(X_pred)[2] != private$num_coef) {
          stop("gpb.GPModel: Number of covariates in ", sQuote("X_pred"), " is not correct")
        }
        
        X_pred <- as.vector(matrix(X_pred))
        
      } else {
        
        if (private$has_covariates){
          stop("gpb.GPModel: Covariate data ", sQuote("X_pred"), " is not provided")
        }
        
      }
      
      # Set IDs for independent processes
      if (!is.null(cluster_ids_pred)) {
        
        if (is.vector(cluster_ids_pred)) {
          
          if (storage.mode(cluster_ids_pred) != "integer") {
            storage.mode(cluster_ids_pred) <- "integer"
          }
          
        } else {
          
          stop("gpb.GPModel.predict: Can only use ", sQuote("vector"), " as ", sQuote("cluster_ids_pred"))
          
        }
        
        if (length(cluster_ids_pred) != num_data_pred) {
          stop("gpb.GPModel.predict: Length of ", sQuote("cluster_ids_pred"), "does not match number of data points")
        }
        
        cluster_ids_pred <- as.vector(cluster_ids_pred)
        
      }
      
      private$num_data_pred <- num_data_pred
      
      gpb.call("GPB_SetPredictionData_R",
               ret=NULL,
               private$handle,
               num_data_pred,
               cluster_ids_pred,
               group_data_pred,
               group_rand_coef_data_pred,
               gp_coords_pred,
               gp_rand_coef_data_pred,
               X_pred)
      
      return(invisible(NULL))
      
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
                       cov_pars = NULL,
                       X_pred = NULL,
                       use_saved_data = FALSE) {
      
      if (!is.null(vecchia_pred_type)) {
        if (!(vecchia_pred_type %in% private$VECCHIA_PRED_TYPES)) {
          stop("gpb.GPModel: ", sQuote("vecchia_pred_type"), " needs to be either: ",
               paste(sQuote(private$VECCHIA_PRED_TYPES),collapse=", "))
        }
        private$vecchia_pred_type <- vecchia_pred_type
      }
      vecchia_pred_type_c_str <- gpb.c_str(private$vecchia_pred_type)
      
      if (!is.null(num_neighbors_pred)) {
        private$num_neighbors_pred <- as.integer(num_neighbors_pred)
      }
      
      if (gpb.is.null.handle(private$handle)) {
        stop("gpb.GPModel: Gaussian process model has not been initialized")
      }
      
      if (!is.null(y)) {
        
        if (!is.vector(y)) {
          
          if (is.matrix(y)) {
            
            if (dim(y)[2] != 1) {
              
              stop("gpb.GPModel.predict: Can only use ", sQuote("vector"), " as ", sQuote("y"))
              
            }
            
          } else {
            
            stop("gpb.GPModel.predict: Can only use ", sQuote("vector"), " as ", sQuote("y"))
            
          }
          
        }
        
        if (storage.mode(y) != "double") {
          storage.mode(y) <- "double"
        }
        
        y <- as.vector(y)
        
        if (length(y) != private$num_data) {
          stop("gpb.GPModel.predict: Number of data points in ", sQuote("y"), " does not match number of data points of initialized model")
        }
        
      }
      
      if (!is.null(cov_pars)) {
        
        if (is.vector(cov_pars)) {
          
          if (storage.mode(cov_pars) != "double") {
            storage.mode(cov_pars) <- "double"
          }
          
          cov_pars <- as.vector(cov_pars)
          
        } else {
          
          stop("gpb.GPModel.predict: Can only use ", sQuote("vector"), " as ", sQuote("cov_pars"))
          
        }
        
        if (length(cov_pars) != private$num_cov_pars) {
          stop("gpb.GPModel.predict: Number of parameters in ", sQuote("cov_pars"), " does not correspond to numbers of parameters of model")
        }
        
      }
      
      if (!use_saved_data) {
        
        num_data_pred <- NULL
        
        # Set data for grouped random effects
        if (private$num_group_re > 0) {
          
          if (is.null(group_data_pred)) {
            stop("gpb.GPModel.predict: ", sQuote("group_data_pred"), " not provided ")
          }
          
          # Check for correct format and set meta data
          if (!(is.data.frame(group_data_pred) | is.matrix(group_data_pred) |
                is.numeric(group_data_pred) | is.character(group_data_pred) |
                is.factor(group_data_pred))) {
            stop("gpb.GPModel.predict: Can only use the following types for as ", sQuote("group_data_pred"), ": ",
                 sQuote("data.frame"), ", ", sQuote("matrix"), ", ", sQuote("character"),
                 ", ", sQuote("numeric"), ", ", sQuote("factor"))
          }
          
          if (is.data.frame(group_data_pred) | is.numeric(group_data_pred) |
              is.character(group_data_pred) | is.factor(group_data_pred)) {
            group_data_pred <- as.matrix(group_data_pred)
          }
          
          if (dim(group_data_pred)[2] != private$num_group_re) {
            stop("gpb.GPModel.predict: Number of grouped random effects in ", sQuote("group_data_pred"), " is not correct")
          }
          
          num_data_pred <- as.integer(dim(group_data_pred)[1])
          group_data_pred <- as.vector(group_data_pred)
          
          if (storage.mode(group_data_pred) != "character") {
            storage.mode(group_data_pred) <- "character"
          }
          
          group_data_pred_c_str <- c()
          for (i in 1:length(group_data_pred)) {
            group_data_pred_c_str <- c(group_data_pred_c_str,gpb.c_str(group_data_pred[i]))
          }
          group_data_pred <- group_data_pred_c_str
          
          # Set data for random coef for grouped REs
          if (private$num_group_rand_coef > 0) {
            
            if (is.null(group_rand_coef_data_pred)) {
              stop("gpb.GPModel.predict: ", sQuote("group_rand_coef_data_pred"), " not provided ")
            }
            
            if (is.numeric(group_rand_coef_data_pred)) {
              group_rand_coef_data_pred <- as.matrix(group_rand_coef_data_pred)
            }
            
            if (is.matrix(group_rand_coef_data_pred)) {
              
              if (storage.mode(group_rand_coef_data_pred) != "double") {
                storage.mode(group_rand_coef_data_pred) <- "double"
              }
              
            } else {
              
              stop("gpb.GPModel.predict: Can only use ", sQuote("matrix"), " as group_rand_coef_data_pred")
              
            }
            
            if (dim(group_rand_coef_data_pred)[1] != num_data_pred) {
              stop("gpb.GPModel.predict: Number of data points in ", sQuote("group_rand_coef_data_pred"), " does not match number of data points in ", sQuote("group_data_pred"))
            }
            
            if (dim(group_rand_coef_data_pred)[2] != private$num_group_rand_coef) {
              stop("gpb.GPModel.predict: Number of covariates in ", sQuote("group_rand_coef_data_pred"), " is not correct")
            }
            
            group_rand_coef_data_pred <- as.vector(matrix(group_rand_coef_data_pred))
            
          }
          
        }
        
        # Set data for Gaussian process
        if (private$num_gp > 0) {
          
          if (is.null(gp_coords_pred)) {
            stop("gpb.GPModel.predict: ", sQuote("gp_coords_pred"), " not provided ")
          }
          
          if (is.numeric(gp_coords_pred)) {
            gp_coords_pred <- as.matrix(gp_coords_pred)
          }
          
          if (is.matrix(gp_coords_pred)) {
            
            # Check whether matrix is the correct type first ("double")
            if (storage.mode(gp_coords_pred) != "double") {
              storage.mode(gp_coords_pred) <- "double"
            }
            
          } else {
            
            stop("gpb.GPModel.predict: Can only use ", sQuote("matrix"), " as ", sQuote("gp_coords_pred"))
            
          }
          
          if (!is.null(num_data_pred)) {
            
            if (dim(gp_coords_pred)[1] != num_data_pred) {
              stop("gpb.GPModel.predict: Number of data points in ", sQuote("gp_coords_pred"), " does not match number of data points in ", sQuote("group_data_pred"))
            }
            
          } else {
            
            num_data_pred <- as.integer(dim(gp_coords_pred)[1])
            
          }
          
          if (dim(gp_coords_pred)[2] != private$dim_coords) {
            stop("gpb.GPModel.predict: Dimension / number of coordinates in ", sQuote("gp_coords_pred"), " is not correct")
          }
          
          gp_coords_pred <- as.vector(matrix(gp_coords_pred))
          
          # Set data for GP random coef
          if (private$num_gp_rand_coef > 0) {
            
            if (is.null(gp_rand_coef_data_pred)) {
              stop("gpb.GPModel.predict: ", sQuote("gp_rand_coef_data_pred"), " not provided ")
            }
            
            if (is.numeric(gp_rand_coef_data_pred)) {
              gp_rand_coef_data_pred <- as.matrix(gp_rand_coef_data_pred)
            }
            
            if (is.matrix(gp_rand_coef_data_pred)) {
              
              if (storage.mode(gp_rand_coef_data_pred) != "double") {
                storage.mode(gp_rand_coef_data_pred) <- "double"
              }
              
            } else {
              
              stop("gpb.GPModel.predict: Can only use ", sQuote("matrix"), " as ", sQuote("gp_rand_coef_data_pred"))
              
            }
            
            if (dim(gp_rand_coef_data_pred)[1] != num_data_pred) {
              stop("gpb.GPModel.predict: Number of data points in ", sQuote("gp_rand_coef_data_pred"), " does not match number of data points")
            }
            
            if (dim(gp_rand_coef_data_pred)[2] != private$num_gp_rand_coef) {
              stop("gpb.GPModel.predict: Number of covariates in ", sQuote("gp_rand_coef_data_pred"), " is not correct")
            }
            
            gp_rand_coef_data_pred <- as.vector(matrix(gp_rand_coef_data_pred))
            
          }
          
        }
        
        # Set data for linear fixed-effects
        if (private$has_covariates){
          
          if (is.null(X_pred)){
            stop("gpb.GPModel: Covariate data ", sQuote("X_pred"), " is not provided")
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
            
            stop("gpb.GPModel: Can only use ", sQuote("matrix"), " as ", sQuote("X_pred"))
            
          }
          
          if (dim(X_pred)[1] != num_data_pred) {
            stop("gpb.GPModel: Number of data points in ", sQuote("X_pred"), " is not correct")
          }
          
          if (dim(X_pred)[2] != private$num_coef) {
            stop("gpb.GPModel: Number of covariates in ", sQuote("X_pred"), " is not correct")
          }
          
          X_pred <- as.vector(matrix(X_pred))
          
        } 
        
        # Set IDs for independent processes
        if (!is.null(cluster_ids_pred)) {
          
          if (is.vector(cluster_ids_pred)) {
            
            if (storage.mode(cluster_ids_pred) != "integer") {
              storage.mode(cluster_ids_pred) <- "integer"
            }
            
          } else {
            
            stop("gpb.GPModel.predict: Can only use ", sQuote("vector"), " as ", sQuote("cluster_ids_pred"))
            
          }
          
          if (length(cluster_ids_pred) != num_data_pred) {
            stop("gpb.GPModel.predict: Length of ", sQuote("cluster_ids_pred"), "does not match number of data points")
          }
          
          cluster_ids_pred <- as.vector(cluster_ids_pred)
          
        }
        
      } else {
        
        cluster_ids_pred <- NULL
        group_data_pred <- NULL
        group_rand_coef_data_pred <- NULL
        gp_coords_pred <- NULL
        gp_rand_coef_data_pred <- NULL
        X_pred <- NULL
        num_data_pred <- private$num_data_pred
        
        if (is.null(private$num_data_pred)) {
          stop("gpb.GPModel.predict: No data has been set for making predictions. Call set_prediction_data first")
        }
        
      }
      
      if (storage.mode(predict_cov_mat) != "logical") {
        stop("gpb.GPModel.predict: Can only use ", sQuote("logical"), " as ", sQuote("predict_cov_mat"))
      }
      
      # Pre-allocate empty vector
      if (predict_cov_mat) {
        preds <- numeric(num_data_pred * (1 + num_data_pred))
      }
      else {
        preds <- numeric(num_data_pred)
      }
      
      preds <- gpb.call("GPB_PredictREModel_R",
                        ret=preds,
                        private$handle,
                        y,
                        num_data_pred,
                        predict_cov_mat,
                        cluster_ids_pred,
                        group_data_pred,
                        group_rand_coef_data_pred,
                        gp_coords_pred,
                        gp_rand_coef_data_pred,
                        cov_pars,
                        X_pred,
                        use_saved_data,
                        vecchia_pred_type_c_str,
                        private$num_neighbors_pred)
      
      pred_mean = preds[1:num_data_pred]
      
      if (predict_cov_mat) {
        pred_cov_mat = matrix(preds[1:(num_data_pred^2) + num_data_pred],ncol=num_data_pred)
      } else {
        pred_cov_mat = NULL
      }
      
      return(list(mu=pred_mean,cov=pred_cov_mat))
      
    },
    
    get_group_data = function() {
      if(isTRUE(private$free_raw_data)){
        stop("gpb.GPModel: cannot return ", sQuote("group_data"), ",
          please set ", sQuote("free_raw_data = FALSE"), " when you create gpb.GPModel")
      }
      return(private$group_data)
    },
    
    get_group_rand_coef_data = function() {
      if(isTRUE(private$free_raw_data)){
        stop("gpb.GPModel: cannot return ", sQuote("group_rand_coef_data"), ",
          please set ", sQuote("free_raw_data = FALSE"), " when you create gpb.GPModel")
      }
      return(private$group_rand_coef_data)
    },
    
    get_gp_coords = function() {
      if(isTRUE(private$free_raw_data)){
        stop("gpb.GPModel: cannot return ", sQuote("gp_coords"), ",
          please set ", sQuote("free_raw_data = FALSE"), " when you create gpb.GPModel")
      }
      return(private$gp_coords)
    },
    
    get_gp_rand_coef_data = function() {
      if(isTRUE(private$free_raw_data)){
        stop("gpb.GPModel: cannot return ", sQuote("gp_rand_coef_data"), ",
          please set ", sQuote("free_raw_data = FALSE"), " when you create gpb.GPModel")
      }
      return(private$gp_rand_coef_data)
    },
    
    get_cluster_ids = function() {
      if(isTRUE(private$free_raw_data)){
        stop("gpb.GPModel: cannot return ", sQuote("cluster_ids"), ",
             please set ", sQuote("free_raw_data = FALSE"), " when you create gpb.GPModel")
      }
      return(private$cluster_ids)
    },
    
    get_cov_function = function() {
      return(private$cov_function)
    },
    
    get_cov_fct_shape = function() {
      return(private$get_cov_fct_shape)
    },
    
    get_ind_effect_group_rand_coef = function() {
      return(private$ind_effect_group_rand_coef)
    },
    
    get_num_data = function() {
      return(private$num_data)
    },
    
    get_num_optim_iter = function() {
      num_it <- integer(1)
      num_it <- gpb.call("GPB_GetNumIt_R",
                         ret = num_it,
                         private$handle)
      return(num_it)
    }
    
  ),
  
  private = list(
    handle = NULL,
    num_data = NULL,
    num_group_re = 0L,
    num_group_rand_coef = 0L,
    num_cov_pars = 1L,
    num_gp = 0L,
    dim_coords = 2L,
    num_gp_rand_coef = 0L,
    has_covariates = FALSE,
    num_coef = NULL,
    std_dev = FALSE,
    group_data = NULL,
    group_rand_coef_data = NULL,
    ind_effect_group_rand_coef = NULL,
    gp_coords = NULL,
    gp_rand_coef_data = NULL,
    cov_function = "exponential",
    cov_fct_shape = 0.,
    vecchia_approx = FALSE,
    num_neighbors = 30L,
    vecchia_ordering = "none",
    vecchia_pred_type = "order_obs_first_cond_obs_only",
    num_neighbors_pred = 30L,
    cov_par_names = NULL,
    coef_names = NULL,
    cluster_ids = NULL,
    free_raw_data = FALSE,
    num_data_pred = NULL,
    params = list(optimizer_cov = "fisher_scoring",
                  optimizer_coef = "wls",
                  maxit = 1000L,
                  delta_rel_conv = 1E-6,
                  init_coef = NULL,
                  init_cov_pars = NULL,
                  lr_coef = 0.01,
                  lr_cov = 0.01,
                  use_nesterov_acc = FALSE,
                  acc_rate_coef = 0.1,
                  acc_rate_cov = 0.5,
                  nesterov_schedule_version = 0L,
                  momentum_offset = 2L,
                  trace = FALSE),
    SUPPORTED_COV_FUNCTIONS = c("exponential", "gaussian", "powered_exponential", "matern"),
    SUPPORTED_VECCHIA_ORDERING = c("none", "random"),
    VECCHIA_PRED_TYPES = c("order_obs_first_cond_obs_only",
                           "order_obs_first_cond_all", "order_pred_first",
                           "latent_order_obs_first_cond_obs_only","latent_order_obs_first_cond_all"),
    
    # Get handle
    get_handle = function() {
      
      if (gpb.is.null.handle(private$handle)) {
        stop("gpb.GPModel: Gaussian process model has not been initialized")
      }
      
      private$handle
      
    }
  )
)

#' Create a \code{gpb.GPModel} object
#'
#' Create a \code{gpb.GPModel} which contains a Gaussian process or mixed effects model with grouped random effects
#'
#' @inheritParams GPModel_shared_params 
#'
#' @return Gaussian process or mixed effects model
#'
#' @examples
#' ## SEE THE HELP OF 'fitGPModel' FOR MORE EXAMPLES
#' library(gpboost)
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' n <- 100 # number of samples
#' m <- 25 # number of categories / levels for grouping variable
#' group <- rep(1,n) # grouping variable
#' for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
#' # Create random effects model
#' gp_model <- GPModel(group_data = group)
#'
#' 
#' #--------------------Gaussian process model----------------
#' n <- 200 # number of samples
#' set.seed(1)
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
#'
#' 
#' #--------------------Combine Gaussian process with grouped random effects----------------
#' n <- 200 # number of samples
#' m <- 25 # number of categories / levels for grouping variable
#' group <- rep(1,n) # grouping variable
#' for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
#' set.seed(1)
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' # Create Gaussian process model
#' gp_model <- GPModel(group_data = group,
#'                     gp_coords = coords, cov_function = "exponential")
#'
#' @export
GPModel <- function(group_data = NULL,
                    group_rand_coef_data = NULL,
                    ind_effect_group_rand_coef = NULL,
                    gp_coords = NULL,
                    gp_rand_coef_data = NULL,
                    cov_function = "exponential",
                    cov_fct_shape = 0,
                    vecchia_approx = FALSE,
                    num_neighbors = 30L,
                    vecchia_ordering = "none",
                    vecchia_pred_type = "order_obs_first_cond_obs_only",
                    num_neighbors_pred = num_neighbors,
                    cluster_ids = NULL,
                    free_raw_data = FALSE) {
  
  # Create new dataset
  invisible(gpb.GPModel$new(group_data = group_data,
                            group_rand_coef_data = group_rand_coef_data,
                            ind_effect_group_rand_coef = ind_effect_group_rand_coef,
                            gp_coords = gp_coords,
                            gp_rand_coef_data = gp_rand_coef_data,
                            cov_function = cov_function,
                            cov_fct_shape = cov_fct_shape,
                            vecchia_approx = vecchia_approx,
                            num_neighbors = num_neighbors,
                            vecchia_ordering = vecchia_ordering,
                            vecchia_pred_type = vecchia_pred_type,
                            num_neighbors_pred = num_neighbors_pred,
                            cluster_ids = cluster_ids,
                            free_raw_data = free_raw_data))
  
}

#' Generic 'fit' method
#'
#' Generic 'fit' method
#' 
#' @param gp_model a \code{gpb.GPModel}
#' @inheritParams GPModel_shared_params
#' 
#' @export 
fit <- function(gp_model, y, X, std_dev = FALSE, params) UseMethod("fit")

#' Fits a \code{gpb.GPModel}
#'
#' Estimates the parameters of a \code{gpb.GPModel} using maximum likelihood estimation
#'
#' @param gp_model a \code{gpb.GPModel}
#' @inheritParams GPModel_shared_params
#'
#' @return A fitted gpb.GPModel
#'
#' @examples
#' ## SEE ALSO THE HELP OF 'fitGPModel' FOR MORE EXAMPLES
#' library(gpboost)
#' 
#' \dontrun{
#' #--------------------Grouped random effects model: single-level random effect----------------
#' n <- 100 # number of samples
#' m <- 25 # number of categories / levels for grouping variable
#' group <- rep(1,n) # grouping variable
#' for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
#' # Create random effects model
#' gp_model <- GPModel(group_data = group)
#'
#' # Simulate data
#' sigma2_1 <- 1^2 # random effect variance
#' sigma2 <- 0.5^2 # error variance
#'  # incidence matrix relating grouped random effects to samples
#' Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
#' set.seed(1)
#' b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
#' eps <- Z1 %*% b1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- eps + xi # observed data
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE)
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(group_data = group, y = y, std_dev = TRUE)
#' summary(gp_model)
#' 
#' # Make predictions
#' group_test <- 1:m
#' pred <- predict(gp_model, group_data_pred = group_test)
#' # Compare true and predicted random effects
#' plot(b1, pred$mu, xlab="truth", ylab="predicted",
#'      main="Comparison of true and predicted random effects")
#' abline(a=0,b=1)
#'
#' # Evaluate negative log-likelihood
#' gp_model$neg_log_likelihood(cov_pars=c(sigma2,sigma2_1),y=y)
#'  
#'  
#' #--------------------Gaussian process model----------------
#' n <- 200 # number of samples
#' set.seed(1)
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
#' ## Other covariance functions:
#' # gp_model <- GPModel(gp_coords = coords, cov_function = "gaussian")
#' # gp_model <- GPModel(gp_coords = coords,
#' #                     cov_function = "matern", cov_fct_shape=1.5)
#' # gp_model <- GPModel(gp_coords = coords,
#' #                     cov_function = "powered_exponential", cov_fct_shape=1.1)
#' # Simulate data
#' sigma2_1 <- 1^2 # marginal variance of GP
#' rho <- 0.1 # range parameter
#' sigma2 <- 0.5^2 # error variance
#' D <- as.matrix(dist(coords))
#' Sigma = sigma2_1*exp(-D/rho)+diag(1E-20,n)
#' C = t(chol(Sigma))
#' b_1=rnorm(n) # simulate random effect
#' eps <- C %*% b_1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- eps + xi
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE,
#'     params = list(optimizer_cov = "gradient_descent",
#'                   lr_cov = 0.1))
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                         y = y, std_dev = TRUE,
#'                         params = list(optimizer_cov = "gradient_descent",
#'                                       lr_cov = 0.1))
#' summary(gp_model)
#' 
#' # Make predictions
#' set.seed(1)
#' ntest <- 5
#' # prediction locations (=features) for Gaussian process
#' coords_test <- cbind(runif(ntest),runif(ntest))/10
#' pred <- predict(gp_model, gp_coords_pred = coords_test,
#'                 predict_cov_mat = TRUE)
#' print("Predicted (posterior/conditional) mean of GP")
#' pred$mu
#' print("Predicted (posterior/conditional) covariance matrix of GP")
#' pred$cov
#' 
#' # Evaluate negative log-likelihood
#' gp_model$neg_log_likelihood(cov_pars=c(sigma2,sigma2_1,rho),y=y)
#' }
#' 
#' @method fit gpb.GPModel 
#' @rdname fit.gpb.GPModel
#' @export
fit.gpb.GPModel <- function(gp_model,
                            y,
                            X = NULL,
                            std_dev = FALSE,
                            params = list(optimizer_cov = "fisher_scoring",
                                          optimizer_coef = "wls",
                                          maxit=1000,
                                          delta_rel_conv=1E-6,
                                          init_coef = NULL,
                                          init_cov_pars = NULL,
                                          lr_coef=0.01,
                                          lr_cov=0.01,
                                          use_nesterov_acc = FALSE,
                                          acc_rate_coef = 0.1,
                                          acc_rate_cov = 0.5,
                                          nesterov_schedule_version = 0L,
                                          momentum_offset = 2L,
                                          trace = FALSE)) {
  
  # Fit model
  invisible(gp_model$fit(y = y,
                         X = X,
                         std_dev = std_dev,
                         params = params))
  
}

#' Fits a \code{gpb.GPModel}
#'
#' Estimates the parameters of a \code{gpb.GPModel} using maximum likelihood estimation
#'
#' @inheritParams GPModel_shared_params 
#'
#' @return A fitted gpb.GPModel
#'
#' @examples
#' library(gpboost)
#' 
#' \dontrun{
#' #--------------------Grouped random effects model: single-level random effect----------------
#' n <- 100 # number of samples
#' m <- 25 # number of categories / levels for grouping variable
#' group <- rep(1,n) # grouping variable
#' for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
#' # Create random effects model
#' gp_model <- GPModel(group_data = group)
#' 
#' # Simulate data
#' sigma2_1 <- 1^2 # random effect variance
#' sigma2 <- 0.5^2 # error variance
#' # incidence matrix relating grouped random effects to samples
#' Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
#' set.seed(1)
#' b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
#' eps <- Z1 %*% b1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- eps + xi # observed data
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE)
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(group_data = group, y = y, std_dev = TRUE)
#' summary(gp_model)
#' 
#' # Make predictions
#' group_test <- 1:m
#' pred <- predict(gp_model, group_data_pred = group_test)
#' # Compare true and predicted random effects
#' plot(b1, pred$mu, xlab="truth", ylab="predicted",
#'      main="Comparison of true and predicted random effects")
#' abline(a=0,b=1)
#' 
#' # Evaluate negative log-likelihood
#' gp_model$neg_log_likelihood(cov_pars=c(sigma2,sigma2_1),y=y)
#' 
#' #--------------------Two crossed random effects and a random slope----------------
#' # NOTE: run the above example first to create the first random effect
#' set.seed(1)
#' x <- runif(n) # covariate data for random slope
#' n_obs_gr <- n/m # number of sampels per group
#' group2 <- rep(1,n) # grouping variable for second random effect
#' for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
#' # Create random effects model
#' gp_model <- GPModel(group_data = cbind(group,group2),
#'                     group_rand_coef_data = x,
#'                     ind_effect_group_rand_coef = 1)# the random slope is for the first random effect
#' 
#' # Simulate data
#' sigma2_2 <- 0.5^2 # variance of second random effect
#' sigma2_3 <- 0.75^2 # variance of random slope for first random effect
#' Z2 <- model.matrix(rep(1,n)~factor(group2)-1) # incidence matrix for second random effect
#' Z3 <- diag(x) %*% Z1 # incidence matrix for random slope for first random effect
#' b2 <- sqrt(sigma2_2) * rnorm(n_obs_gr) # second random effect
#' b3 <- sqrt(sigma2_3) * rnorm(m) # random slope for first random effect
#' eps2 <- Z1%*%b1 + Z2%*%b2 + Z3%*%b3 # sum of all random effects
#' y <- eps2 + xi # observed data
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE)
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(group_data = cbind(group,group2),
#'                         group_rand_coef_data = x,
#'                         ind_effect_group_rand_coef = 1,
#'                         y = y, std_dev = TRUE)
#' summary(gp_model)
#' 
#' 
#' #--------------------Mixed effects model: random effects and linear fixed effects----------------
#' # NOTE: run one of the above examples first to create the random effects part
#' set.seed(1)
#' X <- cbind(rep(1,n),runif(n)) # desing matrix / covariate data for fixed effect
#' beta <- c(3,3) # regression coefficents
#' y <- eps2 + xi + X%*%beta # add fixed effect to observed data
#' # Create random effects model
#' gp_model <- GPModel(group_data = cbind(group,group2),
#'                     group_rand_coef_data = x,
#'                     ind_effect_group_rand_coef = 1)
#' # Fit model
#' fit(gp_model, y = y, X = X, std_dev = TRUE)
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(group_data = cbind(group,group2),
#'                         group_rand_coef_data = x,
#'                         ind_effect_group_rand_coef = 1,
#'                         y = y, X = X, std_dev = TRUE)
#' summary(gp_model)
#' 
#' 
#' #--------------------Gaussian process model----------------
#' n <- 200 # number of samples
#' set.seed(1)
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
#' 
#' # Simulate data
#' sigma2_1 <- 1^2 # marginal variance of GP
#' rho <- 0.1 # range parameter
#' sigma2 <- 0.5^2 # error variance
#' D <- as.matrix(dist(coords))
#' Sigma = sigma2_1*exp(-D/rho)+diag(1E-20,n)
#' C = t(chol(Sigma))
#' b_1=rnorm(n) # simulate random effect
#' eps <- C %*% b_1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- eps + xi
#' 
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE,
#'     params = list(optimizer_cov = "gradient_descent",
#'                   lr_cov = 0.1))
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                         y = y, std_dev = TRUE,
#'                         params = list(optimizer_cov = "gradient_descent",
#'                                       lr_cov = 0.1))
#' summary(gp_model)
#' 
#' # Make predictions
#' set.seed(1)
#' ntest <- 5
#' # prediction locations (=features) for Gaussian process
#' coords_test <- cbind(runif(ntest),runif(ntest))/10
#' pred <- predict(gp_model, gp_coords_pred = coords_test,
#'                 predict_cov_mat = TRUE)
#' print("Predicted (posterior/conditional) mean of GP")
#' pred$mu
#' print("Predicted (posterior/conditional) covariance matrix of GP")
#' pred$cov
#' 
#' # Evaluate negative log-likelihood
#' gp_model$neg_log_likelihood(cov_pars=c(sigma2,sigma2_1,rho),y=y)
#' 
#' 
#' #--------------------Gaussian process model with Vecchia approximation----------------
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
#'                     vecchia_approx = TRUE, num_neighbors = 30)
#' # Fit model
#' fit(gp_model, y = y,
#'     params = list(optimizer_cov = "gradient_descent",
#'                   lr_cov = 0.1, maxit=100))
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                         vecchia_approx = TRUE, num_neighbors = 30,
#'                         y = y, params = list(optimizer_cov = "gradient_descent",
#'                                       lr_cov = 0.1, maxit=100))
#' summary(gp_model)
#' 
#' 
#' #--------------------Gaussian process model with random coefficents----------------
#' n <- 500 # number of samples
#' set.seed(1)
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' set.seed(1)
#' X_SVC=cbind(runif(n),runif(n)) # covariate data for random coeffient
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
#'                     gp_rand_coef_data = X_SVC)
#' ## Other covariance functions:
#' # gp_model <- GPModel(gp_coords = coords, cov_function = "gaussian")
#' # gp_model <- GPModel(gp_coords = coords,
#' #                     cov_function = "matern", cov_fct_shape=1.5)
#' # gp_model <- GPModel(gp_coords = coords,
#' #                     cov_function = "powered_exponential", cov_fct_shape=1.1)
#' # Simulate data
#' sigma2_1 <- 1^2 # marginal variance of GP (for simplicity, all GPs have the same parameters)
#' rho <- 0.1 # range parameter
#' sigma2 <- 0.5^2 # error variance
#' D <- as.matrix(dist(coords))
#' Sigma = sigma2_1*exp(-D/rho)+diag(1E-20,n)
#' C = t(chol(Sigma))
#' b_1=rnorm(n) # simulate random effect
#' b_2=rnorm(n)
#' b_3=rnorm(n)
#' eps <- C %*% b_1 + X_SVC[,1] * C %*% b_2 + X_SVC[,2] * C %*% b_3
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- eps + xi
#' 
#' # Fit model (takes a few seconds)
#' fit(gp_model, y = y, std_dev = TRUE,
#'     params = list(optimizer_cov = "gradient_descent",
#'                   lr_cov = 0.05, use_nesterov_acc = TRUE,
#'                   acc_rate_cov = 0.5))
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                         gp_rand_coef_data = X_SVC,
#'                         y = y, std_dev = TRUE,
#'                         params = list(optimizer_cov = "gradient_descent",
#'                                       lr_cov = 0.05,
#'                                       use_nesterov_acc = TRUE,
#'                                       acc_rate_cov = 0.5))
#' summary(gp_model)
#' 
#' 
#' #--------------------GP model with two independent observations of the GP----------------
#' n <- 200 # number of samples
#' set.seed(1)
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' coords <- rbind(coords,coords) # locations for second observation of GP (same locations)
#' # indices that indicate the GP sample to which an observations belong
#' cluster_ids <- c(rep(1,n),rep(2,n)) 
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
#'                     cluster_ids = cluster_ids)
#' # Simulate data
#' sigma2_1 <- 1^2 # marginal variance of GP
#' rho <- 0.1 # range parameter
#' sigma2 <- 0.5^2 # error variance
#' D <- as.matrix(dist(coords[1:n,]))
#' Sigma = sigma2_1*exp(-D/rho)+diag(1E-20,n)
#' C = t(chol(Sigma))
#' b_1=rnorm(2 * n) # simulate random effect
#' eps <- c(C %*% b_1[1:n], C %*% b_1[1:n + n])
#' xi <- sqrt(sigma2) * rnorm(2 * n) # simulate error term
#' y <- eps + xi
#'
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE,
#'     params = list(optimizer_cov = "gradient_descent",
#'                   lr_cov = 0.05))
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                         cluster_ids = cluster_ids,
#'                         y = y, std_dev = TRUE,
#'                         params = list(optimizer_cov = "gradient_descent",
#'                                       lr_cov = 0.05))
#' summary(gp_model)
#' 
#' 
#' #--------------------Combine Gaussian process with grouped random effects----------------
#' n <- 200 # number of samples
#' m <- 25 # number of categories / levels for grouping variable
#' group <- rep(1,n) # grouping variable
#' for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
#' set.seed(1)
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' # Create Gaussian process model
#' gp_model <- GPModel(group_data = group,
#'                     gp_coords = coords, cov_function = "exponential")
#' 
#' # Simulate data
#' sigma2_1 <- 1^2 # random effect variance
#' sigma2_2 <- 1^2 # marginal variance of GP
#' rho <- 0.1 # range parameter
#' sigma2 <- 0.5^2 # error variance
#' # incidence matrix relating grouped random effects to samples
#' Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1) 
#' set.seed(1)
#' b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
#' D <- as.matrix(dist(coords))
#' Sigma = sigma2_2*exp(-D/rho)+diag(1E-20,n)
#' C = t(chol(Sigma))
#' b_2=rnorm(n) # simulate random effect
#' eps <- Z1 %*% b1 + C %*% b_2
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- eps + xi
#'
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE,
#'     params = list(optimizer_cov = "gradient_descent",
#'                   lr_cov = 0.05, use_nesterov_acc = TRUE))
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(group_data = group,
#'                         gp_coords = coords, cov_function = "exponential",
#'                         y = y, std_dev = TRUE,
#'                         params = list(optimizer_cov = "gradient_descent",
#'                                       lr_cov = 0.05, use_nesterov_acc = TRUE))
#' summary(gp_model)
#' }
#' 
#' @rdname fitGPModel
#' @export fitGPModel
fitGPModel <- function(group_data = NULL,
                       group_rand_coef_data = NULL,
                       ind_effect_group_rand_coef = NULL,
                       gp_coords = NULL,
                       gp_rand_coef_data = NULL,
                       cov_function = "exponential",
                       cov_fct_shape = 0,
                       vecchia_approx = FALSE,
                       num_neighbors = 30L,
                       vecchia_ordering = "none",
                       vecchia_pred_type = "order_obs_first_cond_obs_only",
                       num_neighbors_pred = num_neighbors,
                       cluster_ids = NULL,
                       free_raw_data = FALSE,
                       y,
                       X = NULL,
                       std_dev = FALSE,
                       params = list(optimizer_cov = "fisher_scoring",
                                     optimizer_coef = "wls",
                                     maxit=1000,
                                     delta_rel_conv=1E-6,
                                     init_coef = NULL,
                                     init_cov_pars = NULL,
                                     lr_coef=0.01,
                                     lr_cov=0.01,
                                     use_nesterov_acc = FALSE,
                                     acc_rate_coef = 0.1,
                                     acc_rate_cov = 0.5,
                                     nesterov_schedule_version = 0L,
                                     momentum_offset = 2L,
                                     trace = FALSE)) {
  #Create model
  gpmodel <- gpb.GPModel$new(group_data = group_data,
                             group_rand_coef_data = group_rand_coef_data,
                             ind_effect_group_rand_coef = ind_effect_group_rand_coef,
                             gp_coords = gp_coords,
                             gp_rand_coef_data = gp_rand_coef_data,
                             cov_function = cov_function,
                             cov_fct_shape = cov_fct_shape,
                             vecchia_approx = vecchia_approx,
                             num_neighbors = num_neighbors,
                             vecchia_ordering = vecchia_ordering,
                             vecchia_pred_type = vecchia_pred_type,
                             num_neighbors_pred = num_neighbors_pred,
                             cluster_ids = cluster_ids,
                             free_raw_data = free_raw_data)
  # Fit model
  gpmodel$fit(y = y,
              X = X,
              std_dev = std_dev,
              params = params)
  return(gpmodel)
  
}

#' Summary for a \code{gpb.GPModel}
#'
#' Summary for a \code{gpb.GPModel}
#'
#' @param object a \code{gpb.GPModel}
#' @param ... (not used, ignore this, simply here that there is no CRAN warning)
#'
#' @return A summary of the (fitted) \code{gpb.GPModel}
#'
#' @examples
#' ## SEE ALSO THE HELP OF 'fitGPModel' FOR MORE EXAMPLES
#' library(gpboost)
#'
#' #--------------------Grouped random effects model: single-level random effect----------------
#' n <- 100 # number of samples
#' m <- 25 # number of categories / levels for grouping variable
#' group <- rep(1,n) # grouping variable
#' for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
#' # Create random effects model
#' gp_model <- GPModel(group_data = group)
#'
#' # Simulate data
#' sigma2_1 <- 1^2 # random effect variance
#' sigma2 <- 0.5^2 # error variance
#' # incidence matrix relating grouped random effects to samples
#' Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
#' set.seed(1)
#' b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
#' eps <- Z1 %*% b1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- eps + xi # observed data
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE)
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(group_data = group, y = y, std_dev = TRUE)
#' summary(gp_model)
#'
#' 
#' #--------------------Gaussian process model----------------
#' \dontrun{
#' n <- 200 # number of samples
#' set.seed(1)
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
#'
#' # Simulate data
#' sigma2_1 <- 1^2 # marginal variance of GP
#' rho <- 0.1 # range parameter
#' sigma2 <- 0.5^2 # error variance
#' D <- as.matrix(dist(coords))
#' Sigma = sigma2_1*exp(-D/rho)+diag(1E-20,n)
#' C = t(chol(Sigma))
#' b_1=rnorm(n) # simulate random effect
#' eps <- C %*% b_1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- eps + xi
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE,
#'     params = list(optimizer_cov = "gradient_descent",
#'                   lr_cov = 0.1))
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                         y = y, std_dev = TRUE,
#'                         params = list(optimizer_cov = "gradient_descent",
#'                                       lr_cov = 0.1))
#' summary(gp_model)
#' }
#' 
#' @method summary gpb.GPModel 
#' @rdname summary.gpb.GPModel
#' @export
summary.gpb.GPModel <- function(object, ...){
  cov_pars <- object$get_cov_pars()
  message("Covariance parameters:")
  print(signif(cov_pars,6))
  if (object$.__enclos_env__$private$has_covariates) {
    coef <- object$get_coef()
    cat("\n")
    message("Linear regression coefficients:")
    print(signif(coef,6))
  }
}

#' Make predictions for a \code{gpb.GPModel}
#'
#' Make predictions for a \code{gpb.GPModel}
#'
#' @param object a \code{gpb.GPModel}
#' @param y Observed data (can be NULL, e.g. when the model has been estimated already and the same data is used for making predictions)
#' @param group_data_pred A \code{vector} or \code{matrix} with labels of group levels for which predictions are made (if there are grouped random effects in the gpb.GPModel)
#' @param group_rand_coef_data_pred A \code{vector} or \code{matrix} with covariate data for grouped random coefficients (if there are some in the gpb.GPModel)
#' @param gp_coords_pred A \code{matrix} with prediction coordinates (features) for Gaussian process (if there is a GP in the gpb.GPModel)
#' @param gp_rand_coef_data_pred A \code{vector} or \code{matrix} with covariate data for Gaussian process random coefficients (if there are some in the gpb.GPModel)
#' @param cluster_ids_pred A \code{vector} with IDs / labels indicating the realizations of random effects / Gaussian processes for which predictions are made (set to NULL if you have not specified this when creating the gpb.GPModel)
#' @param predict_cov_mat A \code{boolean}. If TRUE, the (posterior / conditional) predictive covariance is calculated in addition to the (posterior / conditional) predictive mean
#' @param cov_pars A \code{vector} containing covariance parameters (used if the gpb.GPModel has not been trained or if predictions should be made for other parameters than the estimated ones)
#' @param X_pred A \code{matrix} with covariate data for the linear regression term (if there is one in the gpb.GPModel)
#' @param use_saved_data A \code{boolean}. If TRUE, predictions are done using priorly set data via the function '$set_prediction_data'  (this option is not used by users directly)
#' @param ... (not used, ignore this, simply here that there is no CRAN warning)
#' @inheritParams GPModel_shared_params 
#'
#' @return Predictions made using a \code{gpb.GPModel}. It returns a list of length two. The first entry is the predicted mean and the second entry is the predicted covariance matrix (=NULL if 'predict_cov_mat=FALSE')
#'
#' @examples
#' library(gpboost)
#' 
#' #--------------------Grouped random effects model: single-level random effect----------------
#' n <- 100 # number of samples
#' m <- 25 # number of categories / levels for grouping variable
#' group <- rep(1,n) # grouping variable
#' for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
#' # Create random effects model
#' gp_model <- GPModel(group_data = group)
#' 
#' # Simulate data
#' sigma2_1 <- 1^2 # random effect variance
#' sigma2 <- 0.5^2 # error variance
#' # incidence matrix relating grouped random effects to samples
#' Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
#' set.seed(1)
#' b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
#' eps <- Z1 %*% b1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- eps + xi # observed data
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE)
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(group_data = group, y = y, std_dev = TRUE)
#' summary(gp_model)
#' 
#' # Make predictions
#' group_test <- 1:m
#' pred <- predict(gp_model, group_data_pred = group_test)
#' # Compare true and predicted random effects
#' plot(b1, pred$mu, xlab="truth", ylab="predicted",
#'      main="Comparison of true and predicted random effects")
#' abline(a=0,b=1)
#' 
#' 
#' #--------------------Gaussian process model----------------
#' n <- 200 # number of samples
#' set.seed(1)
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
#'
#' # Simulate data
#' sigma2_1 <- 1^2 # marginal variance of GP
#' rho <- 0.1 # range parameter
#' sigma2 <- 0.5^2 # error variance
#' D <- as.matrix(dist(coords))
#' Sigma = sigma2_1*exp(-D/rho)+diag(1E-20,n)
#' C = t(chol(Sigma))
#' b_1=rnorm(n) # simulate random effect
#' eps <- C %*% b_1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- eps + xi
#' # Fit model
#' fit(gp_model, y = y, std_dev = TRUE,
#'     params = list(optimizer_cov = "gradient_descent",
#'                   lr_cov = 0.1))
#' summary(gp_model)
#' # Alternatively, define and fit model directly using fitGPModel
#' gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#'                         y = y, std_dev = TRUE,
#'                         params = list(optimizer_cov = "gradient_descent",
#'                                       lr_cov = 0.1))
#' summary(gp_model)
#' 
#' # Make predictions
#' set.seed(1)
#' ntest <- 5
#' # prediction locations (=features) for Gaussian process
#' coords_test <- cbind(runif(ntest),runif(ntest))/10
#' pred <- predict(gp_model, gp_coords_pred = coords_test,
#'                 predict_cov_mat = TRUE)
#' print("Predicted (posterior/conditional) mean of GP")
#' pred$mu
#' print("Predicted (posterior/conditional) covariance matrix of GP")
#' pred$cov
#' 
#' @rdname predict.gpb.GPModel
#' @export
predict.gpb.GPModel <- function(object,
                                y = NULL,
                                group_data_pred = NULL,
                                group_rand_coef_data_pred = NULL,
                                gp_coords_pred = NULL,
                                gp_rand_coef_data_pred = NULL,
                                cluster_ids_pred = NULL,
                                predict_cov_mat = FALSE,
                                cov_pars = NULL,
                                X_pred = NULL,
                                use_saved_data = FALSE,
                                vecchia_pred_type = NULL,
                                num_neighbors_pred = -1,
                                ...){
  invisible(object$predict(y = y,
                           group_data_pred = group_data_pred,
                           group_rand_coef_data_pred = group_rand_coef_data_pred,
                           gp_coords_pred = gp_coords_pred,
                           gp_rand_coef_data_pred = gp_rand_coef_data_pred,
                           cluster_ids_pred = cluster_ids_pred,
                           predict_cov_mat = predict_cov_mat,
                           cov_pars = cov_pars,
                           X_pred = X_pred,
                           use_saved_data = use_saved_data,
                           vecchia_pred_type = vecchia_pred_type,
                           num_neighbors_pred = num_neighbors_pred))
}
