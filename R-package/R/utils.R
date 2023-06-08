gpb.is.Booster <- function(x) {
  return(gpb.check.r6.class(object = x, name = "gpb.Booster"))
}

gpb.is.Dataset <- function(x) {
  return(gpb.check.r6.class(object = x, name = "gpb.Dataset"))
}

gpb.is.null.handle <- function(x) {
  if (is.null(x)) {
    return(TRUE)
  }
  return(
    isTRUE(.Call(LGBM_HandleIsNull_R, x))
  )
}

gpb.params2str <- function(params) {
  
  # Check for a list as input
  if (!identical(class(params), "list")) {
    stop("params must be a list")
  }
  
  # Split parameter names
  names(params) <- gsub("\\.", "_", names(params))
  
  # Setup temporary variable
  ret <- list()
  
  # Perform key value join
  for (key in names(params)) {
    
    # If a parameter has multiple values, join those values together with commas.
    # trimws() is necessary because format() will pad to make strings the same width
    val <- paste0(
      trimws(
        format(
          x = params[[key]]
          , scientific = FALSE
        )
      )
      , collapse = ","
    )
    if (nchar(val) <= 0L) next # Skip join
    
    # Join key value
    pair <- paste0(c(key, val), collapse = "=")
    ret <- c(ret, pair)
    
  }
  
  # Check ret length
  if (length(ret) == 0L) {
    return("")
  }
  
  return(paste0(ret, collapse = " "))
  
}

gpb.check_interaction_constraints <- function(params, column_names) {
  
  # Convert interaction constraints to feature numbers
  string_constraints <- list()
  
  if (!is.null(params[["interaction_constraints"]])) {
    
    if (!methods::is(params[["interaction_constraints"]], "list")) {
      stop("interaction_constraints must be a list")
    }
    if (!all(sapply(params[["interaction_constraints"]], function(x) {is.character(x) || is.numeric(x)}))) {
      stop("every element in interaction_constraints must be a character vector or numeric vector")
    }
    
    for (constraint in params[["interaction_constraints"]]) {
      
      # Check for character name
      if (is.character(constraint)) {
        
        constraint_indices <- as.integer(match(constraint, column_names) - 1L)
        
        # Provided indices, but some indices are not existing?
        if (sum(is.na(constraint_indices)) > 0L) {
          stop(
            "supplied an unknown feature in interaction_constraints "
            , sQuote(constraint[is.na(constraint_indices)])
          )
        }
        
      } else {
        
        # Check that constraint indices are at most number of features
        if (max(constraint) > length(column_names)) {
          stop(
            "supplied a too large value in interaction_constraints: "
            , max(constraint)
            , " but only "
            , length(column_names)
            , " features"
          )
        }
        
        # Store indices as [0, n-1] indexed instead of [1, n] indexed
        constraint_indices <- as.integer(constraint - 1L)
        
      }
      
      # Convert constraint to string
      constraint_string <- paste0("[", paste0(constraint_indices, collapse = ","), "]")
      string_constraints <- append(string_constraints, constraint_string)
    }
    
  }
  
  return(string_constraints)
  
}

gpb.c_str <- function(x) {
  
  ret <- charToRaw(as.character(x))
  ret <- c(ret, as.raw(0L))
  return(ret)
  
}

gpb.check.r6.class <- function(object, name) {
  
  # Check for non-existence of R6 class or named class
  return(all(c("R6", name) %in% class(object)))
  
}

gpb.check.obj <- function(params, obj) {
  
  # Check whether the objective is empty or not, and take it from params if needed
  if (!is.null(obj)) {
    params$objective <- obj
  }

  if (is.function(params$objective)) {
    
    stop("gpb.check.obj: GPBoost does currently not support custom object functions.")
    
  } else if (!is.null(params$objective)) {
    
    if (!is.character(params$objective)) {
      
      stop("gpb.check.obj: objective should be a character or a function")
      
    }
    
  }
  
  return(params)
  
}

# [description]
#     Take any character values from eval and store them in params$metric.
#     This has to account for the fact that `eval` could be a character vector,
#     a function, a list of functions, or a list with a mix of strings and
#     functions
gpb.check.eval <- function(params, eval) {
  
  if (is.null(params$metric)) {
    params$metric <- list()
  } else if (is.character(params$metric)) {
    params$metric <- as.list(params$metric)
  }
  
  # if 'eval' is a character vector or list, find the character
  # elements and add them to 'metric'
  if (!is.function(eval)) {
    for (i in seq_along(eval)) {
      element <- eval[[i]]
      if (is.character(element)) {
        params$metric <- append(params$metric, element)
      }
    }
  }
  
  # If more than one character metric was given, then "None" should
  # not be included
  if (length(params$metric) > 1L) {
    params$metric <- Filter(
      f = function(metric) {
        !(metric %in% .NO_METRIC_STRINGS())
      }
      , x = params$metric
    )
  }
  
  # duplicate metrics should be filtered out
  params$metric <- as.list(unique(unlist(params$metric)))
  
  return(params)
}


# [description]
#
#     Resolve differences between passed-in keyword arguments, parameters,
#     and parameter aliases. This function exists because some functions in the
#     package take in parameters through their own keyword arguments other than
#     the `params` list.
#
#     If the same underlying parameter is provided multiple
#     ways, the first item in this list is used:
#
#         1. the main (non-alias) parameter found in `params`
#         2. the first alias of that parameter found in `params`
#         3. the keyword argument passed in
#
#     For example, "num_iterations" can also be provided to gpb.train()
#     via keyword "nrounds". gpb.train() will choose one value for this parameter
#     based on the first match in this list:
#
#         1. params[["num_iterations]]
#         2. the first alias of "num_iterations" found in params
#         3. the nrounds keyword argument
#
#     If multiple aliases are found in `params` for the same parameter, they are
#     all removed before returning `params`.
#
# [return]
#     params with num_iterations set to the chosen value, and other aliases
#     of num_iterations removed
gpb.check.wrapper_param <- function(main_param_name, params, alternative_kwarg_value) {
  
  aliases <- .PARAMETER_ALIASES()[[main_param_name]]
  aliases_provided <- names(params)[names(params) %in% aliases]
  aliases_provided <- aliases_provided[aliases_provided != main_param_name]
  
  # prefer the main parameter
  if (!is.null(params[[main_param_name]])) {
    for (param in aliases_provided) {
      params[[param]] <- NULL
    }
    return(params)
  }
  
  # if the main parameter wasn't proovided, prefer the first alias
  if (length(aliases_provided) > 0L) {
    first_param <- aliases_provided[1L]
    params[[main_param_name]] <- params[[first_param]]
    for (param in aliases_provided) {
      params[[param]] <- NULL
    }
    return(params)
  }
  
  # if not provided in params at all, use the alternative value provided
  # through a keyword argument from gpb.train(), gpb.cv(), etc.
  params[[main_param_name]] <- alternative_kwarg_value
  return(params)
}
