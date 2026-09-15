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

#' @title Check whether the modified Bessel function of the second kind is available
#' @description Checks whether the GPBoost library has been compiled with support for
#'              \code{std::cyl_bessel_k}. This is a C++17 feature which is not provided by every
#'              standard library; in particular, libc++ (used by clang on macOS and in the clang
#'              sanitizer containers of R-hub and CRAN) does not provide it. Covariance functions
#'              with a general (i.e. not fixed to 0.5, 1.5, or 2.5) smoothness parameter require it.
#' @return A \code{logical} of length one: \code{TRUE} if \code{std::cyl_bessel_k} is available
#' @keywords internal
#' @noRd
has_std_cyl_bessel_k <- function() {
  has_bessel <- integer(1L)
  .Call(
    GPB_HasStdCylBesselK_R
    , has_bessel
  )
  return(has_bessel[1L] == 1L)
}

#' @title Check a number of threads that has been passed to one of the functions below
#' @description A number of threads is an integer. Rounding a number silently or letting it become
#'              \code{NA_integer_} because it does not fit into an \code{integer} would change which
#'              number of threads is used, and a non-positive number has a meaning of its own
#' @param num_threads The number of threads to check
#' @param name Name of the calling function, for the error message
#' @return The number of threads as an \code{integer} of length one
#' @keywords internal
#' @noRd
gpb.check.num.threads <- function(num_threads, name) {
  if (!is.numeric(num_threads) || length(num_threads) != 1L || !is.finite(num_threads) ||
      num_threads != floor(num_threads) || abs(num_threads) > .Machine$integer.max) {
    stop(name, ": num_threads needs to be an integer of length one")
  }
  return(as.integer(num_threads))
}

#' @title Get the number of threads used for parallelization
#' @description Returns the number of threads that OMP currently uses for parallelization, i.e.
#'              \code{omp_get_max_threads()}. Note that a team of threads can be smaller than this if
#'              the OpenMP runtime limits it, in particular through the limit of the contention group
#'              (\code{OMP_THREAD_LIMIT}).
#'              Note that models for which the number of threads has been specified via the
#'              \code{num_parallel_threads} argument of \code{\link{GPModel}} are not affected by
#'              this number: such models set (and reset) the number of threads themselves whenever
#'              they do calculations.
#' @return An \code{integer} of length one: the current maximum number of threads of OpenMP, as
#'         returned by \code{omp_get_max_threads()}
#' @author Fabio Sigrist
#' @examples
#' num_threads <- gpb.get.num.threads()
#' @rdname gpb.get.num.threads
#' @export
gpb.get.num.threads <- function() {
  num_threads <- integer(1L)
  .Call(
    GPB_GetNumParallelThreads_R
    , num_threads
  )
  return(num_threads[1L])
}

#' @title Set the number of threads used for parallelization
#' @description Sets the number of threads used by OMP and Eigen in the entire R process.
#'              Note that models for which the number of threads has been specified via the
#'              \code{num_parallel_threads} argument of \code{\link{GPModel}} are not affected by
#'              this: such models set (and reset) the number of threads themselves whenever they do
#'              calculations.
#' @param num_threads An \code{integer} specifying the number of threads. If \code{num_threads} is
#'                    not positive, the default number of threads of the session is used, see
#'                    \code{\link{gpb.get.default.num.threads}}. Initially, this is the automatically
#'                    selected number of threads (normally based on the number of physical performance
#'                    cores, subject to the settings of OpenMP and of the system, see the
#'                    \code{num_parallel_threads} argument of \code{\link{GPModel}}), and it is the
#'                    number of threads set by \code{\link{gpb.tune.num.threads}} or
#'                    \code{\link{gpb.set.default.num.threads}} once one of them has been called
#' @return This function does not return anything
#' @author Fabio Sigrist
#' @examples
#' num_threads_old <- gpb.get.num.threads()
#' gpb.set.num.threads(2L)
#' gpb.set.num.threads(num_threads_old)
#' @rdname gpb.set.num.threads
#' @export
gpb.set.num.threads <- function(num_threads) {
  num_threads <- gpb.check.num.threads(num_threads, "gpb.set.num.threads")
  .Call(
    GPB_SetNumParallelThreads_R
    , num_threads
  )
  return(invisible(NULL))
}

#' @title Get the number of threads used by models for which no number of threads is specified
#' @description Returns the number of threads that a \code{\link{GPModel}} uses when no number of
#'              threads has been specified for it via the \code{num_parallel_threads} argument. This
#'              is the number of threads that has been set for the session, either by
#'              \code{\link{gpb.tune.num.threads}} or by \code{\link{gpb.set.default.num.threads}},
#'              and the automatically selected number of threads (normally based on the number of
#'              physical performance cores, subject to the settings of OpenMP and of the system) if no
#'              such number has been set.
#'
#'              This is not the number of threads that OMP currently uses, see
#'              \code{\link{gpb.get.num.threads}}: a model for which no number of threads has been
#'              specified sets the number of threads returned here (and resets it again) whenever it
#'              does calculations, irrespective of the number of threads of the process.
#' @return An \code{integer} of length one: the number of threads used by models for which no number
#'         of threads is specified
#' @author Fabio Sigrist
#' @examples
#' num_threads <- gpb.get.default.num.threads()
#' @rdname gpb.get.default.num.threads
#' @export
gpb.get.default.num.threads <- function() {
  num_threads <- integer(1L)
  .Call(
    GPB_GetDefaultNumParallelThreads_R
    , num_threads
  )
  return(num_threads[1L])
}

#' @title Set the number of threads used by models for which no number of threads is specified
#' @description Sets the number of threads that a \code{\link{GPModel}} uses when no number of
#'              threads has been specified for it via the \code{num_parallel_threads} argument. Use
#'              this to apply a number of threads that has been determined by
#'              \code{\link{gpb.tune.num.threads}} in an earlier session, without running the
#'              benchmark again.
#'
#'              In contrast to \code{\link{gpb.set.num.threads}}, this does not change the number of
#'              threads that OMP currently uses: it only changes the number of threads that models use
#'              when nothing else is requested. The number of threads specified for an individual model
#'              always takes precedence.
#' @param num_threads An \code{integer} specifying the number of threads. It is limited by the largest
#'                    number of threads that GPBoost uses on its own: the number of threads that OMP
#'                    uses when GPBoost determines its default (usually the number of logical
#'                    processors, or the value of the environment variable \code{OMP_NUM_THREADS} if it
#'                    is set), the limit of the contention group of OpenMP (\code{OMP_THREAD_LIMIT}),
#'                    and, unless \code{OMP_NUM_THREADS} is set, a CPU bandwidth limit of a control
#'                    group on Linux. A number of threads that is specified for an individual model via
#'                    \code{num_parallel_threads} is not limited by this. If \code{num_threads} is not
#'                    positive, the automatically selected number of threads is used again
#' @return This function does not return anything
#' @author Fabio Sigrist
#' @examples
#' num_threads_old <- gpb.get.default.num.threads()
#' gpb.set.default.num.threads(2L)
#' gpb.set.default.num.threads(num_threads_old)
#' @rdname gpb.set.default.num.threads
#' @export
gpb.set.default.num.threads <- function(num_threads) {
  num_threads <- gpb.check.num.threads(num_threads, "gpb.set.default.num.threads")
  .Call(
    GPB_SetDefaultNumParallelThreads_R
    , num_threads
  )
  return(invisible(NULL))
}

#' @title Get the automatically selected number of threads
#' @description Returns the number of threads that GPBoost selects on its own, normally based on the
#'              number of physical performance cores and subject to the settings of OpenMP and of the
#'              system, irrespective of a number of threads that has been set for the session
#' @return An \code{integer} of length one
#' @keywords internal
#' @noRd
gpb.get.auto.num.threads <- function() {
  num_threads <- integer(1L)
  .Call(
    GPB_GetAutoNumParallelThreads_R
    , num_threads
  )
  return(num_threads[1L])
}

#' @title Get the largest number of threads that GPBoost uses on its own
#' @description Returns the number of threads that OMP uses when GPBoost determines its default, which
#'              is the upper limit for the number of threads that can be set for the session
#' @return An \code{integer} of length one
#' @keywords internal
#' @noRd
gpb.get.max.num.threads <- function() {
  num_threads <- integer(1L)
  .Call(
    GPB_GetMaxNumParallelThreads_R
    , num_threads
  )
  return(num_threads[1L])
}

#' @title Do not write the message about the automatically selected number of threads
#' @description The message is written once per process when a model uses the automatically selected
#'              number of threads for the first time. Tests call this function so that the message does
#'              not appear in the output of whichever test happens to run first
#' @return This function does not return anything
#' @keywords internal
#' @noRd
gpb.suppress.num.threads.message <- function() {
  .Call(
    GPB_SuppressAutoNumParallelThreadsMessage_R
  )
  return(invisible(NULL))
}
