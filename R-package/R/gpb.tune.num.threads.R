#' @name GPB_THREAD_WORKLOAD_NAMES
#' @title Names of the benchmark workloads
#' @description The workloads of \code{\link{gpb.tune.num.threads}}. They cover the parallel
#'              computational kernels that dominate the different model classes of GPBoost: sparse
#'              operations for grouped random effects, Vecchia and Laplace calculations, and conjugate
#'              gradient and stochastic Lanczos quadrature calculations
#' @keywords internal
#' @noRd
GPB_THREAD_WORKLOAD_NAMES <- c("grouped_re", "vecchia_non_gaussian", "crossed_re_iterative")

#' @title Numbers of threads that are benchmarked
#' @description One thread, the automatically selected number of threads, half of it and the largest
#'              number of threads that GPBoost uses on its own are always benchmarked. Powers of two
#'              are added, the one in the largest gap first, until the benchmark would get too long
#' @param num_threads_auto The automatically selected number of threads
#' @param num_threads_max The largest number of threads
#' @param max_num_candidates Largest number of candidates
#' @return An \code{integer} vector in increasing order
#' @keywords internal
#' @noRd
gpb.thread.candidates <- function(num_threads_auto, num_threads_max, max_num_candidates = 7L) {
  num_threads_auto <- as.integer(num_threads_auto)
  num_threads_max <- as.integer(num_threads_max)
  candidates <- unique(c(1L, as.integer(ceiling(num_threads_auto / 2)), num_threads_auto,
                         num_threads_max))
  candidates <- sort(candidates[candidates >= 1L & candidates <= num_threads_max])
  remaining <- as.integer(2^(0:30))
  remaining <- sort(remaining[remaining >= 1L & remaining <= num_threads_max &
                                !(remaining %in% candidates)])
  while (length(candidates) < max_num_candidates && length(remaining) > 0L) {
    # The power of two that splits the largest gap between two candidates is added next, so that the
    # numbers of threads that are benchmarked are spread out as evenly as possible. Of several equally
    # large gaps, the one with the smallest numbers of threads is split first
    interval <- findInterval(remaining, candidates)
    gaps <- log(candidates[interval + 1L]) - log(candidates[interval])
    next_candidate <- which(gaps >= max(gaps) - 1e-9)[1L]
    candidates <- sort(c(candidates, remaining[next_candidate]))
    remaining <- remaining[-next_candidate]
  }
  return(as.integer(candidates))
}

#' @title Covariance parameters of a repetition of a benchmark workload
#' @description The covariance parameters are varied over the evaluations of a workload, so that every
#'              evaluation does the same work as an evaluation during an optimization. Reusing the same
#'              covariance parameters would allow quantities of a previous evaluation to be reused
#' @param cov_pars Covariance parameters of the workload
#' @param evaluation Number of the evaluation
#' @return A \code{numeric} vector of covariance parameters
#' @keywords internal
#' @noRd
gpb.thread.cov.pars <- function(cov_pars, evaluation) {
  return(cov_pars * (1 + 0.02 * ((evaluation - 1L) %% 5L)))
}

#' @title Data and model of a benchmark workload
#' @description Simulates the data of a workload and returns everything that is needed to time it. The
#'              data are simulated only once, and the models are created outside of the timed sections,
#'              so that neither the simulation nor the setup of a model (e.g. the search for the
#'              neighbors of the Vecchia approximation) is part of a measurement
#' @param workload Name of the workload, see \code{GPB_THREAD_WORKLOAD_NAMES}
#' @param workload_size Either "default" or "small"
#' @return A \code{list} with the response variable, the covariance parameters, the number of
#'         evaluations of a timed section, and a function that creates the model for a number of threads
#' @keywords internal
#' @noRd
gpb.thread.workload <- function(workload, workload_size = "default") {
  small <- workload_size == "small"
  if (workload == "grouped_re") {
    # A single grouped random effect with a Gaussian likelihood. The calculations are sparse and cheap
    # per data point, so that the overhead of the parallelization is visible: this workload is the one
    # that shows when a large number of threads does not pay off any more
    n <- if (small) 20000L else 1000000L
    num_groups <- as.integer(n / 100L)
    set.seed(1L)
    group_data <- rep(seq_len(num_groups), each = n / num_groups)
    y <- stats::rnorm(n) + rep(stats::rnorm(num_groups), each = n / num_groups)
    make_model <- function(num_threads) {
      return(GPModel(group_data = group_data, likelihood = "gaussian",
                     num_parallel_threads = num_threads))
    }
    return(list(name = workload, y = y, cov_pars = c(1, 1),
                num_evaluations = if (small) 2L else 40L, make_model = make_model))
  }
  if (workload == "vecchia_non_gaussian") {
    # A non-Gaussian likelihood with a Vecchia approximation and iterative methods. This exercises the
    # Vecchia calculations, the Laplace approximation and the conjugate gradient and stochastic Lanczos
    # quadrature calculations together
    n <- if (small) 1000L else 10000L
    set.seed(1L)
    gp_coords <- matrix(stats::runif(2 * n), ncol = 2L)
    # A smooth spatial signal, so that the mode finding converges as it does for data from the model
    signal <- sin(2 * pi * gp_coords[, 1L]) + cos(2 * pi * gp_coords[, 2L])
    y <- stats::rbinom(n, size = 1L, prob = 1 / (1 + exp(-signal)))
    make_model <- function(num_threads) {
      gp_model <- GPModel(gp_coords = gp_coords, cov_function = "exponential",
                          likelihood = "bernoulli_logit", gp_approx = "vecchia",
                          num_neighbors = 20L, matrix_inversion_method = "iterative",
                          num_parallel_threads = num_threads)
      gp_model$set_optim_params(params = list(cg_preconditioner_type = "vadu",
                                              num_rand_vec_trace = 20L,
                                              seed_rand_vec_trace = 1L,
                                              reuse_rand_vec_trace = TRUE))
      return(gp_model)
    }
    return(list(name = workload, y = y, cov_pars = c(1, 0.1),
                num_evaluations = 1L, make_model = make_model))
  }
  if (workload == "crossed_re_iterative") {
    # Two crossed grouped random effects with iterative methods. This exercises the products with the
    # sparse matrices of grouped random effects together with conjugate gradient and stochastic Lanczos
    # quadrature calculations, without the dense parts of a Gaussian process
    n <- if (small) 5000L else 200000L
    num_groups <- if (small) 100L else 1000L
    set.seed(1L)
    group_data <- cbind(sample.int(num_groups, n, replace = TRUE),
                        sample.int(num_groups, n, replace = TRUE))
    y <- stats::rnorm(n) + stats::rnorm(num_groups)[group_data[, 1L]] + stats::rnorm(num_groups)[group_data[, 2L]]
    make_model <- function(num_threads) {
      gp_model <- GPModel(group_data = group_data, likelihood = "gaussian",
                          matrix_inversion_method = "iterative",
                          num_parallel_threads = num_threads)
      gp_model$set_optim_params(params = list(cg_preconditioner_type = "ssor",
                                              num_rand_vec_trace = 20L,
                                              seed_rand_vec_trace = 1L,
                                              reuse_rand_vec_trace = TRUE))
      return(gp_model)
    }
    return(list(name = workload, y = y, cov_pars = c(1, 1, 1),
                num_evaluations = if (small) 1L else 5L, make_model = make_model))
  }
  stop("gpb.tune.num.threads: unknown workload ", sQuote(workload))
}

#' @title Select a number of threads from the measured runtimes
#' @description Aggregates the runtimes over the workloads with the geometric mean and selects the
#'              smallest number of threads whose aggregated runtime is within the tolerance of the best
#'              one. A number of threads that is much slower than the best one for a single workload is
#'              not selected, even if the geometric mean over the workloads looks acceptable
#' @param normalized A \code{matrix} with one row per workload and one column per number of threads:
#'                   the runtime relative to the fastest number of threads of the same workload
#' @param candidates An \code{integer} vector with the numbers of threads, in increasing order
#' @param tolerance Relative difference in runtime that is considered negligible
#' @param max_relative_slowdown Largest relative slowdown of a single workload that is accepted
#' @return A \code{list} with the selected number of threads, the aggregated runtimes, which numbers
#'         of threads are acceptable, and whether the safeguard had to be relaxed
#' @keywords internal
#' @noRd
gpb.thread.selection <- function(normalized, candidates, tolerance, max_relative_slowdown) {
  aggregate <- exp(colMeans(log(normalized), na.rm = TRUE))
  acceptable <- apply(normalized <= 1 + max_relative_slowdown, 2L,
                      function(column) all(column, na.rm = TRUE))
  # If no number of threads is fast enough for every workload, the workloads disagree about which one
  # is fast, e.g. because one of them scales and another one does not. The number of threads whose
  # slowest workload is the least slow is then the best compromise. 'acceptable' keeps saying which
  # numbers of threads satisfy the safeguard, which is none of them in that case
  safeguard_was_relaxed <- !any(acceptable)
  if (safeguard_was_relaxed) {
    worst <- apply(normalized, 2L, function(column) max(column, na.rm = TRUE))
    eligible <- worst <= min(worst) * (1 + 1e-9)
  } else {
    eligible <- acceptable
  }
  best <- min(aggregate[eligible])
  # The candidates are in increasing order, so the first one within the tolerance is the smallest one:
  # more threads are not used for a difference in runtime that is negligible
  within_tolerance <- eligible & (aggregate <= best * (1 + tolerance))
  return(list(
    num_threads = candidates[which(within_tolerance)[1L]]
    , aggregate = aggregate
    , acceptable = acceptable
    , safeguard_was_relaxed = safeguard_was_relaxed
    , best = best
  ))
}

#' @title Time one section of a benchmark workload
#' @param gp_model Model of the workload
#' @param workload Workload, see \code{gpb.thread.workload}
#' @return The time in seconds
#' @keywords internal
#' @noRd
gpb.thread.time.section <- function(gp_model, workload) {
  # The memory is cleaned up before the measurement and not during it, so that a garbage collection
  # does not make a single measurement much slower than the others
  gc(verbose = FALSE)
  start <- Sys.time()
  for (evaluation in seq_len(workload[["num_evaluations"]])) {
    invisible(gp_model$neg_log_likelihood(
      cov_pars = gpb.thread.cov.pars(workload[["cov_pars"]], evaluation)
      , y = workload[["y"]]
    ))
  }
  return(as.numeric(difftime(Sys.time(), start, units = "secs")))
}

#' @title Benchmark different numbers of threads and select a tuned default
#' @description Measures how long representative GPBoost calculations take with different numbers of
#'              OpenMP threads and selects a number of threads that is used by all models for which no
#'              number of threads is specified via the \code{num_parallel_threads} argument of
#'              \code{\link{GPModel}}, for the rest of the session.
#'
#'              The benchmark is never run automatically: the number of threads that GPBoost uses
#'              without it is the automatically selected one, which is normally based on the number of
#'              physical performance cores. The selected number
#'              of threads is a tuned default and not an optimal number of threads: the best number of
#'              threads depends on the model, on the size of the data and on the machine. Use the
#'              \code{num_parallel_threads} argument of \code{\link{GPModel}} for a model whose number
#'              of threads should differ from the default.
#'
#'              The measured times are the times of evaluations of the negative log-likelihood, which
#'              is the calculation that dominates the estimation of a model. The simulation of the data
#'              and the setup of the models (e.g. the search for the neighbors of the Vecchia
#'              approximation) happen outside of the timed sections.
#' @param workloads A \code{character} vector specifying the workloads that are benchmarked:
#'                  \itemize{
#'                    \item{"all" (= default): all workloads below}
#'                    \item{"grouped_re": a single-level grouped random effect model. The calculations
#'                    are cheap per data point, which makes the overhead of the parallelization visible}
#'                    \item{"vecchia_non_gaussian": a non-Gaussian Gaussian process model with a Vecchia
#'                    approximation and iterative methods}
#'                    \item{"crossed_re_iterative": crossed grouped random effects with iterative
#'                    methods}
#'                  }
#'                  Restrict the workloads to the model class that you mainly use if you know it
#' @param num_threads_candidates An \code{integer} vector with the numbers of threads that are
#'                               benchmarked. If \code{NULL}, powers of two, the automatically selected
#'                               number of threads and the largest number of threads that GPBoost uses
#'                               on its own are benchmarked
#' @param n_rep An \code{integer} specifying the number of repeated measurements per workload and
#'              number of threads. The median of the repetitions is used. Every workload is measured at
#'              least twice if \code{n_rep} is at least two, since a single measurement has no spread
#'              from which the noise could be estimated
#' @param tolerance A \code{numeric} specifying the relative difference in runtime that is considered
#'                  negligible. The smallest number of threads whose aggregated runtime is within this
#'                  tolerance of the best aggregated runtime is selected, so that more threads are not
#'                  used for a negligible gain. If the measurements are noisier than this, the observed
#'                  noise is used instead
#' @param max_time A \code{numeric} specifying approximately how many seconds the benchmark may take.
#'                 Repetitions are dropped if the measurements take longer. This is a soft limit: every
#'                 workload simulates its data, creates its models and is measured at least once, and
#'                 the remaining time is divided among the workloads that are still to be measured
#' @param max_relative_slowdown A \code{numeric} specifying how much slower than the fastest number of
#'                              threads a number of threads may be for a single workload and still be
#'                              selected. This prevents that a number of threads which is much slower
#'                              for one model class is selected because the aggregate over all
#'                              workloads looks acceptable
#' @param set_default A \code{logical}. If \code{TRUE}, the selected number of threads is set as the
#'                    default of the session, see \code{\link{gpb.set.default.num.threads}}. Set this to
#'                    \code{FALSE} to only measure
#' @param verbose A \code{logical}. If \code{TRUE}, the progress and the results are printed
#' @param workload_size A \code{string}, either "default" or "small". The small workloads run in a few
#'                      seconds, but they are too small for the number of threads to matter, so they
#'                      are only useful for checking that the benchmark runs and the default of the
#'                      session is never changed for them
#' @return A \code{list}, invisibly, with the following elements:
#'         \itemize{
#'           \item{num_threads: the selected number of threads}
#'           \item{num_threads_automatic: the automatically selected number of threads}
#'           \item{num_threads_max: the largest number of threads that GPBoost uses on its own}
#'           \item{default_was_set: whether the default of the session has been changed}
#'           \item{timings: a \code{data.frame} with the measurements per workload and number of
#'           threads (median, minimum and relative median absolute deviation of the repetitions), or
#'           \code{NULL} if only one number of threads is left to benchmark, in which case nothing is
#'           measured and that number of threads is the selected one}
#'           \item{aggregate: a \code{data.frame} with the aggregated relative runtime per number of
#'           threads, i.e. the geometric mean over the workloads of the runtime relative to the fastest
#'           measurement of the workload, and whether the number of threads is acceptable, i.e. whether
#'           it is not slower than \code{max_relative_slowdown} for a single workload. It is
#'           \code{NULL} whenever \code{timings} is}
#'           \item{num_threads_before: the default of the session before the benchmark}
#'           \item{tolerance_used: the tolerance that has been used for the selection}
#'           \item{safeguard_was_relaxed: \code{TRUE} if no number of threads was within
#'           \code{max_relative_slowdown} of the fastest one for every workload, i.e. if the workloads
#'           disagree. The number of threads whose slowest workload is the least slow is selected then}
#'         }
#' @author Fabio Sigrist
#' @examples
#' \donttest{
#' # Benchmark all workloads and use the result for the rest of the session
#' results <- gpb.tune.num.threads()
#'
#' # Benchmark only the model class that is mainly used
#' results <- gpb.tune.num.threads(workloads = "grouped_re")
#'
#' # Only measure, without changing the default
#' results <- gpb.tune.num.threads(set_default = FALSE)
#' }
#' @importFrom stats mad median rbinom rnorm runif
#' @rdname gpb.tune.num.threads
#' @export
gpb.tune.num.threads <- function(workloads = "all",
                                 num_threads_candidates = NULL,
                                 n_rep = 5L,
                                 tolerance = 0.03,
                                 max_time = 120,
                                 max_relative_slowdown = 0.25,
                                 set_default = TRUE,
                                 verbose = TRUE,
                                 workload_size = "default") {

  if (!is.character(workloads) || length(workloads) == 0L || anyNA(workloads)) {
    stop("gpb.tune.num.threads: ", sQuote("workloads"), " needs to be a character vector")
  }
  if (identical(workloads, "all")) {
    workloads <- GPB_THREAD_WORKLOAD_NAMES
  }
  unknown <- setdiff(workloads, GPB_THREAD_WORKLOAD_NAMES)
  if (length(unknown) > 0L) {
    stop("gpb.tune.num.threads: unknown workload(s) ", paste(sQuote(unknown), collapse = ", "),
         ". Possible workloads are ", paste(sQuote(GPB_THREAD_WORKLOAD_NAMES), collapse = ", "),
         " or ", sQuote("all"))
  }
  workloads <- unique(workloads)
  if (!is.numeric(n_rep) || length(n_rep) != 1L || !is.finite(n_rep) || n_rep < 1L ||
      n_rep != floor(n_rep) || n_rep > .Machine$integer.max) {
    stop("gpb.tune.num.threads: ", sQuote("n_rep"), " needs to be a positive integer of length one")
  }
  n_rep <- as.integer(n_rep)
  if (!is.numeric(tolerance) || length(tolerance) != 1L || !is.finite(tolerance) || tolerance < 0) {
    stop("gpb.tune.num.threads: ", sQuote("tolerance"), " needs to be a non-negative number")
  }
  if (!is.numeric(max_time) || length(max_time) != 1L || !is.finite(max_time) || max_time <= 0) {
    stop("gpb.tune.num.threads: ", sQuote("max_time"), " needs to be a positive number")
  }
  if (!is.numeric(max_relative_slowdown) || length(max_relative_slowdown) != 1L ||
      !is.finite(max_relative_slowdown) || max_relative_slowdown < 0) {
    stop("gpb.tune.num.threads: ", sQuote("max_relative_slowdown"),
         " needs to be a non-negative number")
  }
  if (!is.logical(set_default) || length(set_default) != 1L || is.na(set_default)) {
    stop("gpb.tune.num.threads: ", sQuote("set_default"), " needs to be TRUE or FALSE")
  }
  if (!is.logical(verbose) || length(verbose) != 1L || is.na(verbose)) {
    stop("gpb.tune.num.threads: ", sQuote("verbose"), " needs to be TRUE or FALSE")
  }
  if (!is.character(workload_size) || length(workload_size) != 1L ||
      !(workload_size %in% c("default", "small"))) {
    stop("gpb.tune.num.threads: ", sQuote("workload_size"), " needs to be ", sQuote("default"),
         " or ", sQuote("small"))
  }

  # The small workloads are so small that a single thread wins, which would make a single thread the
  # default of the session. They are only there to check that the benchmark runs
  if (workload_size == "small" && set_default) {
    set_default <- FALSE
    if (verbose) {
      cat(paste0("The small workloads are too small for the number of threads to matter, the default"
                 , " of the session is not changed.\n"))
    }
  }

  candidates <- NULL
  if (!is.null(num_threads_candidates)) {
    if (!is.numeric(num_threads_candidates) || any(!is.finite(num_threads_candidates)) ||
        any(num_threads_candidates < 1L) || any(num_threads_candidates != floor(num_threads_candidates)) ||
        any(num_threads_candidates > .Machine$integer.max)) {
      stop("gpb.tune.num.threads: ", sQuote("num_threads_candidates"),
           " needs to be a vector of positive integers")
    }
    candidates <- sort(unique(as.integer(num_threads_candidates)))
  }
  num_threads_auto <- gpb.get.auto.num.threads()
  num_threads_max <- gpb.get.max.num.threads()
  # The default that is active now: it is not necessarily the automatically selected one, an earlier
  # call of this function or of 'gpb.set.default.num.threads()' may have changed it
  num_threads_before <- gpb.get.default.num.threads()
  if (is.null(candidates)) {
    candidates <- gpb.thread.candidates(num_threads_auto, num_threads_max)
  } else {
    above_max <- candidates[candidates > num_threads_max]
    if (length(above_max) > 0L) {
      warning("gpb.tune.num.threads: ", paste(above_max, collapse = ", "),
              " threads are above the largest number of threads that GPBoost uses on its own (",
              num_threads_max, ") and are not benchmarked. That number comes from the OpenMP runtime",
              " (e.g. from ", sQuote("OMP_NUM_THREADS"), " or ", sQuote("OMP_THREAD_LIMIT"),
              ", which are read when OpenMP is initialized and thus have to be set before loading",
              " gpboost) or from a CPU limit of the machine")
      candidates <- candidates[candidates <= num_threads_max]
    }
  }
  if (length(candidates) == 0L) {
    stop("gpb.tune.num.threads: no number of threads left to benchmark")
  }
  if (length(candidates) == 1L) {
    # There is nothing to compare, but the single number of threads is the selected one and is applied
    # like the result of a benchmark
    num_threads_selected <- candidates[1L]
    if (verbose) {
      if (num_threads_max == 1L) {
        cat(paste0("Only one thread is available to the tuner under the current OpenMP and system",
                   " settings, there is nothing to benchmark.\n"))
      } else {
        cat(sprintf(paste0("Only %d thread(s) have been requested, there is nothing to benchmark.\n")
                    , num_threads_selected))
      }
    }
    default_was_set <- FALSE
    if (set_default) {
      gpb.set.default.num.threads(if (num_threads_selected == num_threads_auto) 0L
                                  else num_threads_selected)
      default_was_set <- gpb.get.default.num.threads() != num_threads_before
    }
    # The message that points to this function has done its job once this function has been called
    gpb.suppress.num.threads.message()
    return(invisible(list(
      num_threads = num_threads_selected
      , num_threads_automatic = num_threads_auto
      , num_threads_max = num_threads_max
      , num_threads_before = num_threads_before
      , default_was_set = default_was_set
      , timings = NULL
      , aggregate = NULL
      , tolerance_used = tolerance
      , safeguard_was_relaxed = FALSE
    )))
  }

  # The data of the workloads are simulated with a fixed seed, so that every number of threads gets
  # exactly the same data. The random numbers of the caller must not be affected by that
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    seed_before <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
    on.exit(assign(".Random.seed", seed_before, envir = globalenv()), add = TRUE)
  } else {
    on.exit(suppressWarnings(rm(".Random.seed", envir = globalenv())), add = TRUE)
  }

  # The number of threads of the process is not changed by the benchmark: every model gets its number
  # of threads via 'num_parallel_threads', which is set and reset again by every operation of the model
  if (verbose) {
    cat(sprintf("Benchmarking %d workload(s) with %s thread(s), %d repetition(s) each.\n",
                length(workloads), paste(candidates, collapse = ", "), n_rep))
    cat(sprintf(paste0("GPBoost selects %d thread(s) automatically, the largest number that GPBoost ",
                       "uses on its own is %d.\n"), num_threads_auto, num_threads_max))
  }

  start_time <- Sys.time()
  elapsed <- function() as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  times <- array(NA_real_, dim = c(length(workloads), length(candidates), n_rep))
  time_budget_reached <- FALSE

  reps_done <- integer(length(workloads))
  for (index_workload in seq_along(workloads)) {
    # Every workload gets the same share of the time that is left, and every workload is measured at
    # least once, which is why 'max_time' is a soft limit
    workload_start <- Sys.time()
    budget_workload <- (max_time - elapsed()) / (length(workloads) - index_workload + 1L)
    if (verbose) {
      cat(sprintf("  %s: simulating data ...", workloads[index_workload]))
    }
    workload <- gpb.thread.workload(workloads[index_workload], workload_size)
    # The models are created once per number of threads, so that the setup of a model is not measured,
    # and one section is timed and discarded per model: the first evaluation of the negative
    # log-likelihood does more work than the following ones, e.g. because the mode of the Laplace
    # approximation is initialized
    models <- lapply(candidates, workload[["make_model"]])
    for (index_candidate in seq_along(candidates)) {
      invisible(gpb.thread.time.section(models[[index_candidate]], workload))
    }
    if (verbose) {
      cat(" measuring ...")
    }
    # The protocol matters as much as what is timed: one model per number of threads that is kept
    # for all repetitions, one discarded warm-up per model, a rotating order and several repetitions.
    # Measured against short real fits of the two iterative workloads, this reproduces their ranking
    # (Spearman 1.00 and 0.99, same number of threads selected), while a variant with three
    # repetitions, a new model per measurement and no rotation produced a non-monotone curve and
    # selected 6 instead of 16 threads. Do not simplify this without measuring again
    reps_start <- Sys.time()
    for (rep in seq_len(n_rep)) {
      # The numbers of threads are measured in a rotating order, so that a drift of the speed of the
      # machine (e.g. because of the temperature of the CPU) affects all of them in the same way
      order_candidates <- ((seq_along(candidates) + rep - 2L) %% length(candidates)) + 1L
      for (index_candidate in order_candidates) {
        times[index_workload, index_candidate, rep] <-
          gpb.thread.time.section(models[[index_candidate]], workload)
      }
      reps_done[index_workload] <- rep
      time_per_rep <- as.numeric(difftime(Sys.time(), reps_start, units = "secs")) / rep
      time_workload <- as.numeric(difftime(Sys.time(), workload_start, units = "secs"))
      # The time budget does not stop the second repetition: the noise of a workload cannot be
      # estimated from a single measurement, and the number of threads is selected by comparing
      # differences in runtime with that noise
      if (rep >= 2L && rep < n_rep && time_workload + time_per_rep > budget_workload) {
        time_budget_reached <- TRUE
        break
      }
    }
    rm(models, workload)
    gc(verbose = FALSE)
    if (verbose) {
      cat(" done\n")
    }
  }

  # Median over the repetitions, and the relative median absolute deviation as a measure of the noise
  medians <- apply(times, c(1L, 2L), function(x) stats::median(x, na.rm = TRUE))
  minima <- apply(times, c(1L, 2L), function(x) min(x, na.rm = TRUE))
  relative_mad <- apply(times, c(1L, 2L), function(x) {
    x <- x[!is.na(x)]
    if (length(x) < 2L) {
      return(NA_real_)
    }
    return(stats::mad(x) / stats::median(x))
  })
  dim(medians) <- dim(minima) <- dim(relative_mad) <- c(length(workloads), length(candidates))

  # The runtimes are normalized per workload, so that every workload has the same weight irrespective
  # of how long it takes, and aggregated with the geometric mean. Note that the normalization does not
  # influence which number of threads is selected: it only makes the numbers easier to read
  normalized <- medians / apply(medians, 1L, min)
  # The tolerance is not smaller than the noise of the aggregated runtimes that are compared with it.
  # The relative median absolute deviation is the noise of a single measurement, the aggregated runtime
  # of a number of threads is a median over the repetitions and a mean over the workloads, so that its
  # noise is smaller by the square root of the number of measurements that it summarizes. A workload
  # with fewer repetitions than the others determines this number: it is the noisiest one
  num_measurements <- length(workloads) * max(min(reps_done), 1L)
  noise <- stats::median(relative_mad, na.rm = TRUE) / sqrt(num_measurements)
  # A single repetition has no spread. The noise is then unknown, which is not the same as zero, so it
  # is reported as unknown and only the tolerance that has been asked for is used
  noise_is_known <- is.finite(noise)
  if (!noise_is_known) {
    noise <- NA_real_
  }
  tolerance_used <- if (noise_is_known) max(tolerance, noise) else tolerance
  selection <- gpb.thread.selection(normalized, candidates, tolerance_used, max_relative_slowdown)
  aggregate <- selection[["aggregate"]]
  acceptable <- selection[["acceptable"]]
  safeguard_was_relaxed <- selection[["safeguard_was_relaxed"]]
  num_threads_selected <- selection[["num_threads"]]

  timings <- data.frame(
    workload = rep(workloads, times = length(candidates))
    , num_threads = rep(candidates, each = length(workloads))
    , median = as.vector(medians)
    , min = as.vector(minima)
    , relative_mad = as.vector(relative_mad)
    , relative_runtime = as.vector(normalized)
    , stringsAsFactors = FALSE
  )
  aggregate_table <- data.frame(
    num_threads = candidates
    , relative_runtime = as.vector(aggregate)
    , acceptable = as.vector(acceptable)
    , stringsAsFactors = FALSE
  )

  # The selected number of threads is made the one that models actually use. If it is the automatically
  # selected one, a default of an earlier call has to be removed, which is what a non-positive number
  # does: the default of the session is not necessarily the automatic one when this function is called
  default_was_set <- FALSE
  if (set_default) {
    gpb.set.default.num.threads(if (num_threads_selected == num_threads_auto) 0L
                                else num_threads_selected)
    default_was_set <- gpb.get.default.num.threads() != num_threads_before
  }
  # The message that points to this function has done its job once this function has been called
  gpb.suppress.num.threads.message()

  if (verbose) {
    cat(paste0("\nRuntime relative to the fastest number of threads of the same workload",
               " (smaller is better):\n"))
    print_table <- data.frame(num_threads = candidates, stringsAsFactors = FALSE)
    for (index_workload in seq_along(workloads)) {
      print_table[[workloads[index_workload]]] <- sprintf("%.3f", normalized[index_workload, ])
    }
    print_table[["aggregate"]] <- sprintf("%.3f", as.vector(aggregate))
    print(print_table, row.names = FALSE)
    if (time_budget_reached) {
      cat(sprintf("Fewer than %d repetitions have been measured: the time budget of %g seconds",
                  n_rep, max_time), "has been reached.\n")
    }
    if (noise_is_known) {
      cat(sprintf("Measurement noise: %.1f%%, tolerance used: %.1f%%\n",
                  100 * noise, 100 * tolerance_used))
    } else {
      cat(sprintf(paste0("Measurement noise: not available with one repetition, tolerance used: ",
                         "%.1f%%\n"), 100 * tolerance_used))
    }
    if (safeguard_was_relaxed) {
      cat(sprintf(paste0("No number of threads is within %.0f%% of the fastest one for every ",
                         "workload, the workloads disagree. The number of threads whose slowest ",
                         "workload is the least slow has been selected.\n"),
                  100 * max_relative_slowdown))
    }
    if (default_was_set) {
      cat(sprintf(paste0("GPBoost now uses %d thread(s) as the tuned default of this session ",
                         "(it used %d). This is not an optimal number of threads: the best number ",
                         "of threads depends on the model and on the data.\n"),
                  num_threads_selected, num_threads_before))
      cat(sprintf("Call gpb.set.default.num.threads(%d) to use it again in a later session.\n",
                  num_threads_selected))
    } else if (set_default) {
      cat(sprintf("GPBoost continues to use %d thread(s) by default.\n", num_threads_selected))
    } else {
      cat(sprintf("%d thread(s) would be used, the default has not been changed.\n",
                  num_threads_selected))
    }
  }

  return(invisible(list(
    num_threads = num_threads_selected
    , num_threads_automatic = num_threads_auto
    , num_threads_max = num_threads_max
    , num_threads_before = num_threads_before
    , default_was_set = default_was_set
    , timings = timings
    , aggregate = aggregate_table
    , tolerance_used = tolerance_used
    , safeguard_was_relaxed = safeguard_was_relaxed
  )))
}
