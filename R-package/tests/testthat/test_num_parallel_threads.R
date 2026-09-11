context("num_parallel_threads")

# This test is fast and platform-dependent and is thus not restricted to 'GPBOOST_ALL_TESTS'
test_that("the default number of threads is the number of physical performance cores", {

  num_threads_omp <- gpb.get.num.threads()
  # The number of threads is a property of the entire process and has to be restored even if an
  # expectation below fails
  on.exit(gpb.set.num.threads(num_threads_omp), add = TRUE)
  # A non-positive number of threads resets to the default number of threads
  gpb.set.num.threads(-1L)
  num_threads_default <- gpb.get.num.threads()
  gpb.set.num.threads(num_threads_omp)
  expect_gte(num_threads_default, 1L)
  # The default never exceeds a number of threads that is restricted by, e.g., 'OMP_NUM_THREADS'
  expect_lte(num_threads_default, num_threads_omp)

  if (Sys.getenv("OMP_NUM_THREADS") != "") {
    # An explicitly requested number of threads is used as is
    expect_equal(num_threads_default, num_threads_omp)
  } else {
    # Hyperthreads are not counted separately
    # Note: a default of one thread is correct on a machine with several physical cores if only one of them
    # is available, e.g., with OMP_PLACES={0} or a CPU quota of one CPU, so it cannot be excluded here.
    # The test below covers that OpenMP thread binding does not reduce the default
    num_physical_cores <- tryCatch(parallel::detectCores(logical = FALSE), error = function(e) NA_integer_)
    if (!is.na(num_physical_cores) && num_physical_cores >= 1L) {
      expect_lte(num_threads_default, num_physical_cores)
    }
    # On macOS, performance level 0 is the one of the fastest cores. The corresponding number of cores is
    # only available on CPUs that have cores of different speeds (i.e., on Apple silicon)
    if (Sys.info()[["sysname"]] == "Darwin") {
      num_performance_cores <- tryCatch(
        as.integer(system2("sysctl", c("-n", "hw.perflevel0.physicalcpu"), stdout = TRUE, stderr = FALSE))
        , error = function(e) NA_integer_
        , warning = function(w) NA_integer_
      )
      if (!is.na(num_performance_cores) && num_performance_cores >= 1L) {
        expect_equal(num_threads_default, min(num_performance_cores, num_threads_omp))
      }
    }
  }

})

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

  test_that("'num_parallel_threads' of a model does not change the number of threads of other models ", {

    num_threads_before <- gpb.get.num.threads()
    expect_gte(num_threads_before, 1L)
    # This test changes the number of threads of the entire process, which has to be restored for the test
    # files that run afterwards, also if an expectation below fails
    on.exit(gpb.set.num.threads(num_threads_before), add = TRUE)

    n <- 100
    group <- rep(1:10, each = n / 10)
    y <- rep(c(-1, 1), n / 2) + 0.1 * (1:n)
    # A model that uses only one thread must not change the number of threads used afterwards
    capture.output( gp_model <- fitGPModel(group_data = group, y = y, params = list(maxit = 5),
                                           num_parallel_threads = 1L) , file = 'NUL')
    expect_equal(gpb.get.num.threads(), num_threads_before)
    # ... also not while the model still exists and is used
    capture.output( pred <- predict(gp_model, group_data_pred = group[1:5], predict_var = TRUE) , file = 'NUL')
    expect_equal(gpb.get.num.threads(), num_threads_before)
    # ... and a model created afterwards without 'num_parallel_threads' uses the default number of threads
    capture.output( gp_model2 <- fitGPModel(group_data = group, y = y, params = list(maxit = 5)) , file = 'NUL')
    expect_equal(gpb.get.num.threads(), num_threads_before)
    # Both models give the same results (the number of threads only affects the order of summation)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[1] - gp_model2$get_cov_pars(std_err = FALSE)[1]), 1E-6)

    # Setting the number of threads explicitly
    gpb.set.num.threads(2L)
    expect_equal(gpb.get.num.threads(), 2L)
    # A model with its own number of threads does not change this
    capture.output( gp_model3 <- fitGPModel(group_data = group, y = y, params = list(maxit = 5),
                                            num_parallel_threads = 1L) , file = 'NUL')
    expect_equal(gpb.get.num.threads(), 2L)
    # Non-positive values reset to the default number of threads. This is the number of physical performance
    # cores, which can be smaller than the number of threads that OMP uses when the package is loaded
    gpb.set.num.threads(-1L)
    num_threads_default <- gpb.get.num.threads()
    expect_gte(num_threads_default, 1L)
    expect_lte(num_threads_default, num_threads_before)
    gpb.set.num.threads(2L)
    gpb.set.num.threads(-1L)
    expect_equal(gpb.get.num.threads(), num_threads_default)
    expect_error(gpb.set.num.threads("two"), "num_threads needs to be an integer of length one", fixed = TRUE)

  })

}

# The default number of threads is determined only once per process, so everything that can influence it has
# to be checked in a new process. These tests are small but cover two ways in which the default silently
# became a single thread, so they are not restricted to 'GPBOOST_ALL_TESTS'
#
# Note: the environment variables that OpenMP reads are removed for the subprocesses ('env' only adds
# variables). Otherwise a variable of the calling process, e.g. the OMP_NUM_THREADS = 10 that
# 'helpers/run_tests_coverage_R_package.R' sets, would be inherited, and since an explicitly requested
# number of threads is used as is, the code under test would never run
.gpb_num_threads_in_new_process <- function(commands, env = character(0)) {
  rscript <- file.path(R.home("bin"), "Rscript")
  if (!file.exists(rscript)) {
    return(NA_integer_)
  }
  script <- paste0(".libPaths(", paste0(deparse(.libPaths()), collapse = ""), "); "
                   , "suppressMessages(library(gpboost)); ", commands, "cat(gpb.get.num.threads())")
  omp_variables <- c("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_THREAD_LIMIT")
  values_before <- Sys.getenv(omp_variables, names = TRUE, unset = NA_character_)
  Sys.unsetenv(omp_variables)
  on.exit({
    values_set_before <- values_before[!is.na(values_before)]
    if (length(values_set_before) > 0L) {
      do.call(Sys.setenv, as.list(values_set_before))
    }
  }, add = TRUE)
  output <- suppressWarnings(
    tryCatch(system2(rscript, c("-e", shQuote(script)), stdout = TRUE, stderr = FALSE, env = env)
             , error = function(e) NA_character_)
  )
  suppressWarnings(as.integer(utils::tail(output, 1L)))
}

test_that("setting one thread does not make one thread the default of a new session", {

  num_threads_default <- .gpb_num_threads_in_new_process("gpb.set.num.threads(-1L); ")
  if (is.na(num_threads_default) || num_threads_default < 2L) {
    skip("a machine with at least two default threads is needed to tell the two cases apart")
  }
  # The default is determined once and must not be derived from a number of threads that the library has
  # set before, otherwise a single thread stays the default for the rest of the session
  expect_equal(
    .gpb_num_threads_in_new_process("gpb.set.num.threads(1L); gpb.set.num.threads(-1L); ")
    , num_threads_default
  )

})

test_that("a boosting call with one thread does not make one thread the default of a new session", {

  num_threads_default <- .gpb_num_threads_in_new_process("gpb.set.num.threads(-1L); ")
  if (is.na(num_threads_default) || num_threads_default < 2L) {
    skip("a machine with at least two default threads is needed to tell the two cases apart")
  }
  # The boosting part of the library sets the number of threads of the process as well, so a boosting call
  # with 'num_threads = 1' must not turn a single thread into the default of the session either
  expect_equal(
    .gpb_num_threads_in_new_process(paste0(
      "X <- matrix(runif(100), ncol = 2); "
      , "dataset <- gpb.Dataset(X, params = list(num_threads = 1L, verbose = -1L)); "
      , "invisible(capture.output(dataset$construct())); gpb.set.num.threads(-1L); "
    ))
    , num_threads_default
  )

})

test_that("OpenMP thread binding does not reduce the default number of threads", {

  # OpenMP binds the calling thread, whose processor affinity is then no longer the set of CPUs that the
  # threads of the process may use. Only Linux binds threads through these variables
  if (Sys.info()[["sysname"]] != "Linux") {
    skip("OpenMP thread binding is only tested on Linux")
  }
  num_threads_default <- .gpb_num_threads_in_new_process("gpb.set.num.threads(-1L); ")
  if (is.na(num_threads_default) || num_threads_default < 2L) {
    skip("a machine with at least two default threads is needed to tell the two cases apart")
  }
  expect_equal(
    .gpb_num_threads_in_new_process("gpb.set.num.threads(-1L); ", env = "OMP_PROC_BIND=true")
    , num_threads_default
  )
  expect_equal(
    .gpb_num_threads_in_new_process("gpb.set.num.threads(-1L); "
                                    , env = c("OMP_PLACES=cores", "OMP_PROC_BIND=spread"))
    , num_threads_default
  )

})
