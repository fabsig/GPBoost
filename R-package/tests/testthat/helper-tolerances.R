# Whether the expected values in the tests are compared with the strict tolerances.
#
# Those hold only on the reference platform on which they were calculated. With a
# different compiler or standard library, random number generation and OpenMP
# parallelisation do not reproduce bit-wise identical results, and some of the
# optimization problems are non-convex, so a run can converge to a different
# stationary point with practically the same likelihood. The test files then relax
# their tolerances through 'relax_tolerance*()' and 'TOLERANCE_NON_CONVEX'.
#
# Set GPBOOST_STRICT_TOLERANCES to decide explicitly, "true"/"false" (also "1"/"0"
# and "yes"/"no", in any case). When it is unset, Windows is taken to be the
# reference platform. Note that the value is read as a string, so use
# Sys.setenv(GPBOOST_STRICT_TOLERANCES = "false") and not = FALSE.
gpb_use_strict_tolerances <- function() {
  env <- Sys.getenv("GPBOOST_STRICT_TOLERANCES")
  if (nzchar(env)) {
    tolower(env) %in% c("true", "1", "yes")
  } else {
    .Platform$OS.type == "windows"
  }
}

# Relaxes a tolerance that only holds on the reference platform. The test files use it for the
# comparisons that a different compiler or standard library can move: a doubled tolerance covers a
# slightly different arithmetic, and the lower bound covers a different stationary point, which
# shifts an estimate outright.
#
# 'expected' are the values the tolerance is compared against. Pass them whenever the tolerance
# applies to an absolute difference, as in expect_lt(sum(abs(actual - expected)), ...): the lower
# bound is then half of their total magnitude at most, so that the relaxed comparison still rejects
# a value that is wrong by more than half of the quantity itself. Without them the bound is 0.5,
# which fits quantities of order one but accepts anything at all for a smaller one, such as a
# predictive variance or a standard error. For a relative tolerance, as in
# expect_equal(actual, expected, tolerance = ...), leave 'expected' unset: 0.5 is a relative bound
# of 50% there and does not depend on the magnitude.
relax_tolerance <- function(tol, expected = NULL) {
  if (gpb_use_strict_tolerances()) {
    return(tol)
  }
  lower_bound <- 0.5
  if (!is.null(expected)) {
    lower_bound <- min(lower_bound, 0.5 * sum(abs(expected), na.rm = TRUE))
  }
  max(2 * tol, lower_bound)
}

# Separate helper for absolute differences of negative log-likelihoods: these are on the scale of the
# log-likelihood itself (typically 100-1000 in these tests), so a larger absolute tolerance is still a
# small relative one
relax_tolerance_nll <- function(tol) {
  if (gpb_use_strict_tolerances()) tol else max(3 * tol, 3)
}

# Reported once per test run: testthat sources the helper files once, before the
# test files, for test_dir(), test_file() and test_check() alike
local({
  env <- Sys.getenv("GPBOOST_STRICT_TOLERANCES")
  message(sprintf(
    "[GPBoost tests] strict tolerances are %s  (GPBOOST_STRICT_TOLERANCES = %s)",
    if (gpb_use_strict_tolerances()) "ENABLED" else "DISABLED",
    if (nzchar(env)) sQuote(env) else sprintf("unset, defaulting to %s on this platform",
                                              .Platform$OS.type)))
})
