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
