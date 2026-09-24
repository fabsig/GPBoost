library(testthat)
library(gpboost)

# GPBOOST_TEST_FILTER restricts the run to the test files whose name, without the leading
# "test_" and without the extension, matches this regular expression. The sanitizer workflow
# uses it to spread the suite over several runs, because the instrumented code is too slow to
# run all of it in one job. When the variable is not set, as on CRAN, all files run.
test_filter <- Sys.getenv("GPBOOST_TEST_FILTER")
if (!nzchar(test_filter)) {
    test_filter <- NULL
}

test_check(
    package = "gpboost"
    , filter = test_filter
    , stop_on_failure = TRUE
    , stop_on_warning = FALSE
)
