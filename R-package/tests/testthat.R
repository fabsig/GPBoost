library(testthat)
library(gpboost)

test_check(
    package = "gpboost"
    , stop_on_failure = TRUE
    , stop_on_warning = FALSE
)
