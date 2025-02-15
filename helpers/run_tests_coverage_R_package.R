# Set working directory to GPBoost main folder

# Run unit tests locally
library(testthat)
library(gpboost)
Sys.setenv(GPBOOST_ALL_TESTS = "GPBOOST_ALL_TESTS")
# Sys.setenv(GPBOOST_ADDITIONAL_SLOW_TESTS = "GPBOOST_ADDITIONAL_SLOW_TESTS")
# Sys.setenv(GPBOOST_ALL_TESTS = "NO_GPBOOST_ALGO_TESTS") # If this is set, the (slow) GPBoost algorithm tests are not run
path_tests = paste0(getwd(),.Platform$file.sep,file.path("R-package","tests","testthat"))

system.time({ test_dir(path_tests, reporter = "summary") }) ## Approx. 7 mins (as of 01.02.2024 and on a Laptop with an i7-12800H processor and compiled with MSVC)

# Evaluate coverage of R tests
system("Rscript build_r.R")
library(covr)
coverage  <- covr::package_coverage('./gpboost_r', quiet=FALSE)
print(coverage)
covr::report(coverage, browse = TRUE)

