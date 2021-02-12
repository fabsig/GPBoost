# Set working directory to GPBoost main folder

# Run unit tests locally
library(testthat)
library(gpboost)
Sys.setenv(GPBOOST_ALL_TESTS = "GPBOOST_ALL_TESTS")
path_tests = paste0(getwd(),.Platform$file.sep,file.path("R-package","tests","testthat"))
test_dir(path_tests, reporter = "summary")

# Evaluate coverage of R tests
system("Rscript build_r.R")
library(covr)
coverage  <- covr::package_coverage('./gpboost_r', quiet=FALSE)
print(coverage)
covr::report(coverage, browse = TRUE)

