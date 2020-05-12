# Set working directory to GPBoost main folder

# Evaluate coverage of R tests
system("Rscript build_r.R")
library(covr)
coverage  <- covr::package_coverage('./gpboost_r', quiet=FALSE)
print(coverage)
covr::report(coverage, browse = TRUE)

# Run tests locally
library(testthat)
library(gpboost)
path_tests = paste0(getwd(),.Platform$file.sep,file.path("R-package","tests","testthat"))
test_dir(path_tests, reporter = "summary")
