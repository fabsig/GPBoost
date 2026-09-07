# Set working directory to GPBoost main folder

Sys.setenv(OMP_NUM_THREADS = "10") # can be faster to limit the number of CPU threads

# Run unit tests locally
library(testthat)
library(gpboost)

# Set this to "true" only if the *installed* package was compiled with MSVC, the reference
# platform on which the expected values were calculated. With another compiler (an Rtools/gcc
# build, for instance) random and OMP parallel flukes do not lead to exactly the same results.
# Note that the value is read as a string, so use "true"/"false" and not TRUE/FALSE
Sys.setenv(GPBOOST_STRICT_TOLERANCES = "true")
Sys.setenv(GPBOOST_ALL_TESTS = "GPBOOST_ALL_TESTS")
# Sys.setenv(GPBOOST_ADDITIONAL_SLOW_TESTS = "GPBOOST_ADDITIONAL_SLOW_TESTS")
# Sys.setenv(NO_GPBOOST_ALGO_TESTS = "NO_GPBOOST_ALGO_TESTS") # If this is set, the (slow) GPBoost algorithm tests are not run

options(testthat.summary.max_reports = 100)
path_tests = paste0(getwd(),.Platform$file.sep,file.path("R-package","tests","testthat"))
system.time({ test_dir(path_tests, reporter = "summary") }) ## Approx. 7 mins (as of 01.02.2024 and on a Laptop with an i7-12800H processor and compiled with MSVC)




# Evaluate coverage of R tests
# 'covr' rebuilds the package from source with R's own toolchain, which on Windows is
# Rtools/gcc and never MSVC, so the strict expected values never apply to the coverage
# run even if the installed package above was built with MSVC
Sys.setenv(GPBOOST_STRICT_TOLERANCES = "false")
system("Rscript build_r.R")
library(covr)
coverage  <- covr::package_coverage('./gpboost_r', quiet=FALSE)
print(coverage)
covr::report(coverage, browse = TRUE)

