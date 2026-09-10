# Run unit tests locally

# print(sort(loadedNamespaces()))
if ("gpboost" %in% loadedNamespaces()) stop("gpboost already loaded - restart R before timing (OMP_NUM_THREADS would be ignored)")
Sys.setenv(OMP_NUM_THREADS = "10") # can be faster to limit the number of CPU threads
library(testthat)
library(gpboost)
# Set GPBOOST_STRICT_TOLERANCES to "true" only if the installed package was compiled with MSVC, the reference
# platform on which the expected values were calculated. With another compiler (an Rtools/gcc
# build, for instance) random and OMP parallel flukes do not lead to exactly the same results.
# Note that the value is read as a string, so use "true"/"false" and not TRUE/FALSE
Sys.setenv(GPBOOST_STRICT_TOLERANCES = "true")
Sys.setenv(GPBOOST_ALL_TESTS = "GPBOOST_ALL_TESTS")
# Sys.setenv(GPBOOST_ADDITIONAL_SLOW_TESTS = "GPBOOST_ADDITIONAL_SLOW_TESTS")
# Sys.setenv(NO_GPBOOST_ALGO_TESTS = "NO_GPBOOST_ALGO_TESTS") # If this is set, the (slow) GPBoost algorithm tests are not run
options(testthat.summary.max_reports = 100)

# Locate this script and use the GPBoost main folder as the working directory
gpb_main_folder <- function() {
  path <- NULL
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) { # Rscript
    path <- sub("^--file=", "", file_arg[1])
  } else { # source()
    for (i in rev(seq_len(sys.nframe()))) {
      if (!is.null(sys.frame(i)$ofile)) {
        path <- sys.frame(i)$ofile
        break
      }
    }
  }
  if (is.null(path) && requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    editor_path <- rstudioapi::getSourceEditorContext()$path # run from the RStudio editor
    if (nzchar(editor_path)) {
      path <- editor_path
    }
  }
  if (is.null(path)) {
    stop("cannot locate this script, set the working directory to the GPBoost main folder by hand")
  }
  main_folder <- normalizePath(file.path(dirname(path), ".."), winslash = "/", mustWork = FALSE)
  if (!dir.exists(file.path(main_folder, "R-package")) || !file.exists(file.path(main_folder, "build_r.R"))) {
    stop("'", main_folder, "' is not the GPBoost main folder (this script was located at '", path, "')")
  }
  main_folder
}
setwd(gpb_main_folder())
path_tests = file.path(getwd(), "R-package", "tests", "testthat")
# run tests
system.time({ test_dir(path_tests, reporter = "summary") }) ## Approx. 7 mins (as of 01.02.2024 and on a Laptop with an i7-12800H processor and compiled with MSVC)




# # Evaluate coverage of R tests
# # 'covr' rebuilds the package from source with R's own toolchain, which on Windows is
# # Rtools/gcc and never MSVC, so the strict expected values never apply to the coverage
# # run even if the installed package above was built with MSVC
# Sys.setenv(GPBOOST_STRICT_TOLERANCES = "false")
# system("Rscript build_r.R")
# library(covr)
# coverage  <- covr::package_coverage('./gpboost_r', quiet=FALSE)
# print(coverage)
# covr::report(coverage, browse = TRUE)

