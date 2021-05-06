# tools::package_native_routine_registration_skeleton('.',character_only = FALSE)

# Comment: need to use old version of roxygen2 (<=6.0.1), since newer versions produces error 
# when trying to compile, but we don't need that. It should simply create Rd files.
# (Error in getDLLRegisteredRoutines.DLLInfo(dll, addNames = FALSE : must specify DLL via a "DLLInfo" object. See getLoadedDLLs())
# (See https://github.com/r-lib/roxygen2/issues/822 and https://github.com/r-lib/roxygen2/issues/771)

###########
#  NEED TO RUN R VERSION 3.3.2 (to run this, since an old version Rcpp runs only on older R version)
###########

# remotes::install_version("Rcpp", "0.12.9")##Need old version of Rcpp for old version of roxygen2
# remotes::install_version("xml2", "1.0.0")##Need old version of xml2 for old version of Rcpp and roxygen2
# remotes::install_version("roxygen2", "6.0.1")

setwd("R-package")

library(roxygen2)
roxygen2::roxygenize() ## roclets="rd"





